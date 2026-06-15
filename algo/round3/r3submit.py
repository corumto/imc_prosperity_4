from datamodel import OrderDepth, TradingState, Order
import json
import math
import numpy as np
from math import erf, exp, log, sqrt

OPTION_UNDERLYING_SYMBOL = 'VELVETFRUIT_EXTRACT'

OPTION_SYMBOLS = [
    'VEV_5000', 'VEV_5100', 'VEV_5200',
    'VEV_5300', 'VEV_5400', 'VEV_5500'
]

OPTION_STRIKES = {
    'VEV_5000': 5000.0,
    'VEV_5100': 5100.0,
    'VEV_5200': 5200.0,
    'VEV_5300': 5300.0,
    'VEV_5400': 5400.0,
    'VEV_5500': 5500.0,
}

POS_LIMITS = {
    OPTION_UNDERLYING_SYMBOL: 200,
    **{os: 300 for os in OPTION_SYMBOLS},
}


class HydrogelTrader:
    PRODUCT = "HYDROGEL_PACK"
    POS_LIMIT = 200
    FAIR_VALUE = 10000.0

    ENTRY_THRESHOLD = 10.0
    MAX_TAKE_SIZE = 20

    MM_SIZE = 10
    MM_SKEW = 3

    def __init__(self, state: TradingState):
        self.state = state
        self.orders: list[Order] = []
        self.position = state.position.get(self.PRODUCT, 0)

    def run(self) -> dict[str, list[Order]]:
        if self.PRODUCT not in self.state.order_depths:
            return {}

        od = self.state.order_depths[self.PRODUCT]

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        taker_active = False

        if best_ask is not None and best_ask < self.FAIR_VALUE - self.ENTRY_THRESHOLD:
            taker_active = True
            room = self.POS_LIMIT - self.position
            for price in sorted(od.sell_orders.keys()):
                if price >= self.FAIR_VALUE - self.ENTRY_THRESHOLD or room <= 0:
                    break
                qty = min(-od.sell_orders[price], room, self.MAX_TAKE_SIZE)
                if qty > 0:
                    self.orders.append(Order(self.PRODUCT, price, qty))
                    room -= qty

        if best_bid is not None and best_bid > self.FAIR_VALUE + self.ENTRY_THRESHOLD:
            taker_active = True
            room = self.POS_LIMIT + self.position
            for price in sorted(od.buy_orders.keys(), reverse=True):
                if price <= self.FAIR_VALUE + self.ENTRY_THRESHOLD or room <= 0:
                    break
                qty = min(od.buy_orders[price], room, self.MAX_TAKE_SIZE)
                if qty > 0:
                    self.orders.append(Order(self.PRODUCT, price, -qty))
                    room -= qty

        if not taker_active and best_bid is not None and best_ask is not None:
            inv = self.position / self.POS_LIMIT
            skew = int(inv * self.MM_SKEW)

            mm_bid = best_bid + 1 - skew
            mm_ask = best_ask - 1 - skew

            bid_size = max(0, int(self.MM_SIZE * (1.0 - inv)))
            ask_size = max(0, int(self.MM_SIZE * (1.0 + inv)))

            if mm_bid < mm_ask:
                if bid_size > 0 and (self.POS_LIMIT - self.position) >= bid_size and mm_bid < best_ask:
                    self.orders.append(Order(self.PRODUCT, mm_bid, bid_size))

                if ask_size > 0 and (self.POS_LIMIT + self.position) >= ask_size and mm_ask > best_bid:
                    self.orders.append(Order(self.PRODUCT, mm_ask, -ask_size))

        return {self.PRODUCT: self.orders}


class OptionTrader:
    IV_SCALP_THRESHOLD = 0.02

    def __init__(self, option_symbol: str, underlying_symbol: str, state: TradingState,
                 new_trader_data: dict, strike: float, ttm_days: float):
        self.option_symbol = option_symbol
        self.underlying_symbol = underlying_symbol
        self.strike = strike
        self.ttm_days = ttm_days
        self.tte_years = ttm_days / 365.0

        self.state = state
        self.new_trader_data = new_trader_data
        self.orders: list[Order] = []

        self.position = self.state.position.get(self.option_symbol, 0)
        self.max_position = POS_LIMITS.get(self.option_symbol, 300)
        self.iv_model_poly = None

        self.option_mid_price = self._get_mid_price(self.option_symbol)
        self.underlying_mid_price = self._get_mid_price(self.underlying_symbol)

        self.last_trader_data = self._get_last_trader_data()

    def _get_last_trader_data(self) -> dict:
        try:
            if self.state.traderData:
                return json.loads(self.state.traderData)
        except Exception:
            pass
        return {}

    def _get_mid_price(self, symbol: str) -> float | None:
        order_depth: OrderDepth | None = self.state.order_depths.get(symbol)
        if order_depth is None:
            return None

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is not None and best_ask is not None:
            return 0.5 * (best_bid + best_ask)
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return None

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def _bs_call_price(self, spot: float, sigma: float, rate: float = 0.0) -> float:
        if sigma <= 0 or self.tte_years <= 0:
            return max(spot - self.strike, 0.0)

        vol_sqrt_t = sigma * sqrt(self.tte_years)
        d1 = (log(spot / self.strike) + (rate + 0.5 * sigma * sigma) * self.tte_years) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
        return spot * self._norm_cdf(d1) - self.strike * exp(-rate * self.tte_years) * self._norm_cdf(d2)

    def _implied_vol_from_price(self, price: float, spot: float,
                                 sigma_low: float = 0.0001, sigma_high: float = 5.0,
                                 max_iter: int = 200, tol: float = 1e-6) -> float | None:
        intrinsic = max(spot - self.strike, 0.0)
        time_value = price - intrinsic

        if time_value < 0.01:
            return None

        if not (intrinsic <= price <= spot):
            return None

        low, high = sigma_low, sigma_high
        f_low = self._bs_call_price(spot, low) - price
        f_high = self._bs_call_price(spot, high) - price

        for _ in range(12):
            if f_low * f_high <= 0:
                break
            high *= 1.5
            f_high = self._bs_call_price(spot, high) - price

        if f_low * f_high > 0:
            return None

        for _ in range(max_iter):
            mid = 0.5 * (low + high)
            f_mid = self._bs_call_price(spot, mid) - price

            if abs(f_mid) < tol or abs(high - low) < 1e-5:
                return mid

            if f_low * f_mid <= 0:
                high, f_high = mid, f_mid
            else:
                low, f_low = mid, f_mid

        return 0.5 * (low + high)

    def _predict_fair_iv(self, spot: float) -> float:
        if self.iv_model_poly is None:
            return 0.01

        moneyness = (self.strike / spot - 1.0) if spot > 0 else 0.0
        fair_iv = self.iv_model_poly(moneyness)
        return max(0.01, fair_iv)

    def execute_iv_scalp(self) -> None:
        if self.option_mid_price is None or self.underlying_mid_price is None:
            return

        market_iv = self._implied_vol_from_price(self.option_mid_price, self.underlying_mid_price)
        if market_iv is None:
            return

        fair_iv = self._predict_fair_iv(self.underlying_mid_price)
        iv_diff = market_iv - fair_iv

        if iv_diff < -self.IV_SCALP_THRESHOLD and self.position < self.max_position:
            bid_price = int(self.option_mid_price * 0.99)
            volume = 1
            self.orders.append(Order(self.option_symbol, bid_price, volume))
            self.position += volume

        elif iv_diff > self.IV_SCALP_THRESHOLD and self.position > -self.max_position:
            ask_price = int(self.option_mid_price * 1.01)
            volume = 1
            self.orders.append(Order(self.option_symbol, ask_price, -volume))
            self.position -= volume

    def get_orders(self) -> dict[str, list[Order]]:
        return {self.option_symbol: self.orders}

    def set_iv_model(self, poly) -> None:
        self.iv_model_poly = poly


class Trader:
    def __init__(self):
        self.iv_poly = np.poly1d([1.651954, 0.020983, 0.230133])
        self.ttm_days = 4.0

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        new_trader_data: dict = {}

        # Run Hydrogel trader
        hydro = HydrogelTrader(state)
        result.update(hydro.run())

        # Run option traders
        for option_symbol in OPTION_SYMBOLS:
            if option_symbol not in state.order_depths:
                continue

            try:
                strike = OPTION_STRIKES[option_symbol]
                trader = OptionTrader(
                    option_symbol=option_symbol,
                    underlying_symbol=OPTION_UNDERLYING_SYMBOL,
                    state=state,
                    new_trader_data=new_trader_data,
                    strike=strike,
                    ttm_days=self.ttm_days,
                )
                if self.iv_poly is not None:
                    trader.set_iv_model(self.iv_poly)
                trader.execute_iv_scalp()
                result.update(trader.get_orders())
            except Exception:
                result.setdefault(option_symbol, [])

        try:
            trader_data = json.dumps(new_trader_data)
        except Exception:
            trader_data = ""

        return result, 0, trader_data
