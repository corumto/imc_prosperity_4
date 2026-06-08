from datamodel import TradingState, Order
import json
import math
import numpy as np


OPTION_UNDERLYING_SYMBOL = "VELVETFRUIT_EXTRACT"
OPTION_SYMBOLS = [
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
]

OPTION_STRIKES = {
    "VEV_5000": 5000.0,
    "VEV_5100": 5100.0,
    "VEV_5200": 5200.0,
    "VEV_5300": 5300.0,
    "VEV_5400": 5400.0,
    "VEV_5500": 5500.0,
}

POS_LIMITS = {
    OPTION_UNDERLYING_SYMBOL: 200,
    **{symbol: 300 for symbol in OPTION_SYMBOLS},
}

# Best parameters from notebook optimization.
ENTRY_Z = 0.75
EXIT_Z = 0.50

WINDOW = 250
TARGET_UNIT = 30
MAX_STEP = 40

TTM_YEARS = 7.0 / 365.0
RISK_FREE_RATE = 0.0


class ProductTrader:
    def __init__(self, name: str, state: TradingState):
        self.name = name
        self.state = state
        self.orders: list[Order] = []

        self.position_limit = POS_LIMITS.get(name, 0)
        self.initial_position = state.position.get(name, 0)

        self.buy_orders, self.sell_orders = self._get_order_depth()
        bid_prices = sorted(self.buy_orders.keys(), reverse=True)
        ask_prices = sorted(self.sell_orders.keys())

        self.best_bid = bid_prices[0] if bid_prices else None
        self.best_ask = ask_prices[0] if ask_prices else None

        self.max_allowed_buy_volume = max(self.position_limit - self.initial_position, 0)
        self.max_allowed_sell_volume = max(self.position_limit + self.initial_position, 0)

    def _get_order_depth(self) -> tuple[dict[int, int], dict[int, int]]:
        od = self.state.order_depths.get(self.name)
        if od is None:
            return {}, {}
        buy = {price: abs(volume) for price, volume in sorted(od.buy_orders.items(), reverse=True)}
        sell = {price: abs(volume) for price, volume in sorted(od.sell_orders.items())}
        return buy, sell

    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return 0.5 * (self.best_bid + self.best_ask)

    def bid(self, price: int, volume: int) -> None:
        trade_volume = min(abs(int(volume)), self.max_allowed_buy_volume)
        if trade_volume <= 0:
            return
        self.orders.append(Order(self.name, int(price), trade_volume))
        self.max_allowed_buy_volume -= trade_volume

    def ask(self, price: int, volume: int) -> None:
        trade_volume = min(abs(int(volume)), self.max_allowed_sell_volume)
        if trade_volume <= 0:
            return
        self.orders.append(Order(self.name, int(price), -trade_volume))
        self.max_allowed_sell_volume -= trade_volume


def bs_call_price(spot: float, strike: float, ttm: float, sigma: float, rate: float = 0.0) -> float:
    if sigma <= 0.0 or ttm <= 0.0:
        return max(spot - strike, 0.0)

    vol_sqrt_t = sigma * math.sqrt(ttm)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * ttm) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    cdf_d1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    cdf_d2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
    return spot * cdf_d1 - strike * math.exp(-rate * ttm) * cdf_d2


def implied_vol_call(price: float, spot: float, strike: float, ttm: float) -> float | None:
    intrinsic = max(spot - strike * math.exp(-RISK_FREE_RATE * ttm), 0.0)
    if price <= intrinsic or price >= spot:
        return None

    low = 1e-4
    high = 5.0
    f_low = bs_call_price(spot, strike, ttm, low, RISK_FREE_RATE) - price
    f_high = bs_call_price(spot, strike, ttm, high, RISK_FREE_RATE) - price

    for _ in range(10):
        if f_low * f_high <= 0:
            break
        high *= 1.5
        f_high = bs_call_price(spot, strike, ttm, high, RISK_FREE_RATE) - price

    if f_low * f_high > 0:
        return None

    for _ in range(80):
        mid = 0.5 * (low + high)
        f_mid = bs_call_price(spot, strike, ttm, mid, RISK_FREE_RATE) - price
        if abs(f_mid) < 1e-6:
            return mid
        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    return 0.5 * (low + high)


class RelativeVolAlphaTrader:
    def __init__(self, state: TradingState, trader_data: dict):
        self.state = state
        self.trader_data = trader_data

        self.underlying = ProductTrader(OPTION_UNDERLYING_SYMBOL, state)
        self.option_traders = {symbol: ProductTrader(symbol, state) for symbol in OPTION_SYMBOLS}

        self.history = trader_data.setdefault("residual_history", {})
        self.positions_state = trader_data.setdefault("signal_pos", {})

    def _record_residual(self, symbol: str, residual: float) -> None:
        arr = self.history.setdefault(symbol, [])
        arr.append(float(residual))
        if len(arr) > WINDOW:
            self.history[symbol] = arr[-WINDOW:]

    def _signal_for_symbol(self, symbol: str, residual: float) -> float:
        # Winning family from notebook: mean-revert rel_vol_dev.
        signal_raw = -residual

        series = self.history.get(symbol, [])
        if len(series) < 20:
            return 0.0

        mu = float(np.mean(series))
        sd = float(np.std(series))
        if sd <= 1e-9:
            return 0.0

        z = max(-3.0, min(3.0, (signal_raw - mu) / sd))
        prev = float(self.positions_state.get(symbol, 0.0))

        if abs(z) >= ENTRY_Z:
            pos_signal = max(-2.0, min(2.0, z))
        elif abs(z) <= EXIT_Z:
            pos_signal = 0.0
        else:
            pos_signal = prev

        self.positions_state[symbol] = pos_signal
        return pos_signal

    def _passive_price(self, trader: ProductTrader, side: str) -> int | None:
        if trader.best_bid is None or trader.best_ask is None:
            return None

        if side == "buy":
            return min(trader.best_bid + 1, trader.best_ask - 1)
        return max(trader.best_ask - 1, trader.best_bid + 1)

    def get_orders(self) -> dict[str, list[Order]]:
        result: dict[str, list[Order]] = {symbol: [] for symbol in OPTION_SYMBOLS}

        spot = self.underlying.mid()
        if spot is None or spot <= 0:
            return result

        rows = []
        for symbol, trader in self.option_traders.items():
            opt_mid = trader.mid()
            if opt_mid is None or opt_mid <= 0:
                continue
            strike = OPTION_STRIKES[symbol]
            iv = implied_vol_call(opt_mid, spot, strike, TTM_YEARS)
            if iv is None:
                continue
            m = strike / spot - 1.0
            rows.append((symbol, m, iv))

        if len(rows) < 3:
            return result

        m_arr = np.array([row[1] for row in rows], dtype=float)
        iv_arr = np.array([row[2] for row in rows], dtype=float)
        a2, a1, a0 = np.polyfit(m_arr, iv_arr, 2)

        for symbol, moneyness, iv in rows:
            fitted = float(a2 * moneyness * moneyness + a1 * moneyness + a0)
            residual = iv - fitted
            self._record_residual(symbol, residual)

            signal = self._signal_for_symbol(symbol, residual)
            desired = int(round(signal * TARGET_UNIT))

            trader = self.option_traders[symbol]
            current = trader.initial_position
            delta = desired - current

            if delta > 0:
                buy_volume = min(delta, trader.max_allowed_buy_volume, MAX_STEP)
                price = self._passive_price(trader, "buy")
                if buy_volume > 0 and price is not None:
                    trader.bid(price, buy_volume)
            elif delta < 0:
                sell_volume = min(-delta, trader.max_allowed_sell_volume, MAX_STEP)
                price = self._passive_price(trader, "sell")
                if sell_volume > 0 and price is not None:
                    trader.ask(price, sell_volume)

            result[symbol] = trader.orders

        return result


class Trader:
    def run(self, state: TradingState):
        try:
            trader_data: dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_data = {}

        result: dict[str, list[Order]] = {}

        if OPTION_UNDERLYING_SYMBOL in state.order_depths:
            if all(symbol in state.order_depths for symbol in OPTION_SYMBOLS):
                try:
                    result.update(RelativeVolAlphaTrader(state, trader_data).get_orders())
                except Exception:
                    for symbol in OPTION_SYMBOLS:
                        result.setdefault(symbol, [])

        try:
            encoded = json.dumps(trader_data)
        except Exception:
            encoded = ""

        return result, 0, encoded