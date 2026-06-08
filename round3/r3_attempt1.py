from datamodel import OrderDepth, TradingState, Order
import json
import math
import numpy as np
from math import erf, exp, log, sqrt


OSMIUM_SYMBOL = "ASH_COATED_OSMIUM"
ROOT_SYMBOL = "INTARIAN_PEPPER_ROOT"

# Options trading constants
OPTION_UNDERLYING_SYMBOL = 'VELVETFRUIT_EXTRACT'
OPTION_SYMBOLS = [
    'VEV_4000', 'VEV_4500', 'VEV_5000', 'VEV_5100', 'VEV_5200',
    'VEV_5300', 'VEV_5400', 'VEV_5500', 'VEV_6000', 'VEV_6500'
]

# Strike prices matching option symbols
OPTION_STRIKES = {
    'VEV_4000': 4000.0,
    'VEV_4500': 4500.0,
    'VEV_5000': 5000.0,
    'VEV_5100': 5100.0,
    'VEV_5200': 5200.0,
    'VEV_5300': 5300.0,
    'VEV_5400': 5400.0,
    'VEV_5500': 5500.0,
    'VEV_6000': 6000.0,
    'VEV_6500': 6500.0,
}

POS_LIMITS = {
    OSMIUM_SYMBOL: 80,
    ROOT_SYMBOL: 80,
    OPTION_UNDERLYING_SYMBOL: 200,
    **{os: 300 for os in OPTION_SYMBOLS},
}

HYDROGEL_SYMBOL = "HYDROGEL_PACK"
HYDROGEL_POS_LIMIT = 50          # TODO: verify from contest problem statement
POS_LIMITS[HYDROGEL_SYMBOL] = HYDROGEL_POS_LIMIT

# ── Tune these three after running the EDA notebook ───────────────────────────
# EMA_WINDOW    : set to ~3-5× the half-life in ticks (Cell 6 output)
# ENTRY_ZSCORE  : z-score at which to enter a full position (Cell 7/9 output)
# EXIT_ZSCORE   : z-score at which to close (usually 0.0–0.3)
HYDROGEL_EMA_WINDOW   = 50
HYDROGEL_ENTRY_ZSCORE = 1.5
HYDROGEL_EXIT_ZSCORE  = 0.25


class ProductTrader:
    def __init__(self, name: str, state: TradingState, new_trader_data: dict):
        self.name = name
        self.state = state
        self.new_trader_data = new_trader_data
        self.orders: list[Order] = []
        self.far_bid_price: float | None = None
        self.center_bid_price: float | None = None
        self.far_ask_price: float | None = None
        self.center_ask_price: float | None = None
        self.alt_mid_price: float | None = None
        self.mid_price: float | None = None

        self.last_trader_data = self._get_last_trader_data()
        self._load_level_state()

        self.position_limit = POS_LIMITS.get(self.name, 0)
        self.initial_position = self.state.position.get(self.name, 0)

        self.mkt_buy_orders, self.mkt_sell_orders = self._get_order_depth()
        self.best_bid, self.best_ask = self._get_best_bid_ask()
        self.mid_price = self._compute_mid_price()
        self.reconstruct_side_levels()
        self._persist_level_state()

        self.max_allowed_buy_volume, self.max_allowed_sell_volume = self._get_max_allowed_volume()

    def _get_last_trader_data(self) -> dict:
        try:
            if self.state.traderData:
                return json.loads(self.state.traderData)
        except Exception:
            pass
        return {}

    def _get_order_depth(self) -> tuple[dict[int, int], dict[int, int]]:
        order_depth: OrderDepth | None = self.state.order_depths.get(self.name)
        if order_depth is None:
            return {}, {}

        buy_orders = {
            price: abs(volume)
            for price, volume in sorted(order_depth.buy_orders.items(), key=lambda item: item[0], reverse=True)
        }
        sell_orders = {
            price: abs(volume)
            for price, volume in sorted(order_depth.sell_orders.items(), key=lambda item: item[0])
        }

        return buy_orders, sell_orders

    def _load_level_state(self) -> None:
        levels_by_symbol = self.last_trader_data.get("levels", {})
        if not isinstance(levels_by_symbol, dict):
            return

        state = levels_by_symbol.get(self.name, {})
        if not isinstance(state, dict):
            return

        self.far_bid_price = state.get("far_bid_price")
        self.center_bid_price = state.get("center_bid_price")
        self.far_ask_price = state.get("far_ask_price")
        self.center_ask_price = state.get("center_ask_price")
        self.alt_mid_price = state.get("alt_mid_price")

    def _persist_level_state(self) -> None:
        levels_by_symbol = self.new_trader_data.setdefault("levels", {})
        if not isinstance(levels_by_symbol, dict):
            return

        levels_by_symbol[self.name] = {
            "far_bid_price": self.far_bid_price,
            "center_bid_price": self.center_bid_price,
            "far_ask_price": self.far_ask_price,
            "center_ask_price": self.center_ask_price,
            "alt_mid_price": self.alt_mid_price,
        }

    def _get_best_bid_ask(self) -> tuple[int | None, int | None]:
        best_bid = max(self.mkt_buy_orders.keys()) if self.mkt_buy_orders else None
        best_ask = min(self.mkt_sell_orders.keys()) if self.mkt_sell_orders else None
        return best_bid, best_ask

    def _compute_mid_price(self) -> float | None:
        """True mid price from the top-of-book best bid/ask.

        Returns the arithmetic midpoint between the best bid and best ask
        when both are available, otherwise falls back to whichever side is
        present, and finally None if the book is empty on both sides.
        """
        if self.best_bid is not None and self.best_ask is not None:
            return 0.5 * (self.best_bid + self.best_ask)
        if self.best_bid is not None:
            return float(self.best_bid)
        if self.best_ask is not None:
            return float(self.best_ask)
        return None

    def _get_max_allowed_volume(self) -> tuple[int, int]:
        max_allowed_buy_volume = self.position_limit - self.initial_position
        max_allowed_sell_volume = self.position_limit + self.initial_position
        return max_allowed_buy_volume, max_allowed_sell_volume

    def _reconstruct_one_side(
        self,
        side: str,
        prices: list[float],
        far_state: float | None,
        center_state: float | None,
    ) -> tuple[float | None, float | None]:
        values = sorted([price for price in prices if price > 0])
        far_value = far_state
        center_value = center_state

        if not values:
            return far_value, center_value

        pair_found = False
        if side == "ask" and len(values) >= 2:
            candidate_center = values[-2]
            candidate_far = values[-1]
            if candidate_far - candidate_center <= 3:
                far_value = candidate_far
                center_value = candidate_center
                pair_found = True
        elif side == "bid" and len(values) >= 2:
            candidate_far = values[0]
            candidate_center = values[1]
            if candidate_center - candidate_far <= 3:
                far_value = candidate_far
                center_value = candidate_center
                pair_found = True

        if not pair_found:
            candidate = values[-1] if side == "ask" else values[0]
            if far_state is None or center_state is None:
                far_value = candidate
                center_value = candidate
            elif abs(candidate - far_state) <= 1:
                far_value = candidate
            elif abs(candidate - center_state) <= 1:
                center_value = candidate

        return far_value, center_value

    def reconstruct_side_levels(self) -> None:
        bid_prices = [float(price) for price in list(self.mkt_buy_orders.keys())[:3]]
        ask_prices = [float(price) for price in list(self.mkt_sell_orders.keys())[:3]]

        self.far_bid_price, self.center_bid_price = self._reconstruct_one_side(
            "bid",
            bid_prices,
            self.far_bid_price,
            self.center_bid_price,
        )
        self.far_ask_price, self.center_ask_price = self._reconstruct_one_side(
            "ask",
            ask_prices,
            self.far_ask_price,
            self.center_ask_price,
        )

        bid_values = [v for v in [self.far_bid_price, self.center_bid_price] if v is not None]
        ask_values = [v for v in [self.far_ask_price, self.center_ask_price] if v is not None]

        bid_avg = (sum(bid_values) / len(bid_values)) if bid_values else None
        ask_avg = (sum(ask_values) / len(ask_values)) if ask_values else None

        if bid_avg is not None and ask_avg is not None:
            self.alt_mid_price = (bid_avg + ask_avg) / 2.0
        elif bid_avg is not None:
            self.alt_mid_price = bid_avg + 8
        elif ask_avg is not None:
            self.alt_mid_price = ask_avg - 8
        else:
            self.alt_mid_price = None

    def bid(self, price: int, volume: int) -> None:
        abs_volume = min(abs(int(volume)), self.max_allowed_buy_volume)
        if abs_volume <= 0:
            return
        self.orders.append(Order(self.name, int(price), abs_volume))
        self.max_allowed_buy_volume -= abs_volume

    def ask(self, price: int, volume: int) -> None:
        abs_volume = min(abs(int(volume)), self.max_allowed_sell_volume)
        if abs_volume <= 0:
            return
        self.orders.append(Order(self.name, int(price), -abs_volume))
        self.max_allowed_sell_volume -= abs_volume

    def get_orders(self) -> dict[str, list[Order]]:
        return {self.name: self.orders}


class OptionsTrader:
    """IV Scalping trader using polynomial smile model with TTE dependence."""

    # Polynomial model coefficients from notebook fit (degree 2)
    # IV = a0 + a1*moneyness + a2*moneyness^2 + a3*tte + a4*moneyness*tte + a5*moneyness^2*tte + a6*tte^2
    MODEL_COEFFS = {
        "intercept": 0.0,  # Will be updated from notebook data
        "moneyness": 0.0,
        "moneyness_sq": 0.0,
        "tte": 0.0,
        "moneyness_tte": 0.0,
        "moneyness_sq_tte": 0.0,
        "tte_sq": 0.0,
    }

    # IV scalping threshold (buy if market_iv < fair_iv - threshold, sell if market_iv > fair_iv + threshold)
    IV_SCALP_THRESHOLD = 0.01

    # Risk management
    MAX_POSITION = 50
    MAX_PNL_LOSS = 100

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
        self.pnl = 0.0

        self.option_mid_price = self._get_option_mid_price()
        self.underlying_mid_price = self._get_underlying_mid_price()

        self.last_trader_data = self._get_last_trader_data()
        self._load_position_state()

    def _get_last_trader_data(self) -> dict:
        try:
            if self.state.traderData:
                return json.loads(self.state.traderData)
        except Exception:
            pass
        return {}

    def _load_position_state(self) -> None:
        """Load position tracking from previous trading state."""
        options_state = self.last_trader_data.get("options", {})
        if isinstance(options_state, dict):
            self.pnl = options_state.get(f"{self.option_symbol}_pnl", 0.0)

    def _persist_position_state(self) -> None:
        """Save position tracking for next trading state."""
        options_state = self.new_trader_data.setdefault("options", {})
        options_state[f"{self.option_symbol}_pnl"] = self.pnl

    def _get_option_mid_price(self) -> float | None:
        order_depth: OrderDepth | None = self.state.order_depths.get(self.option_symbol)
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

    def _get_underlying_mid_price(self) -> float | None:
        order_depth: OrderDepth | None = self.state.order_depths.get(self.underlying_symbol)
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
        """Cumulative normal distribution."""
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def _bs_call_price(self, spot: float, sigma: float, rate: float = 0.0) -> float:
        """Black-Scholes call price."""
        if sigma <= 0 or self.tte_years <= 0:
            return max(spot - self.strike, 0.0)

        vol_sqrt_t = sigma * sqrt(self.tte_years)
        d1 = (log(spot / self.strike) + (rate + 0.5 * sigma * sigma) * self.tte_years) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
        return spot * self._norm_cdf(d1) - self.strike * exp(-rate * self.tte_years) * self._norm_cdf(d2)

    def _implied_vol_from_price(self, price: float, spot: float,
                                 sigma_low: float = 0.0001, sigma_high: float = 5.0,
                                 max_iter: int = 200, tol: float = 1e-6) -> float | None:
        """Extract implied vol from option price using bisection."""
        intrinsic = max(spot - self.strike, 0.0)
        time_value = price - intrinsic

        if time_value < 0.01:  # Too little time value
            return None

        if not (intrinsic <= price <= spot):
            return None

        low, high = sigma_low, sigma_high
        f_low = self._bs_call_price(spot, low) - price
        f_high = self._bs_call_price(spot, high) - price

        # Expand bracket if needed
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
        """Predict fair IV using 2D polynomial model with moneyness and TTE."""
        moneyness = (self.strike / spot - 1.0) if spot > 0 else 0.0

        # IV = a0 + a1*m + a2*m^2 + a3*tte + a4*m*tte + a5*m^2*tte + a6*tte^2
        fair_iv = (
            self.MODEL_COEFFS["intercept"] +
            self.MODEL_COEFFS["moneyness"] * moneyness +
            self.MODEL_COEFFS["moneyness_sq"] * (moneyness ** 2) +
            self.MODEL_COEFFS["tte"] * self.tte_years +
            self.MODEL_COEFFS["moneyness_tte"] * moneyness * self.tte_years +
            self.MODEL_COEFFS["moneyness_sq_tte"] * (moneyness ** 2) * self.tte_years +
            self.MODEL_COEFFS["tte_sq"] * (self.tte_years ** 2)
        )

        return max(0.01, fair_iv)  # Floor at 1% to avoid negative vol

    def execute_iv_scalp(self) -> None:
        """Execute IV scalping logic: buy cheap IV, sell expensive IV."""
        if self.option_mid_price is None or self.underlying_mid_price is None:
            return

        # Extract market IV from option price
        market_iv = self._implied_vol_from_price(self.option_mid_price, self.underlying_mid_price)
        if market_iv is None:
            return

        # Predict fair IV
        fair_iv = self._predict_fair_iv(self.underlying_mid_price)

        # IV scalping signals
        iv_diff = market_iv - fair_iv

        # BUY: Market IV too low relative to model prediction
        if iv_diff < -self.IV_SCALP_THRESHOLD and self.position < self.MAX_POSITION:
            bid_price = int(self.option_mid_price * 0.99)  # Bid slightly below mid
            volume = min(10, self.MAX_POSITION - self.position)
            self._bid(bid_price, volume)

        # SELL: Market IV too high relative to model prediction
        if iv_diff > self.IV_SCALP_THRESHOLD and self.position > -self.MAX_POSITION:
            ask_price = int(self.option_mid_price * 1.01)  # Ask slightly above mid
            volume = min(10, self.MAX_POSITION + self.position)
            self._ask(ask_price, volume)

        self._persist_position_state()

    def _bid(self, price: int, volume: int) -> None:
        """Place a buy order."""
        if volume <= 0:
            return
        self.orders.append(Order(self.option_symbol, price, volume))
        self.position += volume

    def _ask(self, price: int, volume: int) -> None:
        """Place a sell order."""
        if volume <= 0:
            return
        self.orders.append(Order(self.option_symbol, price, -volume))
        self.position -= volume

    def get_orders(self) -> dict[str, list[Order]]:
        return {self.option_symbol: self.orders}

    def update_model_coefficients(self, coeffs: dict[str, float]) -> None:
        """Update model coefficients from notebook data."""
        self.MODEL_COEFFS.update(coeffs)



class HydrogelTrader(ProductTrader):
    """
    Mean-reversion trader for HYDROGEL_PACK.

    Uses an online EMA ± EMA-variance z-score to decide when price has
    deviated far enough from fair value to enter, then exits when it
    returns to mean.

    Tune HYDROGEL_EMA_WINDOW, HYDROGEL_ENTRY_ZSCORE, HYDROGEL_EXIT_ZSCORE
    using the EDA output in round3_hydrogel.ipynb.
    """

    _WARMUP_TICKS = HYDROGEL_EMA_WINDOW  # don't trade until EMA is meaningful

    def __init__(self, state: TradingState, new_trader_data: dict):
        super().__init__(HYDROGEL_SYMBOL, state, new_trader_data)

        mid = self._current_mid()
        self.zscore: float | None = None

        if mid is not None:
            ema_mean = self._ema("hydro_ema_mean", HYDROGEL_EMA_WINDOW, mid)
            ema_var  = self._ema("hydro_ema_var",  HYDROGEL_EMA_WINDOW, (mid - ema_mean) ** 2)
            std = math.sqrt(max(ema_var, 1e-8))
            self.zscore = (mid - ema_mean) / std

    # ── helpers ───────────────────────────────────────────────────────────────

    def _current_mid(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return 0.5 * (self.best_bid + self.best_ask)
        return None

    def _ema(self, key: str, window: int, value: float) -> float:
        alpha = 2.0 / (window + 1)
        # Initialise to first observed value so the EMA doesn't start at 0.
        prev = self.last_trader_data.get(key, value)
        new  = alpha * value + (1.0 - alpha) * prev
        self.new_trader_data[key] = new
        return new

    def _ticks_elapsed(self) -> int:
        # Each timestamp step is 100 in Prosperity.
        return int(self.state.timestamp) // 100

    # ── strategy ──────────────────────────────────────────────────────────────

    def get_orders(self) -> dict[str, list[Order]]:
        if self.zscore is None or self._ticks_elapsed() < self._WARMUP_TICKS:
            return {self.name: []}

        z = self.zscore

        if z >= HYDROGEL_ENTRY_ZSCORE:
            # Price too high → sell, hit the best bid (take available liquidity).
            if self.best_bid is not None and self.max_allowed_sell_volume > 0:
                self.ask(self.best_bid, self.max_allowed_sell_volume)

        elif z <= -HYDROGEL_ENTRY_ZSCORE:
            # Price too low → buy, lift the best ask.
            if self.best_ask is not None and self.max_allowed_buy_volume > 0:
                self.bid(self.best_ask, self.max_allowed_buy_volume)

        elif abs(z) <= HYDROGEL_EXIT_ZSCORE:
            # Price near mean → close any open position.
            if self.initial_position > 0 and self.best_bid is not None:
                self.ask(self.best_bid, self.initial_position)
            elif self.initial_position < 0 and self.best_ask is not None:
                self.bid(self.best_ask, -self.initial_position)

        return {self.name: self.orders}


class Trader:
    def __init__(self):
        # Polynomial model coefficients from notebook (will be populated from smile fit data)
        # These should be updated with actual coefficients from your round3_smile.ipynb analysis
        self.model_coeffs = {
            "intercept": 0.28964286,  # TODO: Update with actual value from notebook
            "moneyness": 0.0,
            "moneyness_sq": -0.90852937,
            "tte": -0.18548491,
            "moneyness_tte": 1.70027269,
            "moneyness_sq_tte": 0.0,
            "tte_sq": 12.00763347,
        }
        # Time to maturity in days for all options (constant across options)
        self.ttm_days = 7.0

    def bid():
        return 0

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        new_trader_data: dict = {}

        product_traders = {
            HYDROGEL_SYMBOL: HydrogelTrader,
        }

        # Initialize and execute IV scalping for each option
        option_traders = []
        for option_symbol in OPTION_SYMBOLS:
            if option_symbol not in state.order_depths:
                continue

            try:
                strike = OPTION_STRIKES[option_symbol]
                trader = OptionsTrader(
                    option_symbol=option_symbol,
                    underlying_symbol=OPTION_UNDERLYING_SYMBOL,
                    state=state,
                    new_trader_data=new_trader_data,
                    strike=strike,
                    ttm_days=self.ttm_days,
                )
                # Update with model coefficients
                trader.update_model_coefficients(self.model_coeffs)
                # Execute IV scalping strategy
                trader.execute_iv_scalp()
                # Collect orders
                result.update(trader.get_orders())
                option_traders.append(trader)
            except Exception as e:
                result.setdefault(option_symbol, [])

        for symbol, product_trader in product_traders.items():
            if symbol not in state.order_depths:
                continue

            try:
                trader = product_trader(state, new_trader_data)
                result.update(trader.get_orders())
            except Exception:
                result.setdefault(symbol, [])

        try:
            trader_data = json.dumps(new_trader_data)
        except Exception:
            trader_data = ""

        conversions = 0
        return result, conversions, trader_data
