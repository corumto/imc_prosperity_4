from datamodel import TradingState, Order
import json
import math

HYDROGEL_SYMBOL = "HYDROGEL_PACK"
HYDROGEL_POS_LIMIT = 50  # TODO: confirm from problem statement

# ── Parameters ────────────────────────────────────────────────────────────────
# EMA_WINDOW: lag-1 AC = -0.13 implies half-life ~5 ticks; use 3-5x that.
#             Shorter than v2's 30 so we track the tick-level reversion,
#             not the slow drift that made momentum look right at 30 ticks.
# ENTRY / EXIT: z-score thresholds — tune from EDA hold-time histogram.
HYDROGEL_EMA_WINDOW   = 20
HYDROGEL_ENTRY_ZSCORE = 1.5
HYDROGEL_EXIT_ZSCORE  = 0.25


# ── Base class ────────────────────────────────────────────────────────────────

class ProductTrader:
    def __init__(self, name: str, state: TradingState, new_trader_data: dict):
        self.name            = name
        self.state           = state
        self.new_trader_data = new_trader_data
        self.orders: list[Order] = []

        self.last_trader_data = self._load_trader_data()
        self.position_limit   = HYDROGEL_POS_LIMIT
        self.initial_position = state.position.get(name, 0)

        self.mkt_buy_orders, self.mkt_sell_orders = self._get_order_depth()

        bid_prices = sorted(self.mkt_buy_orders.keys(), reverse=True)
        ask_prices = sorted(self.mkt_sell_orders.keys())

        self.best_bid   = bid_prices[0] if bid_prices else None
        self.best_ask   = ask_prices[0] if ask_prices else None
        # Level-2 prices are the "outer walls" of the order book.
        # Fall back to level 1 when level 2 is absent.
        self.level2_bid = bid_prices[1] if len(bid_prices) >= 2 else self.best_bid
        self.level2_ask = ask_prices[1] if len(ask_prices) >= 2 else self.best_ask

        self.max_allowed_buy_volume  = self.position_limit - self.initial_position
        self.max_allowed_sell_volume = self.position_limit + self.initial_position

    def _load_trader_data(self) -> dict:
        try:
            if self.state.traderData:
                return json.loads(self.state.traderData)
        except Exception:
            pass
        return {}

    def _get_order_depth(self) -> tuple[dict[int, int], dict[int, int]]:
        od = self.state.order_depths.get(self.name)
        if od is None:
            return {}, {}
        buy  = {p: abs(v) for p, v in sorted(od.buy_orders.items(),  reverse=True)}
        sell = {p: abs(v) for p, v in sorted(od.sell_orders.items())}
        return buy, sell

    def bid(self, price: int, volume: int) -> None:
        vol = min(abs(int(volume)), self.max_allowed_buy_volume)
        if vol <= 0:
            return
        self.orders.append(Order(self.name, int(price), vol))
        self.max_allowed_buy_volume -= vol

    def ask(self, price: int, volume: int) -> None:
        vol = min(abs(int(volume)), self.max_allowed_sell_volume)
        if vol <= 0:
            return
        self.orders.append(Order(self.name, int(price), -vol))
        self.max_allowed_sell_volume -= vol


# ── Hydrogel trader ───────────────────────────────────────────────────────────

class HydrogelTrader(ProductTrader):
    """
    Mean-reversion trader using fair_value = (level2_bid + level2_ask) / 2.

    Why outer levels:
      The EDA shows that (bid_price_2 + ask_price_2) / 2 tracks the true
      mid more closely than the inner (bid1 + ask1) / 2. The inner mid is
      noisier because the best-bid/ask quotes move around more.

    Why mean-reversion (not momentum as in v2):
      v2 inverted the signal based on the Hurst result at the 30-tick window.
      But that made things worse, meaning the process is mean-reverting over
      the EMA horizon when using the cleaner fair_value signal. We also use a
      shorter window (20 ticks ≈ 4x the ~5-tick half-life implied by
      lag-1 AC = -0.13) so the EMA tracks the drift without lagging.
    """

    def __init__(self, state: TradingState, new_trader_data: dict):
        super().__init__(HYDROGEL_SYMBOL, state, new_trader_data)

        self.fair_value: float | None = None
        self.zscore: float | None = None

        if self.level2_bid is not None and self.level2_ask is not None:
            fv       = (self.level2_bid + self.level2_ask) / 2.0
            self.fair_value = fv
            ema_mean = self._ema("h_mean", HYDROGEL_EMA_WINDOW, fv)
            ema_var  = self._ema("h_var",  HYDROGEL_EMA_WINDOW, (fv - ema_mean) ** 2)
            std      = math.sqrt(max(ema_var, 1e-8))
            self.zscore = (fv - ema_mean) / std

    def _ema(self, key: str, window: int, value: float) -> float:
        alpha = 2.0 / (window + 1)
        prev  = self.last_trader_data.get(key, value)
        new   = alpha * value + (1.0 - alpha) * prev
        self.new_trader_data[key] = new
        return new

    def get_orders(self) -> dict[str, list[Order]]:
        if self.zscore is None:
            return {self.name: []}

        if self.state.timestamp < HYDROGEL_EMA_WINDOW * 100:
            return {self.name: []}

        z   = self.zscore
        pos = self.initial_position

        if z >= HYDROGEL_ENTRY_ZSCORE:
            # fair_value above EMA → expect reversion down → sell.
            self.ask(self.best_bid, self.max_allowed_sell_volume)

        elif z <= -HYDROGEL_ENTRY_ZSCORE:
            # fair_value below EMA → expect reversion up → buy.
            self.bid(self.best_ask, self.max_allowed_buy_volume)

        elif pos > 0 and z >= -HYDROGEL_EXIT_ZSCORE:
            # Long, price returned to/above mean → close.
            self.ask(self.best_bid, pos)

        elif pos < 0 and z <= HYDROGEL_EXIT_ZSCORE:
            # Short, price returned to/below mean → close.
            self.bid(self.best_ask, -pos)

        return {self.name: self.orders}


# ── Entry point ───────────────────────────────────────────────────────────────

class Trader:
    def run(self, state: TradingState):
        new_trader_data: dict = {}
        result: dict[str, list[Order]] = {}

        if HYDROGEL_SYMBOL in state.order_depths:
            try:
                result.update(HydrogelTrader(state, new_trader_data).get_orders())
            except Exception:
                result.setdefault(HYDROGEL_SYMBOL, [])

        try:
            trader_data = json.dumps(new_trader_data)
        except Exception:
            trader_data = ""

        return result, 0, trader_data
