from datamodel import TradingState, Order
import json
import math

HYDROGEL_SYMBOL = "HYDROGEL_PACK"
HYDROGEL_POS_LIMIT = 50  # TODO: confirm from problem statement

# ── Tune with EDA notebook output ────────────────────────────────────────────
# EMA_WINDOW    : momentum lookback window (Hurst ≈ 0.98 → trend at this scale)
# ENTRY_ZSCORE  : enter when price has trended this far above/below the EMA
# EXIT_ZSCORE   : exit when the trend has faded back to this z-score
HYDROGEL_EMA_WINDOW   = 30
HYDROGEL_ENTRY_ZSCORE = 1.5
HYDROGEL_EXIT_ZSCORE  = 0.25


# ── Minimal base class ────────────────────────────────────────────────────────

class ProductTrader:
    def __init__(self, name: str, state: TradingState, new_trader_data: dict):
        self.name             = name
        self.state            = state
        self.new_trader_data  = new_trader_data
        self.orders: list[Order] = []

        self.last_trader_data = self._load_trader_data()
        self.position_limit   = HYDROGEL_POS_LIMIT
        self.initial_position = state.position.get(name, 0)

        self.mkt_buy_orders, self.mkt_sell_orders = self._get_order_depth()
        self.best_bid = max(self.mkt_buy_orders)  if self.mkt_buy_orders  else None
        self.best_ask = min(self.mkt_sell_orders) if self.mkt_sell_orders else None

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


# ── Mean-reversion trader ─────────────────────────────────────────────────────

class HydrogelTrader(ProductTrader):
    """
    Trades HYDROGEL_PACK using momentum at the EMA time scale.

    EDA findings:
      - Hurst ≈ 0.98: at the 30-tick EMA scale price TRENDS, not reverts.
        Mean-reversion logic (short when z>0) shorts into momentum → loses.
      - Lag-1 AC = -0.13: tick-level reversion is real but below the EMA scale.
      - Correct read: when z >= ENTRY the price has been moving up and
        continues up → BUY. Exit when the trend fades (z returns toward 0).
    """

    def __init__(self, state: TradingState, new_trader_data: dict):
        super().__init__(HYDROGEL_SYMBOL, state, new_trader_data)

        self.zscore: float | None = None

        if self.best_bid is not None and self.best_ask is not None:
            mid      = 0.5 * (self.best_bid + self.best_ask)
            ema_mean = self._ema("h_mean", HYDROGEL_EMA_WINDOW, mid)
            ema_var  = self._ema("h_var",  HYDROGEL_EMA_WINDOW, (mid - ema_mean) ** 2)
            std      = math.sqrt(max(ema_var, 1e-8))
            self.zscore = (mid - ema_mean) / std

    def _ema(self, key: str, window: int, value: float) -> float:
        alpha = 2.0 / (window + 1)
        prev  = self.last_trader_data.get(key, value)  # init to current value, not 0
        new   = alpha * value + (1.0 - alpha) * prev
        self.new_trader_data[key] = new
        return new

    def get_orders(self) -> dict[str, list[Order]]:
        if self.zscore is None:
            return {self.name: []}

        # Wait for EMA variance to converge before trading.
        if self.state.timestamp < HYDROGEL_EMA_WINDOW * 100:
            return {self.name: []}

        z   = self.zscore
        pos = self.initial_position

        if z >= HYDROGEL_ENTRY_ZSCORE:
            # Trending up → go max long (closes any short automatically).
            self.bid(self.best_ask, self.max_allowed_buy_volume)

        elif z <= -HYDROGEL_ENTRY_ZSCORE:
            # Trending down → go max short (closes any long automatically).
            self.ask(self.best_bid, self.max_allowed_sell_volume)

        elif pos > 0 and z <= HYDROGEL_EXIT_ZSCORE:
            # Long, uptrend faded back → close.
            self.ask(self.best_bid, pos)

        elif pos < 0 and z >= -HYDROGEL_EXIT_ZSCORE:
            # Short, downtrend faded back → close.
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
