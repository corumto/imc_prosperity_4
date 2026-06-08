from datamodel import TradingState, Order
import json

HYDROGEL_SYMBOL = "HYDROGEL_PACK"
HYDROGEL_POS_LIMIT = 50  # TODO: confirm from problem statement

# ── Parameters ────────────────────────────────────────────────────────────────
# QUOTE_EDGE   : ticks from fair_value to post on each side.
#                Inner spread is ~16 ticks (best_bid ~8 below fv, best_ask ~8
#                above). Posting at fv ± 2 puts us well inside the spread so
#                aggressive bots prefer our quotes over the existing walls.
# SKEW_FACTOR  : ticks of quote shift per unit of open position.
#                Shifts both quotes down when long (makes sells easier, buys
#                harder) to rebalance inventory and limit drift exposure.
QUOTE_EDGE   = 2
SKEW_FACTOR  = 0.1   # at max position (50) → 5-tick skew


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


# ── Market-making trader ──────────────────────────────────────────────────────

class HydrogelTrader(ProductTrader):
    """
    Passive market maker for HYDROGEL_PACK.

    Why passive (not directional z-score):
      All directional attempts (v1 mean-reversion, v2 momentum, v3 mean-
      reversion with outer levels) lost money because they paid the ~16-tick
      spread on every entry and exit. The reversion edge (~1-2 ticks from a
      lag-1 AC of -0.13) can never overcome that cost.

    How it works:
      1. Compute fair_value = (level2_bid + level2_ask) / 2.
         The outer walls are more stable than the inner best bid/ask.
      2. Post a passive BID at fair_value - QUOTE_EDGE and a passive ASK at
         fair_value + QUOTE_EDGE, both inside the existing spread.
      3. Aggressive bots and the exchange fill against our quotes; we collect
         the spread instead of paying it.
      4. Skew both quotes toward zero inventory to manage drift exposure
         (Hurst ≈ 0.98 means the price can trend across a whole day).
    """

    def get_orders(self) -> dict[str, list[Order]]:
        if self.level2_bid is None or self.level2_ask is None:
            return {self.name: []}

        fair_value = (self.level2_bid + self.level2_ask) / 2.0
        pos        = self.initial_position

        # Inventory skew: positive pos → shift quotes down (easier to sell).
        skew = round(-pos * SKEW_FACTOR)

        bid_price = round(fair_value) - QUOTE_EDGE + skew
        ask_price = round(fair_value) + QUOTE_EDGE + skew

        # Guard: quotes must not cross (can happen at extreme skew).
        if bid_price >= ask_price:
            bid_price = ask_price - 1

        # Guard: don't accidentally post an aggressive order that crosses the
        # existing book (would turn into an immediate market order).
        if self.best_ask is not None:
            bid_price = min(bid_price, self.best_ask - 1)
        if self.best_bid is not None:
            ask_price = max(ask_price, self.best_bid + 1)

        if self.max_allowed_buy_volume > 0:
            self.bid(bid_price, self.max_allowed_buy_volume)

        if self.max_allowed_sell_volume > 0:
            self.ask(ask_price, self.max_allowed_sell_volume)

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
