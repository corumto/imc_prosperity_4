from datamodel import TradingState, Order
import json


OPTION_UNDERLYING_SYMBOL = "VELVETFRUIT_EXTRACT"
OPTION_SYMBOLS = [
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
]

POS_LIMITS = {
    OPTION_UNDERLYING_SYMBOL: 200,
    **{symbol: 300 for symbol in OPTION_SYMBOLS},
}

TARGET_SYMBOL = "VEV_5400"
LEFT_HEDGE_SYMBOL = "VEV_5300"
RIGHT_HEDGE_SYMBOL = "VEV_5500"

ENTRY_EDGE = 3.0
EXIT_EDGE = 1.0
QUOTE_EDGE = 1
MAX_ORDER_SIZE = 8
MAX_HEDGE_REBALANCE = 2
AGGRESSIVE_HEDGE_TRIGGER = 2


class ProductTrader:
    def __init__(self, name: str, state: TradingState, new_trader_data: dict):
        self.name = name
        self.state = state
        self.new_trader_data = new_trader_data
        self.orders: list[Order] = []

        self.last_trader_data = self._load_trader_data()
        self.position_limit = POS_LIMITS.get(name, 0)
        self.initial_position = state.position.get(name, 0)

        self.mkt_buy_orders, self.mkt_sell_orders = self._get_order_depth()

        bid_prices = sorted(self.mkt_buy_orders.keys(), reverse=True)
        ask_prices = sorted(self.mkt_sell_orders.keys())

        self.best_bid = bid_prices[0] if bid_prices else None
        self.best_ask = ask_prices[0] if ask_prices else None

        self.max_allowed_buy_volume = max(self.position_limit - self.initial_position, 0)
        self.max_allowed_sell_volume = max(self.position_limit + self.initial_position, 0)

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

        buy = {price: abs(volume) for price, volume in sorted(od.buy_orders.items(), reverse=True)}
        sell = {price: abs(volume) for price, volume in sorted(od.sell_orders.items())}
        return buy, sell

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


class VEV5400Trader:
    """Trade the 5400 strike against a simple synthetic curve from 5300 and 5500.

    The notebook showed that the 5400 contract is consistently the quietest and
    cheapest strike in the smile. In live trading terms, that shows up as a
    persistent spread versus a nearby-strike interpolation:

        fair(5400) ~= 0.5 * mid(5300) + 0.5 * mid(5500)

    The target strike is traded directly, and neighbor strikes are used both
    for fair-value estimation and for inventory hedging.
    """

    def __init__(self, state: TradingState, new_trader_data: dict):
        self.state = state
        self.new_trader_data = new_trader_data

        self.target = ProductTrader(TARGET_SYMBOL, state, new_trader_data)
        self.left = ProductTrader(LEFT_HEDGE_SYMBOL, state, new_trader_data)
        self.right = ProductTrader(RIGHT_HEDGE_SYMBOL, state, new_trader_data)

    def _mid_price(self, trader: ProductTrader) -> float | None:
        if trader.best_bid is None or trader.best_ask is None:
            return None
        return (trader.best_bid + trader.best_ask) / 2.0

    def _synthetic_fair_value(self) -> float | None:
        left_mid = self._mid_price(self.left)
        right_mid = self._mid_price(self.right)
        if left_mid is None or right_mid is None:
            return None
        return 0.5 * left_mid + 0.5 * right_mid

    def _quote_prices(self, fair_value: float) -> tuple[int, int]:
        target_mid = self._mid_price(self.target)
        if target_mid is None:
            target_mid = fair_value

        inventory = self.target.initial_position
        inventory_skew = round(-inventory / 30)
        signal_skew = 0

        spread = fair_value - target_mid
        if spread > ENTRY_EDGE:
            signal_skew = 1
        elif spread < -ENTRY_EDGE:
            signal_skew = -1

        bid_price = round(fair_value) - QUOTE_EDGE + inventory_skew + signal_skew
        ask_price = round(fair_value) + QUOTE_EDGE + inventory_skew + signal_skew

        if self.target.best_ask is not None:
            bid_price = min(bid_price, self.target.best_ask - 1)
        if self.target.best_bid is not None:
            ask_price = max(ask_price, self.target.best_bid + 1)

        if bid_price >= ask_price:
            bid_price = ask_price - 1

        return bid_price, ask_price

    def _passive_hedge_prices(self, trader: ProductTrader) -> tuple[int | None, int | None]:
        bid_price = trader.best_bid
        ask_price = trader.best_ask

        if bid_price is None or ask_price is None:
            return None, None

        passive_bid = min(bid_price + 1, ask_price - 1)
        passive_ask = max(ask_price - 1, bid_price + 1)

        if passive_bid >= passive_ask:
            passive_bid = bid_price
            passive_ask = ask_price

        return passive_bid, passive_ask

    def _rebalance_hedges(self) -> None:
        # Hedge the projected 5400 exposure, including this tick's newly placed
        # target orders, so hedges can participate in the same timestamp.
        projected_target_pos = self.target.initial_position + sum(order.quantity for order in self.target.orders)

        target_pos = projected_target_pos
        desired_left = int(round(-target_pos / 2.0))
        desired_right = int(round(-target_pos / 2.0))

        left_delta = desired_left - self.left.initial_position
        right_delta = desired_right - self.right.initial_position

        left_bid, left_ask = self._passive_hedge_prices(self.left)
        right_bid, right_ask = self._passive_hedge_prices(self.right)

        if left_delta > 0 and left_bid is not None:
            volume = min(left_delta, self.left.max_allowed_buy_volume, MAX_HEDGE_REBALANCE)
            if volume > 0:
                price = self.left.best_ask if left_delta >= AGGRESSIVE_HEDGE_TRIGGER and self.left.best_ask is not None else left_bid
                self.left.bid(price, volume)
        elif left_delta < 0 and left_ask is not None:
            volume = min(-left_delta, self.left.max_allowed_sell_volume, MAX_HEDGE_REBALANCE)
            if volume > 0:
                price = self.left.best_bid if -left_delta >= AGGRESSIVE_HEDGE_TRIGGER and self.left.best_bid is not None else left_ask
                self.left.ask(price, volume)

        if right_delta > 0 and right_bid is not None:
            volume = min(right_delta, self.right.max_allowed_buy_volume, MAX_HEDGE_REBALANCE)
            if volume > 0:
                price = self.right.best_ask if right_delta >= AGGRESSIVE_HEDGE_TRIGGER and self.right.best_ask is not None else right_bid
                self.right.bid(price, volume)
        elif right_delta < 0 and right_ask is not None:
            volume = min(-right_delta, self.right.max_allowed_sell_volume, MAX_HEDGE_REBALANCE)
            if volume > 0:
                price = self.right.best_bid if -right_delta >= AGGRESSIVE_HEDGE_TRIGGER and self.right.best_bid is not None else right_ask
                self.right.ask(price, volume)

    def get_orders(self) -> dict[str, list[Order]]:
        fair_value = self._synthetic_fair_value()
        target_mid = self._mid_price(self.target)

        if fair_value is None or target_mid is None:
            return {
                TARGET_SYMBOL: [],
                LEFT_HEDGE_SYMBOL: [],
                RIGHT_HEDGE_SYMBOL: [],
            }

        spread = fair_value - target_mid

        bid_price, ask_price = self._quote_prices(fair_value)

        if spread > ENTRY_EDGE:
            buy_size = min(MAX_ORDER_SIZE, self.target.max_allowed_buy_volume)
            sell_size = min(2, self.target.max_allowed_sell_volume)
        elif spread < -ENTRY_EDGE:
            buy_size = min(2, self.target.max_allowed_buy_volume)
            sell_size = min(MAX_ORDER_SIZE, self.target.max_allowed_sell_volume)
        else:
            buy_size = min(3, self.target.max_allowed_buy_volume)
            sell_size = min(3, self.target.max_allowed_sell_volume)

        if buy_size > 0:
            self.target.bid(bid_price, buy_size)
        if sell_size > 0:
            self.target.ask(ask_price, sell_size)

        self._rebalance_hedges()

        self.new_trader_data["last_spread"] = spread

        return {
            TARGET_SYMBOL: self.target.orders,
            LEFT_HEDGE_SYMBOL: self.left.orders,
            RIGHT_HEDGE_SYMBOL: self.right.orders,
        }


class Trader:
    def run(self, state: TradingState):
        try:
            new_trader_data: dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            new_trader_data = {}
        result: dict[str, list[Order]] = {}

        if TARGET_SYMBOL in state.order_depths and LEFT_HEDGE_SYMBOL in state.order_depths and RIGHT_HEDGE_SYMBOL in state.order_depths:
            try:
                result.update(VEV5400Trader(state, new_trader_data).get_orders())
            except Exception:
                result.setdefault(TARGET_SYMBOL, [])
                result.setdefault(LEFT_HEDGE_SYMBOL, [])
                result.setdefault(RIGHT_HEDGE_SYMBOL, [])

        try:
            trader_data = json.dumps(new_trader_data)
        except Exception:
            trader_data = ""

        return result, 0, trader_data