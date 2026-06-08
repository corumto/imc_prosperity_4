from datamodel import OrderDepth, TradingState, Order
import json


OSMIUM_SYMBOL = "ASH_COATED_OSMIUM"
ROOT_SYMBOL = "INTARIAN_PEPER_ROOT"

POS_LIMITS = {
    OSMIUM_SYMBOL: 80,
    ROOT_SYMBOL: 80,
}


class ProductTrader:
    def __init__(self, name: str, state: TradingState, new_trader_data: dict):
        self.name = name
        self.state = state
        self.new_trader_data = new_trader_data
        self.orders: list[Order] = []

        self.last_trader_data = self._get_last_trader_data()

        self.position_limit = POS_LIMITS.get(self.name, 0)
        self.initial_position = self.state.position.get(self.name, 0)

        self.mkt_buy_orders, self.mkt_sell_orders = self._get_order_depth()
        self.best_bid, self.best_ask = self._get_best_bid_ask()

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

    def _get_best_bid_ask(self) -> tuple[int | None, int | None]:
        best_bid = max(self.mkt_buy_orders.keys()) if self.mkt_buy_orders else None
        best_ask = min(self.mkt_sell_orders.keys()) if self.mkt_sell_orders else None
        return best_bid, best_ask

    def _get_max_allowed_volume(self) -> tuple[int, int]:
        max_allowed_buy_volume = self.position_limit - self.initial_position
        max_allowed_sell_volume = self.position_limit + self.initial_position
        return max_allowed_buy_volume, max_allowed_sell_volume

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


class OsmiumTrader(ProductTrader):
    def __init__(self, state: TradingState, new_trader_data: dict):
        super().__init__(OSMIUM_SYMBOL, state, new_trader_data)

    def get_orders(self) -> dict[str, list[Order]]:
        fair_value = 10_000

        for ask_price, ask_volume in self.mkt_sell_orders.items():
            if ask_price < fair_value:
                self.bid(ask_price, ask_volume)
            else:
                break

        for bid_price, bid_volume in self.mkt_buy_orders.items():
            if bid_price > fair_value:
                self.ask(bid_price, bid_volume)
            else:
                break

        return {self.name: self.orders}


class RootTrader(ProductTrader):
    def __init__(self, state: TradingState, new_trader_data: dict):
        super().__init__(ROOT_SYMBOL, state, new_trader_data)

    def get_orders(self) -> dict[str, list[Order]]:
        # Long-only: buy immediately and keep accumulating up to the position limit.
        for ask_price, ask_volume in self.mkt_sell_orders.items():
            if self.max_allowed_buy_volume <= 0:
                break
            self.bid(ask_price, ask_volume)

        return {self.name: self.orders}


class Trader:
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        new_trader_data: dict = {}

        product_traders = {
            OSMIUM_SYMBOL: OsmiumTrader,
            ROOT_SYMBOL: RootTrader,
        }

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