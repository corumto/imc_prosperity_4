from datamodel import OrderDepth, TradingState, Order
import json


OSMIUM_SYMBOL = "ASH_COATED_OSMIUM"
ROOT_SYMBOL = "INTARIAN_PEPPER_ROOT"

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
        self.far_bid_price: float | None = None
        self.center_bid_price: float | None = None
        self.far_ask_price: float | None = None
        self.center_ask_price: float | None = None
        self.alt_mid_price: float | None = None

        self.last_trader_data = self._get_last_trader_data()
        self._load_level_state()

        self.position_limit = POS_LIMITS.get(self.name, 0)
        self.initial_position = self.state.position.get(self.name, 0)

        self.mkt_buy_orders, self.mkt_sell_orders = self._get_order_depth()
        self.best_bid, self.best_ask = self._get_best_bid_ask()
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
        elif bid_avg is not None: # should only happen at very start
            self.alt_mid_price = bid_avg + 8
        elif ask_avg is not None: # should only happen at very start
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
    SPREAD_TAKE_MAX_DEVIATION = 10  # Max units to deviate from position_limit; tune as needed

    def __init__(self, state: TradingState, new_trader_data: dict):
        super().__init__(ROOT_SYMBOL, state, new_trader_data)
        root_state = self.last_trader_data.get("root", {})
        if not isinstance(root_state, dict):
            root_state = {}
        self.initial_buy_finished: bool = bool(root_state.get("initial_buy_finished", False))

    def _persist_root_state(self) -> None:
        root_state = self.new_trader_data.setdefault("root", {})
        if not isinstance(root_state, dict):
            return
        root_state["initial_buy_finished"] = self.initial_buy_finished

    def get_orders(self) -> dict[str, list[Order]]:
        if not self.initial_buy_finished and self.alt_mid_price is not None:
            threshold = self.alt_mid_price + 8
            for ask_price, ask_volume in self.mkt_sell_orders.items():
                if self.max_allowed_buy_volume <= 0:
                    break
                if ask_price < threshold:
                    self.bid(ask_price, ask_volume)
                else:
                    break

        if self.max_allowed_buy_volume <= 0:
            self.initial_buy_finished = True

        # Take spreads by selling when initial buy is finished
        if self.initial_buy_finished and self.center_ask_price is not None:
            # Sell quantity is capped by max deviation from position_limit
            spread_ask_qty = min(self.SPREAD_TAKE_MAX_DEVIATION, self.max_allowed_sell_volume)
            self.ask(self.center_ask_price - 1, spread_ask_qty)

        # Always provide liquidity on the bid side, 1 above center bid
        if self.center_bid_price is not None:
            self.bid(self.center_bid_price + 1, self.max_allowed_buy_volume)

        self._persist_root_state()

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