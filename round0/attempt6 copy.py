from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json
import math


EMERALDS = "EMERALDS"
TOMATOES = "TOMATOES"

POS_LIMITS = {
    EMERALDS: 80,
    TOMATOES: 80,
}


class ProductTrader:
    def __init__(self, product: str, state: TradingState, new_trader_data: Dict):
        self.product = product
        self.state = state
        self.new_trader_data = new_trader_data
        self.orders: List[Order] = []

        self.last_trader_data = self._load_last_trader_data()
        self.position_limit = POS_LIMITS.get(product, 0)
        self.initial_position = int(self.state.position.get(product, 0))

        self.buy_orders, self.sell_orders = self._get_sorted_order_depth()
        self.best_bid = max(self.buy_orders.keys()) if self.buy_orders else None
        self.best_ask = min(self.sell_orders.keys()) if self.sell_orders else None
        self.wall_mid = self._get_wall_mid()

        self.max_allowed_buy, self.max_allowed_sell = self._get_max_allowed_volume()

    def _load_last_trader_data(self) -> Dict:
        if not self.state.traderData:
            return {}
        try:
            return json.loads(self.state.traderData)
        except Exception:
            return {}

    def _get_sorted_order_depth(self):
        order_depth: OrderDepth = self.state.order_depths.get(self.product, OrderDepth())
        buy_orders = dict(sorted(order_depth.buy_orders.items(), key=lambda kv: kv[0], reverse=True))
        sell_orders = dict(sorted(order_depth.sell_orders.items(), key=lambda kv: kv[0]))
        return buy_orders, sell_orders

    def _get_wall_mid(self):
        if not self.buy_orders or not self.sell_orders:
            return None
        bid_wall = min(self.buy_orders.keys())
        ask_wall = max(self.sell_orders.keys())
        return 0.5 * (bid_wall + ask_wall)

    def _get_max_allowed_volume(self):
        max_buy = self.position_limit - self.initial_position
        max_sell = self.position_limit + self.initial_position
        return max_buy, max_sell

    def bid(self, price: int, volume: int):
        size = min(max(int(volume), 0), self.max_allowed_buy)
        if size <= 0:
            return
        self.orders.append(Order(self.product, int(price), size))
        self.max_allowed_buy -= size

    def ask(self, price: int, volume: int):
        size = min(max(int(volume), 0), self.max_allowed_sell)
        if size <= 0:
            return
        self.orders.append(Order(self.product, int(price), -size))
        self.max_allowed_sell -= size

    def get_orders(self) -> Dict[str, List[Order]]:
        return {self.product: self.orders}


class EmeraldTrader(ProductTrader):
    FAIR_VALUE = 10000

    def __init__(self, state: TradingState, new_trader_data: Dict):
        super().__init__(EMERALDS, state, new_trader_data)

    def get_orders(self) -> Dict[str, List[Order]]:
        # Take favorable liquidity.
        for ask_price, ask_vol in self.sell_orders.items():
            if ask_price <= self.FAIR_VALUE - 1:
                self.bid(ask_price, -ask_vol)

        for bid_price, bid_vol in self.buy_orders.items():
            if bid_price >= self.FAIR_VALUE + 1:
                self.ask(bid_price, bid_vol)

        # Quote around fair value.
        quote_bid = self.FAIR_VALUE - 1
        quote_ask = self.FAIR_VALUE + 1

        if self.best_bid is not None and self.best_bid < self.FAIR_VALUE:
            quote_bid = self.best_bid + 1
        if self.best_ask is not None and self.best_ask > self.FAIR_VALUE:
            quote_ask = self.best_ask - 1

        self.bid(quote_bid, self.max_allowed_buy)
        self.ask(quote_ask, self.max_allowed_sell)

        return super().get_orders()


class TomatoTrader(ProductTrader):
    PREV_MID_KEY = "tomatoes_prev_mid"
    PREV_RET_KEY = "tomatoes_prev_ret"
    AR_NUM_KEY = "tomatoes_ar_num"
    AR_DEN_KEY = "tomatoes_ar_den"
    RET_VAR_KEY = "tomatoes_ret_var"

    AR_UPDATE_BETA = 0.04
    DEFAULT_PHI = -0.4
    ENTRY_Z = 1.0
    OPEN_SPREAD = 1

    def __init__(self, state: TradingState, new_trader_data: Dict):
        super().__init__(TOMATOES, state, new_trader_data)

    def _get_mid(self):
        if self.buy_orders and self.sell_orders:
            worst_bid = min(self.buy_orders.keys())
            worst_ask = max(self.sell_orders.keys())
            return 0.5 * (worst_bid + worst_ask)
        return None

    def _estimate_fair(self):
        mid = self._get_mid()
        if mid is None:
            return None, 0.0

        prev_mid = self.last_trader_data.get(self.PREV_MID_KEY)
        if prev_mid is None:
            self.new_trader_data[self.PREV_MID_KEY] = mid
            self.new_trader_data[self.PREV_RET_KEY] = 0.0
            self.new_trader_data[self.AR_NUM_KEY] = 0.0
            self.new_trader_data[self.AR_DEN_KEY] = 1.0
            self.new_trader_data[self.RET_VAR_KEY] = 1.0
            return mid, 0.0

        ret = mid - float(prev_mid)
        prev_ret = self.last_trader_data.get(self.PREV_RET_KEY)

        ar_num = float(self.last_trader_data.get(self.AR_NUM_KEY, 0.0))
        ar_den = float(self.last_trader_data.get(self.AR_DEN_KEY, 1.0))
        ret_var = float(self.last_trader_data.get(self.RET_VAR_KEY, 1.0))

        beta = self.AR_UPDATE_BETA
        ret_var = (1.0 - beta) * ret_var + beta * (ret * ret)

        if prev_ret is not None:
            lag = float(prev_ret)
            ar_num = (1.0 - beta) * ar_num + beta * (ret * lag)
            ar_den = (1.0 - beta) * ar_den + beta * (lag * lag)

        if ar_den <= 1e-9:
            phi = self.DEFAULT_PHI
        else:
            phi = ar_num / ar_den

        phi = max(-0.95, min(0.0, phi))
        pred_ret = phi * ret
        fair = mid + pred_ret

        ret_std = math.sqrt(max(ret_var, 1e-9))
        if ret_std <= 1e-9:
            strength = 0.0
        else:
            strength = min(1.0, abs(pred_ret) / (self.ENTRY_Z * ret_std))

        self.new_trader_data[self.PREV_MID_KEY] = mid
        self.new_trader_data[self.PREV_RET_KEY] = ret
        self.new_trader_data[self.AR_NUM_KEY] = ar_num
        self.new_trader_data[self.AR_DEN_KEY] = ar_den
        self.new_trader_data[self.RET_VAR_KEY] = ret_var
        self.new_trader_data["tomatoes_phi"] = phi
        self.new_trader_data["tomatoes_pred_ret"] = pred_ret

        return fair, strength

    def get_orders(self) -> Dict[str, List[Order]]:
        fair, strength = self._estimate_fair()
        if fair is None:
            return super().get_orders()

        take_buy_cap = int(round(self.max_allowed_buy * strength))
        take_sell_cap = int(round(self.max_allowed_sell * strength))

        # Take favorable prices around the estimated fair value.
        for ask_price, ask_vol in self.sell_orders.items():
            if ask_price <= fair - self.OPEN_SPREAD and take_buy_cap > 0:
                size = min(-ask_vol, take_buy_cap)
                self.bid(ask_price, size)
                take_buy_cap -= size

        for bid_price, bid_vol in self.buy_orders.items():
            if bid_price >= fair + self.OPEN_SPREAD and take_sell_cap > 0:
                size = min(bid_vol, take_sell_cap)
                self.ask(bid_price, size)
                take_sell_cap -= size

        # Quote passively near fair.
        quote_bid = int(fair - 1)
        quote_ask = int(fair + 1)

        if self.best_bid is not None:
            quote_bid = self.best_bid + 1
        if self.best_ask is not None:
            quote_ask = self.best_ask - 1

        if quote_bid < quote_ask:
            passive_scale = 0.25 + 0.75 * strength
            self.bid(quote_bid, int(round(self.max_allowed_buy * passive_scale)))
            self.ask(quote_ask, int(round(self.max_allowed_sell * passive_scale)))

        return super().get_orders()


class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        new_trader_data: Dict = {}

        trader_classes = {
            EMERALDS: EmeraldTrader,
            TOMATOES: TomatoTrader,
        }

        for symbol, trader_cls in trader_classes.items():
            if symbol in state.order_depths:
                result.update(trader_cls(state, new_trader_data).get_orders())

        trader_data = json.dumps(new_trader_data)
        conversions = 0
        return result, conversions, trader_data