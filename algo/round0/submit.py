from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json


EMERALDS = "EMERALDS"
TOMATOES = "TOMATOES"

POS_LIMITS = {
    EMERALDS: 80,
    TOMATOES: 80,
}

DRAWDOWN_SOFT_LIMIT = 0.08
DRAWDOWN_HARD_LIMIT = 0.2
MIN_RISK_SCALE = 0.6


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

        self.drawdown_rate, self.risk_scale = self._get_drawdown_controls()
        self.passive_risk_scale = 1.0

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

    def _get_mark_to_market_pnl(self):
        mid = self.wall_mid
        if mid is None:
            if self.best_bid is not None and self.best_ask is not None:
                mid = 0.5 * (self.best_bid + self.best_ask)
            elif self.best_bid is not None:
                mid = float(self.best_bid)
            elif self.best_ask is not None:
                mid = float(self.best_ask)
            else:
                return None

        cash_ledger = getattr(self.state, "_cash", {})
        cash = float(cash_ledger.get(self.product, 0.0))
        position = float(self.state.position.get(self.product, 0))
        return cash + (position * float(mid))

    def _get_drawdown_controls(self):
        pnl = self._get_mark_to_market_pnl()
        if pnl is None:
            self._store_drawdown_state(0.0, 1.0, 0.0)
            return 0.0, 1.0

        last_drawdown_state = self.last_trader_data.get("drawdown", {}).get(self.product, {})
        previous_peak = float(last_drawdown_state.get("peak_pnl", pnl))
        peak_pnl = max(previous_peak, pnl)
        drawdown = max(0.0, peak_pnl - pnl)
        drawdown_rate = drawdown / max(abs(peak_pnl), 1.0)

        if drawdown_rate <= DRAWDOWN_SOFT_LIMIT:
            risk_scale = 1.0
        elif drawdown_rate >= DRAWDOWN_HARD_LIMIT:
            risk_scale = MIN_RISK_SCALE
        else:
            span = DRAWDOWN_HARD_LIMIT - DRAWDOWN_SOFT_LIMIT
            shrink = (drawdown_rate - DRAWDOWN_SOFT_LIMIT) / span
            risk_scale = 1.0 - (1.0 - MIN_RISK_SCALE) * shrink

        self._store_drawdown_state(peak_pnl, risk_scale, drawdown_rate)
        return drawdown_rate, risk_scale

    def _store_drawdown_state(self, peak_pnl: float, risk_scale: float, drawdown_rate: float):
        drawdown_state = self.new_trader_data.setdefault("drawdown", {})
        drawdown_state[self.product] = {
            "peak_pnl": peak_pnl,
            "drawdown_rate": drawdown_rate,
            "risk_scale": risk_scale,
        }

    def _get_max_allowed_volume(self):
        max_buy = self.position_limit - self.initial_position
        max_sell = self.position_limit + self.initial_position
        return max_buy, max_sell

    def bid(self, price: int, volume: int):
        size = min(max(int(round(volume * self.passive_risk_scale)), 0), self.max_allowed_buy)
        if size <= 0:
            return
        self.orders.append(Order(self.product, int(price), size))
        self.max_allowed_buy -= size

    def ask(self, price: int, volume: int):
        size = min(max(int(round(volume * self.passive_risk_scale)), 0), self.max_allowed_sell)
        if size <= 0:
            return
        self.orders.append(Order(self.product, int(price), -size))
        self.max_allowed_sell -= size

    def take_bid(self, price: int, volume: int):
        size = min(max(int(round(volume * self.risk_scale)), 0), self.max_allowed_buy)
        if size <= 0:
            return
        self.orders.append(Order(self.product, int(price), size))
        self.max_allowed_buy -= size

    def take_ask(self, price: int, volume: int):
        size = min(max(int(round(volume * self.risk_scale)), 0), self.max_allowed_sell)
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
                self.take_bid(ask_price, -ask_vol)

        for bid_price, bid_vol in self.buy_orders.items():
            if bid_price >= self.FAIR_VALUE + 1:
                self.take_ask(bid_price, bid_vol)

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
    EMA_KEY = "tomatoes_ema"
    EMA_WINDOW = 12
    OPEN_SPREAD = 1

    def __init__(self, state: TradingState, new_trader_data: Dict):
        super().__init__(TOMATOES, state, new_trader_data)

    def _estimate_fair(self):
        if self.best_bid is not None and self.best_ask is not None:
            mid = 0.5 * (self.best_bid + self.best_ask)
        elif self.best_bid is not None:
            mid = float(self.best_bid)
        elif self.best_ask is not None:
            mid = float(self.best_ask)
        else:
            return None
                
        prev_mid = self.last_trader_data.get("prev_mid")
        if prev_mid is not None:
            last_return = mid - prev_mid
            # fade the last move (AC = -0.4)
            adjustment = -0.4 * last_return
            fair = mid + adjustment
        else:
            fair = mid
        
        self.new_trader_data["prev_mid"] = mid
        return fair

    def get_orders(self) -> Dict[str, List[Order]]:
        fair = self._estimate_fair()
        if fair is None:
            return super().get_orders()

        # Take favorable prices around the estimated fair value.
        for ask_price, ask_vol in self.sell_orders.items():
            if ask_price <= fair - self.OPEN_SPREAD:
                self.take_bid(ask_price, -ask_vol)

        for bid_price, bid_vol in self.buy_orders.items():
            if bid_price >= fair + self.OPEN_SPREAD:
                self.take_ask(bid_price, bid_vol)

        # Quote passively near fair.
        quote_bid = int(fair - 1)
        quote_ask = int(fair + 1)

        if self.best_bid is not None:
            quote_bid = self.best_bid + 1
        if self.best_ask is not None:
            quote_ask = self.best_ask - 1

        if quote_bid < quote_ask:
            self.bid(quote_bid, self.max_allowed_buy)
            self.ask(quote_ask, self.max_allowed_sell)

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