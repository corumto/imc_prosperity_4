from datamodel import TradingState, Order
import json
import math
from statistics import NormalDist

_N = NormalDist()

# ── Products ──────────────────────────────────────────────────────────────────
UNDERLYING = "VELVETFRUIT_EXTRACT"
OPTION_SYMBOLS = [
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5500",
]

POS_LIMITS = {
    UNDERLYING: 300,
    **{s: 200 for s in OPTION_SYMBOLS},
}

ORDER_LIMIT = 100  # max total order volume per tick per product

# ── Time ──────────────────────────────────────────────────────────────────────
DAYS_REMAINING = 7   # total days at start of round (day 0 tte = 8/365, day 3 tte = 5/365)
DAY            = 1   # which competition day we're running (update each submission)
DAYS_PER_YEAR  = 365

# ── Signal ────────────────────────────────────────────────────────────────────
BUY_THR   = -1.0   # buy when option_price − bs(cma_iv) < BUY_THR
SELL_THR  =  1.0   # sell when option_price − bs(cma_iv) > SELL_THR
CLOSE_THR =  0.0   # close when signal reverts past zero

MIN_SAMPLES = 20   # ignore signal until CMA has this many data points


# ── Black-Scholes helpers ─────────────────────────────────────────────────────
def bs_call(S, K, tte, sigma, r=0.0):
    if sigma <= 0 or tte <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * tte) / (sigma * math.sqrt(tte))
    d2 = d1 - sigma * math.sqrt(tte)
    return S * _N.cdf(d1) - K * math.exp(-r * tte) * _N.cdf(d2)


def implied_vol(price, S, K, tte, r=0.0):
    intrinsic = max(S - K * math.exp(-r * tte), 0.0)
    if not (intrinsic < price < S):
        return float("nan")
    lo, hi = 1e-4, 3.0
    fl = bs_call(S, K, tte, lo, r) - price
    fh = bs_call(S, K, tte, hi, r) - price
    if fl * fh > 0:
        return float("nan")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        fm = bs_call(S, K, tte, mid, r) - price
        if abs(fm) < 1e-6:
            return mid
        if fl * fm <= 0:
            hi, fh = mid, fm
        else:
            lo, fl = mid, fm
    return 0.5 * (lo + hi)


# ── Trader ────────────────────────────────────────────────────────────────────
class VEVOptionTrader:

    def __init__(self, state: TradingState, new_trader_data: dict):
        self.state = state
        self.new_trader_data = new_trader_data
        self.td = self._load_td()
        # Carry forward all stored state so nothing is dropped this tick
        self.new_trader_data.update(self.td)

    def _load_td(self):
        try:
            if self.state.traderData:
                return json.loads(self.state.traderData)
        except:
            pass
        return {}

    def _mid(self, symbol):
        od = self.state.order_depths.get(symbol)
        if od is None:
            return None
        bids, asks = od.buy_orders, od.sell_orders
        if not bids or not asks:
            return None
        return (max(bids) + min(asks)) / 2

    def _best_bid_ask(self, symbol):
        od = self.state.order_depths.get(symbol)
        if od is None:
            return None, None
        bids, asks = od.buy_orders, od.sell_orders
        return (max(bids) if bids else None), (min(asks) if asks else None)

    def _update_cma(self, key, value):
        cma, n = self.td.get(key, [0.0, 0])
        n += 1
        cma = cma + (value - cma) / n
        self.new_trader_data[key] = [cma, n]
        return cma, n

    def _tte(self):
        return (DAYS_REMAINING - DAY - self.state.timestamp / 1_000_000) / DAYS_PER_YEAR

    def get_orders(self):
        orders = {}

        spot = self._mid(UNDERLYING)
        if spot is None:
            return orders

        tte = self._tte()
        if tte <= 0:
            return orders

        for symbol in OPTION_SYMBOLS:
            strike = float(symbol.split("_")[-1])
            option_price = self._mid(symbol)
            if option_price is None:
                continue

            # Back-solve IV and update cumulative mean
            iv = implied_vol(option_price, spot, strike, tte)
            if math.isnan(iv):
                continue

            cma_iv, n = self._update_cma(f"{symbol}_cma", iv)
            if n < MIN_SAMPLES:
                continue

            # Price signal: observed - fair value at cumulative mean IV
            fair = bs_call(spot, strike, tte, cma_iv)
            resid = option_price - fair

            pos = self.state.position.get(symbol, 0)
            lim = POS_LIMITS.get(symbol, 200)
            best_bid, best_ask = self._best_bid_ask(symbol)

            option_orders = []

            # Close: use mid-based signal, hit the market to exit promptly
            if pos > 0 and resid >= CLOSE_THR and best_bid is not None:
                qty = min(pos, ORDER_LIMIT)
                option_orders.append(Order(symbol, best_bid, -qty))

            elif pos < 0 and resid <= CLOSE_THR and best_ask is not None:
                qty = min(-pos, ORDER_LIMIT)
                option_orders.append(Order(symbol, best_ask, qty))

            else:
                # Open: check execution prices directly so the edge clears the spread.
                # Buy only if the ask itself is below fair + BUY_THR (= fair - 1).
                # Sell only if the bid itself is above fair + SELL_THR (= fair + 1).
                if best_ask is not None and best_ask < fair + BUY_THR and pos < lim:
                    qty = min(lim - pos, ORDER_LIMIT)
                    option_orders.append(Order(symbol, best_ask, qty))

                if best_bid is not None and best_bid > fair + SELL_THR and pos > -lim:
                    qty = min(lim + pos, ORDER_LIMIT)
                    option_orders.append(Order(symbol, best_bid, -qty))

            if option_orders:
                orders[symbol] = option_orders

        return orders


class Trader:

    def run(self, state: TradingState):
        new_trader_data = {}
        trader = VEVOptionTrader(state, new_trader_data)
        result = trader.get_orders()

        try:
            final_td = json.dumps(new_trader_data)
        except:
            final_td = ""

        return result, 0, final_td
