import json
from datamodel import OrderDepth, TradingState, Order

POSITION_LIMIT = 10

MM_PRODUCTS = [
    'GALAXY_SOUNDS_DARK_MATTER', 'GALAXY_SOUNDS_PLANETARY_RINGS', 'GALAXY_SOUNDS_SOLAR_WINDS', 'GALAXY_SOUNDS_SOLAR_FLAMES',
    'SLEEP_POD_SUEDE', 'SLEEP_POD_LAMB_WOOL', 'SLEEP_POD_POLYESTER', 'SLEEP_POD_NYLON', 'SLEEP_POD_COTTON',
    'MICROCHIP_CIRCLE', 'MICROCHIP_SQUARE', 'MICROCHIP_RECTANGLE', 'MICROCHIP_TRIANGLE',
    'PEBBLES_S', 'PEBBLES_M', 'PEBBLES_L',
    'ROBOT_MOPPING', 'ROBOT_LAUNDRY',
    'UV_VISOR_YELLOW', 'UV_VISOR_ORANGE', 'UV_VISOR_RED', 'UV_VISOR_MAGENTA',
    'TRANSLATOR_SPACE_GRAY', 'TRANSLATOR_ASTRO_BLACK', 'TRANSLATOR_ECLIPSE_CHARCOAL', 'TRANSLATOR_GRAPHITE_MIST', 'TRANSLATOR_VOID_BLUE',
    'PANEL_1X2', 'PANEL_2X2', 'PANEL_1X4', 'PANEL_2X4', 'PANEL_4X4',
    'OXYGEN_SHAKE_MORNING_BREATH', 'OXYGEN_SHAKE_EVENING_BREATH', 'OXYGEN_SHAKE_MINT', 'OXYGEN_SHAKE_CHOCOLATE',
    'SNACKPACK_RASPBERRY',
]

LONG_PRODUCTS = [
    'OXYGEN_SHAKE_GARLIC',
    'GALAXY_SOUNDS_BLACK_HOLES',
    'PEBBLES_XL',
    'ROBOT_DISHES',
]

SHORT_PRODUCTS = [
    'MICROCHIP_OVAL',
    'PEBBLES_XS',
    'ROBOT_IRONING',
    'ROBOT_VACUUMING',
    'UV_VISOR_AMBER',
]

# Mean reversion pairs
# Spread trade: signal = VANILLA - CHOCOLATE (positively correlated, ~0.9)
SPREAD_PAIR   = ('SNACKPACK_VANILLA', 'SNACKPACK_CHOCOLATE')
SPREAD_THR    = 300    # open when |deviation| exceeds this — tune to ~1 std dev of the spread
SPREAD_CLOSE  = 5     # close when |deviation| falls `below this

# Sum trade: signal = PISTACHIO + STRAWBERRY (negatively correlated, ~-0.9)
SUM_PAIR      = ('SNACKPACK_PISTACHIO', 'SNACKPACK_STRAWBERRY')
SUM_MEAN      = 20_000  # fixed structural mean — no EMA needed
SUM_THR       = 300    # open when |deviation| exceeds this — tune to ~1 std dev of the sum
SUM_CLOSE     = 5     # close when |deviation| falls below this

MR_EMA_WINDOW = 50


def get_mid(od: OrderDepth):
    best_bid = max(od.buy_orders)  if od.buy_orders  else None
    best_ask = min(od.sell_orders) if od.sell_orders else None
    if best_bid and best_ask:
        return (best_bid + best_ask) / 2
    return best_bid or best_ask


def update_ema(td, key, value, window):
    alpha = 2 / (window + 1)
    old = td.get(key, value)  # seed with first observation
    new = alpha * value + (1 - alpha) * old
    td[key] = new
    return new


class Trader:

    def run(self, state: TradingState):
        result = {}

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        # ── market make ────────────────────────────────────────────────────────
        for product in MM_PRODUCTS:
            if product not in state.order_depths:
                continue

            od = state.order_depths[product]
            position = state.position.get(product, 0)

            buy_orders  = {p: abs(v) for p, v in od.buy_orders.items()}
            sell_orders = {p: abs(v) for p, v in od.sell_orders.items()}

            if not buy_orders or not sell_orders:
                continue

            best_bid = max(buy_orders)
            best_ask = min(sell_orders)

            if best_ask <= best_bid:
                continue

            bid_price = best_bid + 1
            ask_price = best_ask - 1

            if bid_price >= ask_price:
                continue

            orders = []
            if (cap := POSITION_LIMIT - position) > 0:
                orders.append(Order(product, bid_price,  cap))
            if (cap := POSITION_LIMIT + position) > 0:
                orders.append(Order(product, ask_price, -cap))

            result[product] = orders

        # ── directional ────────────────────────────────────────────────────────
        for product in LONG_PRODUCTS:
            if product not in state.order_depths:
                continue
            od = state.order_depths[product]
            position = state.position.get(product, 0)
            cap = POSITION_LIMIT - position
            if cap <= 0 or not od.sell_orders:
                continue
            result[product] = [Order(product, min(od.sell_orders), cap)]

        for product in SHORT_PRODUCTS:
            if product not in state.order_depths:
                continue
            od = state.order_depths[product]
            position = state.position.get(product, 0)
            cap = POSITION_LIMIT + position
            if cap <= 0 or not od.buy_orders:
                continue
            result[product] = [Order(product, max(od.buy_orders), -cap)]

        # ── mean reversion: spread (VANILLA - CHOCOLATE) ───────────────────────
        prod_a, prod_b = SPREAD_PAIR
        if prod_a in state.order_depths and prod_b in state.order_depths:
            mid_a = get_mid(state.order_depths[prod_a])
            mid_b = get_mid(state.order_depths[prod_b])

            if mid_a is not None and mid_b is not None:
                signal   = mid_a - mid_b
                ema      = update_ema(td, 'spread_ema', signal, MR_EMA_WINDOW)
                dev      = signal - ema

                od_a, od_b = state.order_depths[prod_a], state.order_depths[prod_b]
                pos_a = state.position.get(prod_a, 0)
                pos_b = state.position.get(prod_b, 0)

                if dev > SPREAD_THR:
                    # spread too wide: short A, long B
                    if (cap := POSITION_LIMIT + pos_a) > 0 and od_a.buy_orders:
                        result[prod_a] = [Order(prod_a, max(od_a.buy_orders), -cap)]
                    if (cap := POSITION_LIMIT - pos_b) > 0 and od_b.sell_orders:
                        result[prod_b] = [Order(prod_b, min(od_b.sell_orders), cap)]

                elif dev < -SPREAD_THR:
                    # spread too tight: long A, short B
                    if (cap := POSITION_LIMIT - pos_a) > 0 and od_a.sell_orders:
                        result[prod_a] = [Order(prod_a, min(od_a.sell_orders), cap)]
                    if (cap := POSITION_LIMIT + pos_b) > 0 and od_b.buy_orders:
                        result[prod_b] = [Order(prod_b, max(od_b.buy_orders), -cap)]

                elif abs(dev) < SPREAD_CLOSE:
                    # back near mean: close
                    if pos_a > 0 and od_a.buy_orders:
                        result[prod_a] = [Order(prod_a, max(od_a.buy_orders), -pos_a)]
                    elif pos_a < 0 and od_a.sell_orders:
                        result[prod_a] = [Order(prod_a, min(od_a.sell_orders), -pos_a)]
                    if pos_b > 0 and od_b.buy_orders:
                        result[prod_b] = [Order(prod_b, max(od_b.buy_orders), -pos_b)]
                    elif pos_b < 0 and od_b.sell_orders:
                        result[prod_b] = [Order(prod_b, min(od_b.sell_orders), -pos_b)]

        # ── mean reversion: sum (PISTACHIO + STRAWBERRY) ──────────────────────
        prod_a, prod_b = SUM_PAIR
        if prod_a in state.order_depths and prod_b in state.order_depths:
            mid_a = get_mid(state.order_depths[prod_a])
            mid_b = get_mid(state.order_depths[prod_b])

            if mid_a is not None and mid_b is not None:
                signal   = mid_a + mid_b
                dev      = signal - SUM_MEAN

                od_a, od_b = state.order_depths[prod_a], state.order_depths[prod_b]
                pos_a = state.position.get(prod_a, 0)
                pos_b = state.position.get(prod_b, 0)

                if dev > SUM_THR:
                    # sum too high: short both
                    if (cap := POSITION_LIMIT + pos_a) > 0 and od_a.buy_orders:
                        result[prod_a] = [Order(prod_a, max(od_a.buy_orders), -cap)]
                    if (cap := POSITION_LIMIT + pos_b) > 0 and od_b.buy_orders:
                        result[prod_b] = [Order(prod_b, max(od_b.buy_orders), -cap)]

                elif dev < -SUM_THR:
                    # sum too low: long both
                    if (cap := POSITION_LIMIT - pos_a) > 0 and od_a.sell_orders:
                        result[prod_a] = [Order(prod_a, min(od_a.sell_orders), cap)]
                    if (cap := POSITION_LIMIT - pos_b) > 0 and od_b.sell_orders:
                        result[prod_b] = [Order(prod_b, min(od_b.sell_orders), cap)]

                elif abs(dev) < SUM_CLOSE:
                    # back near mean: close
                    if pos_a > 0 and od_a.buy_orders:
                        result[prod_a] = [Order(prod_a, max(od_a.buy_orders), -pos_a)]
                    elif pos_a < 0 and od_a.sell_orders:
                        result[prod_a] = [Order(prod_a, min(od_a.sell_orders), -pos_a)]
                    if pos_b > 0 and od_b.buy_orders:
                        result[prod_b] = [Order(prod_b, max(od_b.buy_orders), -pos_b)]
                    elif pos_b < 0 and od_b.sell_orders:
                        result[prod_b] = [Order(prod_b, min(od_b.sell_orders), -pos_b)]

        return result, 0, json.dumps(td)
