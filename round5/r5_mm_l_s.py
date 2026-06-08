from datamodel import OrderDepth, TradingState, Order

POSITION_LIMIT = 10


MM_PRODUCTS = [
    'GALAXY_SOUNDS_DARK_MATTER', 'GALAXY_SOUNDS_PLANETARY_RINGS', 'GALAXY_SOUNDS_SOLAR_WINDS', 'GALAXY_SOUNDS_SOLAR_FLAMES',
    'SLEEP_POD_SUEDE', 'SLEEP_POD_LAMB_WOOL', 'SLEEP_POD_POLYESTER', 'SLEEP_POD_NYLON', 'SLEEP_POD_COTTON',
    'MICROCHIP_CIRCLE', 'MICROCHIP_SQUARE', 'MICROCHIP_RECTANGLE', 'MICROCHIP_TRIANGLE',
    'PEBBLES_S', 'PEBBLES_M', 'PEBBLES_L'
    'ROBOT_MOPPING', 'ROBOT_LAUNDRY',
    'UV_VISOR_YELLOW', 'UV_VISOR_ORANGE', 'UV_VISOR_RED', 'UV_VISOR_MAGENTA',
    'TRANSLATOR_SPACE_GRAY', 'TRANSLATOR_ASTRO_BLACK', 'TRANSLATOR_ECLIPSE_CHARCOAL', 'TRANSLATOR_GRAPHITE_MIST', 'TRANSLATOR_VOID_BLUE',
    'PANEL_1X2', 'PANEL_2X2', 'PANEL_1X4', 'PANEL_2X4', 'PANEL_4X4',
    'OXYGEN_SHAKE_MORNING_BREATH', 'OXYGEN_SHAKE_EVENING_BREATH', 'OXYGEN_SHAKE_MINT', 'OXYGEN_SHAKE_CHOCOLATE',
    'SNACKPACK_CHOCOLATE', 'SNACKPACK_VANILLA' , 'SNACKPACK_PISTACHIO', 'SNACKPACK_STRAWBERRY',
    'SNACKPACK_RASPBERRY'
]
LONG_PRODUCTS = [
    'OXYGEN_SHAKE_GARLIC',
    'GALAXY_SOUNDS_BLACK_HOLES',
    'PEBBLES_XL',
    'ROBOT_DISHES'
]

SHORT_PRODUCTS = [
    'MICROCHIP_OVAL',
    'PEBBLES_XS',
    'ROBOT_IRONING',
    'ROBOT_VACUUMING',
    'UV_VISOR_AMBER'
]


class Trader:

    def run(self, state: TradingState):
        result = {}

        for product in MM_PRODUCTS:
            if product not in state.order_depths:
                continue

            od: OrderDepth = state.order_depths[product]
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
            buy_capacity  = POSITION_LIMIT - position
            sell_capacity = POSITION_LIMIT + position

            if buy_capacity > 0:
                orders.append(Order(product, bid_price, buy_capacity))
            if sell_capacity > 0:
                orders.append(Order(product, ask_price, -sell_capacity))

            result[product] = orders

        for product in LONG_PRODUCTS:
            if product not in state.order_depths:
                continue

            od: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)
            buy_capacity = POSITION_LIMIT - position

            if buy_capacity <= 0:
                continue

            sell_orders = {p: abs(v) for p, v in od.sell_orders.items()}
            if not sell_orders:
                continue

            best_ask = min(sell_orders)
            result[product] = [Order(product, best_ask, buy_capacity)]

        for product in SHORT_PRODUCTS:
            if product not in state.order_depths:
                continue

            od: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)
            sell_capacity = POSITION_LIMIT + position

            if sell_capacity <= 0:
                continue

            buy_orders = {p: abs(v) for p, v in od.buy_orders.items()}
            if not buy_orders:
                continue

            best_bid = max(buy_orders)
            result[product] = [Order(product, best_bid, -sell_capacity)]

        return result, 0, ""
