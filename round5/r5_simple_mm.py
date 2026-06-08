from datamodel import OrderDepth, TradingState, Order
import json

POSITION_LIMIT = 10

PRODUCTS = [
    'GALAXY_SOUNDS_DARK_MATTER', 'GALAXY_SOUNDS_BLACK_HOLES', 'GALAXY_SOUNDS_PLANETARY_RINGS', 'GALAXY_SOUNDS_SOLAR_WINDS', 'GALAXY_SOUNDS_SOLAR_FLAMES'
    'SLEEP_POD_SUEDE', 'SLEEP_POD_LAMB_WOOL', 'SLEEP_POD_POLYESTER', 'SLEEP_POD_NYLON', 'SLEEP_POD_COTTON'
    'MICROCHIP_CIRCLE', 'MICROCHIP_OVAL', 'MICROCHIP_SQUARE', 'MICROCHIP_RECTANGLE', 'MICROCHIP_TRIANGLE'
    'PEBBLES_XS', 'PEBBLES_S', 'PEBBLES_M', 'PEBBLES_L', 'PEBBLES_XL'
    'ROBOT_VACUUMING', 'ROBOT_MOPPING', 'ROBOT_DISHES', 'ROBOT_LAUNDRY', 'ROBOT_IRONING'
    'UV_VISOR_YELLOW', 'UV_VISOR_AMBER', 'UV_VISOR_ORANGE', 'UV_VISOR_RED', 'UV_VISOR_MAGENTA'
    'TRANSLATOR_SPACE_GRAY', 'TRANSLATOR_ASTRO_BLACK', 'TRANSLATOR_ECLIPSE_CHARCOAL', 'TRANSLATOR_GRAPHITE_MIST', 'TRANSLATOR_VOID_BLUE'
    'PANEL_1X2', 'PANEL_2X2', 'PANEL_1X4', 'PANEL_2X4', 'PANEL_4X4'
    'OXYGEN_SHAKE_MORNING_BREATH', 'OXYGEN_SHAKE_EVENING_BREATH', 'OXYGEN_SHAKE_MINT', 'OXYGEN_SHAKE_CHOCOLATE', 'OXYGEN_SHAKE_GARLIC'
    'SNACKPACK_CHOCOLATE', 'SNACKPACK_VANILLA' , 'SNACKPACK_PISTACHIO', 'SNACKPACK_STRAWBERRY',
    'SNACKPACK_RASPBERRY'
]


class Trader:

    def run(self, state: TradingState):
        result = {}

        for product in PRODUCTS:
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

        return result, 0, ""
