from datamodel import OrderDepth, TradingState, Order
import json
from math import erf, log, sqrt

# ─── Symbols & limits ─────────────────────────────────────────────────────────

UNDERLYING = "VELVETFRUIT_EXTRACT"
HYDROGEL   = "HYDROGEL_PACK"

OPTION_STRIKES: dict[str, int] = {
    "VEV_4000": 4000, "VEV_4500": 4500,
    "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
    "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}

POS_LIMITS: dict[str, int] = {
    UNDERLYING: 200,
    HYDROGEL:   50,
    **{sym: 200 for sym in OPTION_STRIKES},
}

# ─── Time-to-expiry ───────────────────────────────────────────────────────────

EXPIRY_DAY    = 7
TICKS_PER_DAY = 10_000

def compute_tte(day: int, timestamp: int) -> float:
    remaining = (EXPIRY_DAY - day) * TICKS_PER_DAY - timestamp
    return max(remaining / (TICKS_PER_DAY * 365), 0.0)

# ─── Black-Scholes ────────────────────────────────────────────────────────────

def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def bs_price(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    v  = sigma * sqrt(T)
    d1 = (log(S / K) + 0.5 * sigma * sigma * T) / v
    return S * _ncdf(d1) - K * _ncdf(d1 - v)

def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    v  = sigma * sqrt(T)
    d1 = (log(S / K) + 0.5 * sigma * sigma * T) / v
    return _ncdf(d1)

def implied_vol(mkt: float, S: float, K: float, T: float) -> float | None:
    tv = mkt - max(S - K, 0.0)
    if tv < 0.5 or T <= 0:
        return None
    lo, hi = 1e-4, 8.0
    for _ in range(60):
        mid  = 0.5 * (lo + hi)
        diff = bs_price(S, K, T, mid) - mkt
        if abs(diff) < 1e-5:
            return mid
        hi = mid if diff > 0 else hi
        lo = lo  if diff > 0 else mid
    return 0.5 * (lo + hi)

# ─── Parameters ───────────────────────────────────────────────────────────────

# Market-making (near-ATM only: 5000..5500)
MM_STRIKES = {"VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"}

# We quote 1 tick inside the current best on each side.
# Skip quoting a side when the resulting bid >= ask (spread too tight).
MM_SIZE = 10

# Only quote when the current market spread is at least this many ticks,
# so quoting 1 inside still leaves a positive gap.
MIN_SPREAD_TO_QUOTE = 3

# Delta hedge dead-band: only trade underlying when |portfolio delta| exceeds this.
# Saves the 5-tick underlying spread on small fluctuations.
HEDGE_THRESHOLD = 5

# ITM arb: buy when ask < intrinsic - this margin
ARB_STRIKES  = {"VEV_4000", "VEV_4500"}
ARB_MIN_EDGE = 1.0
ARB_SIZE     = 5

# Fallback vol; live calibration refines this each tick.
FALLBACK_VOL = 0.35

# Vol calibration: options with reliable time value
CALIB_STRIKES = {"VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"}

# Hydrogel
HYDROGEL_EMA_WINDOW   = 50
HYDROGEL_ENTRY_ZSCORE = 1.5
HYDROGEL_EXIT_ZSCORE  = 0.25

# ─── Order-book helpers ───────────────────────────────────────────────────────

def _best_bid_ask(od: OrderDepth | None) -> tuple[int | None, int | None]:
    if od is None:
        return None, None
    bb = max(od.buy_orders)  if od.buy_orders  else None
    ba = min(od.sell_orders) if od.sell_orders else None
    return bb, ba

def _mid(od: OrderDepth | None) -> float | None:
    bb, ba = _best_bid_ask(od)
    if bb is not None and ba is not None:
        return 0.5 * (bb + ba)
    if bb is not None:
        return float(bb)
    if ba is not None:
        return float(ba)
    return None

# ─── Live vol calibration ─────────────────────────────────────────────────────

def calibrate_vol(state: TradingState, T: float, prev_vol: float) -> float:
    S = _mid(state.order_depths.get(UNDERLYING))
    if S is None or T <= 0:
        return prev_vol
    ivs = []
    for sym in CALIB_STRIKES:
        p = _mid(state.order_depths.get(sym))
        if p is None:
            continue
        v = implied_vol(p, S, OPTION_STRIKES[sym], T)
        if v is not None:
            ivs.append(v)
    if len(ivs) < 2:
        return prev_vol
    ivs.sort()
    n = len(ivs)
    median = ivs[n // 2] if n % 2 else 0.5 * (ivs[n // 2 - 1] + ivs[n // 2])
    # Smooth 80/20 so a single mispriced tick doesn't whipsaw the vol estimate
    return 0.8 * median + 0.2 * prev_vol

# ─── Base ProductTrader ───────────────────────────────────────────────────────

class ProductTrader:
    def __init__(self, name: str, state: TradingState, new_data: dict):
        self.name     = name
        self.state    = state
        self.new_data = new_data
        self.orders: list[Order] = []

        try:
            self.prev_data: dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            self.prev_data = {}

        self.position_limit = POS_LIMITS.get(name, 0)
        self.position       = state.position.get(name, 0)
        self.max_buy        = self.position_limit - self.position
        self.max_sell       = self.position_limit + self.position

        od               = state.order_depths.get(name)
        self.best_bid, self.best_ask = _best_bid_ask(od)
        self.mid         = _mid(od)

    def bid(self, price: int, volume: int) -> None:
        vol = min(abs(int(volume)), self.max_buy)
        if vol <= 0:
            return
        self.orders.append(Order(self.name, int(price), vol))
        self.max_buy -= vol

    def ask(self, price: int, volume: int) -> None:
        vol = min(abs(int(volume)), self.max_sell)
        if vol <= 0:
            return
        self.orders.append(Order(self.name, int(price), -vol))
        self.max_sell -= vol

    def get_orders(self) -> dict[str, list[Order]]:
        return {self.name: self.orders}

# ─── Hydrogel mean-reversion ──────────────────────────────────────────────────

class HydrogelTrader(ProductTrader):
    def __init__(self, state: TradingState, new_data: dict):
        super().__init__(HYDROGEL, state, new_data)
        self.zscore: float | None = None
        self._warmed_up = False

        if self.mid is None:
            return

        import math
        alpha = 2.0 / (HYDROGEL_EMA_WINDOW + 1)
        ema_m  = self.prev_data.get("hyd_ema_m", self.mid)
        ema_v  = self.prev_data.get("hyd_ema_v", 0.0)
        n_ticks = self.prev_data.get("hyd_ticks", 0) + 1

        ema_m = alpha * self.mid + (1 - alpha) * ema_m
        ema_v = alpha * (self.mid - ema_m) ** 2 + (1 - alpha) * ema_v

        new_data["hyd_ema_m"]  = ema_m
        new_data["hyd_ema_v"]  = ema_v
        new_data["hyd_ticks"]  = n_ticks

        if n_ticks >= HYDROGEL_EMA_WINDOW:
            std = math.sqrt(max(ema_v, 1e-8))
            self.zscore     = (self.mid - ema_m) / std
            self._warmed_up = True

    def get_orders(self) -> dict[str, list[Order]]:
        if not self._warmed_up or self.zscore is None:
            return {self.name: []}
        z = self.zscore
        if z >= HYDROGEL_ENTRY_ZSCORE and self.best_bid is not None:
            self.ask(self.best_bid, self.max_sell)
        elif z <= -HYDROGEL_ENTRY_ZSCORE and self.best_ask is not None:
            self.bid(self.best_ask, self.max_buy)
        elif abs(z) <= HYDROGEL_EXIT_ZSCORE:
            if self.position > 0 and self.best_bid is not None:
                self.ask(self.best_bid, self.position)
            elif self.position < 0 and self.best_ask is not None:
                self.bid(self.best_ask, -self.position)
        return {self.name: self.orders}

# ─── Options strategy ─────────────────────────────────────────────────────────

class OptionsStrategy:
    def __init__(self, state: TradingState, new_data: dict,
                 T: float, sigma: float):
        self.state    = state
        self.new_data = new_data
        self.T        = T
        self.sigma    = sigma

        self.opt_orders: dict[str, list[Order]] = {}
        self.und_orders: list[Order]            = []

        und_od           = state.order_depths.get(UNDERLYING)
        self.S_bid, self.S_ask = _best_bid_ask(und_od)
        self.S_mid       = _mid(und_od)
        self.und_pos     = state.position.get(UNDERLYING, 0)
        self.und_lim     = POS_LIMITS[UNDERLYING]

        self._opt_delta  = self._portfolio_delta()

    # ── book helpers ──────────────────────────────────────────────────────────

    def _portfolio_delta(self) -> float:
        if self.S_mid is None or self.T <= 0:
            return 0.0
        total = 0.0
        for sym, K in OPTION_STRIKES.items():
            pos = self.state.position.get(sym, 0)
            if pos != 0:
                total += pos * bs_delta(self.S_mid, K, self.T, self.sigma)
        return total

    def _opt_bid(self, sym: str, price: int, vol: int) -> None:
        pos   = self.state.position.get(sym, 0)
        avail = POS_LIMITS[sym] - pos
        v     = min(abs(vol), avail)
        if v > 0:
            self.opt_orders.setdefault(sym, []).append(Order(sym, price, v))

    def _opt_ask(self, sym: str, price: int, vol: int) -> None:
        pos   = self.state.position.get(sym, 0)
        avail = POS_LIMITS[sym] + pos
        v     = min(abs(vol), avail)
        if v > 0:
            self.opt_orders.setdefault(sym, []).append(Order(sym, price, -v))

    def _und_buy(self, price: int, vol: int) -> None:
        avail = self.und_lim - self.und_pos
        v     = min(abs(vol), avail)
        if v > 0:
            self.und_orders.append(Order(UNDERLYING, price, v))
            self.und_pos += v

    def _und_sell(self, price: int, vol: int) -> None:
        avail = self.und_lim + self.und_pos
        v     = min(abs(vol), avail)
        if v > 0:
            self.und_orders.append(Order(UNDERLYING, price, -v))
            self.und_pos -= v

    # ── strategy 1: near-ATM market-making ────────────────────────────────────
    # Quote 1 tick inside the current best bid/ask on each side.
    # Skip if posting inside would cross (spread < 3 after posting).

    def run_mm(self) -> None:
        if self.S_mid is None or self.T <= 0:
            return

        for sym in MM_STRIKES:
            od = self.state.order_depths.get(sym)
            if od is None:
                continue

            best_bid, best_ask = _best_bid_ask(od)
            if best_bid is None or best_ask is None:
                continue

            spread = best_ask - best_bid
            if spread < MIN_SPREAD_TO_QUOTE:
                continue

            our_bid = best_bid + 1
            our_ask = best_ask - 1

            # Sanity-check against BS fair: only post a side if it's on the
            # correct side of fair value (avoid buying above fair or selling below).
            fair = bs_price(self.S_mid, OPTION_STRIKES[sym], self.T, self.sigma)
            if our_bid >= fair:
                our_bid = max(1, int(fair) - 1)
            if our_ask <= fair:
                our_ask = int(fair) + 1

            if our_bid >= our_ask:
                continue

            self._opt_bid(sym, our_bid, MM_SIZE)
            self._opt_ask(sym, our_ask, MM_SIZE)

    # ── strategy 2: ITM intrinsic arbitrage ───────────────────────────────────

    def run_arb(self) -> None:
        if self.S_mid is None:
            return
        for sym in ARB_STRIKES:
            od = self.state.order_depths.get(sym)
            if od is None or not od.sell_orders:
                continue
            best_ask = min(od.sell_orders)
            K        = OPTION_STRIKES[sym]
            intrinsic = max(self.S_mid - K, 0.0)
            if intrinsic - best_ask >= ARB_MIN_EDGE:
                self._opt_bid(sym, best_ask, ARB_SIZE)
                if self.S_bid is not None:
                    self._und_sell(self.S_bid, ARB_SIZE)

    # ── strategy 3: delta hedge ────────────────────────────────────────────────
    # Only trade when net delta exposure exceeds HEDGE_THRESHOLD to avoid
    # paying the 5-6 tick underlying spread on every minor fluctuation.

    def run_delta_hedge(self) -> None:
        if self.S_mid is None:
            return
        delta = self._opt_delta
        if abs(delta) < HEDGE_THRESHOLD:
            return

        target = -round(delta)
        target = max(-self.und_lim, min(self.und_lim, target))
        hedge  = target - self.und_pos

        if hedge > 0 and self.S_ask is not None:
            self._und_buy(self.S_ask, hedge)
        elif hedge < 0 and self.S_bid is not None:
            self._und_sell(self.S_bid, -hedge)

    # ── run all ───────────────────────────────────────────────────────────────

    def execute(self) -> dict[str, list[Order]]:
        self.run_arb()
        self.run_mm()
        self.run_delta_hedge()
        result: dict[str, list[Order]] = dict(self.opt_orders)
        if self.und_orders:
            result[UNDERLYING] = self.und_orders
        return result

# ─── Trader ───────────────────────────────────────────────────────────────────

class Trader:
    def run(self, state: TradingState) -> tuple[dict, int, str]:
        try:
            prev: dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            prev = {}

        new_data: dict = {}

        # ── day / TTE tracking ────────────────────────────────────────────────
        ts      = int(state.timestamp)
        prev_ts = prev.get("last_ts", -1)
        day     = prev.get("day", 0)
        if ts < prev_ts:   # timestamp rolled back → new day started
            day += 1
        new_data["last_ts"] = ts
        new_data["day"]     = day

        T = compute_tte(day, ts)

        # ── vol calibration ───────────────────────────────────────────────────
        prev_vol = prev.get("sigma", FALLBACK_VOL)
        sigma    = calibrate_vol(state, T, prev_vol)
        sigma    = max(0.05, min(3.0, sigma))
        new_data["sigma"] = sigma

        # ── options ───────────────────────────────────────────────────────────
        result: dict[str, list[Order]] = {}
        result.update(OptionsStrategy(state, new_data, T, sigma).execute())

        # ── hydrogel ──────────────────────────────────────────────────────────
        if HYDROGEL in state.order_depths:
            try:
                new_data.update({k: v for k, v in prev.items()
                                 if k.startswith("hyd_")})
                result.update(HydrogelTrader(state, new_data).get_orders())
            except Exception:
                pass

        try:
            trader_data = json.dumps(new_data)
        except Exception:
            trader_data = ""

        return result, 0, trader_data
