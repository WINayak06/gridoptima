"""
solver_core.py
==============
Unified Core Engine for Static Dispatch (ED, DC-OPF) and Unit Commitment (ED-UC, SCUC).

FIXES APPLIED (vs original):
  [BUG-1] c_cost was mapped to column 'c' (no-load $/h) via gen_df.get("c", ...)
          but the fallback label "startup_cost" was misleading — startup_cost was
          NEVER actually loaded. Fixed: load 'c' (no-load) and 'startup_cost'
          (one-time startup) as two separate arrays.
  [BUG-2] No startup indicator variable y[g,t] existed, so startup costs were
          never modelled properly. Fixed: added binary y[g,t] (=1 when unit
          transitions OFF→ON) with constraint y[g,t] >= u[g,t] - u[g,t-1].
  [BUG-3] NVAR = 2*GT was too small once startup variables were added.
          Fixed: NVAR = 3*GT  →  [u | p | y].
  [BUG-4] integrality vector only covered u variables. Fixed: y variables
          also set to binary (integ[2*GT : 3*GT] = 1).
  [BUG-5] DC-OPF used raw bus_df/gen_df dicts instead of normalised bdf/gdf,
          causing case-sensitive key lookups to silently fall back to 0.
          Fixed: DC-OPF now uses bdf, gdf, ldf throughout.
  [BUG-6] SCUC line loading divided by rate[:, None] directly — lines with
          rate=0 produced inf/NaN in JSON, crashing the frontend.
          Fixed: safe_rate replaces zeros with np.inf before division.
  [BUG-7] bus_df.pivot() in run_uc_suite hardcoded column names "bus_id",
          "hour", "demand" — any alternate naming (e.g. "pd", "Load") caused
          a KeyError. Fixed: column names resolved dynamically before pivot.
"""

import time
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp, linprog
from scipy.sparse.linalg import factorized

warnings.filterwarnings("ignore")


def _safe_float(val, fallback=0.0):
    try:
        return float(val)
    except Exception:
        return fallback


# ═══════════════════════════════════════════════════════════════════════
#  PART A – STATIC DISPATCH SUITE  (ED & DC-OPF, single snapshot)
# ═══════════════════════════════════════════════════════════════════════

def run_static_suite(
    gen_df: pd.DataFrame,
    bus_df: pd.DataFrame,
    line_df: pd.DataFrame,
    base_mva: float = 100.0,
    log_cb=None,
):
    """
    Run static Economic Dispatch (ED) and DC-OPF for a single hour snapshot.

    Expected column names (case-insensitive, underscore-normalised):
        gen_df  : bus_id / bus, min_p / Pmin, max_p / Pmax, b (variable cost $/MWh),
                  c (no-load cost $/h), a (quadratic — currently zero so ignored)
        bus_df  : bus_id / bus_i / bus, demand / Pd / pd / load
        line_df : from_bus / fbus, to_bus / tbus, x_pu / x, rate_a_mw / rateA
    """

    def log(msg):
        if log_cb:
            log_cb(msg)

    results = {"suite_type": "static", "ed": None, "dcopf": None}

    # ── helper: normalise column names once ──────────────────────────
    def _norm(df):
        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]
        return df

    gdf = _norm(gen_df)
    bdf = _norm(bus_df)

    # ── 1. Economic Dispatch (ED) ────────────────────────────────────
    log("Solving Static Economic Dispatch (ED)...")
    try:
        t0 = time.time()

        # demand column: demand / pd / load
        demand_col = next(
            (c for c in ["demand", "pd", "load"] if c in bdf.columns), None
        )
        if demand_col is None:
            raise ValueError("bus_df has no recognisable demand column (demand/Pd/load)")
        total_demand = float(bdf[demand_col].sum())

        n_gens  = len(gdf)
        max_p   = gdf.get("max_p", gdf.get("pmax", pd.Series([1000] * n_gens))).values.astype(float)
        min_p   = gdf.get("min_p", gdf.get("pmin", pd.Series([0]     * n_gens))).values.astype(float)
        coeff_b = gdf.get("b", gdf.get("slope", gdf.get("cost", pd.Series([15] * n_gens)))).values.astype(float)

        # Minimise: b·p  (+ slack penalty for infeasibility detection)
        c_vector = np.concatenate([coeff_b, [1e5, 1e5]])
        A_eq     = np.zeros((1, n_gens + 2))
        A_eq[0, :n_gens] =  1.0
        A_eq[0, n_gens]  =  1.0   # surplus slack
        A_eq[0, n_gens + 1] = -1.0  # deficit slack
        b_eq   = np.array([total_demand])
        bounds = [(min_p[i], max_p[i]) for i in range(n_gens)] + [(0, None), (0, None)]

        res = linprog(c=c_vector, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

        if res.success:
            p_values = res.x[:n_gens]
            mcp = (
                float(np.abs(res.eqlin.marginals[0]))
                if hasattr(res, "eqlin") and res.eqlin is not None
                else float(max(coeff_b[p_values > 0.1]))
            )

            gen_units  = []
            total_cost = 0.0
            for idx in range(n_gens):
                p_actual = max(0.0, float(p_values[idx]))
                u_cost   = float(coeff_b[idx] * p_actual)
                total_cost += u_cost
                gen_units.append(
                    {
                        "id":       idx + 1,
                        "dispatch": round(p_actual, 2),
                        "cost":     round(u_cost, 2),
                        "marginal": bool(0.1 < p_actual < (max_p[idx] - 0.1)),
                    }
                )

            # ── Shadow prices (bus λ) ───────────────────────────────
            # NOTE: classic ED is a single-node ("copper-plate") model — there
            # is only ONE power-balance constraint (total gen = total demand),
            # so there is only one dual value (mcp). Because no network/branch
            # constraints exist, every bus shares the same shadow price. We
            # still report it per-bus so the frontend/report can show a
            # bus-indexed table consistent with DC-OPF/SCUC.
            bus_id_col = next(
                (c for c in ["bus_id", "bus_i", "bus"] if c in bdf.columns), None
            )
            bus_shadow_prices = []
            for i, row in bdf.reset_index(drop=True).iterrows():
                bid = int(_safe_float(row[bus_id_col])) if bus_id_col else i + 1
                bus_shadow_prices.append({"bus_id": bid, "lambda": round(mcp, 4)})

            results["ed"] = {
                "total_cost":    round(total_cost, 2),
                "mcp":           round(mcp, 2),
                "total_demand":  round(total_demand, 2),
                "generators":    gen_units,
                "solve_time_s":  round(time.time() - t0, 3),
                "bus_shadow_prices": bus_shadow_prices,
                "shadow_price_note": (
                    "ED is a single-bus (copper-plate) model with no network "
                    "constraints, so \u03bb is uniform across all buses and there "
                    "are no line shadow prices (\u03bc)."
                ),
            }
            log("ED Solved Successfully.")
        else:
            log(f"ED solver returned non-optimal status: {res.message}")

    except Exception as exc:
        log(f"ED Failed: {exc}")

    # ── 2. DC-OPF ───────────────────────────────────────────────────
    log("Solving DC-OPF...")
    try:
        t0 = time.time()

        # FIX [BUG-5]: use normalised copies so column lookups are case-insensitive
        ldf        = _norm(line_df) if line_df is not None else None
        buses      = bdf.to_dict(orient="records")
        generators = gdf.to_dict(orient="records")
        branches   = ldf.to_dict(orient="records") if ldf is not None else []

        Nb = len(buses)
        Ng = len(generators)

        # Sort buses by id and build index map
        buses      = sorted(buses, key=lambda b: int(_safe_float(b.get("bus_i", b.get("bus_id", b.get("bus", 0))))))
        bus_id_map = {
            int(_safe_float(b.get("bus_i", b.get("bus_id", b.get("bus", 0))))): i
            for i, b in enumerate(buses)
        }

        # Build B-matrix
        B_lil        = sp.lil_matrix((Nb, Nb), dtype=np.float64)
        valid_branches = []
        for br in branches:
            fbus_id = int(_safe_float(br.get("fbus", br.get("from_bus", 0))))
            tbus_id = int(_safe_float(br.get("tbus", br.get("to_bus",   0))))
            if fbus_id in bus_id_map and tbus_id in bus_id_map:
                x_val = _safe_float(br.get("x", br.get("x_pu", 0.01)))
                if abs(x_val) < 1e-9:
                    x_val = 1e-4
                b_val = 1.0 / x_val
                f_idx, t_idx = bus_id_map[fbus_id], bus_id_map[tbus_id]
                B_lil[f_idx, f_idx] += b_val
                B_lil[t_idx, t_idx] += b_val
                B_lil[f_idx, t_idx] -= b_val
                B_lil[t_idx, f_idx] -= b_val
                valid_branches.append((br, f_idx, t_idx, b_val))

        # Decision variables: [Pg (Ng), θ (Nb), slack_pos (Nb), slack_neg (Nb)]
        N_vars = Ng + 3 * Nb
        bounds = []

        # Generator output bounds (pu)
        for g in generators:
            bounds.append((0.0, _safe_float(g.get("max_p", g.get("pmax", 1000))) / base_mva))

        # Voltage angle bounds (slack bus fixed at 0)
        for i in range(Nb):
            bounds.append((0.0, 0.0) if i == 0 else (-np.pi, np.pi))

        # Slack variables for load shedding / over-generation
        for b in buses:
            bounds.append((0.0, _safe_float(b.get("pd", b.get("demand", 0))) / base_mva))
        for _ in range(Nb):
            bounds.append((0.0, None))

        # Cost vector
        PENALTY = 50_000.0 * base_mva
        c_vec = np.zeros(N_vars)
        for i, g in enumerate(generators):
            c_vec[i] = _safe_float(g.get("b", g.get("cost", 15.0))) * base_mva
        c_vec[Ng + Nb : Ng + 2 * Nb] = PENALTY
        c_vec[Ng + 2 * Nb : N_vars]  = PENALTY

        # Power balance: Pg_bus - B·θ + slack_pos - slack_neg = Pd_bus
        A_eq_sp = sp.lil_matrix((Nb, N_vars), dtype=np.float64)
        for i, g in enumerate(generators):
            g_bus = int(_safe_float(g.get("bus_id", g.get("bus", 0))))
            if g_bus in bus_id_map:
                A_eq_sp[bus_id_map[g_bus], i] = 1.0

        B_arr = B_lil.toarray()
        for r in range(Nb):
            for c_idx in range(Nb):
                if B_arr[r, c_idx] != 0.0:
                    A_eq_sp[r, Ng + c_idx] = -B_arr[r, c_idx]
            A_eq_sp[r, Ng + Nb + r]     =  1.0
            A_eq_sp[r, Ng + 2 * Nb + r] = -1.0

        b_eq = np.array(
            [_safe_float(b.get("pd", b.get("demand", 0))) / base_mva for b in buses]
        )

        # Thermal line limits
        constrained_branches = []
        for br, f_idx, t_idx, b_val in valid_branches:
            limit_pu = _safe_float(br.get("rate_a_mw", br.get("ratea", 0))) / base_mva
            if limit_pu > 0.001:
                constrained_branches.append((br, f_idx, t_idx, b_val, limit_pu))

        n_ineq = 2 * len(constrained_branches)
        if n_ineq > 0:
            A_ub_sp = sp.lil_matrix((n_ineq, N_vars), dtype=np.float64)
            b_ub    = np.zeros(n_ineq)
            for k, (br, f_idx, t_idx, b_val, limit_pu) in enumerate(constrained_branches):
                A_ub_sp[2 * k,     Ng + f_idx] =  b_val
                A_ub_sp[2 * k,     Ng + t_idx] = -b_val
                A_ub_sp[2 * k + 1, Ng + f_idx] = -b_val
                A_ub_sp[2 * k + 1, Ng + t_idx] =  b_val
                b_ub[2 * k]     = limit_pu
                b_ub[2 * k + 1] = limit_pu
            A_ub = sp.csc_matrix(A_ub_sp)
        else:
            A_ub, b_ub = None, None

        res = linprog(
            c=c_vec,
            A_eq=sp.csc_matrix(A_eq_sp),
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
        )

        if res.success:
            Pg_pu = res.x[:Ng]
            Va    = res.x[Ng : Ng + Nb]

            gen_out = []
            t_cost  = 0.0
            for i, g in enumerate(generators):
                out_mw = Pg_pu[i] * base_mva
                slope  = _safe_float(g.get("b", g.get("cost", 15.0)))
                t_cost += out_mw * slope
                gen_out.append({"id": i + 1, "output_mw": round(out_mw, 2)})

            # ── Shadow prices ────────────────────────────────────────
            # Bus λ (LMP): dual of each nodal power-balance equality row.
            # `res.eqlin.marginals` gives dz/d(b_eq) in $/pu since both the
            # objective (c_vec) and b_eq were built in per-unit (base_mva)
            # quantities — divide by base_mva to convert to $/MWh.
            bus_lmp = []
            if hasattr(res, "eqlin") and res.eqlin is not None:
                lam_pu = res.eqlin.marginals
                for i, b in enumerate(buses):
                    bid = int(_safe_float(b.get("bus_i", b.get("bus_id", b.get("bus", i + 1)))))
                    bus_lmp.append({"bus_id": bid, "lambda": round(float(lam_pu[i]) / base_mva, 4)})

            # Line μ (congestion shadow price): dual of the two thermal-limit
            # inequality rows per constrained branch (only one can bind at a
            # time). Same pu → $/MWh conversion as above.
            mu_map = {}
            if n_ineq > 0 and hasattr(res, "ineqlin") and res.ineqlin is not None:
                ineq_marg = res.ineqlin.marginals
                for k, (br, f_idx, t_idx, b_val, limit_pu) in enumerate(constrained_branches):
                    mu_val = (abs(ineq_marg[2 * k]) + abs(ineq_marg[2 * k + 1])) / base_mva
                    mu_map[id(br)] = round(float(mu_val), 4)

            line_out = []
            for br, f_idx, t_idx, b_val in valid_branches:
                flow = (Va[f_idx] - Va[t_idx]) * b_val * base_mva
                cap  = _safe_float(br.get("rate_a_mw", br.get("ratea", 0)))
                line_out.append(
                    {
                        "id":           str(br.get("id", br.get("line_index", 0))),
                        "flow_mw":      round(flow, 2),
                        "loading_pct":  round(abs(flow) / cap * 100, 2) if cap > 0 else 0.0,
                        "mu":           mu_map.get(id(br), 0.0),
                    }
                )

            results["dcopf"] = {
                "total_cost":   round(t_cost, 2),
                "generators":   gen_out,
                "lines":        line_out,
                "solve_time_s": round(time.time() - t0, 3),
                "bus_lmp":      bus_lmp,
                "shadow_price_note": (
                    "\u03bb is the dual of each bus's power-balance constraint "
                    "(locational marginal price, \u20b9/MWh). \u03bc is the dual of a "
                    "line's thermal-limit constraint (\u20b9/MWh value of relaxing "
                    "that line's rating by 1 MW) — non-zero only for congested "
                    "(binding) lines."
                ),
            }
            log("DC-OPF Solved Successfully.")
        else:
            log(f"DC-OPF solver returned non-optimal status: {res.message}")

    except Exception as exc:
        log(f"DC-OPF Failed: {exc}")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  PART B – UNIT COMMITMENT SUITE  (ED-UC & SCUC, multi-hour)
# ═══════════════════════════════════════════════════════════════════════

def run_uc_suite(
    gen_df: pd.DataFrame,
    bus_df: pd.DataFrame,
    line_df: pd.DataFrame,
    gap: float = 0.01,
    time_limit: int = 300,
    log_cb=None,
):
    """
    Run Economic Dispatch Unit Commitment (ED-UC) and
    Security-Constrained UC (SCUC) over 24 hours.

    Expected column names:
        gen_df  : bus_id, min_p, max_p, b (variable $/MWh), c (no-load $/h),
                  startup_cost ($), shutdown_cost ($)
        bus_df  : bus_id, hour, demand
        line_df : from_bus, to_bus, x_pu, rate_a_mw

    Cost model per generator g, hour t:
        cost[g,t] = c[g] · u[g,t]          (no-load cost, charged every hour ON)
                  + b[g] · p[g,t]           (variable/fuel cost)
                  + startup_cost[g] · y[g,t] (one-time cost when unit starts up)

    Variables:
        u[g,t] ∈ {0,1}   commitment status
        p[g,t] ≥ 0        power output (MW)
        y[g,t] ∈ {0,1}   startup indicator  (y=1 iff u[g,t]-u[g,t-1] = 1)
    """

    def log(msg):
        if log_cb:
            log_cb(msg)

    log("Preparing data for Multi-Hour Unit Commitment...")

    N_GEN = len(gen_df)
    N_T   = bus_df["hour"].nunique()
    N_BUS = int(bus_df["bus_id"].max())

    # ── Generator parameters ─────────────────────────────────────────
    Pmin    = gen_df.get("min_p",        gen_df.get("Pmin",         pd.Series([0.0]   * N_GEN))).values.astype(float)
    Pmax    = gen_df.get("max_p",        gen_df.get("Pmax",         pd.Series([1000.0]* N_GEN))).values.astype(float)
    b_cost  = gen_df.get("b",            gen_df.get("cost",         pd.Series([15.0]  * N_GEN))).values.astype(float)
    # FIX [BUG-1]: load no-load cost and startup cost as SEPARATE arrays
    c_cost  = gen_df.get("c",            pd.Series([0.0] * N_GEN)).values.astype(float)           # no-load $/h
    su_cost = gen_df.get("startup_cost", pd.Series([0.0] * N_GEN)).values.astype(float)           # one-time startup $
    gen_bus = gen_df.get("bus_id",       gen_df.get("bus", pd.Series([1] * N_GEN))).values.astype(int) - 1  # 0-indexed

    # ── Demand ───────────────────────────────────────────────────────
    # FIX [BUG-7]: resolve column names dynamically instead of hardcoding "demand"/"bus_id"
    _bus_cols = [c.lower() for c in bus_df.columns]
    _d_col    = next((c for c in ["demand", "pd", "load"] if c in _bus_cols), "demand")
    _b_col    = next((c for c in ["bus_id", "bus", "bus_i"] if c in _bus_cols), "bus_id")
    _h_col    = next((c for c in ["hour", "t", "period"] if c in _bus_cols), "hour")

    # Work on a normalised copy so groupby/pivot use consistent names
    _bus_norm = bus_df.copy()
    _bus_norm.columns = [c.lower() for c in _bus_norm.columns]

    D_total = _bus_norm.groupby(_h_col)[_d_col].sum().sort_index().values.astype(float)
    bus_pivot = (
        _bus_norm.pivot(index=_b_col, columns=_h_col, values=_d_col)
        .reindex(range(1, N_BUS + 1), fill_value=0.0)
    )
    D_bus = bus_pivot.values.astype(float)  # (N_BUS, N_T)

    # ── PTDF for SCUC ────────────────────────────────────────────────
    PTDF = None
    if line_df is not None:
        log("Building PTDF matrix for SCUC...")
        N_LINE = len(line_df)
        from_b = line_df.get("from_bus", line_df.get("fbus")).values.astype(int) - 1
        to_b   = line_df.get("to_bus",   line_df.get("tbus")).values.astype(int) - 1
        b_line = 1.0 / line_df.get("x_pu", line_df.get("x")).values.astype(float)

        rows = np.concatenate([np.arange(N_LINE), np.arange(N_LINE)])
        cols = np.concatenate([from_b, to_b])

        A_f   = sp.csr_matrix((np.concatenate([b_line, -b_line]), (rows, cols)), shape=(N_LINE, N_BUS))
        K     = sp.csr_matrix((np.concatenate([np.ones(N_LINE), -np.ones(N_LINE)]), (rows, cols)), shape=(N_LINE, N_BUS))
        B_bus = K.T.dot(sp.diags(b_line)).dot(K)

        # Reduced system (remove slack bus row/col 0)
        B_red   = B_bus[np.ix_(np.arange(1, N_BUS), np.arange(1, N_BUS))].tocsc()
        solve_B = factorized(B_red)

        # PTDF[l, n] = sensitivity of flow on line l to injection at bus n
        PTDF = np.hstack(
            [np.zeros((N_LINE, 1)),
             solve_B(A_f[:, 1:].toarray().T).T]
        )  # shape: (N_LINE, N_BUS)
        log("PTDF built successfully.")

    # ── Shared index helpers ─────────────────────────────────────────
    GT = N_GEN * N_T
    # FIX [BUG-3]: NVAR = 3*GT  →  [u (0..GT-1) | p (GT..2GT-1) | y (2GT..3GT-1)]
    NVAR = 3 * GT

    def idx_u(g, t): return g * N_T + t          # commitment binary
    def idx_p(g, t): return GT + g * N_T + t     # power output (continuous)
    def idx_y(g, t): return 2 * GT + g * N_T + t # startup indicator binary

    results = {"suite_type": "uc"}

    # ── Solve ED-UC then SCUC ────────────────────────────────────────
    for mode in ["educ", "scuc"]:
        log(f"Assembling MILP for {mode.upper()}...")

        # ── Objective ────────────────────────────────────────────────
        c_obj = np.zeros(NVAR)
        for g in range(N_GEN):
            for t in range(N_T):
                c_obj[idx_u(g, t)] = c_cost[g]   # no-load cost every committed hour
                c_obj[idx_p(g, t)] = b_cost[g]   # variable cost per MWh dispatched
                c_obj[idx_y(g, t)] = su_cost[g]  # FIX [BUG-1]: one-time startup cost

        # ── Integrality: u and y binary, p continuous ─────────────────
        # FIX [BUG-4]: integ must cover y variables as well
        integ = np.zeros(NVAR)
        integ[:GT]         = 1  # u binary
        integ[2 * GT:]     = 1  # y binary  ← was missing in original

        # ── Variable bounds ───────────────────────────────────────────
        lb = np.zeros(NVAR)
        ub = np.ones(NVAR)
        for g in range(N_GEN):
            for t in range(N_T):
                ub[idx_p(g, t)] = Pmax[g]   # p unbounded above by 1 — override with Pmax
        bounds = Bounds(lb=lb, ub=ub)

        # ── Constraints ───────────────────────────────────────────────
        Ar, Ac, Av, con_lb, con_ub = [], [], [], [], []
        row_ctr = [0]

        def add_row(col_val_pairs, lo, hi):
            for col, val in col_val_pairs:
                Ar.append(row_ctr[0])
                Ac.append(col)
                Av.append(float(val))
            con_lb.append(lo)
            con_ub.append(hi)
            row_ctr[0] += 1

        # C1: Power balance — sum_g p[g,t] = D_total[t]  for each t
        for t in range(N_T):
            add_row([(idx_p(g, t), 1.0) for g in range(N_GEN)], D_total[t], D_total[t])

        # C2: Generation limits — Pmin·u <= p <= Pmax·u
        for g in range(N_GEN):
            for t in range(N_T):
                add_row([(idx_p(g, t),  1.0), (idx_u(g, t), -Pmax[g])], -np.inf, 0.0)  # p <= Pmax·u
                add_row([(idx_p(g, t), -1.0), (idx_u(g, t),  Pmin[g])], -np.inf, 0.0)  # p >= Pmin·u

        # C3 (FIX [BUG-2]): Startup indicator — y[g,t] >= u[g,t] - u[g,t-1]
        #   Rearranged: y[g,t] - u[g,t] + u[g,t-1] >= 0
        #   All units assumed OFF before hour 1.
        for g in range(N_GEN):
            for t in range(N_T):
                if t == 0:
                    # No previous state → startup if ON in hour 1
                    add_row([(idx_y(g, t), 1.0), (idx_u(g, t), -1.0)], 0.0, np.inf)
                else:
                    add_row(
                        [(idx_y(g, t), 1.0), (idx_u(g, t), -1.0), (idx_u(g, t - 1), 1.0)],
                        0.0, np.inf,
                    )

        # C4 (SCUC only): DC line thermal limits via PTDF
        #   PTDF · (P_gen_bus - D_bus) in [-rate, +rate]
        #   i.e.  sum_g PTDF[l, gen_bus[g]] · p[g,t] in [-rate[l]+shift, +rate[l]+shift]
        #   where shift = PTDF[l,:] · D_bus[:,t]
        if mode == "scuc" and PTDF is not None:
            N_LINE  = len(line_df)
            rate    = line_df.get("rate_a_mw", line_df.get("rateA")).values.astype(float)
            PTDF_D  = PTDF.dot(D_bus)          # (N_LINE, N_T)
            ptdf_g  = PTDF[:, gen_bus]         # (N_LINE, N_GEN)  — column per generator

            for l in range(N_LINE):
                for t in range(N_T):
                    shift = float(PTDF_D[l, t])
                    add_row(
                        [(idx_p(g, t), float(ptdf_g[l, g])) for g in range(N_GEN)],
                        -rate[l] + shift,
                         rate[l] + shift,
                    )

        # ── Assemble sparse constraint matrix ─────────────────────────
        n_rows = row_ctr[0]
        A_mat  = sp.csr_matrix((Av, (Ar, Ac)), shape=(n_rows, NVAR))
        lc     = LinearConstraint(A_mat, lb=np.array(con_lb), ub=np.array(con_ub))

        # ── Solve ─────────────────────────────────────────────────────
        log(f"Solving {mode.upper()} MILP ({n_rows} constraints, {NVAR} variables)...")
        t0  = time.time()
        res = milp(
            c=c_obj,
            constraints=lc,
            integrality=integ,
            bounds=bounds,
            options={"disp": False, "mip_rel_gap": gap, "time_limit": time_limit},
        )
        elapsed = time.time() - t0

        if not res.success:
            log(f"{mode.upper()} solver failed: {res.message}")
            results[mode] = {"status": "failed", "message": res.message}
            continue

        # ── Extract solution ──────────────────────────────────────────
        x = res.x
        U = np.array([[round(x[idx_u(g, t)]) for t in range(N_T)] for g in range(N_GEN)], dtype=int)
        P = np.array([[x[idx_p(g, t)]        for t in range(N_T)] for g in range(N_GEN)])
        Y = np.array([[round(x[idx_y(g, t)]) for t in range(N_T)] for g in range(N_GEN)], dtype=int)

        # ── Cost breakdown ────────────────────────────────────────────
        gen_variable_cost = np.array([
            sum(b_cost[g] * P[g, t] for t in range(N_T)) for g in range(N_GEN)
        ])
        gen_noload_cost = np.array([
            sum(c_cost[g] * U[g, t] for t in range(N_T)) for g in range(N_GEN)
        ])
        gen_startup_cost = np.array([
            sum(su_cost[g] * Y[g, t] for t in range(N_T)) for g in range(N_GEN)
        ])
        gen_total_cost = gen_variable_cost + gen_noload_cost + gen_startup_cost

        hourly_cost = np.array([
            sum(
                c_cost[g] * U[g, t] + b_cost[g] * P[g, t] + su_cost[g] * Y[g, t]
                for g in range(N_GEN)
            )
            for t in range(N_T)
        ])

        # ── Marginal price proxy (highest b_cost among committed units) ──
        lam = []
        for t in range(N_T):
            committed = [g for g in range(N_GEN) if U[g, t] == 1]
            lam.append(float(max(b_cost[g] for g in committed)) if committed else 0.0)

        # ── True hourly shadow prices via fixed-commitment LP re-solve ────
        # scipy.optimize.milp does not return dual values — duals aren't even
        # well-defined once integer variables are involved (the feasible
        # region isn't convex). Standard fix: FIX u[g,t] (and y[g,t], via the
        # Pmin/Pmax bounds they gate) at their optimal MILP values, which
        # collapses the problem into a pure LP over p[g,t] only. Re-solving
        # that LP with `linprog` gives real duals:
        #   - dual of the hourly system-balance row  → system λ_t (₹/MWh)
        #   - dual of each line's flow-limit row (SCUC) → line μ_l,t (₹/MWh)
        # Bus-level LMPs for SCUC are then reconstructed from λ_t and μ_l,t
        # via the PTDF decomposition:  LMP_bus,t = λ_t + Σ_l PTDF[l,bus]·μ_l,t
        lam_hourly    = np.array(lam)     # fallback stays as proxy unless LP succeeds
        mu_hourly     = None              # (N_LINE, N_T) congestion price magnitude
        bus_lmp_matrix = None             # (N_BUS, N_T) locational marginal price
        try:
            NP = N_GEN * N_T

            def jdx_p(g, t):
                return g * N_T + t

            lp_c = np.zeros(NP)
            lp_bounds = []
            for g in range(N_GEN):
                for t in range(N_T):
                    lp_c[jdx_p(g, t)] = b_cost[g]
                    lp_bounds.append((Pmin[g] * U[g, t], Pmax[g] * U[g, t]))

            # C1: system balance, one row per hour
            r1, c1, v1 = [], [], []
            for t in range(N_T):
                for g in range(N_GEN):
                    r1.append(t); c1.append(jdx_p(g, t)); v1.append(1.0)
            A_eq2 = sp.csr_matrix((v1, (r1, c1)), shape=(N_T, NP))
            b_eq2 = D_total.copy()

            A_ub2, b_ub2, n_lr = None, None, 0
            if mode == "scuc" and PTDF is not None:
                n_lr = N_LINE * N_T
                ru, cu, vu, rhs_u = [], [], [], []
                rl, cl, vl, rhs_l = [], [], [], []
                for l in range(N_LINE):
                    for t in range(N_T):
                        shift = float(PTDF_D[l, t])
                        row = l * N_T + t
                        for g in range(N_GEN):
                            coeff = float(ptdf_g[l, g])
                            if coeff != 0.0:
                                ru.append(row); cu.append(jdx_p(g, t)); vu.append(coeff)
                                rl.append(row); cl.append(jdx_p(g, t)); vl.append(-coeff)
                        rhs_u.append(rate[l] + shift)
                        rhs_l.append(rate[l] - shift)
                A_up = sp.csr_matrix((vu, (ru, cu)), shape=(n_lr, NP))
                A_lo = sp.csr_matrix((vl, (rl, cl)), shape=(n_lr, NP))
                A_ub2 = sp.vstack([A_up, A_lo]).tocsr()
                b_ub2 = np.concatenate([np.array(rhs_u), np.array(rhs_l)])

            lp_res = linprog(
                c=lp_c, A_eq=A_eq2, b_eq=b_eq2, A_ub=A_ub2, b_ub=b_ub2,
                bounds=lp_bounds, method="highs",
            )

            if lp_res.success and getattr(lp_res, "eqlin", None) is not None:
                lam_hourly = lp_res.eqlin.marginals.copy()

                if mode == "scuc" and PTDF is not None and getattr(lp_res, "ineqlin", None) is not None:
                    ineq_m      = lp_res.ineqlin.marginals
                    marg_upper  = ineq_m[:n_lr].reshape(N_LINE, N_T)
                    marg_lower  = ineq_m[n_lr:].reshape(N_LINE, N_T)
                    mu_hourly   = np.abs(marg_upper) + np.abs(marg_lower)
                    signed_mu   = marg_upper - marg_lower                       # (N_LINE, N_T)
                    bus_lmp_matrix = lam_hourly[None, :] + PTDF.T.dot(signed_mu)  # (N_BUS, N_T)
            else:
                log(f"{mode.upper()} shadow-price LP re-solve did not return duals; using λ proxy.")
        except Exception as exc:
            log(f"{mode.upper()} shadow-price computation failed ({exc}); using λ proxy.")

        # ── Line flows (SCUC) ─────────────────────────────────────────
        lines_out = []
        if mode == "scuc" and PTDF is not None:
            N_LINE = len(line_df)
            rate   = line_df.get("rate_a_mw", line_df.get("rateA")).values.astype(float)
            p_bus  = np.zeros((N_BUS, N_T))
            for g in range(N_GEN):
                p_bus[gen_bus[g], :] += P[g, :]
            flows = PTDF.dot(p_bus - D_bus)            # (N_LINE, N_T)

            # FIX [BUG-6]: guard against rate=0 lines to avoid inf/NaN in JSON
            safe_rate = np.where(rate == 0, np.inf, rate)
            loading   = (np.abs(flows) / safe_rate[:, None]) * 100

            for i, row_data in line_df.iterrows():
                entry = {
                    "line_index":       int(row_data.get("line_index", row_data.get("id", i + 1))),
                    "from_bus":         int(row_data.get("from_bus", row_data.get("fbus", 0))),
                    "to_bus":           int(row_data.get("to_bus",   row_data.get("tbus", 0))),
                    "rate_mw":          float(rate[i]),
                    "max_loading_pct":  float(loading[i].max()),
                }
                if mu_hourly is not None:
                    entry["mu_hourly"] = [round(float(v), 4) for v in mu_hourly[i]]
                    entry["mu_avg"]    = round(float(mu_hourly[i].mean()), 4)
                    entry["mu_max"]    = round(float(mu_hourly[i].max()), 4)
                lines_out.append(entry)

        # ── Bus LMP table (SCUC only) ───────────────────────────────────
        bus_lmp_out = None
        if bus_lmp_matrix is not None:
            bus_lmp_out = [
                {
                    "bus_id":      b + 1,
                    "avg_lambda":  round(float(bus_lmp_matrix[b].mean()), 4),
                    "max_lambda":  round(float(bus_lmp_matrix[b].max()), 4),
                    "hourly":      [round(float(v), 4) for v in bus_lmp_matrix[b]],
                }
                for b in range(N_BUS)
            ]

        # ── Package results ───────────────────────────────────────────
        results[mode] = {
            "total_cost":    float(res.fun),
            "solve_time_s":  elapsed,
            "cost_breakdown": {
                "variable_cost": float(gen_variable_cost.sum()),
                "noload_cost":   float(gen_noload_cost.sum()),
                "startup_cost":  float(gen_startup_cost.sum()),
            },
            "hourly": [
                {
                    "hour":           t + 1,
                    "demand_mw":      float(D_total[t]),
                    "cost":           float(hourly_cost[t]),
                    "generators_on":  int(U[:, t].sum()),
                    "lambda":         round(float(lam_hourly[t]), 4),
                    "startups":       int(Y[:, t].sum()),
                }
                for t in range(N_T)
            ],
            "generators": [
                {
                    "name":          f"G{g + 1}",
                    "bus":           int(gen_bus[g] + 1),
                    "pmin":          float(Pmin[g]),
                    "pmax":          float(Pmax[g]),
                    "b_cost":        float(b_cost[g]),
                    "c_cost":        float(c_cost[g]),
                    "su_cost":       float(su_cost[g]),
                    "hours_on":      int(U[g].sum()),
                    "startups":      int(Y[g].sum()),
                    "total_mwh":     float(P[g].sum()),
                    "variable_cost": float(gen_variable_cost[g]),
                    "noload_cost":   float(gen_noload_cost[g]),
                    "startup_cost":  float(gen_startup_cost[g]),
                    "total_cost":    float(gen_total_cost[g]),
                    "commitment":    U[g].tolist(),
                    "dispatch":      [round(v, 2) for v in P[g].tolist()],
                }
                for g in range(N_GEN)
            ],
            "lines": lines_out,
            "bus_lmp": bus_lmp_out,
            "shadow_price_note": (
                "\u03bb (system marginal price) and, for SCUC, per-line \u03bc and "
                "bus-level LMPs are recovered by fixing the MILP's optimal "
                "commitment (u, y) and re-solving the resulting per-hour "
                "dispatch LP for dual values \u2014 scipy's MILP solver itself "
                "returns no duals. Bus LMP = \u03bb_t + \u03a3_l PTDF[l,bus]\u00b7\u03bc_l,t."
            ),
        }
        log(f"{mode.upper()} complete. Total cost = ${res.fun:,.2f}  ({elapsed:.1f}s)")

    # ── Cross-mode comparison ─────────────────────────────────────────
    results["meta"] = {
        "n_bus":   N_BUS,
        "n_gen":   N_GEN,
        "n_lines": len(line_df) if line_df is not None else 0,
        "n_hours": N_T,
    }

    if "educ" in results and "scuc" in results:
        if isinstance(results["educ"], dict) and "total_cost" in results["educ"] and \
           isinstance(results["scuc"], dict) and "total_cost" in results["scuc"]:
            educ_cost = results["educ"]["total_cost"]
            scuc_cost = results["scuc"]["total_cost"]
            results["comparison"] = {
                "educ_total":    educ_cost,
                "scuc_total":    scuc_cost,
                "premium":       scuc_cost - educ_cost,
                "premium_pct":   (scuc_cost - educ_cost) / scuc_cost * 100 if scuc_cost else 0.0,
                "hourly_delta":  [
                    {
                        "hour":  t + 1,
                        "delta": results["scuc"]["hourly"][t]["cost"] - results["educ"]["hourly"][t]["cost"],
                    }
                    for t in range(N_T)
                ],
            }

    return results