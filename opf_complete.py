"""
Optimal Power Sharing in DC Microgrids — Complete Implementation
================================================================

Implements SOCP-relaxed OPF for DC microgrids with prosumer nanogrids.

References:
  [1] Khan, Nasir, Schulz. IEEE Access, vol. 11, 2023.
  [2] Khan, Akande, Schulz. 2024.
  [3] Li, Liu, Wang, Low, Mei. IEEE Trans. Power Syst., 2018.
  [4] Gan & Low. IEEE Trans. Power Syst., 2014.

Author: Aaron Alves (FYP 2025/2026, ECNG 3020)
Supervisor: Dr. Arvind Singh
"""

import numpy as np
import cvxpy as cp
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import warnings

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Branch:
    """Distribution line between two buses."""
    from_bus: int   # 0-indexed
    to_bus: int     # 0-indexed
    r: float        # resistance [Ω or p.u.]
    I_max: float = 10.0  # current rating [A]

@dataclass
class Bus:
    """Nanogrid bus."""
    P_G: float = 0.0       # PV generation [p.u.]
    P_L: float = 0.0       # load demand [p.u.]
    SOC_init: float = 0.70
    C_B: float = 1.2       # battery capacity [p.u.-h]
    P_G_max: float = 1.0
    P_G_min: float = 0.0
    SOC_max: float = 0.90
    SOC_min: float = 0.50
    has_pv: bool = True
    has_battery: bool = True

@dataclass
class ConverterParams:
    """DC-DC converter quadratic loss model: P_c = α + β·P_o + γ·P_o²"""
    alpha: float = 0.001   # no-load loss (very small for modern converters)
    beta: float = 0.02     # linear loss coefficient  
    gamma: float = 0.03    # quadratic loss coefficient
    # Efficiency polynomial (for reporting): η = k0 + k1·x + k2·x²
    k0: float = 0.60
    k1: float = 0.65
    k2: float = -0.31
    P_R: float = 1.0       # rated power

@dataclass
class SystemData:
    """Complete DC microgrid system."""
    buses: list
    branches: list
    converter: ConverterParams = field(default_factory=ConverterParams)
    v_min: float = 0.9025    # (0.95)²
    v_max: float = 1.1025    # (1.05)²
    name: str = "unnamed"

    @property
    def n_buses(self): return len(self.buses)
    @property
    def n_branches(self): return len(self.branches)

@dataclass
class OPFResult:
    """Single time-step OPF result."""
    status: str
    objective: float
    v: np.ndarray
    P_G: np.ndarray
    P_B: np.ndarray
    P_inj: np.ndarray
    SOC: np.ndarray
    P_conv: np.ndarray
    P_ij: np.ndarray
    P_ji: np.ndarray
    l_ij: np.ndarray
    dist_loss: float
    conv_loss: float
    total_loss: float
    socp_gap: np.ndarray

@dataclass
class MultiPeriodResult:
    """24-hour OPF result."""
    status: str
    objective: float
    T: int
    # All arrays: [T, n_buses] or [T, n_branches]
    v: np.ndarray
    P_G: np.ndarray
    P_B: np.ndarray
    P_inj: np.ndarray
    SOC: np.ndarray
    P_conv: np.ndarray
    P_ij: np.ndarray
    P_ji: np.ndarray
    l_ij: np.ndarray
    dist_loss: np.ndarray      # per time step
    conv_loss: np.ndarray      # per time step
    total_loss: np.ndarray     # per time step
    socp_gap: np.ndarray       # [T, n_branches]


# =============================================================================
# NREL-STYLE PV AND LOAD PROFILES
# =============================================================================

def generate_pv_profile(T: int = 24) -> np.ndarray:
    """Generate normalized PV generation profile (0-1) over T hours.
    
    Based on typical tropical/subtropical irradiance curve.
    Sunrise ~6am, peak ~12pm, sunset ~18pm.
    """
    hours = np.arange(T)
    pv = np.zeros(T)
    for t in range(T):
        if 6 <= t <= 18:
            # Bell-shaped curve centered at hour 12
            pv[t] = np.exp(-0.5 * ((t - 12) / 2.5) ** 2)
    return pv / max(pv.max(), 1e-8)  # normalize to [0, 1]


def generate_load_profile(T: int = 24) -> np.ndarray:
    """Generate normalized residential load profile (0-1) over T hours.
    
    Two peaks: morning (~8am) and evening (~19pm).
    Minimum overnight.
    """
    hours = np.arange(T)
    load = np.zeros(T)
    for t in range(T):
        # Base load
        base = 0.3
        # Morning peak
        morning = 0.3 * np.exp(-0.5 * ((t - 8) / 1.5) ** 2)
        # Evening peak (larger)
        evening = 0.7 * np.exp(-0.5 * ((t - 19) / 2.0) ** 2)
        # Overnight dip
        night = -0.15 * np.exp(-0.5 * ((t - 3) / 2.0) ** 2)
        load[t] = base + morning + evening + night
    return load / load.max()  # normalize to [0, 1]


# =============================================================================
# STATIC OPF SOLVER (SINGLE TIME STEP)
# =============================================================================

def solve_opf_static(
    sys: SystemData,
    formulation: str = "OPF2",
    solver: str = "CLARABEL",
    verbose: bool = False,
) -> OPFResult:
    """Solve SOCP-relaxed OPF for a single time step.
    
    OPF-1: min Σ r_ij · l_ij                    (distribution losses only)
    OPF-2: min Σ r_ij · l_ij + Σ P_conv_i       (distribution + converter losses)
    
    Subject to BFM constraints, battery limits, voltage limits.
    """
    N = sys.n_buses
    E = sys.n_branches

    # Decision variables
    v = cp.Variable(N, name="v")
    P_G = cp.Variable(N, name="P_G")
    P_B = cp.Variable(N, name="P_B")
    P_inj = cp.Variable(N, name="P_inj")
    SOC = cp.Variable(N, name="SOC")
    P_ij = cp.Variable(E, name="P_ij")
    P_ji = cp.Variable(E, name="P_ji")
    l_ij = cp.Variable(E, name="l_ij", nonneg=True)

    use_conv = formulation.upper() == "OPF2"
    if use_conv:
        P_conv = cp.Variable(N, name="P_conv", nonneg=True)
        P_o = cp.Variable(N, name="P_o", nonneg=True)
    else:
        P_conv = np.zeros(N)

    # Parameters
    r = np.array([br.r for br in sys.branches])
    I_max_sq = np.array([br.I_max**2 for br in sys.branches])
    P_L = np.array([bus.P_L for bus in sys.buses])
    from_idx = np.array([br.from_bus for br in sys.branches])
    to_idx = np.array([br.to_bus for br in sys.branches])

    constraints = []

    # (C1) Power balance: P_G - P_L - P_B - P_conv = P_inj
    for i in range(N):
        constraints.append(P_G[i] - P_L[i] - P_B[i] - P_conv[i] == P_inj[i])

    # (C2) Net injection = sum of outgoing flows from bus i
    # For bus i: P_inj_i = Σ_{e: from=i} P_ij[e] + Σ_{e: to=i} P_ji[e]
    for i in range(N):
        flow_terms = []
        for e in range(E):
            if from_idx[e] == i:
                flow_terms.append(P_ij[e])
            if to_idx[e] == i:
                flow_terms.append(P_ji[e])
        if flow_terms:
            constraints.append(P_inj[i] == sum(flow_terms))
        else:
            constraints.append(P_inj[i] == 0)

    # (C3) Branch power loss: P_ij + P_ji = r_ij · l_ij
    for e in range(E):
        constraints.append(P_ij[e] + P_ji[e] == r[e] * l_ij[e])

    # (C4) Voltage drop: v_i - v_j = r_ij · (P_ij - P_ji)
    for e in range(E):
        constraints.append(v[from_idx[e]] - v[to_idx[e]] == r[e] * (P_ij[e] - P_ji[e]))

    # (C5) SOCP relaxation: P_ij² ≤ v_i · l_ij
    for e in range(E):
        constraints.append(cp.quad_over_lin(P_ij[e], v[from_idx[e]]) <= l_ij[e])

    # (C6) Current limits
    for e in range(E):
        constraints.append(l_ij[e] <= I_max_sq[e])

    # (C7) Voltage limits
    constraints += [v >= sys.v_min, v <= sys.v_max]

    # (C8) Generation limits
    for i in range(N):
        if sys.buses[i].has_pv:
            constraints += [P_G[i] >= sys.buses[i].P_G_min,
                           P_G[i] <= sys.buses[i].P_G_max]
        else:
            constraints.append(P_G[i] == 0.0)

    # (C9) Battery SOC
    for i in range(N):
        if sys.buses[i].has_battery:
            constraints.append(SOC[i] == sys.buses[i].SOC_init + P_B[i] / sys.buses[i].C_B)
            constraints += [SOC[i] >= sys.buses[i].SOC_min, SOC[i] <= sys.buses[i].SOC_max]
        else:
            constraints += [P_B[i] == 0.0, SOC[i] == 0.0]

    # (C10) Converter loss model (OPF-2)
    if use_conv:
        conv = sys.converter
        for i in range(N):
            constraints += [P_o[i] >= P_inj[i], P_o[i] >= -P_inj[i]]
            constraints.append(P_conv[i] >= conv.alpha + conv.beta * P_o[i] + conv.gamma * cp.square(P_o[i]))

    # Objective
    dist_loss_expr = cp.sum(cp.multiply(r, l_ij))
    if use_conv:
        objective = cp.Minimize(dist_loss_expr + cp.sum(P_conv))
    else:
        objective = cp.Minimize(dist_loss_expr)

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=verbose)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        nan_n, nan_e = np.full(N, np.nan), np.full(E, np.nan)
        return OPFResult(problem.status, np.nan, nan_n, nan_n, nan_n, nan_n,
                        nan_n, nan_n, nan_e, nan_e, nan_e, np.nan, np.nan, np.nan, nan_e)

    # Extract
    v_val, P_G_val, P_B_val = v.value, P_G.value, P_B.value
    P_inj_val, SOC_val = P_inj.value, SOC.value
    P_ij_val, P_ji_val, l_ij_val = P_ij.value, P_ji.value, l_ij.value
    P_conv_val = P_conv.value if use_conv else np.zeros(N)

    dist_loss_val = float(np.sum(r * l_ij_val))
    conv_loss_val = float(np.sum(P_conv_val))
    socp_gap = np.array([v_val[from_idx[e]] * l_ij_val[e] - P_ij_val[e]**2 for e in range(E)])

    return OPFResult(
        status=problem.status, objective=float(problem.value),
        v=v_val, P_G=P_G_val, P_B=P_B_val, P_inj=P_inj_val,
        SOC=SOC_val, P_conv=P_conv_val, P_ij=P_ij_val, P_ji=P_ji_val,
        l_ij=l_ij_val, dist_loss=dist_loss_val, conv_loss=conv_loss_val,
        total_loss=dist_loss_val + conv_loss_val, socp_gap=socp_gap)


# =============================================================================
# MULTI-PERIOD OPF SOLVER (24-HOUR)
# =============================================================================

def solve_opf_multiperiod(
    sys: SystemData,
    pv_profile: np.ndarray,
    load_profile: np.ndarray,
    formulation: str = "OPF2",
    solver: str = "CLARABEL",
    verbose: bool = False,
) -> MultiPeriodResult:
    """Solve SOCP-relaxed OPF over T time steps with SOC coupling.
    
    Parameters
    ----------
    sys : SystemData
        System description (P_G, P_L on buses are PEAK values scaled by profiles).
    pv_profile : array (T,)
        Normalized PV multiplier per hour [0,1].
    load_profile : array (T,)
        Normalized load multiplier per hour [0,1].
    formulation : str
        "OPF1" or "OPF2".
    """
    N = sys.n_buses
    E = sys.n_branches
    T = len(pv_profile)
    use_conv = formulation.upper() == "OPF2"

    # Parameters
    r = np.array([br.r for br in sys.branches])
    I_max_sq = np.array([br.I_max**2 for br in sys.branches])
    from_idx = np.array([br.from_bus for br in sys.branches])
    to_idx = np.array([br.to_bus for br in sys.branches])

    P_G_peak = np.array([bus.P_G for bus in sys.buses])
    P_L_peak = np.array([bus.P_L for bus in sys.buses])

    # Decision variables: [T, ...]
    v = cp.Variable((T, N), name="v")
    P_G = cp.Variable((T, N), name="P_G")
    P_B = cp.Variable((T, N), name="P_B")
    P_inj = cp.Variable((T, N), name="P_inj")
    SOC = cp.Variable((T + 1, N), name="SOC")  # T+1 for initial + T steps
    P_flow = cp.Variable((T, E), name="P_ij")
    P_rev = cp.Variable((T, E), name="P_ji")
    l = cp.Variable((T, E), name="l_ij", nonneg=True)

    if use_conv:
        P_conv = cp.Variable((T, N), name="P_conv", nonneg=True)
        P_o = cp.Variable((T, N), name="P_o", nonneg=True)
    else:
        P_conv = np.zeros((T, N))

    constraints = []

    # Initial SOC
    for i in range(N):
        constraints.append(SOC[0, i] == sys.buses[i].SOC_init)

    for t in range(T):
        # Time-varying profiles
        P_G_max_t = P_G_peak * pv_profile[t]  # available PV at time t
        P_L_t = P_L_peak * load_profile[t]      # load at time t

        for i in range(N):
            # (C1) Power balance
            constraints.append(P_G[t, i] - P_L_t[i] - P_B[t, i] - P_conv[t, i] == P_inj[t, i])

            # (C2) Net injection
            flow_terms = []
            for e in range(E):
                if from_idx[e] == i:
                    flow_terms.append(P_flow[t, e])
                if to_idx[e] == i:
                    flow_terms.append(P_rev[t, e])
            if flow_terms:
                constraints.append(P_inj[t, i] == sum(flow_terms))
            else:
                constraints.append(P_inj[t, i] == 0)

            # (C8) Generation limits
            if sys.buses[i].has_pv:
                constraints += [P_G[t, i] >= 0, P_G[t, i] <= P_G_max_t[i]]
            else:
                constraints.append(P_G[t, i] == 0.0)

            # (C9) SOC dynamics: SOC[t+1] = SOC[t] + P_B[t] / C_B
            if sys.buses[i].has_battery:
                constraints.append(SOC[t + 1, i] == SOC[t, i] + P_B[t, i] / sys.buses[i].C_B)
                constraints += [SOC[t + 1, i] >= sys.buses[i].SOC_min,
                               SOC[t + 1, i] <= sys.buses[i].SOC_max]
            else:
                constraints += [P_B[t, i] == 0.0, SOC[t + 1, i] == 0.0]

            # (C10) Converter loss (OPF-2)
            if use_conv:
                conv = sys.converter
                constraints += [P_o[t, i] >= P_inj[t, i], P_o[t, i] >= -P_inj[t, i]]
                constraints.append(P_conv[t, i] >= conv.alpha + conv.beta * P_o[t, i]
                                  + conv.gamma * cp.square(P_o[t, i]))

        for e in range(E):
            # (C3) Branch loss
            constraints.append(P_flow[t, e] + P_rev[t, e] == r[e] * l[t, e])
            # (C4) Voltage drop
            constraints.append(v[t, from_idx[e]] - v[t, to_idx[e]]
                             == r[e] * (P_flow[t, e] - P_rev[t, e]))
            # (C5) SOCP relaxation
            constraints.append(cp.quad_over_lin(P_flow[t, e], v[t, from_idx[e]]) <= l[t, e])
            # (C6) Current limit
            constraints.append(l[t, e] <= I_max_sq[e])

        # (C7) Voltage limits
        constraints += [v[t, :] >= sys.v_min, v[t, :] <= sys.v_max]

    # Objective: sum over all time steps
    dist_loss_expr = cp.sum(cp.multiply(r[np.newaxis, :], l))
    if use_conv:
        objective = cp.Minimize(dist_loss_expr + cp.sum(P_conv))
    else:
        objective = cp.Minimize(dist_loss_expr)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=verbose)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        warnings.warn(f"Solver status: {problem.status}")
        return None

    # Extract per-timestep losses
    v_val = v.value
    P_conv_val = P_conv.value if use_conv else np.zeros((T, N))
    l_val = l.value

    dist_loss_t = np.array([np.sum(r * l_val[t]) for t in range(T)])
    conv_loss_t = np.array([np.sum(P_conv_val[t]) for t in range(T)])

    socp_gap = np.zeros((T, E))
    for t in range(T):
        for e in range(E):
            socp_gap[t, e] = v_val[t, from_idx[e]] * l_val[t, e] - P_flow.value[t, e]**2

    return MultiPeriodResult(
        status=problem.status, objective=float(problem.value), T=T,
        v=v_val, P_G=P_G.value, P_B=P_B.value, P_inj=P_inj.value,
        SOC=SOC.value, P_conv=P_conv_val, P_ij=P_flow.value, P_ji=P_rev.value,
        l_ij=l_val, dist_loss=dist_loss_t, conv_loss=conv_loss_t,
        total_loss=dist_loss_t + conv_loss_t, socp_gap=socp_gap)


# =============================================================================
# TEST CASE BUILDERS
# =============================================================================

def build_tc1_7bus() -> SystemData:
    """TC1: 7-bus radial (single cluster, debug/validation).
    
    Topology:  1 --- 2 --- 3 --- 4
                     |
                     5 --- 6 --- 7
    Bus 1: community load. Buses 2-7: prosumers.
    Sized so total PV energy > total load energy over 24h.
    """
    buses = [
        Bus(P_G=0.0, P_L=0.30, SOC_init=0.0, has_pv=False, has_battery=False, P_G_max=0.0),
        Bus(P_G=0.50, P_L=0.10, SOC_init=0.80, C_B=2.0, P_G_max=0.60),
        Bus(P_G=0.55, P_L=0.08, SOC_init=0.75, C_B=2.0, P_G_max=0.65),
        Bus(P_G=0.45, P_L=0.12, SOC_init=0.70, C_B=2.0, P_G_max=0.55),
        Bus(P_G=0.50, P_L=0.09, SOC_init=0.78, C_B=2.0, P_G_max=0.60),
        Bus(P_G=0.48, P_L=0.11, SOC_init=0.72, C_B=2.0, P_G_max=0.58),
        Bus(P_G=0.52, P_L=0.10, SOC_init=0.76, C_B=2.0, P_G_max=0.62),
    ]
    branches = [
        Branch(0, 1, r=0.05), Branch(1, 2, r=0.04), Branch(2, 3, r=0.06),
        Branch(1, 4, r=0.05), Branch(4, 5, r=0.04), Branch(5, 6, r=0.05),
    ]
    return SystemData(buses=buses, branches=branches, name="TC1-7bus")


def build_tc2_14bus() -> SystemData:
    """TC2: Modified IEEE 14-bus DC microgrid (Khan et al. validation).
    
    Bus 14: community load. Buses 1-13: prosumers with varied PV/load.
    Branch data from IEEE 14-bus with reactance zeroed, resistance reduced 10%.
    """
    # Prosumer data from Khan et al. Table 1 (static case, varied)
    prosumer_data = [
        # (P_G_peak, P_L, SOC_init) — sized for 24h energy balance
        (0.50, 0.10, 0.83), (0.55, 0.12, 0.70), (0.45, 0.08, 0.84),
        (0.48, 0.11, 0.72), (0.52, 0.09, 0.84), (0.50, 0.12, 0.83),
        (0.45, 0.10, 0.82), (0.55, 0.12, 0.70), (0.48, 0.10, 0.82),
        (0.50, 0.08, 0.83), (0.52, 0.12, 0.70), (0.45, 0.10, 0.85),
        (0.48, 0.09, 0.79),
    ]
    buses = []
    for pg, pl, soc in prosumer_data:
        buses.append(Bus(P_G=pg, P_L=pl, SOC_init=soc, C_B=2.0, P_G_max=pg * 1.2))
    # Bus 14: community load
    buses.append(Bus(P_G=0.0, P_L=0.35, SOC_init=0.0, has_pv=False,
                    has_battery=False, P_G_max=0.0))

    branch_data = [
        (0,1,0.01938), (0,4,0.05403), (1,2,0.04699), (1,3,0.05811),
        (1,4,0.05695), (2,3,0.06701), (3,4,0.01335), (3,6,0.02),
        (3,8,0.02), (4,5,0.02), (5,10,0.09498), (5,11,0.12291),
        (5,12,0.06615), (6,7,0.02), (6,8,0.11001), (7,13,0.02),
        (8,9,0.03181), (8,13,0.12711), (9,10,0.08205), (11,12,0.19797),
    ]
    branches = [Branch(f, t, r=max(rv * 0.9, 0.005)) for f, t, rv in branch_data]
    return SystemData(buses=buses, branches=branches, name="TC2-14bus")


def build_tc3_20bus() -> SystemData:
    """TC3: 20-bus clustered ring microgrid (4×5 clusters).
    
    Novel contribution. 4 clusters of 5 nanogrids each.
    Within each cluster: ring topology (5 nodes in a ring).
    Between clusters: gateway nodes form a higher-level ring.
    Bus 20: community load at gateway.
    
    Cluster 1: buses 0-4  (gateway: bus 0)
    Cluster 2: buses 5-9  (gateway: bus 5)
    Cluster 3: buses 10-14 (gateway: bus 10)
    Cluster 4: buses 15-18 (gateway: bus 15)
    Bus 19: community load
    """
    np.random.seed(42)
    buses = []
    for i in range(19):
        pg = 0.35 + 0.20 * np.random.rand()
        pl = 0.08 + 0.08 * np.random.rand()
        soc = 0.70 + 0.15 * np.random.rand()
        buses.append(Bus(P_G=round(pg, 3), P_L=round(pl, 3),
                        SOC_init=round(soc, 3), C_B=2.5, P_G_max=round(pg * 1.3, 3)))
    # Bus 20: community load
    buses.append(Bus(P_G=0.0, P_L=0.50, SOC_init=0.0, has_pv=False,
                    has_battery=False, P_G_max=0.0))

    branches = []
    r_intra = 0.04  # intra-cluster resistance
    r_inter = 0.06  # inter-cluster resistance

    # Cluster rings
    for c_start in [0, 5, 10, 15]:
        c_size = 5 if c_start < 15 else 4
        for k in range(c_size):
            i = c_start + k
            j = c_start + (k + 1) % c_size
            branches.append(Branch(i, j, r=r_intra + 0.01 * np.random.rand()))

    # Inter-cluster ring: gateway nodes 0-5-10-15-0
    gateways = [0, 5, 10, 15]
    for k in range(4):
        branches.append(Branch(gateways[k], gateways[(k + 1) % 4], r=r_inter + 0.01 * np.random.rand()))

    # Community load connected to gateway 0
    branches.append(Branch(0, 19, r=0.03))

    return SystemData(buses=buses, branches=branches, name="TC3-20bus")


# =============================================================================
# VERIFICATION UTILITIES
# =============================================================================

def verify_socp_exactness(result, tol=1e-4):
    """Check SOCP relaxation tightness."""
    if isinstance(result, MultiPeriodResult):
        gaps = result.socp_gap
    else:
        gaps = result.socp_gap
    max_gap = np.max(np.abs(gaps))
    return max_gap < tol, max_gap


def compute_converter_efficiency(P_inj, conv: ConverterParams):
    """Compute converter efficiency at each bus from dispatch."""
    x = np.abs(P_inj) / conv.P_R
    eta = conv.k0 + conv.k1 * x + conv.k2 * x**2
    eta = np.clip(eta, 0.01, 1.0)
    return eta


# =============================================================================
# GRAPH GENERATION
# =============================================================================

def generate_all_figures(sys: SystemData, res1: MultiPeriodResult, res2: MultiPeriodResult,
                         pv_prof, load_prof, output_dir: str = "/home/claude/figures"):
    """Generate all publication-quality figures for the manuscript."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os
    os.makedirs(output_dir, exist_ok=True)

    T = res1.T
    hours = np.arange(T)
    N = sys.n_buses

    # Consistent styling
    plt.rcParams.update({
        'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
        'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'lines.linewidth': 1.5,
        'font.family': 'serif',
    })
    colors_opf1 = '#2196F3'
    colors_opf2 = '#E53935'

    # --- Fig 1: PV and Load Profiles ---
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(hours, pv_prof, 'o-', color='#FFA000', label='PV Generation Multiplier', markersize=4)
    ax.plot(hours, load_prof, 's-', color='#5C6BC0', label='Load Demand Multiplier', markersize=4)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Normalized Multiplier')
    ax.set_title(f'PV and Load Profiles — {sys.name}')
    ax.legend()
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    fig.savefig(f'{output_dir}/fig1_profiles_{sys.name}.png')
    plt.close()

    # --- Fig 2: Total System Losses Over Time (OPF-1 vs OPF-2) ---
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(hours, res1.total_loss, 'o-', color=colors_opf1, label='OPF-1 (dist. only)', markersize=4)
    ax.plot(hours, res2.total_loss, 's-', color=colors_opf2, label='OPF-2 (dist. + conv.)', markersize=4)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Total System Losses (p.u.)')
    ax.set_title(f'Total Losses: OPF-1 vs OPF-2 — {sys.name}')
    ax.legend()
    ax.set_xlim(0, T - 1)
    ax.grid(True, alpha=0.3)
    fig.savefig(f'{output_dir}/fig2_total_losses_{sys.name}.png')
    plt.close()

    # --- Fig 3: Distribution vs Converter Losses Stacked ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for ax, res, label in [(axes[0], res1, 'OPF-1'), (axes[1], res2, 'OPF-2')]:
        ax.bar(hours, res.dist_loss, color='#42A5F5', label='Distribution', alpha=0.85)
        ax.bar(hours, res.conv_loss, bottom=res.dist_loss, color='#EF5350', label='Converter', alpha=0.85)
        ax.set_xlabel('Hour of Day')
        ax.set_title(f'{label}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Losses (p.u.)')
    fig.suptitle(f'Loss Breakdown — {sys.name}', fontsize=13, y=1.02)
    fig.savefig(f'{output_dir}/fig3_loss_breakdown_{sys.name}.png')
    plt.close()

    # --- Fig 4: Power Scheduled at Each Bus (OPF-1 and OPF-2) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, res, label in [(axes[0], res1, 'OPF-1'), (axes[1], res2, 'OPF-2')]:
        for i in range(min(N, 10)):  # plot first 10 buses
            ax.plot(hours, res.P_inj[:, i], label=f'Bus {i+1}', alpha=0.7, linewidth=1.0)
        ax.set_xlabel('Hour of Day')
        ax.set_title(f'Power Injection — {label}')
        ax.legend(fontsize=6, ncol=2, loc='upper left')
        ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Net Power Injection (p.u.)')
    fig.suptitle(f'Scheduled Power Dispatch — {sys.name}', fontsize=13, y=1.02)
    fig.savefig(f'{output_dir}/fig4_power_dispatch_{sys.name}.png')
    plt.close()

    # --- Fig 5: Converter Efficiency Over Time ---
    conv = sys.converter
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, res, label in [(axes[0], res1, 'OPF-1'), (axes[1], res2, 'OPF-2')]:
        for i in range(min(N, 10)):
            eff_t = []
            for t in range(T):
                x = abs(res.P_inj[t, i]) / conv.P_R
                eta = conv.k0 + conv.k1 * x + conv.k2 * x**2
                eff_t.append(max(eta * 100, 0))
            ax.scatter([hours] * 1, eff_t, s=12, alpha=0.6, label=f'Bus {i+1}')
        ax.set_xlabel('Hour of Day')
        ax.set_title(f'Converter Efficiency — {label}')
        ax.set_ylim(50, 100)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Converter Efficiency (%)')
    fig.suptitle(f'Converter Efficiency Comparison — {sys.name}', fontsize=13, y=1.02)
    fig.savefig(f'{output_dir}/fig5_converter_eff_{sys.name}.png')
    plt.close()

    # --- Fig 6: Number of Active Converters ---
    thresh = 0.01
    active1 = np.sum(np.abs(res1.P_inj) > thresh, axis=1)
    active2 = np.sum(np.abs(res2.P_inj) > thresh, axis=1)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    width = 0.35
    ax.bar(hours - width/2, active1, width, color=colors_opf1, label='OPF-1', alpha=0.85)
    ax.bar(hours + width/2, active2, width, color=colors_opf2, label='OPF-2', alpha=0.85)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Number of Active Converters')
    ax.set_title(f'Operating Converters — {sys.name}')
    ax.legend()
    ax.set_xlim(-0.5, T - 0.5)
    ax.grid(True, alpha=0.3)
    fig.savefig(f'{output_dir}/fig6_active_converters_{sys.name}.png')
    plt.close()

    # --- Fig 7: Battery SOC Trajectories ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, res, label in [(axes[0], res1, 'OPF-1'), (axes[1], res2, 'OPF-2')]:
        for i in range(min(N, 10)):
            if sys.buses[i].has_battery:
                ax.plot(np.arange(T + 1), res.SOC[:, i] * 100, label=f'Bus {i+1}', alpha=0.7)
        ax.axhline(y=sys.buses[0].SOC_max * 100, color='r', linestyle='--', linewidth=0.8, label='SOC limits')
        ax.axhline(y=sys.buses[0].SOC_min * 100, color='r', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Hour')
        ax.set_title(f'Battery SOC — {label}')
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('State of Charge (%)')
    fig.suptitle(f'Battery SOC Trajectories — {sys.name}', fontsize=13, y=1.02)
    fig.savefig(f'{output_dir}/fig7_soc_{sys.name}.png')
    plt.close()

    # --- Fig 8: Voltage Profiles ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, res, label in [(axes[0], res1, 'OPF-1'), (axes[1], res2, 'OPF-2')]:
        # Plot voltage at peak load hour
        t_peak = np.argmax(load_prof)
        V_pu = np.sqrt(np.maximum(res.v[t_peak], 0))
        ax.bar(np.arange(N), V_pu, color='#66BB6A', alpha=0.85)
        ax.axhline(y=np.sqrt(sys.v_min), color='r', linestyle='--', linewidth=1, label='V limits')
        ax.axhline(y=np.sqrt(sys.v_max), color='r', linestyle='--', linewidth=1)
        ax.set_xlabel('Bus Number')
        ax.set_title(f'Voltage at Peak Load (t={t_peak}) — {label}')
        ax.set_ylim(0.93, 1.07)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Voltage (p.u.)')
    fig.suptitle(f'Bus Voltage Profiles — {sys.name}', fontsize=13, y=1.02)
    fig.savefig(f'{output_dir}/fig8_voltage_{sys.name}.png')
    plt.close()

    # --- Fig 9: SOCP Gap Verification ---
    fig, ax = plt.subplots(figsize=(7, 3.5))
    max_gaps_1 = np.max(np.abs(res1.socp_gap), axis=1)
    max_gaps_2 = np.max(np.abs(res2.socp_gap), axis=1)
    ax.semilogy(hours, max_gaps_1 + 1e-15, 'o-', color=colors_opf1, label='OPF-1', markersize=4)
    ax.semilogy(hours, max_gaps_2 + 1e-15, 's-', color=colors_opf2, label='OPF-2', markersize=4)
    ax.axhline(y=1e-4, color='gray', linestyle='--', label='Exactness tol (1e-4)')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Max SOCP Gap')
    ax.set_title(f'SOCP Relaxation Tightness — {sys.name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(f'{output_dir}/fig9_socp_gap_{sys.name}.png')
    plt.close()

    print(f"  [✓] Generated 9 figures in {output_dir}/")


def generate_summary_table(results_dict: dict, output_dir: str = "/home/claude/figures"):
    """Generate LaTeX-ready summary comparison table."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')

    col_labels = ['Test Case', 'Buses', 'Branches',
                  'OPF-1\nDist. Loss', 'OPF-1\nTotal Loss',
                  'OPF-2\nDist. Loss', 'OPF-2\nConv. Loss', 'OPF-2\nTotal Loss',
                  'Δ Total\nLoss (%)']
    rows = []
    for name, (sys, r1, r2) in results_dict.items():
        dl1 = np.sum(r1.dist_loss)
        tl1 = np.sum(r1.total_loss)
        dl2 = np.sum(r2.dist_loss)
        cl2 = np.sum(r2.conv_loss)
        tl2 = np.sum(r2.total_loss)
        pct = (tl2 - tl1) / max(tl1, 1e-10) * 100
        rows.append([name, sys.n_buses, sys.n_branches,
                    f'{dl1:.4f}', f'{tl1:.4f}',
                    f'{dl2:.4f}', f'{cl2:.4f}', f'{tl2:.4f}',
                    f'{pct:+.1f}%'])

    table = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#E3F2FD')
            cell.set_text_props(weight='bold')
    fig.savefig(f'{output_dir}/fig_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [✓] Summary table saved")


# =============================================================================
# MAIN — RUN ALL TEST CASES
# =============================================================================

if __name__ == "__main__":
    import time
    import os

    print("=" * 72)
    print("  OPTIMAL POWER SHARING IN DC MICROGRIDS")
    print("  OPF-1 vs OPF-2 Comparison — All Test Cases")
    print("=" * 72)

    pv_prof = generate_pv_profile(24)
    load_prof = generate_load_profile(24)

    test_cases = {
        "TC1": build_tc1_7bus(),
        "TC2": build_tc2_14bus(),
        "TC3": build_tc3_20bus(),
    }

    all_results = {}

    for tc_name, sys in test_cases.items():
        print(f"\n{'─' * 72}")
        print(f"  {tc_name}: {sys.name} ({sys.n_buses} buses, {sys.n_branches} branches)")
        print(f"{'─' * 72}")

        t0 = time.time()

        print(f"  Solving OPF-1 (24h, distribution losses only)...")
        res1 = solve_opf_multiperiod(sys, pv_prof, load_prof, "OPF1", solver="CLARABEL")
        if res1 is None:
            print(f"  [✗] OPF-1 FAILED for {tc_name}")
            continue
        exact1, gap1 = verify_socp_exactness(res1)
        print(f"    Status: {res1.status}, Obj: {res1.objective:.6f}, "
              f"SOCP exact: {exact1} (gap={gap1:.2e})")

        print(f"  Solving OPF-2 (24h, distribution + converter losses)...")
        res2 = solve_opf_multiperiod(sys, pv_prof, load_prof, "OPF2", solver="CLARABEL")
        if res2 is None:
            print(f"  [✗] OPF-2 FAILED for {tc_name}")
            continue
        exact2, gap2 = verify_socp_exactness(res2)
        print(f"    Status: {res2.status}, Obj: {res2.objective:.6f}, "
              f"SOCP exact: {exact2} (gap={gap2:.2e})")

        dt = time.time() - t0
        print(f"  Solve time: {dt:.1f}s")

        # Summary
        print(f"\n  {'Metric':<25s}  {'OPF-1':>10s}  {'OPF-2':>10s}")
        print(f"  {'─' * 47}")
        print(f"  {'Total dist. loss (24h)':<25s}  {np.sum(res1.dist_loss):>10.4f}  {np.sum(res2.dist_loss):>10.4f}")
        print(f"  {'Total conv. loss (24h)':<25s}  {np.sum(res1.conv_loss):>10.4f}  {np.sum(res2.conv_loss):>10.4f}")
        print(f"  {'Total system loss (24h)':<25s}  {np.sum(res1.total_loss):>10.4f}  {np.sum(res2.total_loss):>10.4f}")

        all_results[tc_name] = (sys, res1, res2)

        # Generate figures
        print(f"\n  Generating figures...")
        generate_all_figures(sys, res1, res2, pv_prof, load_prof)

    # Summary table
    if all_results:
        generate_summary_table(all_results)

    # Copy all figures to output
    os.makedirs("/mnt/user-data/outputs/figures", exist_ok=True)
    os.system("cp /home/claude/figures/*.png /mnt/user-data/outputs/figures/ 2>/dev/null")
    print(f"\n{'=' * 72}")
    print(f"  ALL DONE — figures in /mnt/user-data/outputs/figures/")
    print(f"{'=' * 72}")
