import numpy as np
import matplotlib.pyplot as plt
from iapws import IAPWS97

# ==========================================
# 1. 基础设计参数 (AP1000)
# ==========================================
params = {
    "P_sys": 15.51,            # 系统压力 MPa
    "Q_total": 3400e6,         # 额定热功率 W
    "W_total": 14314,          # 总冷却剂流量 kg/s
    "T_in": 279.4,             # 堆芯入口温度 °C
    "H_core": 4.2672,          # 堆芯有效高度 m
    "N_assy": 157,
    "N_rods": 264,
    "D_out": 0.0095,           # 包壳外径 m
    "D_in": 0.00836,           # 包壳内径 m
    "D_pellet": 0.00819,       # 芯块直径 m
    "Pitch": 0.0126,           # 栅距 m
    "F_u": 0.974,              # 燃料发热份额
    "Bypass": 0.059,           # 旁流系数
    "Fq_N": 2.524,             # 核热点因子 (极值约束)
    "FdH_N": 1.65,             # 核焓升因子 (径向因子)
    "Fq_E": 1.03,              # 工程热点因子 (局部制造偏差)
    "FdH_E": 1.085,            # 流量不确定性因子
    # ---- 压降参数 ----
    "N_grids": 8,
    "K_grid": 0.6,
    "g": 9.81,
}

# ==========================================
# 2. 物理关联式模块
# ==========================================
def get_uo2_k(T_c):
    """UO2 导热系数 (W/m·K)，IAEA推荐，95%TD新鲜燃料"""
    Tk = T_c + 273.15
    return 100.0 / (6.75 + 0.0235 * Tk) + 6.4e9 * np.exp(-16350.0 / Tk) / Tk**2


def get_gap_conductance(q_linear):
    """气隙等效导热系数 (W/m²K)，随线功率变化"""
    if q_linear > 25000:
        return 10000.0
    elif q_linear > 15000:
        frac = (q_linear - 15000.0) / 10000.0
        return 5000.0 + 5000.0 * frac
    else:
        return 5000.0


def jens_lottes_wall_temp(q_flux, P_mpa, T_sat):
    """
    Jens-Lottes 过冷核态沸腾壁温
    ΔT_sat = 25.0 * (q''/1e6)^0.25 * exp(-P/6.2)
    """
    dT = 25.0 * (q_flux / 1e6)**0.25 * np.exp(-P_mpa / 6.2)
    return T_sat + dT


def calculate_w3_full(P, h_local, h_in, q_local, D_h, G):
    """W-3 CHF关联式，返回DNBR"""
    if q_local < 1.0:
        return 99.9
    try:
        sat_f = IAPWS97(P=P, x=0)
        sat_g = IAPWS97(P=P, x=1)
        h_f = sat_f.h * 1000.0
        h_g = sat_g.h * 1000.0
        x = (h_local - h_f) / (h_g - h_f)

        P_psi = P * 145.038
        G_Mlb = (G * 0.2048 * 3600) / 1e6
        De_inch = D_h * 39.37
        h_f_btu = h_f * 0.0004299
        h_in_btu = h_in * 0.0004299

        term1 = 2.022 - 0.0004302 * P_psi
        term2 = (0.1722 - 0.0000984 * P_psi) * np.exp(
            (18.177 - 0.004129 * P_psi) * x
        )
        term3 = (0.1484 - 1.596 * x + 0.1729 * x * abs(x)) * G_Mlb + 1.037
        term4 = 1.157 - 0.869 * x
        term5 = 0.2664 + 0.8357 * np.exp(-3.151 * De_inch)
        term6 = 0.8258 + 0.000794 * (h_f_btu - h_in_btu)

        q_chf = (term1 + term2) * term3 * term4 * term5 * term6 * 1e6 * 3.15459
        return q_chf / q_local
    except Exception:
        return 1.3


# ==========================================
# 3. 轴向功率分布构造
# ==========================================
def build_axial_profile(z_nodes, H, Fq_N, FdH_N):
    """
    构造轴向功率分布 φ(z)，满足：
      - ∫φ dz / H = 1  (归一化)
      - φ_max = F_q / F_dH = F_z  (峰化因子约束)
    
    方法：使用截断余弦 + 功率偏移因子
      φ(z) = A * cos(π(z - z_peak)/H_e) + B
    其中 z_peak 允许轻微偏离中心（模拟燃耗/毒物效应）
    
    这里采用更简洁的方法：对正弦分布施加展平变换
      φ_raw = sin^α(πz/H)，调节 α 使峰值因子匹配
    """
    F_z_target = Fq_N / FdH_N  # = 2.524 / 1.65 = 1.530

    # 用 sin^α 分布：峰值始终为1，均值随 α 变化
    # 需要找到 α 使得 max/mean = F_z_target
    # 即 1/mean(sin^α) = F_z_target → mean(sin^α) = 1/F_z_target

    target_mean = 1.0 / F_z_target

    # 二分法求 α
    alpha_lo, alpha_hi = 0.01, 5.0
    for _ in range(100):
        alpha_mid = (alpha_lo + alpha_hi) / 2.0
        phi_test = np.sin(np.pi * z_nodes / H) ** alpha_mid
        mean_test = np.mean(phi_test)
        if mean_test > target_mean:
            alpha_lo = alpha_mid
        else:
            alpha_hi = alpha_mid
        if abs(mean_test - target_mean) < 1e-8:
            break

    alpha = (alpha_lo + alpha_hi) / 2.0
    phi_raw = np.sin(np.pi * z_nodes / H) ** alpha
    phi = phi_raw / np.mean(phi_raw)  # 归一化使均值 = 1

    return phi, alpha


# ==========================================
# 4. 主计算流程
# ==========================================
def run_simulation():
    n = 100  # 增加节点数以提高精度
    z_nodes = np.linspace(0, params["H_core"], n)
    dz = params["H_core"] / n

    # --------------------------------------------------
    # 【关键修正】构造满足 F_q 约束的轴向分布
    # --------------------------------------------------
    phi_z, alpha = build_axial_profile(
        z_nodes, params["H_core"], params["Fq_N"], params["FdH_N"]
    )

    F_z_actual = np.max(phi_z)
    F_z_target = params["Fq_N"] / params["FdH_N"]
    Fq_check = params["FdH_N"] * F_z_actual

    print(f"[轴向分布] 展平指数 α = {alpha:.4f}")
    print(f"[轴向分布] φ_max = {F_z_actual:.4f} (目标 F_z = {F_z_target:.4f})")
    print(f"[轴向分布] φ_mean = {np.mean(phi_z):.4f} (应为 1.0)")
    print(f"[校核] F_dH × φ_max = {params['FdH_N']:.3f} × {F_z_actual:.3f} = {Fq_check:.3f}")
    print(f"[校核] F_q 设计限值 = {params['Fq_N']:.3f}")
    if abs(Fq_check - params["Fq_N"]) < 0.01:
        print(f"  ✓ F_q 约束满足 (偏差 {abs(Fq_check - params['Fq_N']):.4f})")
    else:
        print(f"  ⚠ F_q 约束偏差 = {abs(Fq_check - params['Fq_N']):.4f}")

    # 几何与流量
    total_rods = params["N_assy"] * params["N_rods"]
    A_flow = params["Pitch"]**2 - np.pi / 4 * params["D_out"]**2
    De = 4 * A_flow / (np.pi * params["D_out"])

    # 热通道流量（FdH_E 作为流量不确定性因子）
    W_rod_avg = params["W_total"] * (1 - params["Bypass"]) / total_rods
    W_rod = W_rod_avg / params["FdH_E"]
    G = W_rod / A_flow

    # 堆芯平均表面热流密度
    q_avg_surf = (params["Q_total"] * params["F_u"]) / (
        total_rods * np.pi * params["D_out"] * params["H_core"]
    )

    # 饱和参数
    sat_liq = IAPWS97(P=params["P_sys"], x=0)
    T_sat = sat_liq.T - 273.15
    h_f_sys = sat_liq.h * 1000.0

    # 入口焓
    h_in = IAPWS97(P=params["P_sys"], T=params["T_in"] + 273.15).h * 1000.0
    h_curr = h_in

    results = []
    dp_friction = 0.0
    dp_gravity = 0.0

    print(f"\n[参数] q''_avg = {q_avg_surf / 1e3:.2f} kW/m²")
    print(f"[参数] G = {G:.2f} kg/m²s")
    print(f"[参数] De = {De * 1000:.2f} mm")
    print(f"[参数] T_sat = {T_sat:.2f} °C")
    print(f"[参数] W_rod (热通道) = {W_rod:.4f} kg/s")
    print(f"[参数] q''_peak = {q_avg_surf * params['FdH_N'] * F_z_actual * params['Fq_E'] / 1e6:.3f} MW/m²")

    for i in range(n):
        # 热通道局部热流密度
        q_hot = q_avg_surf * params["FdH_N"] * phi_z[i]  # 用于焓升
        q_loc = q_hot * params["Fq_E"]                     # 用于温度（含工程因子）

        # 焓升
        dh = (q_hot * np.pi * params["D_out"] * dz) / W_rod
        h_mid = h_curr + dh / 2.0
        h_curr += dh

        # 冷却剂物性
        if h_mid >= h_f_sys:
            T_b = T_sat
            rho = sat_liq.rho
            mu = sat_liq.mu
            kw = sat_liq.k
            Pr = getattr(sat_liq, 'Prandtl', getattr(sat_liq, 'Prandt', 4.0))
        else:
            water = IAPWS97(P=params["P_sys"], h=h_mid / 1000.0)
            T_b = water.T - 273.15
            rho = water.rho
            mu = water.mu
            kw = water.k
            Pr = getattr(water, 'Prandtl', getattr(water, 'Prandt', 1.0))

        # Dittus-Boelter 对流换热
        Re = (G * De) / mu
        h_conv = 0.023 * Re**0.8 * Pr**0.4 * kw / De
        T_co_conv = T_b + q_loc / h_conv

        # 壁温确定：对流 vs 过冷沸腾
        T_co_jl = jens_lottes_wall_temp(q_loc, params["P_sys"], T_sat)
        if T_co_conv > T_sat:
            # 对流预测壁温超过饱和温度 → 过冷沸腾发生
            # 沸腾增强换热，壁温被拉低到 Jens-Lottes 预测值
            # 但不能低于饱和温度
            t_co = max(T_co_jl, T_sat)
        else:
            t_co = T_co_conv

        # 包壳导热
        k_clad = 15.0
        t_ci = t_co + q_loc * params["D_out"] * np.log(
            params["D_out"] / params["D_in"]
        ) / (2.0 * k_clad)

        # 气隙
        q_linear = q_loc * np.pi * params["D_out"]
        h_gap = get_gap_conductance(q_linear)
        t_us = t_ci + q_loc * (params["D_out"] / params["D_pellet"]) / h_gap

        # 芯块中心温度迭代
        t_center = t_us + 400.0
        for _ in range(50):
            # 分别计算表面、中点、中心的导热系数
            k_us = get_uo2_k(t_us)
            k_mid = get_uo2_k((t_center + t_us) / 2.0)
            k_center = get_uo2_k(t_center)
            
            # 辛普森积分近似求等效 k
            k_eff = (k_us + 4.0 * k_mid + k_center) / 6.0
            
            new_t0 = t_us + q_linear / (4.0 * np.pi * k_eff)
            if abs(new_t0 - t_center) < 0.01:
                break
            t_center = 0.6 * new_t0 + 0.4 * t_center
        t_center = new_t0

        # DNBR
        dnbr = calculate_w3_full(params["P_sys"], h_mid, h_in, q_loc, De, G)

        # 压降
        f = 0.184 * Re**(-0.2)
        dp_friction += f * (dz / De) * (G**2 / (2.0 * rho))
        dp_gravity += rho * params["g"] * dz

        results.append({
            "z": z_nodes[i],
            "T_b": T_b,
            "T_co": t_co,
            "T_ci": t_ci,
            "T_us": t_us,
            "T_0": t_center,
            "q_loc": q_loc,
            "q_linear": q_linear,
            "dnbr": dnbr,
            "h_gap": h_gap,
        })

    # 格架压降（用平均密度）
    rho_avg = np.mean([IAPWS97(P=params["P_sys"], h=h_in/1000).rho, sat_liq.rho])
    dp_grid = params["N_grids"] * params["K_grid"] * (G**2 / (2.0 * rho_avg))
    total_dp = dp_friction + dp_gravity + dp_grid

    return results, dp_friction, dp_gravity, dp_grid, total_dp


# ==========================================
# 5. 绘图与输出
# ==========================================
if __name__ == "__main__":
    res, dp_f, dp_g, dp_grid, dp_total = run_simulation()
    z = [d['z'] for d in res]

    T0_max = max(d['T_0'] for d in res)
    Tco_max = max(d['T_co'] for d in res)
    Tci_max = max(d['T_ci'] for d in res)
    Tus_max = max(d['T_us'] for d in res)
    T_out = res[-1]['T_b']
    dnbr_min = min(d['dnbr'] for d in res)
    ql_max = max(d['q_linear'] for d in res)

    # ---- 四面板图 ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) 温度分布
    ax1 = axes[0, 0]
    ax1.plot(z, [d['T_0'] for d in res], 'r-', linewidth=2, label=f'Pellet Center T₀ (max={T0_max:.0f}°C)')
    ax1.plot(z, [d['T_us'] for d in res], 'm-', linewidth=1.5, label=f'Pellet Surface T_us (max={Tus_max:.0f}°C)')
    ax1.plot(z, [d['T_ci'] for d in res], 'orange', linewidth=1.5, label=f'Clad Inner T_ci (max={Tci_max:.0f}°C)')
    ax1.plot(z, [d['T_co'] for d in res], 'g-', linewidth=2, label=f'Clad Outer T_co (max={Tco_max:.0f}°C)')
    ax1.plot(z, [d['T_b'] for d in res], 'b--', linewidth=2, label=f'Coolant T_b (out={T_out:.0f}°C)')
    ax1.axhline(y=2200, color='gray', linestyle=':', linewidth=1.5, label='Fuel Melting Limit (2200°C)')
    ax1.set_xlabel('Axial Position (m)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('AP1000 Hot Channel — Temperature Profiles')
    ax1.legend(fontsize=7, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # (b) 热流密度 & 线功率
    ax2 = axes[0, 1]
    color_q = 'red'
    ax2.plot(z, [d['q_loc'] / 1e6 for d in res], color=color_q, linewidth=2, label="q'' (MW/m²)")
    ax2.set_xlabel('Axial Position (m)')
    ax2.set_ylabel('Surface Heat Flux (MW/m²)', color=color_q)
    ax2.tick_params(axis='y', labelcolor=color_q)
    ax2b = ax2.twinx()
    ax2b.plot(z, [d['q_linear'] / 1e3 for d in res], 'b--', linewidth=2, label='q_L (kW/m)')
    ax2b.set_ylabel('Linear Power (kW/m)', color='blue')
    ax2b.tick_params(axis='y', labelcolor='blue')
    ax2.set_title('Heat Flux & Linear Power')
    ax2.grid(True, alpha=0.3)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

    # (c) DNBR
    ax3 = axes[1, 0]
    ax3.plot(z, [d['dnbr'] for d in res], 'k-', linewidth=2)
    ax3.axhline(y=1.3, color='r', linestyle='--', linewidth=1.5, label='Safety Limit (1.3)')
    ax3.set_xlabel('Axial Position (m)')
    ax3.set_ylabel('DNBR')
    ax3.set_title(f'DNBR Distribution (min = {dnbr_min:.2f})')
    ax3.set_ylim(0, min(20, max(d['dnbr'] for d in res) * 1.1))
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # (d) 气隙导热系数
    ax4 = axes[1, 1]
    ax4.plot(z, [d['h_gap'] for d in res], 'darkorange', linewidth=2)
    ax4.set_xlabel('Axial Position (m)')
    ax4.set_ylabel('Gap Conductance (W/m²K)')
    ax4.set_title('Gap Conductance vs Axial Position')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('AP1000_corrected_v2.png', dpi=150)
    plt.show()

    # ---- 数值结果 ----
    print("\n" + "=" * 60)
    print("  AP1000 热通道分析结果 (修正版 v2)")
    print("=" * 60)
    print(f"  最大芯块中心温度:       {T0_max:.1f} °C", end="")
    print(f"  {'  ✓ < 2200°C' if T0_max < 2200 else '  ✗ 超限!'}")
    print(f"  最大芯块表面温度:       {Tus_max:.1f} °C")
    print(f"  最大包壳内表面温度:     {Tci_max:.1f} °C")
    print(f"  最大包壳外表面温度:     {Tco_max:.1f} °C")
    print(f"  热通道出口冷却剂温度:   {T_out:.1f} °C")
    print(f"  最大线功率密度:         {ql_max / 1e3:.2f} kW/m")
    print(f"  最小 DNBR:              {dnbr_min:.2f}", end="")
    print(f"  {'  ✓ > 1.3' if dnbr_min > 1.3 else '  ✗ 低于限值!'}")
    print("-" * 60)
    print(f"  摩擦压降:   {dp_f / 1e3:.2f} kPa")
    print(f"  重力压降:   {dp_g / 1e3:.2f} kPa")
    print(f"  格架压降:   {dp_grid / 1e3:.2f} kPa")
    print(f"  总压降:     {dp_total / 1e3:.2f} kPa")
    print("=" * 60)

    # ---- 温度分解表（峰值位置）----
    i_peak = np.argmax([d['T_0'] for d in res])
    d = res[i_peak]
    print(f"\n  峰值位置 z = {d['z']:.3f} m 处的温度分解:")
    print(f"  {'─' * 45}")
    print(f"  冷却剂温度 T_b:         {d['T_b']:.1f} °C")
    print(f"  对流温升 ΔT_conv:       {d['T_co'] - d['T_b']:.1f} °C")
    print(f"  包壳温升 ΔT_clad:       {d['T_ci'] - d['T_co']:.1f} °C")
    print(f"  气隙温升 ΔT_gap:        {d['T_us'] - d['T_ci']:.1f} °C")
    print(f"  芯块温升 ΔT_fuel:       {d['T_0'] - d['T_us']:.1f} °C")
    print(f"  {'─' * 45}")
    print(f"  芯块中心温度 T_0:       {d['T_0']:.1f} °C")
    print(f"  局部线功率:             {d['q_linear'] / 1e3:.2f} kW/m")
    print(f"  局部 DNBR:              {d['dnbr']:.2f}")
