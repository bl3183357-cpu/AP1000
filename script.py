import numpy as np
import matplotlib.pyplot as plt
from iapws import IAPWS97




# =============================================================================
# 1. 基础物性与经验关联式
# =============================================================================
P_SYS = 15.51       # MPa
T_SAT = 344.8       # ℃

def get_h_from_T(T):
    """使用 iapws 根据温度计算冷却剂焓值 (kJ/kg)"""
    water = IAPWS97(P=P_SYS, T=T + 273.15)
    return water.h

def get_T_from_h(h):
    """使用 iapws 根据焓值计算冷却剂温度 (℃)"""
    water = IAPWS97(P=P_SYS, h=h)
    return water.T - 273.15

def get_density(T):
    """使用 iapws 根据温度计算冷却剂密度 (kg/m^3)"""
    water = IAPWS97(P=P_SYS, T=T + 273.15)
    return water.rho

def get_uo2_k(T_c):
    """UO2 导热系数 (W/m·K) """
    Tk = T_c + 273.15
    return 100.0 / (6.75 + 0.0235 * Tk) + 6.4e9 * np.exp(-16350.0 / Tk) / Tk**2

def jens_lottes_wall_temp(q_flux_W_m2, T_sat):
    """Jens-Lottes 过冷沸腾壁温 """
    # q_flux 单位为 W/m^2, 转换为 MW/m^2 计算
    dT = 25.0 * (q_flux_W_m2 / 1e6)**0.25 * np.exp(-P_SYS / 6.2)
    return T_sat + dT

def get_chf_w3(P_MPa, G_SI, De_SI, h_local_SI, h_in_SI):
    """
    使用 W-3 经验关联式计算临界热流密度 (CHF)
    输入参数为 SI 单位，内部转换为英制计算，返回 MW/m^2
    """
    # 1. 单位转换 (SI -> 英制)
    P_psia = P_MPa * 145.038                           # 系统压力 (psia)
    G_Mlb = (G_SI * 737.338) / 1e6                     # 质量流速 (10^6 lb/hr·ft^2)
    De_in = De_SI * 39.3701                            # 当量直径 (in)
    H_in_Btu = h_in_SI * 0.4299                        # 入口焓 (Btu/lb)
    H_local_Btu = h_local_SI * 0.4299                  # 局部焓 (Btu/lb)
    
    # 15.51 MPa 下饱和水焓与汽化潜热 (转为 Btu/lb)
    H_f_Btu = 1630.0 * 0.4299
    H_fg_Btu = 966.0 * 0.4299
    
    # 局部平衡干度 (热通道内通常为负值，即过冷状态)
    X_loc = (H_local_Btu - H_f_Btu) / H_fg_Btu
    
    # 2. W-3 公式计算项分解
    term1 = (2.022 - 0.0004302 * P_psia) + (0.1722 - 0.0000984 * P_psia) * np.exp((18.177 - 0.004129 * P_psia) * X_loc)
    term2 = (0.1484 - 1.596 * X_loc + 0.1729 * X_loc * abs(X_loc)) * G_Mlb + 1.037
    term3 = 1.157 - 0.869 * X_loc
    term4 = 0.2664 + 0.8357 * np.exp(-3.151 * De_in)
    term5 = 0.8258 + 0.000794 * (H_f_Btu - H_in_Btu)
    
    # 临界热流密度 (10^6 Btu/hr·ft^2)
    q_chf_Mbtu = term1 * term2 * term3 * term4 * term5
    
    # 3. 单位转换回 SI (10^6 Btu/hr·ft^2 = 3.15459 MW/m^2)
    q_chf_MW_m2 = q_chf_Mbtu * 3.15459
    
    return q_chf_MW_m2


# =============================================================================
# 2. 主程序：AP1000 热工水力单通道校核
# =============================================================================
def main():
    # --- 反应堆设计参数 ---
    Q_total = 3400e6          # 堆芯总热功率 (W)
    W_total = 14314.0         # 冷却剂总流量 (kg/s)
    T_in = 279.4              # 入口温度 (℃)
    H_core = 4.2672           # 堆芯高度 (m)
    N_assy = 157              # 组件数
    N_rods_per_assy = 264     # 每组件燃料棒数
    D_co = 9.5e-3             # 包壳外径 (m)
    D_ci = 8.36e-3            # 包壳内径 (m)
    D_p = 8.19e-3             # 芯块直径 (m)
    Pitch = 12.6e-3           # 栅距 (m)
    Bypass_ratio = 0.059      # 旁流系数
    Heat_frac = 0.974         # 燃料元件发热份额
    
    # 热点因子
    F_q_N = 2.524             # 热流量核热点因子
    F_q_E = 1.03              # 热流量工程热点因子
    F_dH_E = 1.085            # 焓升工程热点因子
    
    # 局部阻力系数
    K_in = 0.75
    K_out = 1.0
    K_grid = 1.05
    N_grid = 10               # 假设定位格架数量
    
    # --- 几何与流量计算 ---
    N_rods_total = N_assy * N_rods_per_assy
    A_channel = Pitch**2 - np.pi/4 * D_co**2  # 单通道流通面积 (m^2)
    De = 4 * A_channel / (np.pi * D_co)       # 当量直径 (m)
    
    W_core_active = W_total * (1 - Bypass_ratio)
    G_avg = W_core_active / (N_rods_total * A_channel)  # 平均质量流速 (kg/m^2·s)
    
    # --- 任务 1 & 4: 全堆芯平均参数计算 ---
    h_in = get_h_from_T(T_in)
    delta_h_avg = (Q_total * Heat_frac * 1e-3) / W_core_active # 平均焓升 (kJ/kg)
    h_out_avg = h_in + delta_h_avg
    T_out_avg = get_T_from_h(h_out_avg)
    
    rho_in = get_density(T_in)
    rho_out_avg = get_density(T_out_avg)
    v_avg = G_avg / ((rho_in + rho_out_avg) / 2.0)
    
    print("="*60)
    print(" 任务 1 & 4: 全堆芯平均参数")
    print("="*60)
    print(f"堆芯平均冷却剂出口温度: {T_out_avg:.2f} ℃")
    print(f"冷却剂平均流速:         {v_avg:.2f} m/s")
    
    # --- 任务 2: 热通道功率分布与离散化 ---
    n_nodes = 100  # 轴向控制体数量 (满足 >= 10 的要求)
    z = np.linspace(0, H_core, n_nodes)
    dz = H_core / n_nodes
    
    # 构造轴向功率分布 (截断正弦分布)
    alpha = 0.8844
    phi_z = np.sin(np.pi * z / H_core)**alpha
    # 强制归一化，确保离散后均值严格为 1.0 (总和为 n)
    phi_z = phi_z / np.mean(phi_z) 
    
    F_z_max = np.max(phi_z)
    F_xy = F_q_N / F_z_max  # 径向核热点因子推导
    
    # 热通道参数
    q_avg_rod = (Q_total * Heat_frac) / N_rods_total  # 单棒平均功率 (W)
    q_L_avg = q_avg_rod / H_core                      # 平均线功率 (W/m)
    q_flux_avg = q_avg_rod / (np.pi * D_co * H_core)  # 平均热流密度 (W/m^2)
    
    # 热通道最大值
    q_L_max = q_L_avg * F_q_N * F_q_E
    q_flux_max = q_flux_avg * F_q_N * F_q_E
    
    print("\n" + "="*60)
    print(" 任务 2: 功率与热流密度")
    print("="*60)
    print(f"平均线功率:             {q_L_avg/1000:.2f} kW/m")
    print(f"最大线功率:             {q_L_max/1000:.2f} kW/m")
    print(f"平均表面热流密度:       {q_flux_avg/1e6:.3f} MW/m^2")
    print(f"最大表面热流密度:       {q_flux_max/1e6:.3f} MW/m^2")

    # --- 任务 3 & 4 & 5: 轴向温度场与 DNBR 计算 ---
    h_arr = np.zeros(n_nodes)
    T_b_arr = np.zeros(n_nodes)
    T_co_arr = np.zeros(n_nodes)
    T_ci_arr = np.zeros(n_nodes)
    T_us_arr = np.zeros(n_nodes)
    T_0_arr = np.zeros(n_nodes)
    DNBR_arr = np.zeros(n_nodes)
    q_flux_arr = np.zeros(n_nodes)
    q_L_arr = np.zeros(n_nodes)
    
    # 热通道入口焓
    h_current = h_in
    
    # 传热热阻参数
    h_gap = 10000.0  # 气隙导热系数 W/m^2·K
    k_clad = 16.0    # 包壳导热系数 W/m·K
    
    for i in range(n_nodes):
        # 局部热流与线功率 (考虑工程热点因子)
        q_L_local = q_L_avg * F_xy * phi_z[i] * F_q_E
        q_flux_local = q_flux_avg * F_xy * phi_z[i] * F_q_E
        
        q_L_arr[i] = q_L_local
        q_flux_arr[i] = q_flux_local
        
        # 1. 冷却剂焓升与温度 (考虑焓升热点因子)
        # 局部热量输入到冷却剂
        dq = q_L_local * dz / F_q_E * F_dH_E  # 焓升因子作用于整体热量积分
        h_current += (dq / 1000.0) / (G_avg * A_channel) # kJ/kg
        h_arr[i] = h_current
        T_b = get_T_from_h(h_current)
        T_b_arr[i] = T_b
        
        # 2. 包壳外表面温度 (单相强制对流 or 过冷沸腾)
        # 简化单相对流换热系数 h_conv ~ 35000 W/m2K
        h_conv = 35000.0 
        T_co_sp = T_b + q_flux_local / h_conv
        T_co_boil = jens_lottes_wall_temp(q_flux_local, T_SAT)
        T_co = min(T_co_sp, T_co_boil) # 取两者较小值（沸腾会强化换热限制壁温）
        T_co_arr[i] = T_co
        
        # 3. 包壳内表面温度 (圆筒壁导热)
        T_ci = T_co + (q_L_local / (2 * np.pi * k_clad)) * np.log(D_co / D_ci)
        T_ci_arr[i] = T_ci
        
        # 4. 芯块外表面温度 (气隙导热)
        T_us = T_ci + q_flux_local * (D_co / D_p) / h_gap
        T_us_arr[i] = T_us
        
        # 5. 芯块中心温度 
        t_center = T_us + 400.0 # 初始猜测
        for _ in range(50):
            k_us = get_uo2_k(T_us)
            k_mid = get_uo2_k((t_center + T_us) / 2.0)
            k_center = get_uo2_k(t_center)
            
            # 辛普森积分等效导热系数
            k_eff = (k_us + 4.0 * k_mid + k_center) / 6.0
            
            new_t0 = T_us + q_L_local / (4.0 * np.pi * k_eff)
            if abs(new_t0 - t_center) < 0.01: # 满足迭代误差 < 0.1% 的要求
                break
            t_center = 0.6 * new_t0 + 0.4 * t_center
        T_0_arr[i] = new_t0
        
        # 6. DNBR 计算 (使用 W-3 公式)
        chf = get_chf_w3(P_SYS, G_avg, De, h_current, h_in) * 1e6 # 转换为 W/m^2
        DNBR_arr[i] = chf / max(q_flux_local, 1e-6)

    print("\n" + "="*60)
    print(" 任务 3 & 4 & 5: 核心安全限值校核")
    print("="*60)
    print(f"最大包壳外表面温度:     {np.max(T_co_arr):.1f} ℃")
    print(f"最大芯块中心温度:       {np.max(T_0_arr):.1f} ℃  (限制 < 2200℃)")
    print(f"最小 DNBR (MDNBR):      {np.min(DNBR_arr):.2f}  (限制 > 1.3)")

    # --- 任务 6: 堆芯压降计算 ---
    rho_out = get_density(T_b_arr[-1])
    rho_avg_core = (rho_in + rho_out) / 2.0
    
    # 动压头 (Pa)
    dyn_p_in = (G_avg**2) / (2.0 * rho_in)
    dyn_p_out = (G_avg**2) / (2.0 * rho_out)
    dyn_p_avg = (G_avg**2) / (2.0 * rho_avg_core)
    
    # 摩擦压降 (假设 f=0.015)
    f_fric = 0.015
    dp_fric = f_fric * (H_core / De) * dyn_p_avg
    
    # 重力压降
    dp_grav = rho_avg_core * 9.81 * H_core
    
    # 局部压降
    dp_in = K_in * dyn_p_in
    dp_out = K_out * dyn_p_out
    dp_grid = N_grid * K_grid * dyn_p_avg
    
    dp_total = dp_fric + dp_grav + dp_in + dp_out + dp_grid
    
    print("\n" + "="*60)
    print(" 任务 6: 堆芯压降分布")
    print("="*60)
    print(f"入口局部压降:   {dp_in/1000:.2f} kPa")
    print(f"出口局部压降:   {dp_out/1000:.2f} kPa")
    print(f"格架局部压降:   {dp_grid/1000:.2f} kPa")
    print(f"摩擦压降:       {dp_fric/1000:.2f} kPa")
    print(f"重力压降:       {dp_grav/1000:.2f} kPa")
    print(f"堆芯总压降:     {dp_total/1000:.2f} kPa")

    # =========================================================================
    # 绘图部分 
    # =========================================================================
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    fig = plt.figure(figsize=(14, 10))
    
    # 图 1: 温度分布
    ax1 = fig.add_subplot(221)
    ax1.plot(z, T_0_arr, 'r-', lw=2, label=f'芯块中心 T_0 (max={np.max(T_0_arr):.0f}℃)')
    ax1.plot(z, T_us_arr, 'm-', lw=1.5, label=f'芯块表面 T_us')
    ax1.plot(z, T_ci_arr, 'orange', lw=1.5, label=f'包壳内表面 T_ci')
    ax1.plot(z, T_co_arr, 'g-', lw=2, label=f'包壳外表面 T_co')
    ax1.plot(z, T_b_arr, 'b--', lw=2, label=f'冷却剂 T_b (out={T_b_arr[-1]:.0f}℃)')
    ax1.axhline(2200, color='gray', linestyle=':', label='燃料熔化限值 (2200℃)')
    ax1.set_title("热通道轴向温度分布")
    ax1.set_xlabel("轴向高度 (m)")
    ax1.set_ylabel("温度 (℃)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    
    # 图 2: 热流密度与线功率
    ax2 = fig.add_subplot(222)
    line1 = ax2.plot(z, q_flux_arr/1e6, 'r-', lw=2, label="表面热流密度 q''")
    ax2.set_ylabel("表面热流密度 (MW/m^2)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    ax2_twin = ax2.twinx()
    line2 = ax2_twin.plot(z, q_L_arr/1000, 'b--', lw=2, label="线功率 q_L")
    ax2_twin.set_ylabel("线功率 (kW/m)", color='b')
    ax2_twin.tick_params(axis='y', labelcolor='b')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')
    ax2.set_title("热流密度与线功率轴向分布")
    ax2.set_xlabel("轴向高度 (m)")
    ax2.grid(True, alpha=0.3)
    
    # 图 3: DNBR 分布
    ax3 = fig.add_subplot(223)
    ax3.plot(z, DNBR_arr, 'k-', lw=2, label=f'DNBR (min={np.min(DNBR_arr):.2f})')
    ax3.axhline(1.3, color='r', linestyle='--', label='安全限值 (1.3)')
    ax3.set_title("DNBR 轴向分布")
    ax3.set_xlabel("轴向高度 (m)")
    ax3.set_ylabel("DNBR")
    ax3.set_ylim(0, max(10, np.min(DNBR_arr)*3))
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 图 4: 冷却剂焓值分布 
    ax4 = fig.add_subplot(224)
    ax4.plot(z, h_arr, 'c-', lw=2, label='冷却剂焓值')
    ax4.set_title("冷却剂焓值轴向分布")
    ax4.set_xlabel("轴向高度 (m)")
    ax4.set_ylabel("焓值 (kJ/kg)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 标注入口和出口焓值
    ax4.text(z[0], h_arr[0], f" 入口: {h_arr[0]:.1f}", verticalalignment='bottom')
    ax4.text(z[-1], h_arr[-1], f" 出口: {h_arr[-1]:.1f}", verticalalignment='top')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
