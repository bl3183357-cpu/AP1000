import numpy as np
import matplotlib.pyplot as plt
from iapws import IAPWS97
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

# =============================================================================
# 1. 基础物性与经验关联式 (全局变量由UI动态更新)
# =============================================================================
P_SYS = 15.51       # MPa
T_SAT = 344.8       # ℃

def get_h_from_T(T):
    water = IAPWS97(P=P_SYS, T=T + 273.15)
    return water.h

def get_T_from_h(h):
    water = IAPWS97(P=P_SYS, h=h)
    return water.T - 273.15

def get_density(T):
    water = IAPWS97(P=P_SYS, T=T + 273.15)
    return water.rho

def get_uo2_k(T_c):
    Tk = T_c + 273.15
    return 100.0 / (6.75 + 0.0235 * Tk) + 6.4e9 * np.exp(-16350.0 / Tk) / Tk**2

def jens_lottes_wall_temp(q_flux_W_m2, T_sat):
    dT = 25.0 * (q_flux_W_m2 / 1e6)**0.25 * np.exp(-P_SYS / 6.2)
    return T_sat + dT

def get_chf_w3(P_MPa, G_SI, De_SI, h_local_SI, h_in_SI):
    P_psia = P_MPa * 145.038
    G_Mlb = (G_SI * 737.338) / 1e6
    De_in = De_SI * 39.3701
    H_in_Btu = h_in_SI * 0.4299
    H_local_Btu = h_local_SI * 0.4299
    
    H_f_Btu = 1630.0 * 0.4299
    H_fg_Btu = 966.0 * 0.4299
    
    X_loc = (H_local_Btu - H_f_Btu) / H_fg_Btu
    
    term1 = (2.022 - 0.0004302 * P_psia) + (0.1722 - 0.0000984 * P_psia) * np.exp((18.177 - 0.004129 * P_psia) * X_loc)
    term2 = (0.1484 - 1.596 * X_loc + 0.1729 * X_loc * abs(X_loc)) * G_Mlb + 1.037
    term3 = 1.157 - 0.869 * X_loc
    term4 = 0.2664 + 0.8357 * np.exp(-3.151 * De_in)
    term5 = 0.8258 + 0.000794 * (H_f_Btu - H_in_Btu)
    
    q_chf_Mbtu = term1 * term2 * term3 * term4 * term5
    return q_chf_Mbtu * 3.15459


# =============================================================================
# 2. 核心计算逻辑
# =============================================================================
def run_calculation(params, output_text):
    global P_SYS, T_SAT
    
    # 更新全局压力和温度
    P_SYS = params['P_SYS']
    T_SAT = params['T_SAT']
    
    # 提取参数
    Q_total = params['Q_total']
    W_total = params['W_total']
    T_in = params['T_in']
    H_core = params['H_core']
    N_assy = int(params['N_assy'])
    N_rods_per_assy = int(params['N_rods_per_assy'])
    D_co = params['D_co']
    D_ci = params['D_ci']
    D_p = params['D_p']
    Pitch = params['Pitch']
    Bypass_ratio = params['Bypass_ratio']
    Heat_frac = params['Heat_frac']
    F_q_N = params['F_q_N']
    F_q_E = params['F_q_E']
    F_dH_E = params['F_dH_E']
    
    K_in, K_out, K_grid, N_grid = 0.75, 1.0, 1.05, 10
    f_fric = 0.015            
    h_gap = 10000.0           
    k_clad = 16.0             

    # --- 计算有关堆参数 ---
    N_rods_total = N_assy * N_rods_per_assy
    A_channel = Pitch**2 - np.pi/4 * D_co**2  
    De = 4 * A_channel / (np.pi * D_co)       
    
    W_core_active = W_total * (1 - Bypass_ratio)
    h_in = get_h_from_T(T_in)
    rho_in = get_density(T_in)

    n_nodes = 100
    dz = H_core / n_nodes
    z = np.linspace(dz/2, H_core - dz/2, n_nodes)
    alpha = 0.8844
    phi_z = np.sin(np.pi * z / H_core)**alpha
    phi_z = phi_z / np.mean(phi_z) 
    
    F_z_max = np.max(phi_z)
    F_xy = F_q_N / F_z_max
    
    q_avg_rod = (Q_total * Heat_frac) / N_rods_total
    q_L_avg = q_avg_rod / H_core
    q_flux_avg = q_avg_rod / (np.pi * D_co * H_core)
    
    q_L_max = q_L_avg * F_q_N * F_q_E
    q_flux_max = q_flux_avg * F_q_N * F_q_E

    G_avg = W_core_active / (N_rods_total * A_channel)

    # --- 计算平均管冷却剂的焓场分布 ---
    h_avg_arr = np.zeros(n_nodes)
    h_current_avg = h_in
    for i in range(n_nodes):
        dq_avg = q_L_avg * phi_z[i] * dz
        h_current_avg += (dq_avg / 1000.0) / (G_avg * A_channel)
        h_avg_arr[i] = h_current_avg
    
    T_out_avg = get_T_from_h(h_avg_arr[-1])
    rho_out_avg = get_density(T_out_avg)
    rho_avg_core = (rho_in + rho_out_avg) / 2.0

    # --- 计算平均管的压降 ---
    dyn_p_in_avg = (G_avg**2) / (2.0 * rho_in)
    dyn_p_out_avg = (G_avg**2) / (2.0 * rho_out_avg)
    dyn_p_avg_core = (G_avg**2) / (2.0 * rho_avg_core)
    
    dp_fric_avg = f_fric * (H_core / De) * dyn_p_avg_core
    dp_grav_avg = rho_avg_core * 9.81 * H_core
    dp_in_avg = K_in * dyn_p_in_avg
    dp_out_avg = K_out * dyn_p_out_avg
    dp_grid_avg = N_grid * K_grid * dyn_p_avg_core
    
    dp_avg_total = dp_fric_avg + dp_grav_avg + dp_in_avg + dp_out_avg + dp_grid_avg
    dp_h_e = dp_avg_total

    # --- 热管等压降迭代 ---
    G_hot = G_avg * 0.95  
    epsilon = 1e-4
    h_hot_arr = np.zeros(n_nodes)
    
    for iteration in range(100):
        h_current_hot = h_in
        for i in range(n_nodes):
            dq_hot = q_L_avg * F_xy * phi_z[i] * F_dH_E * dz 
            h_current_hot += (dq_hot / 1000.0) / (G_hot * A_channel)
            h_hot_arr[i] = h_current_hot
            
        T_out_hot = get_T_from_h(h_hot_arr[-1])
        rho_out_hot = get_density(T_out_hot)
        rho_avg_hot = (rho_in + rho_out_hot) / 2.0
        
        dyn_p_in_hot = (G_hot**2) / (2.0 * rho_in)
        dyn_p_out_hot = (G_hot**2) / (2.0 * rho_out_hot)
        dyn_p_avg_hot = (G_hot**2) / (2.0 * rho_avg_hot)
        
        dp_hot_total = (f_fric * (H_core / De) * dyn_p_avg_hot + 
                        rho_avg_hot * 9.81 * H_core + 
                        K_in * dyn_p_in_hot + K_out * dyn_p_out_hot + 
                        N_grid * K_grid * dyn_p_avg_hot)
        
        if abs((dp_hot_total - dp_h_e) / dp_h_e) <= epsilon:
            break
        G_hot = G_hot * np.sqrt(dp_h_e / dp_hot_total)

    # --- 计算 t0,max 与 MDNBR ---
    T_b_arr, T_co_arr, T_ci_arr, T_us_arr, T_0_arr = [np.zeros(n_nodes) for _ in range(5)]
    DNBR_arr, q_flux_arr, q_L_arr = [np.zeros(n_nodes) for _ in range(3)]
    
    for i in range(n_nodes):
        q_L_local = q_L_avg * F_xy * phi_z[i] * F_q_E
        q_flux_local = q_flux_avg * F_xy * phi_z[i] * F_q_E
        
        q_L_arr[i] = q_L_local
        q_flux_arr[i] = q_flux_local
        
        h_current = h_hot_arr[i]
        T_b = get_T_from_h(h_current)
        T_b_arr[i] = T_b
        
        h_conv = 35000.0 
        T_co_sp = T_b + q_flux_local / h_conv
        T_co_boil = jens_lottes_wall_temp(q_flux_local, T_SAT)
        T_co = min(T_co_sp, T_co_boil)
        T_co_arr[i] = T_co
        
        T_ci = T_co + (q_L_local / (2 * np.pi * k_clad)) * np.log(D_co / D_ci)
        T_ci_arr[i] = T_ci
        
        T_us = T_ci + q_flux_local * (D_co / D_p) / h_gap
        T_us_arr[i] = T_us
        
        t_center = T_us + 400.0 
        for _ in range(50):
            k_us = get_uo2_k(T_us)
            k_mid = get_uo2_k((t_center + T_us) / 2.0)
            k_center = get_uo2_k(t_center)
            k_eff = (k_us + 4.0 * k_mid + k_center) / 6.0
            
            new_t0 = T_us + q_L_local / (4.0 * np.pi * k_eff)
            if abs(new_t0 - t_center) < 0.01: 
                break
            t_center = 0.6 * new_t0 + 0.4 * t_center
        T_0_arr[i] = new_t0
        
        chf = get_chf_w3(P_SYS, G_hot, De, h_current, h_in) * 1e6 
        DNBR_arr[i] = chf / max(q_flux_local, 1e-6)

    # --- 输出报告到 UI ---
    report = f"""============================================================
 堆芯热工水力计算结果报告
============================================================
1. 堆芯冷却剂出口温度:     {T_out_avg:.2f} ℃

2. 燃料棒表面平均热流密度: {q_flux_avg/1e6:.3f} MW/m^2
   燃料棒表面最大热流密度: {q_flux_max/1e6:.3f} MW/m^2
   平均线功率:             {q_L_avg/1000:.2f} kW/m
   最大线功率:             {q_L_max/1000:.2f} kW/m

3. 热管的焓、包壳内外表面温度、芯块中心温度随轴向的分布: (详见生成图表1和图表4)

4. 包壳表面最高温度:       {np.max(T_co_arr):.1f} ℃
   芯块中心最高温度:       {np.max(T_0_arr):.1f} ℃

5. DNBR在轴向上的变化:     (详见生成图表3), 最小DNBR值为 {np.min(DNBR_arr):.2f}

6. 计算堆芯压降:           {dp_avg_total/1000:.2f} kPa
============================================================"""
    
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, report)

    # --- 绘图输出 ---
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False    
    
    fig = plt.figure(figsize=(14, 10))
    fig.canvas.manager.set_window_title('热工水力计算图表')
    
    ax1 = fig.add_subplot(221)
    ax1.plot(z, T_0_arr, 'r-', lw=2, label=f'芯块中心 T_0 (max={np.max(T_0_arr):.0f}℃)')
    ax1.plot(z, T_ci_arr, 'orange', lw=1.5, label=f'包壳内表面 T_ci')
    ax1.plot(z, T_co_arr, 'g-', lw=2, label=f'包壳外表面 T_co')
    ax1.plot(z, T_b_arr, 'b--', lw=2, label=f'冷却剂 T_b')
    ax1.set_title("图1: 热通道轴向温度分布")
    ax1.set_xlabel("轴向高度 (m)")
    ax1.set_ylabel("温度 (℃)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    ax2 = fig.add_subplot(222)
    line1 = ax2.plot(z, q_flux_arr/1e6, 'r-', lw=2, label="表面热流密度 q''")
    ax2.set_ylabel("表面热流密度 (MW/m^2)", color='r')
    ax2_twin = ax2.twinx()
    line2 = ax2_twin.plot(z, q_L_arr/1000, 'b--', lw=2, label="线功率 q_L")
    ax2_twin.set_ylabel("线功率 (kW/m)", color='b')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')
    ax2.set_title("图2: 热流密度与线功率轴向分布")
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(223)
    ax3.plot(z, DNBR_arr, 'k-', lw=2, label=f'DNBR (MDNBR={np.min(DNBR_arr):.2f})')
    ax3.axhline(1.3, color='r', linestyle='--', label='安全限值 (1.3)')
    ax3.set_title("图3: DNBR 轴向分布")
    ax3.set_xlabel("轴向高度 (m)")
    ax3.set_ylabel("DNBR")
    ax3.set_ylim(0, max(10, np.min(DNBR_arr)*3))
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    ax4 = fig.add_subplot(224)
    ax4.plot(z, h_hot_arr, 'c-', lw=2, label='热管冷却剂焓值')
    ax4.set_title("图4: 热管冷却剂焓值轴向分布")
    ax4.set_xlabel("轴向高度 (m)")
    ax4.set_ylabel("焓值 (kJ/kg)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# 3. UI 界面构建
# =============================================================================
def create_ui():
    root = tk.Tk()
    root.title("堆芯热工水力计算程序")
    root.geometry("850x650")
    
    # 默认参数字典
    default_params = {
        "系统压力 P_SYS (MPa)": ("P_SYS", 15.51),
        "饱和温度 T_SAT (℃)": ("T_SAT", 344.8),
        "堆芯总热功率 Q_total (W)": ("Q_total", 3400e6),
        "冷却剂总流量 W_total (kg/s)": ("W_total", 14314.0),
        "入口温度 T_in (℃)": ("T_in", 279.4),
        "堆芯高度 H_core (m)": ("H_core", 4.2672),
        "组件数 N_assy": ("N_assy", 157),
        "每组件燃料棒数 N_rods": ("N_rods_per_assy", 264),
        "包壳外径 D_co (m)": ("D_co", 9.5e-3),
        "包壳内径 D_ci (m)": ("D_ci", 8.36e-3),
        "芯块直径 D_p (m)": ("D_p", 8.19e-3),
        "栅距 Pitch (m)": ("Pitch", 12.6e-3),
        "旁流系数 Bypass_ratio": ("Bypass_ratio", 0.059),
        "燃料元件发热份额": ("Heat_frac", 0.974),
        "热流量核热点因子 F_q_N": ("F_q_N", 2.524),
        "热流量工程热点因子 F_q_E": ("F_q_E", 1.03),
        "焓升工程热点因子 F_dH_E": ("F_dH_E", 1.085)
    }

    entries = {}

    # 顶部参数输入区
    frame_inputs = ttk.LabelFrame(root, text="输入参数", padding=(10, 10))
    frame_inputs.pack(fill="x", padx=10, pady=10)

    row, col = 0, 0
    for label_text, (key, default_val) in default_params.items():
        ttk.Label(frame_inputs, text=label_text).grid(row=row, column=col*2, sticky="e", padx=5, pady=5)
        entry = ttk.Entry(frame_inputs, width=15)
        entry.insert(0, str(default_val))
        entry.grid(row=row, column=col*2+1, sticky="w", padx=5, pady=5)
        entries[key] = entry
        
        col += 1
        if col > 2:  # 每行显示3个参数
            col = 0
            row += 1

    # 底部结果显示区
    frame_output = ttk.LabelFrame(root, text="计算结果报告", padding=(10, 10))
    frame_output.pack(fill="both", expand=True, padx=10, pady=5)
    
    text_output = scrolledtext.ScrolledText(frame_output, wrap=tk.WORD, font=("Consolas", 10))
    text_output.pack(fill="both", expand=True)

    # 运行按钮回调函数
    def on_run():
        try:
            # 读取所有输入框的值并转换为浮点数
            params = {key: float(entry.get()) for key, entry in entries.items()}
            text_output.delete(1.0, tk.END)
            text_output.insert(tk.END, "正在计算中，请稍候...\n")
            root.update()
            
            # 执行计算
            run_calculation(params, text_output)
            
        except ValueError:
            messagebox.showerror("输入错误", "请确保所有参数均输入了合法的数字！")
        except Exception as e:
            messagebox.showerror("计算错误", f"计算过程中发生异常:\n{str(e)}")

    # 运行按钮
    btn_run = ttk.Button(root, text="🚀 开始计算并生成图表", command=on_run)
    btn_run.pack(pady=10, ipadx=20, ipady=5)

    root.mainloop()

if __name__ == "__main__":
    create_ui()
