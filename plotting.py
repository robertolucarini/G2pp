import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np
from src.utils import parse_string_tenor


def plot_residuals_heatmap(hw_model, df_vols):

    t_exps = np.array([parse_string_tenor(str(x)) for x in df_vols.index])
    t_durs = np.array([parse_string_tenor(str(x)) for x in df_vols.columns])

    # create 2d grids (meshgrid)
    T_DUR, T_EXP = np.meshgrid(t_durs, t_exps)
    T_MAT = T_EXP + T_DUR
    
    def get_market_vars(t_exp, t_dur, t_mat):
        
        coupon_times = np.arange(1.0, t_dur + 1e-6, 1.0) + t_exp
        discounts = np.array([hw_model.zc_price_market(t) for t in coupon_times])
        
        ann = np.sum(discounts)
        if ann < 1e-10: 
            ann = hw_model.zc_price_market(t_mat)
            
        P_exp = hw_model.zc_price_market(t_exp)
        P_mat = hw_model.zc_price_market(t_mat)
        
        S0 = (P_exp - P_mat) / ann
        return S0, ann

    vec_market_func = np.vectorize(get_market_vars)
    S0_grid, Ann_grid = vec_market_func(T_EXP, T_DUR, T_MAT)

    def get_brigo_vol(t_exp, t_mat, s0, ann):
        inputs = {'T_exp': t_exp, 'T_mat': t_mat, 'S0': s0, 'annuity': ann}
        return hw_model.calculate_model_vol_brigo(inputs)

    # vectorize the vol call
    vec_brigo_func = np.vectorize(get_brigo_vol)
    # model vol    
    mod_vol_grid = vec_brigo_func(T_EXP, T_MAT, S0_grid, Ann_grid)

    # residuals
    mkt_vol_grid = df_vols.values
    residuals = mod_vol_grid - mkt_vol_grid
    
    # handle nan
    residuals[np.isnan(mkt_vol_grid)] = np.nan

    # plot
    plt.figure(figsize=(10, 6))
    sns.heatmap(residuals, 
                xticklabels=df_vols.columns, 
                yticklabels=df_vols.index, 
                annot=True, fmt=".1f", cmap="RdBu_r", center=0)
    plt.title("Volatility Residuals in bps (Model - Market)")
    plt.show()

def plot_vol_diagnostics(hw_model, df_vols, fixed_res_scale=None):
   
    expiries = np.array([parse_string_tenor(str(x)) for x in df_vols.index])
    tenors = np.array([parse_string_tenor(str(x)) for x in df_vols.columns])
    X, Y = np.meshgrid(tenors, expiries)
    
    market_vals = df_vols.values
    model_vals = np.zeros_like(market_vals)
    residuals = np.zeros_like(market_vals)
    
    print("Calculating model surface for diagnostics...")

    for i, T_exp in enumerate(expiries):
        for j, T_dur in enumerate(tenors):
            if np.isnan(market_vals[i, j]):
                model_vals[i, j] = np.nan
                residuals[i, j] = np.nan
                continue
                
            T_mat = T_exp + T_dur
            coupon_times = np.arange(1.0, T_dur + 1e-6, 1.0)
            
            discount_sum = 0.0
            for t in coupon_times:
                discount_sum += hw_model.zc_price_market(T_exp + t)
            
            ann = discount_sum if discount_sum > 1e-10 else hw_model.zc_price_market(T_mat)
            
            P_exp = hw_model.zc_price_market(T_exp)
            P_mat = hw_model.zc_price_market(T_mat)
            S0 = (P_exp - P_mat) / ann
            
            inputs = {'T_exp': T_exp, 'T_mat': T_mat, 'S0': S0, 'annuity': ann}
            mod_vol = hw_model.calculate_model_vol_brigo(inputs)
            
            model_vals[i, j] = mod_vol
            residuals[i, j] = mod_vol - market_vals[i, j]

    max_vol = np.nanmax(market_vals)
    z_lim_vol = (0, max_vol * 1.1) # Add 10% padding

    if fixed_res_scale:
        res_lim = fixed_res_scale
    else:
        max_err = np.nanmax(np.abs(residuals))
        res_lim = np.ceil(max_err * 1.2) 

    fig1 = plt.figure(figsize=(20, 7))
    
    # Plot A: Market Data
    ax1 = fig1.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, market_vals, cmap=cm.viridis, alpha=0.9)
    ax1.set_title("Market Volatility Surface")
    ax1.set_xlabel("Tenor"); ax1.set_ylabel("Expiry")
    ax1.set_zlabel("Vol (bps)")
    ax1.set_zlim(z_lim_vol)

    # Plot B: Model Fit
    ax2 = fig1.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, model_vals, cmap=cm.viridis, alpha=0.9)
    ax2.set_title(f"G2++ Model Surface")
    ax2.set_xlabel("Tenor"); ax2.set_ylabel("Expiry")
    ax2.set_zlim(z_lim_vol)

    # Plot C: Residuals
    ax3 = fig1.add_subplot(133, projection='3d')
    surf = ax3.plot_surface(X, Y, residuals, cmap=cm.coolwarm, alpha=0.9)
    ax3.plot_surface(X, Y, np.zeros_like(residuals), color='black', alpha=0.2)
    
    ax3.set_title("Residuals (Model - Market)")
    ax3.set_xlabel("Tenor"); ax3.set_ylabel("Expiry")
    ax3.set_zlabel("Error (bps)")
    
    ax3.set_zlim(-res_lim, res_lim) 
    
    fig1.tight_layout()
    plt.show()

    # Overlay Plot
    fig2 = plt.figure(figsize=(12, 10))
    ax_ov = fig2.add_subplot(111, projection='3d')
    ax_ov.plot_surface(X, Y, model_vals, cmap=cm.viridis, alpha=0.6, label='Model')
    ax_ov.plot_wireframe(X, Y, market_vals, color='black', rstride=1, cstride=1, linewidth=1.5, label='Market')
    ax_ov.set_title("Model Fit Overlay")
    ax_ov.set_xlabel("Tenor"); ax_ov.set_ylabel("Expiry")
    ax_ov.set_zlim(z_lim_vol)
    plt.show()

def plot_exercise_boundary(model, t_ex, T_mat, K, x_limit=0.06, y_limit=0.06):
    if t_ex not in model.lsm_coeffs:
        print(f"No coefficients found for T={t_ex}.")
        return

    coeffs = model.lsm_coeffs[t_ex]
    var_x = (model.sigma**2 / (2 * model.a)) * (1 - np.exp(-2 * model.a * t_ex))
    var_y = (model.eta**2 / (2 * model.b)) * (1 - np.exp(-2 * model.b * t_ex))
    
    x_grid = np.linspace(-x_limit, x_limit, 100)
    y_grid = np.linspace(-y_limit, y_limit, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    intrinsic = model.get_swaption_payoff_grid(t_ex, T_mat, K, X.flatten(), Y.flatten()).reshape(X.shape)
    A_all = np.column_stack([np.ones(X.size), X.flatten(), Y.flatten(), X.flatten()**2, Y.flatten()**2, X.flatten()*Y.flatten()])
    continuation = (A_all @ coeffs).reshape(X.shape)
    
    valid_mask = (np.abs(X) < 4.5 * np.sqrt(var_x)) & (np.abs(Y) < 4.5 * np.sqrt(var_y))
    exercise_region = (intrinsic > continuation) & (intrinsic > 0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(intrinsic, extent=[-x_limit, x_limit, -y_limit, y_limit], origin='lower', cmap='Blues', alpha=0.3, aspect='auto')
    ax.imshow(~valid_mask, extent=[-x_limit, x_limit, -y_limit, y_limit], origin='lower', cmap='binary', alpha=0.4, aspect='auto')
    
    exercise_plot = exercise_region & valid_mask
    if np.any(exercise_plot):
        ax.contourf(X, Y, exercise_plot, levels=[0.5, 1.5], colors=['#2ca02c'], alpha=0.6)
    
    ax.set_title(f"Exercise Boundary @ T={t_ex}Y")
    ax.set_xlabel("Factor X")
    ax.set_ylabel("Factor Y")
    plt.show()

