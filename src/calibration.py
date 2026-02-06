import numpy as np
from scipy.optimize import minimize, differential_evolution
from src.utils import parse_string_tenor
from src.config import GLOBAL_OPT, LOCAL_OPT_INIT_GUESS, SEED

def calibrate_(hw_model, df_vols, seed=SEED):
    """ calibrate model parameters to swaptin market vol"""

    market_data = []
    for i, exp_label in enumerate(df_vols.index):
        # expiries
        T_exp = parse_string_tenor(str(exp_label))

        if T_exp <= 0.25: 
            continue

        for j, T_dur_label in enumerate(df_vols.columns):
            # tenors
            T_dur = parse_string_tenor(str(T_dur_label))
            # market vol
            vol_bps = df_vols.iloc[i, j]
            if np.isnan(vol_bps) or vol_bps <= 0: 
                continue

            # total time period from zero
            T_mat = T_exp + T_dur

            # annuity
            ann = sum(hw_model.zc_price_market(T_exp + t) for t in np.arange(1.0, T_dur + 1e-6, 1.0))
            if ann < 1e-10: 
                ann = hw_model.zc_price_market(T_mat)

            # swap rate
            S0 = (hw_model.zc_price_market(T_exp) - hw_model.zc_price_market(T_mat)) / ann

            market_data.append({'T_exp': T_exp, 'T_mat': T_mat, 'S0': S0, 'annuity': ann, 'mkt_vol': vol_bps})
    
    # optimization bounds    
    bounds = [(0.01, 1.2), (0.001, 0.2), (0.001, 0.1), (0.001, 0.05), (-0.99, -0.5)]

    best_rmse, iteration = np.inf, 0

    # minimize volatility RMSE
    def objective(params):
        nonlocal best_rmse, iteration
        hw_model.a, hw_model.b, hw_model.sigma, hw_model.eta, hw_model.rho = params
        
        penalty = 0.0
        if hw_model.a < hw_model.b + 0.02: 
            penalty += 5000.0 * ((hw_model.b + 0.02) - hw_model.a)**2

        err_sq = 0.0
        for pt in market_data:
            # model vol
            mod_vol = hw_model.calculate_model_vol_brigo(pt)
            # squared error
            err_sq += (mod_vol - pt['mkt_vol'])**2
        
        rmse = np.sqrt(err_sq / len(market_data))
        if (rmse + penalty) < best_rmse:
            best_rmse = rmse + penalty
            print(f"Iter {iteration:4d} | RMSE: {rmse:8.4f} | a:{params[0]:.3f} b:{params[1]:.3f} rho:{params[4]:.3f}")
        
        iteration += 1
        return rmse + penalty
    
    # two steps optimization: first Gloabl opt, then Local opt    
    current_x0 = LOCAL_OPT_INIT_GUESS

    if GLOBAL_OPT:
        print("Starting Global Optimization (Differential Evolution)...")
        res_global = differential_evolution(objective, bounds, popsize=10, seed=seed, maxiter=3, workers=1)
        current_x0 = res_global.x

    print("Starting Local Optimization (L-BFGS-B)...")
    res_final = minimize(objective, current_x0, method='L-BFGS-B', bounds=bounds)

    return res_final