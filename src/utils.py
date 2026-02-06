import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from src.config import SEED, MC_N_PATH, T_MAT, STRIKE, EXERCISE_DATES
import polars as pl


def parse_string_tenor(s):
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).upper().replace(' ', '')
    try:
        if 'M' in s:
            numeric_part = ''.join(c for c in s if c.isdigit() or c == '.')
            return float(numeric_part) / 12.0
        if 'Y' in s:
            numeric_part = ''.join(c for c in s if c.isdigit() or c == '.')
            return float(numeric_part)
        return float(s)
    except Exception as e:
        print(f"Error parsing tenor '{s}': {e}")
        return 0.0

def save_parameters(model, filename="logs/opt_params.json", rmse=None):
    params = {
        "timestamp": datetime.now().strftime("%Y-%m-%d"),
        "a": float(model.a), "b": float(model.b),
        "sigma": float(model.sigma), "eta": float(model.eta),
        "rho": float(model.rho),
    }
    if rmse is not None: params["rmse"] = float(rmse)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f: json.dump(params, f, indent=4)
    print(f"Parameters saved to {filename}")

def load_parameters(model, filename="logs/opt_params.json"):
    if not os.path.exists(filename): return False, None
    try:
        with open(filename, 'r') as f: params = json.load(f)
        file_date = params.get("timestamp", "")
        today = datetime.now().strftime("%Y-%m-%d")
        if file_date != today:
            print(f"--- ALERT: Parameters are from {file_date} ---")
        
        model.a = params["a"]
        model.b = params["b"]
        model.sigma = params["sigma"]
        model.eta = params["eta"]
        model.rho = params["rho"]
        return True, params.get("rmse")
    except Exception as e:
        print(f"Load failed: {e}")
        return False, None

def write_final_report(filename, model, calibration_res, pricing_res, greeks_res, martingale_stats, mc_stats):
    
    with open(filename, 'w') as f:
        # --- HEADER ---
        f.write("="*80 + "\n")
        f.write(f"G2++ MODEL: FINAL VALUATION REPORT\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        # --- 1. CALIBRATION RESULTS ---
        f.write("-" * 30 + "\n")
        f.write("1. CALIBRATION SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Optimization Status: {getattr(calibration_res, 'message', 'N/A')}\n")
        f.write(f"Final RMSE (bps): {(getattr(calibration_res, 'fun', 0.0)):.4f}\n\n")
        
        f.write("Calibrated Parameters:\n")
        f.write(f"  a (Mean Rev 1): {model.a:.6f}\n")
        f.write(f"  b (Mean Rev 2): {model.b:.6f}\n")
        f.write(f"  rho (Correlation): {model.rho:.6f}\n")
        f.write(f"  sigma (Vol 1): {model.sigma:.6f}\n")
        f.write(f"  eta (Vol 2): {model.eta:.6f}\n\n")

        # --- 2. MONTE CARLO ---
        f.write("-" * 30 + "\n")
        f.write("2. SIMULATION DIAGNOSTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Paths Generated: {mc_stats.get('n_paths', 'N/A')}\n")
        f.write(f"Time Steps: {mc_stats.get('n_steps', 'N/A')}\n")
        f.write(f"Random Number Gen: {mc_stats.get('rng_type', 'Sobol')}\n")
        f.write(f"Standard Error: {mc_stats.get('std_error', 0.0):.6f}\n")
        
        lower, upper = mc_stats.get('conf_interval_95', (0,0))
        f.write(f"95% Conf. Interval: [{lower:.4f}, {upper:.4f}]\n\n")

        # --- 3. MARTINGALE CHECK ---
        f.write("-" * 30 + "\n")
        f.write("3. MARTINGALE TEST\n")
        f.write("-" * 30 + "\n")
        f.write("Check: Is E[Discounted Asset] == Initial Asset?\n")
        f.write(f"Target Value (ZC): {martingale_stats.get('target_value', 0.0):.6f}\n")
        f.write(f"Simulated Mean: {martingale_stats.get('mean_discounted_payoff', 0.0):.6f}\n")
        f.write(f"Discrepancy (bps): {martingale_stats.get('error_bps', 0.0):.4f} bps\n")
        
        is_pass = abs(martingale_stats.get('error_bps', 999)) < 5.0
        status = "PASSED" if is_pass else "WARNING: HIGH BIAS"
        f.write(f"Status: {status}\n\n")

        # --- 4. PRICING RESULTS ---
        f.write("-" * 30 + "\n")
        f.write("4. PRICING RESULTS (bps)\n")
        f.write("-" * 30 + "\n")
        f.write(f"Structure: {pricing_res.get('product_name', 'Swaption')}\n")
        f.write(f"Strike: {pricing_res.get('strike', 0.0):.4%}\n")
        f.write("-" * 20 + "\n")
        f.write(f"Bermudan Price: {pricing_res.get('bermudan_price', 0.0):.4f}\n")
        f.write(f"European (MC): {pricing_res.get('european_price_mc', 0.0):.4f}\n")
        f.write(f"European (Analytic): {pricing_res.get('european_price_ana', 0.0):.4f}\n")
        f.write(f"Control Var Adj: {pricing_res.get('cv_adjustment', 0.0):.4f}\n\n")

        # --- 5. GREEKS ---
        f.write("-" * 30 + "\n")
        f.write("5. RISK SENSITIVITIES (Greeks)\n")
        f.write("-" * 30 + "\n")
        f.write(f"DV01 (Rate Delta): {greeks_res.get('dv01', 0.0):.4f}\n")
        f.write(f"Vega (Sigma): {greeks_res.get('vega_sigma', 0.0):.4f}\n")
        f.write(f"Vega (Eta): {greeks_res.get('vega_eta', 0.0):.4f}\n")
        f.write("="*80 + "\n")
        
    print(f"Report successfully generated at: {filename}")

def load_market_data(yield_path, vol_path):
    if not os.path.exists(yield_path) or not os.path.exists(vol_path):
        raise FileNotFoundError(f"Data files not found:\n{yield_path}\n{vol_path}")

    # yields
    df_yields = pd.read_csv(yield_path, sep=";")
    
    def parse_unit(row):
        mapping = {'WK': 1/52, 'MO': 1/12, 'YR': 1.0}
        return row['Term'] * mapping.get(row['Unit'].upper(), 1.0)
    
    df_yields['Years'] = df_yields.apply(parse_unit, axis=1)
    df_yields['Mid'] = (df_yields['Final Bid Rate'] + df_yields['Final Ask Rate']) / 200.0
    
    # vols
    df_vols = pd.read_csv(vol_path, index_col=0, sep=";")
    
    return df_yields['Years'].values, df_yields['Mid'].values, df_vols

def run_model_validation(model, T_val=T_MAT, n_paths=MC_N_PATH):
    """
    Runs Martingale tests and MC convergence checks
    """
    print(f"Running validation (Martingale Check @ {T_val}Y)...")
    
    # ruun pure simulation
    _, _, disc_factors = model.simulate(
        sim_times=np.linspace(0, T_val, 50), 
        n_paths=n_paths, 
        seed=SEED
    )

    # martingale stats
    sim_zc = np.mean(disc_factors[-1, :])
    ana_zc = model.zc_price_market(T_val)
    
    martingale_stats = {
        'target_value': ana_zc,
        'mean_discounted_payoff': sim_zc,
        'error_bps': (sim_zc - ana_zc) * 10000}

    # 2. MC Stats
    std_err = np.std(disc_factors[-1, :]) / np.sqrt(n_paths)
    mc_stats = {
        'n_paths': n_paths, 'n_steps': 50, 'rng_type': 'Sobol',
        'std_error': std_err,
        'conf_interval_95': (sim_zc - 1.96*std_err, sim_zc + 1.96*std_err)}
    
    return martingale_stats, mc_stats

def prepare_pricing_report(model, bermudan_price, T_exercises=EXERCISE_DATES, T_mat=T_MAT, K=STRIKE):
    """
    Calculates European 
    """
    print("Calculating benchmark European prices...")
    
    T_euro = T_exercises[-1]
    
    # analytic european (Bachelier)
    ann_euro = sum(model.zc_price_market(T_euro + t) for t in np.arange(1.0, T_mat - T_euro + 1e-6, 1.0))
    if ann_euro < 1e-10: 
        ann_euro = model.zc_price_market(T_mat)
    
    S0_euro = (model.zc_price_market(T_euro) - model.zc_price_market(T_mat)) / ann_euro
    vol_in = {'T_exp': T_euro, 'T_mat': T_mat, 'S0': S0_euro, 'annuity': ann_euro}
    vol_bps = model.calculate_model_vol_brigo(vol_in)
    
    euro_ana = model.price_bachelier_swaption(T_euro, T_mat, K, vol_bps, ann_euro)
    
    # MC European
    x, y, d = model.simulate([0.0, T_euro], n_paths=5000, seed=SEED)
    payoff = model.get_swaption_payoff_grid(T_euro, T_mat, K, x[-1], y[-1])
    euro_mc = np.mean(payoff * d[-1])
    
    return {
        'product_name': f'Bermudan {int(T_exercises[0])}Y-{int(T_mat)}Y',
        'strike': K,
        'bermudan_price': bermudan_price * 10000,
        'european_price_mc': euro_mc * 10000,
        'european_price_ana': euro_ana * 10000,
        'cv_adjustment': (euro_mc - euro_ana) * 10000
    }

