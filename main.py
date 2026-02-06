import os
import sys
import numpy as np
from types import SimpleNamespace
# model
from src.model import G2pp
# calibration
from src.calibration import calibrate_
# plot
from src.plotting import plot_residuals_heatmap, plot_exercise_boundary, plot_vol_diagnostics
# config
from src.config import EXERCISE_DATES, STRIKE, T_MAT, VOL_PLOTS
# helpers   
from src.utils import load_market_data,load_parameters,save_parameters,write_final_report,run_model_validation,prepare_pricing_report


def run_pipeline():
    print("=" * 80); print("STARTING HW2F MODEL PIPELINE"); print("=" * 80)

    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    ROOT = os.path.dirname(os.path.abspath(__file__))
    years, rates, df_vols = load_market_data(os.path.join(ROOT, "data", "estr.csv"),os.path.join(ROOT, "data", "estr_vol_surface.csv"))

    # ---------------------------------------------------------
    # 2. Instantiate and Calibrate
    # ---------------------------------------------------------
    hw_model = G2pp(years, rates)
    
    success, rmse = load_parameters(hw_model, "logs/opt_params.json")
    
    if success:
        print(f"Parameters loaded (RMSE: {rmse:.4f})")

        res = SimpleNamespace(success=True, message="Loaded from JSON", fun=rmse)
    else:
        print("No parameters found. Calibrating...")
        res = calibrate_(hw_model, df_vols)
        if res.success:
            save_parameters(hw_model, "logs/opt_params.json", rmse=res.fun)
            plot_residuals_heatmap(hw_model, df_vols)
            plot_vol_diagnostics(hw_model, df_vols)

    if not res.success:
        print("Calibration failed. Exiting."); sys.exit(1)

    # ---------------------------------------------------------
    # 3. Pricing and Greeks
    # ---------------------------------------------------------
    print(f"\nPricing Bermudan Swaption (Strike: {STRIKE:.2%})...")

    price_raw = hw_model.price_bermudan_lsm(EXERCISE_DATES, T_MAT, STRIKE)
    print(f">> Price: {price_raw*10000:.2f} bps")
    
    print("Calculating Greeks...")
    greeks = hw_model.calculate_greeks(EXERCISE_DATES, T_MAT, STRIKE, years, rates)

    # ---------------------------------------------------------
    # 4. Diagnostics & Reporting
    # ---------------------------------------------------------
    if VOL_PLOTS:
        plot_exercise_boundary(hw_model, 1.0, T_MAT, STRIKE)
        plot_residuals_heatmap(hw_model, df_vols)
        plot_vol_diagnostics(hw_model, df_vols, fixed_res_scale=None)
        
    # model tests
    martingale_stats, mc_stats = run_model_validation(hw_model, T_val=T_MAT)

    # writer helper
    pricing_res = prepare_pricing_report(hw_model, greeks['price'])

    # save txt report in local 
    write_final_report("logs/Report.txt", hw_model, res, pricing_res, greeks, martingale_stats, mc_stats)


if __name__ == "__main__":
    run_pipeline()