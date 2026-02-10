# Two-Factors Gaussian model (Hull-White 2F)

This project implements a Two-Factor Gaussian model (G2++) model. It features a Bermudan Swaptions pricer with Longstaff-Schwartz Least-Squares Monte Carlo (LSM) method. 

## Features
**Calibration**
   * **Analytical Formula**: Fits model parameters ($a, b, \sigma, \eta, \rho$) to market Swaption volatility surfaces with analytical approximation.
   * **Optimization**: Supports both Global (Differential Evolution) and Local (L-BFGS-B) optimization strategies. Default is first Global, then Local.

**Monte Carlo Simulation**
   * **Quasi-Monte Carlo**: Utilizes Sobol Sequences for faster convergence.
   * **Brownian Bridge**: Implements Brownian Bridge path construction to further improve the distribution of path generation.
   * **Antithetic Variates**: systematic variance reduction for the stochastic drivers.

**Pricing:**
   * Implements the **Longstaff-Schwartz (LSM)** algorithm to approximate the optimal exercise boundary.
   * **Control Variate**: Corrects the Bermudan Monte Carlo price using the residual error between the Monte Carlo European price and the Analytical European price (Bachelier/Normal model).
   * **Greeks**: Calculates DV01 (Delta) and Vega sensitivities via Finite Difference.

**Diagnostics and Visualization**
   * **Validation**: Automated Martingale Tests to ensure the model correctly recovers the initial Zero Coupon curve.
   * **Plots**:
     * 3D Volatility Surface (Market vs. Model vs. Residuals).
     * Heatmap of calibration errors.
     * Exercise Boundary: Visualizes the "Stop vs. Continue" regions in the 2D factor space ($x, y$).

## Project Structure
```text

├── data/
│   ├── estr.csv               # Yield curve data (ESTR)
│   └── estr_vol_surface.csv   # Swaption Volatility Surface
├── logs/
│   ├── opt_params.json        # Cached calibrated parameters
│   └── Report.txt             # Final report
├── src/
│   ├── calibration.py         # Optimization
│   ├── config.py              # Global configuration parameters
│   ├── model.py               # G2++ class
│   ├── plotting.py            # Plots
│   └── utils.py               # Helpers
└── main.py                    # Entry point
```
## Output
* **Console**: Real-time logging of calibration progress (RMSE) and pricing results.

* **Plots**: If ```VOL_PLOTS = True``` in config, interactive matplotlib windows will show the calibration quality and exercise boundaries.

* **Report**: A detailed ```logs/Report.txt``` is generated, containing:
  * Calibration statistics (RMSE, parameters).
  * Martingale validation status (Pass/Fail).
  * Monte Carlo confidence intervals.
  * Final Prices (Bermudan vs European) and Greeks.
 
## References
* Andersen, L. (2000). _"A Simple Approach to the Pricing of Bermudan Swaptions in the Multi-Factor LIBOR Market Model"._ Journal of Computational Finance.
* Brigo, D., & Mercurio, F. (2001). _"A deterministic-shift extension of analytically tractable and time-homogeneous short-rate models."_ Finance and Stochastics.
* Brigo, D., & Mercurio, F. (2006). _"Interest Rate Models - Theory and Practice: With Smile, Inflation and Credit"._ Springer-Verlag.
* Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). _"A Limited Memory Algorithm for Bound Constrained Optimization"._ SIAM Journal on Scientific Computing.
* Glasserman, P. (2003). _"Monte Carlo Methods in Financial Engineering"._ Springer-Verlag.
* Glasserman, P., & Yu, B. (2004). _"Number of Paths Versus Number of Basis Functions in American Option Pricing"._ Operations Research.
* Hull, J., & White, A. (1990). _"Pricing Interest-Rate-Derivative Securities."_ The Review of Financial Studies. 
* Jäckel, P. (2002). _"Monte Carlo Methods in Finance"._ John Wiley & Sons.
* Joy, C., Boyle, P. P., & Tan, K. S. (1996). _"Quasi-Monte Carlo Methods in Numerical Finance"._ Management Science.
* Longstaff, F. A., & Schwartz, E. S. (2001). _"Valuing American Options by Simulation: A Simple Least-Squares Approach."_ The Review of Financial Studies. 
* Moskowitz, B., & Caflisch, R. E. (1996). _"Smoothness and Dimension Reduction in Quasi-Monte Carlo Methods"._ Mathematical and Computer Modelling.
* Storn, R., & Price, K. (1997). _"Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces."_ Journal of Global Optimization.
