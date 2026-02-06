import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm, qmc
from src.config import ANTITHETIC_VAR, SOBOL, B_BRIDGE, SEED, GREEKS_N_PATH, MC_N_PATH, BUMP


class G2pp:
    def __init__(self, times, yields):
        # CubicSpline (PCHIP) -> interpolate market yields
        self.curve_spline = PchipInterpolator(times, yields)
        
        # spline on log-discount factors log(P(0,t)) = - y(t) * t
        log_p_values = -np.array(yields) * np.array(times)
        self.log_p_spline = PchipInterpolator(times, log_p_values)

        # derivative spline: d/dt log P(0,t) = -f(0,t)        
        self.d_log_p_dt = self.log_p_spline.derivative(nu=1)
        
        # default parameters
        self.a = 0.1
        self.b = 0.05
        self.sigma = 0.01
        self.eta = 0.01
        self.rho = -0.7
        
        # store LSM regression coeffs
        self.lsm_coeffs = {}

    def update_curve_data(self, times, new_yields):
        """
        rebuilds internal splines using provided times and new_yields
        used for Greek calculation
        """
        self.curve_spline = PchipInterpolator(times, new_yields)
        log_p_values = -np.array(new_yields) * np.array(times)
        self.log_p_spline = PchipInterpolator(times, log_p_values)        
        self.d_log_p_dt = self.log_p_spline.derivative(nu=1)

    # ---------------------------
    # Components
    # ---------------------------
    def instantaneous_forward_rate(self, t):
        # f(0,t) = - d/dt log P(0,t)
        if t <= 1e-8: 
            return -self.d_log_p_dt(1e-8)
        return -self.d_log_p_dt(t)
       
    def phi(self, t):
        """
        time-dependent drift adjustment -> fit initial curve
        phi(t) = f(0,t) + convexity terms
        """
        f0t = self.instantaneous_forward_rate(t)

        # convexity adjustment for factor x
        conv_x = (self.sigma**2 / (2 * self.a**2)) * (1 - np.exp(-self.a * t))**2
        # convexity adjustment for factor y
        conv_y = (self.eta**2 / (2 * self.b**2)) * (1 - np.exp(-self.b * t))**2
        # correlation term
        conv_xy = 0.0
        if abs(self.rho) > 1e-10:
            conv_xy = self.rho * (self.sigma * self.eta / (self.a * self.b)) * \
                      (1 - np.exp(-self.a * t)) * (1 - np.exp(-self.b * t))

        return f0t + conv_x + conv_y + conv_xy

    def zc_price_market(self, T):
        # market price of ZCB P(0, T)
        if T <= 1e-8: 
            return 1.0
        return np.exp(self.log_p_spline(T))

    def B_func(self, k, t, T):
        # sensi B(z, t, T) = (1 - exp(-k(T-t))) / k
        return (1.0 - np.exp(-k * (T - t))) / k

    def bond_price_2f(self, t, T, x_t, y_t):
        """analytical ZCB price"""
        if T <= t + 1e-8: 
            return 1.0
        dt = T - t
        Ba = self.B_func(self.a, t, T)
        Bb = self.B_func(self.b, t, T)
        
        # variances        
        def V_func(tau):
            term_a = (self.sigma**2 / self.a**2) * (tau + (2.0/self.a)*np.exp(-self.a*tau) - (1.0/(2.0*self.a))*np.exp(-2.0*self.a*tau) - (3.0/(2.0*self.a)))
            term_b = (self.eta**2 / self.b**2) * (tau + (2.0/self.b)*np.exp(-self.b*tau) - (1.0/(2.0*self.b))*np.exp(-2.0*self.b*tau) - (3.0/(2.0*self.b)))
            term_rho = 2.0*self.rho*(self.sigma*self.eta/(self.a*self.b)) * (tau + (np.exp(-self.a*tau)-1.0)/self.a + (np.exp(-self.b*tau)-1.0)/self.b - (np.exp(-(self.a+self.b)*tau)-1.0)/(self.a+self.b))
            return term_a + term_b + term_rho

        # convexity adjustment
        V = 0.5 * (V_func(dt) - V_func(T) + V_func(t))

        discount_ratio = self.zc_price_market(T) / self.zc_price_market(t) if t > 1e-8 else self.zc_price_market(T)
        
        return discount_ratio * np.exp(V - Ba * x_t - Bb * y_t)

    def calculate_model_vol_brigo(self, pt):
        """ model-implied Normal swaption vol """
        T_exp, T_mat = pt['T_exp'], pt['T_mat']
        # coupon dates
        coupon_times = np.arange(T_exp + 1.0, T_mat + 1e-6, 1.0)
        if len(coupon_times) == 0: 
            return 0.0

        # weights and annuity
        p_exp = self.zc_price_market(T_exp)
        p_mat = self.zc_price_market(T_mat)
        zcs = np.array([self.zc_price_market(t) for t in coupon_times])
        annuity = np.sum(zcs)
         
        S_0 = (p_exp - p_mat) / annuity
        
        # coeffs c_i for the linear combination
        weights = {T_exp: 1.0, T_mat: -(1.0 + S_0)}
        for t in coupon_times[:-1]: 
            weights[t] = -S_0

        relevant_times = sorted(weights.keys())
        
        # covariance matrix helper
        def get_covariance(Ti, Tj):
            Ba_i, Ba_j = self.B_func(self.a, T_exp, Ti), self.B_func(self.a, T_exp, Tj)
            Bb_i, Bb_j = self.B_func(self.b, T_exp, Ti), self.B_func(self.b, T_exp, Tj)

            var_x = (self.sigma**2 / (2 * self.a)) * (1 - np.exp(-2 * self.a * T_exp)) * Ba_i * Ba_j
            var_y = (self.eta**2 / (2 * self.b)) * (1 - np.exp(-2 * self.b * T_exp)) * Bb_i * Bb_j
            cov_xy = self.rho * (self.sigma * self.eta / (self.a + self.b)) * \
                     (1 - np.exp(-(self.a + self.b) * T_exp)) * (Ba_i * Bb_j + Ba_j * Bb_i)
            return var_x + var_y + cov_xy

        # sum weighted covariances
        total_var = 0.0
        for Ti in relevant_times:
            for Tj in relevant_times:
                total_var += weights[Ti] * weights[Tj] * self.zc_price_market(Ti) * self.zc_price_market(Tj) * get_covariance(Ti, Tj)

        # from price variance to rate vol
        vol_n = np.sqrt(max(total_var, 0.0) / T_exp) / annuity
        return vol_n * 10000

    def calculate_model_vol_hw(self, pt):
        """
        Calculates model implied Normal vol using Rebonato's effective duration approximation
        """
        T_exp = pt['T_exp']
        T_mat = pt['T_mat']
        annuity = pt['annuity']
        S0 = pt['S0']
        tau = 1.0 # Assumes annual payment frequency
        
        # effective sensi
        B_a_eff, B_b_eff = 0.0, 0.0
        payment_dates = np.arange(T_exp + tau, T_mat + 1e-6, tau)
        
        for Ti in payment_dates:
            # weight w_i = tau * P(0, Ti) / A(0)
            weight = tau * self.zc_price_market(Ti) / annuity
            B_a_eff += weight * self.B_func(self.a, T_exp, Ti)
            B_b_eff += weight * self.B_func(self.b, T_exp, Ti)
        
        # add terminal weight correction
        weight_mat = self.zc_price_market(T_mat) / annuity
        B_a_eff += weight_mat * self.B_func(self.a, T_exp, T_mat)
        B_b_eff += weight_mat * self.B_func(self.b, T_exp, T_mat)

        # integrated factor variance/covariance matrix
        v1 = (self.sigma**2 / (2 * self.a)) * (1.0 - np.exp(-2.0 * self.a * T_exp))
        v2 = (self.eta**2 / (2 * self.b)) * (1.0 - np.exp(-2.0 * self.b * T_exp))
        v12 = (self.rho * self.sigma * self.eta / (self.a + self.b)) * (1.0 - np.exp(-(self.a + self.b) * T_exp))
        
        # total variance swap rate
        total_var = (B_a_eff**2 * v1 + B_b_eff**2 * v2 + 2.0 * B_a_eff * B_b_eff * v12)
        
        # variance to Normal vol
        return np.sqrt(max(total_var, 1e-15) / T_exp) * 10000

    def price_bachelier_swaption(self, T_exp, T_mat, K, vol_bps, annuity):
        """analytical swaption normal price"""
        # fwd swap rate
        S0 = (self.zc_price_market(T_exp) - self.zc_price_market(T_mat)) / annuity
        # abs vol
        sigma_n = vol_bps / 10000.0
        # moneyness
        d = (S0 - K) / (sigma_n * np.sqrt(T_exp))

        # payer = A * [(S-K)*cdf(d) + sigma*sqrt(T)*pdf(d)]
        return annuity * ((S0 - K) * norm.cdf(d) + sigma_n * np.sqrt(T_exp) * norm.pdf(d))

    # ---------------------------
    # Simulation
    # ---------------------------
    def get_sobol_paths(self, n_paths, n_steps, seed=SEED):
        """ sobol Quasi Monte Carlo"""
        # two processes
        dim = 2 * n_steps

        # sobol generator scramble=True!!!
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed) 

        # generate sapmle paths
        u_vec = sampler.random(n=n_paths)
        # from Uniform[0,1] to standard normal 
        z_vec = norm.ppf(u_vec)

        # initialize the 3d array with zeros -> shape: 2 dim for 2 factors, time steps, siml run
        z_reshaped = np.zeros((2, n_steps, n_paths))
        # rows -> time steps 
        # cols -> paths
        z_vec_T = z_vec.T

        for t in range(n_steps):
            # split sobol vector in half -> one per process
            z_reshaped[0, t, :] = z_vec_T[t, :]
            z_reshaped[1, t, :] = z_vec_T[n_steps + t, :]

        return z_reshaped

    def get_brownian_bridge(self, z_input, sim_times):
        """ build brownian bridge"""

        n_steps, n_paths = z_input.shape
        W = np.zeros((n_steps + 1, n_paths))

        # breadth-first search BFS -> in which order time steps are calculated
        # fix end point
        bridge_indices = [n_steps]
        # it maps the indexes used to solve current mid index
        map_dependency = {n_steps: (0, 0)}

        # start with full interval
        queue = [(0, n_steps)]
        # organize the time steps hierachically -> end, mid, quarters, eights
        while queue:
            # extrat first point -> get mid points before any qaurter point (BFS)
            left, right = queue.pop(0)
            # average
            mid = (left + right) // 2

            if mid != left and mid != right:
                # add new mid time step
                bridge_indices.append(mid)
                # it maps the indexes used to solve current mid index
                map_dependency[mid] = (left, right)
                # add sub-interval to the back of the queue 
                queue.append((left, mid))
                queue.append((mid, right))
                
        for i, target_idx in enumerate(bridge_indices):
            # best quality sobol (bridge indices) to first time step (target_idex)            
            # initial shocks
            z_val = z_input[i, :]
            if i == 0:
                # full period
                T = sim_times[n_steps]
                # W_T = sqrt(T) * Z
                W[target_idx, :] = np.sqrt(T) * z_val
            else:
                # indexes (start/end, quartes, etc)
                left, right = map_dependency[target_idx]
                # current, left, right time values
                ti, tj, tk = sim_times[left], sim_times[target_idx], sim_times[right]
                # delta time steps -> periods
                dt_ik, dt_ij, dt_jk = tk - ti, tj - ti, tk - tj
                # epxected value -> average bwt left and right
                mu = (dt_jk / dt_ik) * W[left, :] + (dt_ij / dt_ik) * W[right, :]
                # conditional var
                sigma = np.sqrt((dt_ij * dt_jk) / dt_ik)
                # inetrmediate point -> expectation + noise
                W[target_idx, :] = mu + sigma * z_val

        return np.diff(W, axis=0)

    def simulate(self, sim_times, n_paths, rng_type=SOBOL, seed=SEED, antithetic=ANTITHETIC_VAR, use_bb=B_BRIDGE):
        """ Monte Carlo engine"""
        
        n_steps = len(sim_times) - 1
        # antithetic variates
        n_gen = n_paths // 2 if antithetic else n_paths
        
        if rng_type:
            z_gen = self.get_sobol_paths(n_gen, n_steps, seed=seed)
        else:
            np.random.seed(seed)
            z_gen = np.random.normal(0, 1, (2, n_steps, n_gen))
        
        # antithetic variates
        z = np.concatenate([z_gen, -z_gen], axis=2) if antithetic else z_gen
        actual_paths = z.shape[2]
        x, y = np.zeros((n_steps + 1, actual_paths)), np.zeros((n_steps + 1, actual_paths))
        discount_to_t0 = np.ones((n_steps + 1, actual_paths))

        if use_bb:
            # sobol returns increments alredy scaled - dW
            dW1 = self.get_brownian_bridge(z[0, :, :], sim_times)
            dW2 = self.get_brownian_bridge(z[1, :, :], sim_times)
        else:
            # Z * dt
            dt_arr = np.diff(sim_times).reshape(-1, 1)
            dW1 = z[0, :, :] * np.sqrt(dt_arr)
            dW2 = z[1, :, :] * np.sqrt(dt_arr)

        # cholesky for 2x2 corr matrix 
        dW2_corr = self.rho * dW1 + np.sqrt(1.0 - self.rho**2) * dW2 if abs(self.rho) > 1e-9 else dW2
            
        for t in range(n_steps):
            dt = sim_times[t+1] - sim_times[t]
            exp_a, exp_b = np.exp(-self.a * dt), np.exp(-self.b * dt)
            
            if dt > 1e-9:
                # OU std dev
                sd_ou_x = self.sigma * np.sqrt((1 - np.exp(-2 * self.a * dt)) / (2 * self.a))
                sd_ou_y = self.eta * np.sqrt((1 - np.exp(-2 * self.b * dt)) / (2 * self.b))
                # divided by sqrt(dt) because later we shock increments dW that are already scaled
                scale_x = sd_ou_x / np.sqrt(dt)
                scale_y = sd_ou_y / np.sqrt(dt)
            else:
                scale_x, scale_y = 0.0, 0.0
            
            # processes evolution
            x[t+1] = x[t] * exp_a + dW1[t, :] * scale_x
            y[t+1] = y[t] * exp_b + dW2_corr[t, :] * scale_y

            # stochastic disocunt factor -> numerically integrate r over dt
            # trapezoidal rule -> averaging phi (det) and r (stoc)
            phi_term = 0.5 * (self.phi(sim_times[t]) + self.phi(sim_times[t+1]))    
            r_avg = 0.5 * (x[t] + y[t] + x[t+1] + y[t+1]) + phi_term
            # accumulate discount factor
            discount_to_t0[t+1] = discount_to_t0[t] * np.exp(-r_avg * dt)

        return x, y, discount_to_t0

    # ---------------------------
    # Pricing
    # ---------------------------
    def get_swaption_payoff_grid(self, t, T_mat, K, x_t, y_t):
        """ instantaneous swaptions payoff at time t"""

        # assume annual payments
        coupon_times = np.arange(t + 1.0, T_mat + 1e-6, 1.0)
        if len(coupon_times) == 0: 
            return np.zeros_like(x_t)
        
        # sum all past disc factors
        annuity = np.sum([self.bond_price_2f(t, cp, x_t, y_t) for cp in coupon_times], axis=0)
        # zcb price at swap maturity
        p_mat = self.bond_price_2f(t, T_mat, x_t, y_t)
        # par arte
        swap_rate = (1.0 - p_mat) / annuity
        # payoff
        return np.maximum(annuity * (swap_rate - K), 0)

    def price_bermudan_lsm(self, T_exercises, T_mat, K, n_paths=MC_N_PATH, rng_type=SOBOL, seed=SEED, antithetic=ANTITHETIC_VAR):
        """ bermudan swaption pricer with Longstaff-Swartz and european Control Variate""" 

        self.lsm_coeffs = {}
        # exercise dates and pay dates
        unique_times = set(T_exercises) | {0.0} | set(np.arange(min(T_exercises) + 1.0, T_mat + 0.001, 1.0))

        sim_times = np.array(sorted(list(unique_times)))
        ex_indices = np.searchsorted(sim_times, T_exercises)
        
        # generate processes and associated discount factors
        x_p, y_p, discount_to_t0 = self.simulate(sim_times, n_paths, rng_type=rng_type, seed=seed, antithetic=antithetic)
        
        # control: antithetic variates could generate a different num of paths than requested
        actual_paths = x_p.shape[1]

        # start from last
        idx_last = ex_indices[-1]
        # payoff at expiry
        payoffs = self.get_swaption_payoff_grid(sim_times[idx_last], T_mat, K, x_p[idx_last], y_p[idx_last])
        
        # backward induction
        for i in range(len(T_exercises) - 2, -1, -1):
            idx_curr, idx_next = ex_indices[i], ex_indices[i+1]
            # discount future values
            payoffs *= (discount_to_t0[idx_next] / discount_to_t0[idx_curr])
            # exercise now
            intrinsic = self.get_swaption_payoff_grid(T_exercises[i], T_mat, K, x_p[idx_curr], y_p[idx_curr])
            
            # indexing in-the-money paths
            itm = intrinsic > 1e-6
            
            # continuation values
            coeffs = np.zeros(6)
            if np.sum(itm) > 50:
                # only ITM
                X_itm = x_p[idx_curr, itm]
                Y_itm = y_p[idx_curr, itm]
                # basis function (quadratic)-> prepare reg variables, only ITM paths
                A = np.column_stack([np.ones_like(X_itm), X_itm, Y_itm, X_itm**2, Y_itm**2, X_itm*Y_itm])
                # regression
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(A, payoffs[itm], rcond=None)
                except: 
                    pass
                # prepare basis function, all paths!!
                A_all = np.column_stack([np.ones(actual_paths), x_p[idx_curr], y_p[idx_curr], x_p[idx_curr]**2, y_p[idx_curr]**2, x_p[idx_curr]*y_p[idx_curr]])
                
                # continuation value -> apply reg coeffs to basis function 
                continuation = np.maximum(A_all @ coeffs, 0.0)
                # decision mask
                exercise = (intrinsic > continuation) & itm
                # update payoff only for exercising paths  
                payoffs[exercise] = intrinsic[exercise]
            
            # store reg coeffs
            self.lsm_coeffs[T_exercises[i]] = coeffs
        # avg of discounted prices across scenarios
        bermudan_price = np.mean(payoffs * discount_to_t0[ex_indices[0]])
        
        # Control Variate
        T_exp_euro = T_exercises[-1]
        # eur swapt price with same random process as before 
        mc_euro = np.mean(self.get_swaption_payoff_grid(T_exp_euro, T_mat, K, x_p[idx_last], y_p[idx_last]) * discount_to_t0[idx_last])
        
        # annuity eur swapt 
        ann_euro = sum(self.zc_price_market(T_exp_euro + tau) for tau in np.arange(1.0, T_mat - T_exp_euro + 1e-6, 1.0))
        if ann_euro < 1e-10: 
            ann_euro = self.zc_price_market(T_mat)
        # swap rate
        S0_euro = (self.zc_price_market(T_exp_euro) - self.zc_price_market(T_mat)) / ann_euro
        # market-vol
        vol_euro_bps = self.calculate_model_vol_brigo({'T_exp': T_exp_euro, 'T_mat': T_mat, 'S0': S0_euro, 'annuity': ann_euro})
        # eur swapt analytical price
        euro_analytical = self.price_bachelier_swaption(T_exp_euro, T_mat, K, vol_euro_bps, ann_euro)

        # scale bermudan MC price for the error in european MC price
        return bermudan_price - (mc_euro - euro_analytical)

    def calculate_greeks(self, T_exercises, T_mat, K, times, yields, stress=BUMP, n_paths=GREEKS_N_PATH):
        """ finite difference method"""

        # base price
        times_arr, yields_arr = np.array(times), np.array(yields)
        self.update_curve_data(times_arr, yields_arr)
        base = self.price_bermudan_lsm(T_exercises, T_mat, K, n_paths, seed=SEED)
        
        # delta DV01
        # add bumbps to curves -> recalibrate phi
        self.update_curve_data(times_arr, yields_arr + stress)
        # new swpt price
        delta_p = self.price_bermudan_lsm(T_exercises, T_mat, K, n_paths, seed=SEED)
        # restore curves
        self.update_curve_data(times_arr, yields_arr)
        
        # vega X
        old_sigma = self.sigma
        self.sigma += stress
        vega_s_p = self.price_bermudan_lsm(T_exercises, T_mat, K, n_paths, seed=SEED)
        # reset to old vega
        self.sigma = old_sigma
        
        # vega Y
        old_eta = self.eta
        self.eta += stress
        vega_e_p = self.price_bermudan_lsm(T_exercises, T_mat, K, n_paths, seed=SEED)
        self.eta = old_eta
        
        return {
            "price": base, 
            "dv01": (delta_p - base) * 10000, 
            "vega_sigma": (vega_s_p - base) * 10000, 
            "vega_eta": (vega_e_p - base) * 10000
        }