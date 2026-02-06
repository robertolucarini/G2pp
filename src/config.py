

# calibration
GLOBAL_OPT = False
LOCAL_OPT_INIT_GUESS = [0.1, 0.05, 0.01, 0.01, -0.7]

# volatility diagnostic plots
VOL_PLOTS = True

# simulation
SOBOL = True
ANTITHETIC_VAR = True
B_BRIDGE = True
SEED = 56
MC_N_PATH = 2**13 

# bermudan static data
EXERCISE_DATES = [1.0, 2.0, 3.0, 4.0, 5.0]
STRIKE = 0.03
T_MAT = 10.0

# greeks
GREEKS_N_PATH = 2**13
BUMP = 0.0001