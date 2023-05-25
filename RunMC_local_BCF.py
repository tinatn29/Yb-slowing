import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import sys
from SSMCSolver_BCF import MCSolver
import RandomInitialize as RI


'''
RunMC_BCF_local.py
This file is for local testing of SSMCSolver_BCF.
Make sure to only run a few atoms
'''

delta = 50 # bichromatic detuning
omega = delta * np.sqrt(3/2) # optimal force condition
v_detuning = 0
loading_time = 1e-3 # loading time of 1ms

MC = MCSolver(delta=delta, omega=omega, phi=-np.pi/4, loading_time=loading_time)

# load v_initial from file to start with the same distribution
# 2mm oven radius
t_input = np.load('./input_files/t_old_new_input_2mm.npy')
print(len(t_input))

## Uncomment to select chirp or fixed detuning
'''
# Test fixed detuning
MC.args['detuning'] = MC.args['k'] * v_detuning # velocity detuning
print(f"Fixed detuning at {v_detuning:.1f} m/s, applied for {loading_time*1000:.1f} ms")
'''

'''
# Test chirp
chirp_period = 1e-3 # 1ms
vmax = 10
vmin = 0
MC.SetChirping(chirp_period, vmax, vmin)
print("Chirping between {0} m/s and {1} m/s at period of {2:.2f} ms".format(vmax, vmin, chirp_period * 1000))
'''

v_initial = np.array([0.0, 0.0, 10.0])
r_initial = np.array([0.0, 0.0, 0.0])
t_initial = 0
output = MC.SolveMC_single_GaussianBeam_InputAll(v_initial, r_initial, t_initial)
print(output)

