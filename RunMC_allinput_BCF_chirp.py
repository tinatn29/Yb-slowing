import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import sys
from SSParameters import GreenLaser
from SSMCSolver_2 import MCSolver
import RandomInitialize as RI


'''
RunMC_BCF_allinput_chirp.py [1]<Delta/gamma> [2]<result_file.csv> 
[3]<chirp vmax> [4]<chirp vmin> [5]<chirp period> [6 optional]<beam radius in m>
- Gaussian beam profile
- with chirp
'''


# Set laser parameters
delta = float(sys.argv[1]) # bichromatic detuning from argument
filename = sys.argv[2] # filename from argument

MC = MCSolver(delta=delta, omega=omega, phi=-np.pi/4)

# Set up chirp, sweeping velocity detuning from vmax to vmin in chirp_period
vmax = float(sys.argv[3]) # vmax (m/s) chirp from argument e.g. 140
vmin = float(sys.argv[4]) # vmin (m/s) chirp from argument e.g. 10
chirp_period = float(sys.argv[5]) # chirp period e.g. 3e-3 = 3 ms
MC.SetChirping(chirp_period, vmax, vmin) # set up chirp parameters
print("Chirping between {0} m/s and {1} m/s at period of {2:.2f} ms".format(vmax, vmin, chirp_period * 1000))
print("Delta = {0:.0f} * gamma".format(delta))

# optional argument: specify beam radius (in case we want a radius that is different from 4mm)
if len(sys.argv) > 6:
    MC.beam_radius = float(sys.argv[6])

print("Beam radius = {0:.0f} mm".format(MC.beam_radius * 1000))

# load v_initial from file to start with the same distribution
# 2mm oven radius
v_input = np.load('./input_files/v_input_2mm_5000.npy')
r_input = np.load('./input_files/r_input_2mm_5000.npy')
t_input = np.load('./input_files/t_input_2mm_5000.npy')

if __name__ == '__main__':
    results = MC.SolveMC_parallel_InputAll(v_input, r_input, t_input)
    with open(filename, 'a', newline='') as file:  # 'a' for append
        writer = csv.writer(file)
        writer.writerows([result for result in results])







