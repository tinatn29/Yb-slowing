import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import sys
from SSMCSolver_square import MCSolver


'''
RunMC_allinput_squareAM_chirp.py [1]<Delta/gamma> [2]<Rabi shape>
[3]<chirp vmax m/s> [4]<chirp vmin m/s> [5]<chirp period in s> 
[6]<beam radius in m> 
[7]<filename> .csv filename for outputs 
- Gaussian beam
- chirp
'''


# Set laser parameters
delta = float(sys.argv[1])  # delta in gamma
omega = delta * np.pi / 4 # optimal force condition

Rabi_shape = int(sys.argv[2])   # 0 = square, 1 = cosine, 2+ = square Fourier series

# set chirp parameters
vmax = float(sys.argv[3])  # vmax (m/s) upper chirp limit
vmin = float(sys.argv[4]) # vmin (m/s) lower chirp limit
chirp_period = float(sys.argv[5]) # chirp period in s e.g. 1e-3 for 1 ms

beam_radius = float(sys.argv[6]) # laser beam radius in m e.g. 0.004 = 4 mm
filename = sys.argv[7] # filename.csv for output

# Set up the solver
MC = MCSolver(delta=delta, omega=omega, Rabi_shape=Rabi_shape, phi=0.36*np.pi, beam_radius=beam_radius, Gaussian_beam=True)
MC.SetChirping(chirp_period, vmax, vmin) # chirping
print("Chirping between {0} m/s and {1} m/s at period of {2:.2f} ms".format(vmax, vmin, chirp_period * 1000))

# load v_initial from file to start with the same distribution
# 2mm oven radius
v_input = np.load('./input_files/v_input_2mm_5000.npy')
r_input = np.load('./input_files/r_input_2mm_5000.npy')
t_input = np.load('./input_files/t_input_2mm_5000.npy')

if __name__ == '__main__':
    results = MC.SolveMC_parallel_chirp_InputAll(v_input, r_input, t_input)
    with open(filename, 'a', newline='') as file:  # 'a' for append
        writer = csv.writer(file)
        writer.writerows([result for result in results])







