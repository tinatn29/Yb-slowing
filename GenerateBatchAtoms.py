import numpy as np
import RandomInitialize as RI


oven_radius = 0.002
flux = 5000  # no. of atoms coming out per ms --> FLUX (/s) = flux * 1000
N_new = flux * 3  # New atoms coming out between 0 to 3 ms

# Old atoms: left the oven anytime between -10 ms to 0.
N_old_out = flux * 10
v_out = np.array(RI.rand_velocity_3D(450, num=N_old_out))
t_flight = 1e-3 * np.random.uniform(low=0.0, high=10.0, size=N_old_out)  # time of flight is between 0 to 10 ms

v = np.zeros_like(v_out)
v[:, 0] = v_out[:, 0]  
v[:, 1] = v_out[:, 1] - 9.81 * t_flight  # v_y changes due to gravity
v[:, 2] = v_out[:, 2]

r = RI.rand_position(num=N_old_out, rad=oven_radius)
r[:, 0] += v_out[:, 0] * t_flight
r[:, 1] += v_out[:, 1] * t_flight - 0.5 * 9.81 * t_flight**2
r[:, 2] += v_out[:, 2] * t_flight

v_old = v[r[:, 2] <= 0.33]
r_old = r[r[:, 2] <= 0.33]
t_old = np.zeros_like(v_old[:, 2])

v_new = np.array(RI.rand_velocity_3D(450, num=N_new))
r_new = RI.rand_position(num=N_new, rad=oven_radius)  # start at z = 0
t_new = 1e-3 * np.random.uniform(low=0.0, high=3.0, size=N_new)

# combine old and new atoms and save to files
v_final = np.concatenate((v_old, v_new), axis=0)
np.save(f'v_input_{oven_radius * 1000:.0f}mm_{flux}.npy', v_final)

r_final = np.concatenate((r_old, r_new), axis=0)
np.save(f'r_input_{oven_radius * 1000:.0f}mm_{flux}.npy', r_final)

t_final = np.concatenate((t_old, t_new), axis=None)
np.save(f't_input_{oven_radius * 1000:.0f}mm_{flux}.npy', t_final)

print(np.size(t_final))