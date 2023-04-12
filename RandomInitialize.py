import numpy as np
import scipy.special
import mpmath
from numpy import linalg as LA

'''
This file contains helper functions to sample atoms' velocities and positions from known distributions
'''

# Probability distribution functions
def MB_dist(v, Tc=450):
    # PDF of Maxwell-Boltzmann distribution
    T = Tc + 273.15
    m = 174 * 1.67e-27
    k = 1.38e-23
    bins = v
    return (4 * np.pi * bins**2) * (m / (2 * np.pi * k * T))**(3 / 2)\
        * np.exp(- m * bins**2 / (2 * k * T))


def eff_dist(v, Tc=450):
    # PDF of effusion distribution (velocity distribution for atoms leaving the oven)
    T = Tc + 273.15
    m = 174 * 1.67e-27
    k = 1.38e-23
    bins = v
    return (m**2 * bins**3 / (2 * k**2 * T**2)) * \
        np.exp(-m * bins**2 / (2 * k * T))


def gauss_dist(bins, mu=0, sigma=15):
    # PDF of 1D gaussian distribution
    return (1 / np.sqrt(2 * np.pi * sigma**2)) * \
        np.exp(-(bins**2) / (2 * sigma**2))


# Transverse velocity profile for a collimated atomic beam (non-Gaussian)
def transv_dist(vs, Tc=450, d=8, reg=True):
    T = Tc + 273.15
    m = 174 * 1.67e-27
    k_B = 1.38e-23
    a = 0.8
    # import scipy.special
    # G_half = scipy.special.gamma(-1/2)
    # G = scipy.special.gammaincc(1/2, x) + scipy.special.gammainc()
    sigma = np.sqrt(k_B * T / m)
    pdfs = []

    def pdf(vs, a, d):
        return np.real(np.abs(vs) * np.exp(-(vs**2) / (2 * sigma**2)) *
                       scipy.special.gamma(- 1 / 2) * G /
                       (4 * np.sqrt(np.pi * sigma**2 *
                                    np.sqrt(1 + (a / d)**2))))

    if np.size(vs) == 1:
        x = (1 / 2) * (vs * d / (sigma * a))**2
        G = mpmath.gammainc(- 1 / 2, x, regularized=reg)
        pdf_final = np.array(pdf(vs, a, d)) / 0.921

    if np.size(vs) > 1:
        for v in vs:
            x = (1 / 2) * (v * d / (sigma * a))**2
            G = mpmath.gammainc(-1 / 2, x, regularized=reg)
            pdfs.append(pdf(v, a, d))
        pdf_final = np.array(pdfs) / 0.921

    return pdf_final

'''
The following functions sample atoms' velocities from given distributions
'''
def rand_velocity_MB(Tc, num=1):
    # Input: T in celcius, output: random 3D velocity vectors
    # under Maxwell-Boltzmann distribution (vx, vy, vz are Gaussian)
    k = 1.38e-23
    m = 174 * 1.67e-27
    mu = 0
    sigma = np.sqrt(k * (Tc + 273.15) / m)
    v = np.random.normal(mu, sigma, size=(num, 3)) # sample vx,vy,vz independently from 3 Gaussian distributions
    return v


def rand_speed_MB(Tc, num=1):
    # Sample speeds from Maxwell-Boltzmann distribution  (PDF = v^2 * Gaussian)
    vi = rand_velocity_MB(Tc, num=num)
    v = np.sqrt(vi[:, 0]**2 + vi[:, 1]**2 + vi[:, 2]**2)
    return v


def rand_speed_eff(Tc, num=1, c=3):
    # Sample speeds from effusion velocity distribution (PDF = v^3 * Gaussian)
    # using acceptance-rejection sampling
    count = 0
    v_out = np.zeros(num)
    while count < num:
        v = rand_speed_MB(Tc, num=1)
        u = np.random.rand()

        if c * MB_dist(v) * u < eff_dist(v):
            v_out[count] = v
            count += 1
    return v_out


def rand_speed_transv(Tc, num=1, c=1.7):
    # Sample transverse speeds (non-Gaussian transverse distribution)
    count = 0
    v_out = np.zeros(num)
    mu = 0
    sigma = 15

    while count < num:
        v = np.random.normal(mu, sigma)
        # sampled v from Gaussian distribution (mu=0, sigma=40)
        u = np.random.rand()

        if c * gauss_dist(v) * u < transv_dist(v, Tc=Tc):
            v_out[count] = np.abs(v)
            count += 1
    return v_out


def rand_transv_gauss(Tc, num=1):
    # Sample transverse velocity (assuming Gaussian transverse distribution)
    k = 1.38e-23
    T = Tc + 273.15
    m = 174 * 1.67e-27
    sigma = np.sqrt(k * T / (2 * m)) * 2 * 0.017 # width = width of longitudinal vz (effusion) * 2 * half-angle of divergence
    vx = np.random.normal(0, sigma, num)
    return vx


# Generate n random emission directions (d dimensions, d=3 by default): output = num x dim array
def rand_direction(d=3, num=1):
    dp = np.random.normal(0, 1, size=(num, d))
    norm = LA.norm(dp, axis=1)
    norm_tile = np.tile(np.array([norm]).transpose(), (1, d))
    dp_norm = dp / norm_tile
    return dp_norm


def rand_velocity_1D(Tc, num=1):
    # Return num x 1 array of [vz] for num atoms
    count = 0
    v_trans_store = []
    v_z_store = []

    while count < num:
        v_trans = rand_speed_transv(Tc)
        v = rand_speed_eff(Tc)
        v_z = np.sqrt(v**2 - v_trans**2)

        if np.abs(v_trans / v_z) <= 17e-3:
            # Only accept trial if v_trans / v_z <= 17mrad
            v_trans_store.append(v_trans)
            v_z_store.append(v_z)
            count += 1
    return np.array(v_z_store).flatten()


def rand_velocity_1D_gauss(Tc, num=1):
    v_trans = rand_transv_gauss(Tc, num=num)
    v = rand_speed_eff(Tc, num=num)
    # Return num x 1 array of longitudinal velocity [vz] for num atoms
    return np.sqrt(v**2 - v_trans**2)


def rand_velocity_3D(Tc, num=1):
    # Return num x 3 array of [vx, vy, vz] for num atoms
    count = 0
    v_trans_store = []
    v_z_store = []
    v_out = []

    while count < num:
        v_trans = rand_speed_transv(Tc)[0]
        D = rand_direction(d=2).flatten()
        v_x_ = v_trans * D[0]
        v_y_ = v_trans * D[1]
        v = rand_speed_eff(Tc)[0]
        v_z = np.sqrt(np.max([0.01, v**2 - v_trans**2]))

        if np.abs(v_trans / v_z) <= 17e-3:
            # Only accept trial if v_trans / v_z <= 17mrad
            v_trans_store.append(v_trans)
            v_z_store.append(v_z)
            v_out.append([v_x_, v_y_, v_z])
            count += 1
    return v_out


def rand_velocity_3D_gauss(Tc, num=1):
    # Return num x 3 array of [vx, vy, vz] for num atoms
    v_trans = rand_speed_transv_gauss(Tc, num=num)
    v = rand_speed_eff(Tc, num=num)
    D = rand_direction(d=2, num=num)

    v_z = np.sqrt(v**2 - v_trans**2)
    v_x = v_trans * D[:, 0]
    v_y = v_trans * D[:, 1]

    v_3D = np.array([v_x, v_y, v_z]).T
    return v_3D


def rand_position(num=1, rad=0.004, starting_z=0):
    # oven radius = 4 mm
    u = np.random.uniform(0, 2 * np.pi, num)
    r = rad * np.sqrt(np.random.rand(num))
    x = r * np.cos(u)
    y = r * np.sin(u)
    z = np.ones_like(x) * starting_z
    pos = np.transpose([x, y, z])
    return pos


def rand_position_length(num=1, rad=0.004, L=0.33, loading_time=3e-3):
    # oven radius = 4 mm
    # slower length = 33 cm
    u = np.random.uniform(0, 2 * np.pi, num)
    r = rad * np.sqrt(np.random.rand(num))
    x = r * np.cos(u)
    y = r * np.sin(u)
    z = np.random.uniform(-loading_time * 1000 * L, L, num)
    pos = np.transpose([x, y, z])
    return pos


def uniform_velocity_3D(Tc, vmin=260, vmax=340, num=1):
    # Return num x 3 array of [vx, vy, vz] for num atoms
    count = 0
    v_trans_store = []
    v_z_store = []
    v_out = []

    while count < num:
        v_trans = rand_speed_transv(Tc)[0]
        D = rand_direction(d=2).flatten()
        v_x_ = v_trans * D[0]
        v_y_ = v_trans * D[1]
        v = np.random.uniform(low=vmin, high=vmax)
        v_z = np.sqrt(np.max([0.01, v**2 - v_trans**2]))

        if np.abs(v_trans / v_z) <= 17e-3:
            # Only accept trial if v_trans / v_z <= 17mrad
            v_trans_store.append(v_trans)
            v_z_store.append(v_z)
            v_out.append([v_x_, v_y_, v_z])
            count += 1

    return v_out


def generate_allinput(flux=1000, oven_radius=0.002):
    N_new = flux * 3  # new atoms coming out for the first 3 ms
    # Old atoms: left the oven anytime between -10 ms to 0.

    N_old_out = flux * 10
    v_out = np.array(rand_velocity_3D(450, num=N_old_out))
    t_flight = 1e-3 * np.random.uniform(low=0.0, high=10.0, size=N_old_out)  # time of flight is between 0 to 10 ms

    v = np.zeros_like(v_out)
    v[:, 0] = v_out[:, 0]  
    v[:, 1] = v_out[:, 1] - 9.81 * t_flight  # v_y changes due to gravity
    v[:, 2] = v_out[:, 2]

    r = rand_position(num=N_old_out, rad=oven_radius)
    r[:, 0] += v_out[:, 0] * t_flight
    r[:, 1] += v_out[:, 1] * t_flight - 0.5 * 9.81 * t_flight**2
    r[:, 2] += v_out[:, 2] * t_flight

    v_old = v[r[:, 2] <= 0.33]
    r_old = r[r[:, 2] <= 0.33]
    t_old = np.zeros_like(v_old[:, 2])

    v_new = np.array(rand_velocity_3D(450, num=N_new))
    r_new = rand_position(num=N_new, rad=oven_radius)  # start at z = 0
    t_new = 1e-3 * np.random.uniform(low=0.0, high=3.0, size=N_new)

    # combine old and new atoms and save to files
    v_final = np.concatenate((v_old, v_new), axis=0)
    r_final = np.concatenate((r_old, r_new), axis=0)
    t_final = np.concatenate((t_old, t_new), axis=None)
    
    return v_final, r_final, t_final
