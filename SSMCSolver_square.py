import numpy as np
from qutip import *
from scipy import signal
import RandomInitialize as RI
import time
import copy


# Compute a truncated Fourier series of square waves (containing num_terms)
def sum_harmonic_wave(t, args, num_terms, phi=0):
    def harmonic_wave(t, args, h=1, phi=0):
        return np.sin(h * args['Delta'] * t + phi)
    wave = 0
    for i in range(num_terms):
        h = 2 * i + 1
        wave += (4 / (h * np.pi)) * harmonic_wave(t, args, h=h, phi=h * phi)
    return wave


'''
Envelope function for Rabi frequency modulation
Modulation shape set by args['Rabi_shape']
Rabi freq modulation with (0) square waves OR (1+) Fourier series with one or more harmonics
No amplitude modulation: args['Rabi_shape'] = -1
'''
def Rabi_freq(t, args, phi=0):
    # -1: (constant Rabi freq - no amplitude modulation)
    if args['Rabi_shape'] == -1:
        return np.ones_like(t)

    # 0: near-ideal square wave AM
    if args['Rabi_shape'] == 0:
        y = signal.square(args['Delta'] * t + phi, duty=args['duty'])
    
    # 1: cosine AM
    if args['Rabi_shape'] == 1:
        y = np.sin(args['Delta'] * t + phi)

    # > 1: truncated Fourier series of square waves with no. of terms = args['Rabi_shape']
    # e.g. args['Rabi_shape'] = 2 means the Fourier series will contain the first harmonic and third harmonic 
    if args['Rabi_shape'] >1:
        y = sum_harmonic_wave(t, args, args['Rabi_shape'], phi=phi)

    # if non-neg = True, the envelope stays non-negative [0, 1], otherwise it's [-1, 1]
    if args['non_neg']:
        return 0.5 * (y + 1)
    else: 
        return y


'''
Phase modulation function
Matching case envelope (+/-) using constant amplitude (No AM)
'''
def phase_mod_match(t, args):
   # 0: return square wave PM
    if args['PM_shape'] == 0:
        phase_mod_right = np.pi * 0.5 * (-signal.square(args['Delta'] * t + args['phi_right'], duty=0.5) + 1)
        phase_mod_left = -np.pi * 0.5 * (-signal.square(args['Delta'] * t + args['phi_left'], duty=0.5) + 1)
    
    # >= 1: return Fourier series of square waves with no. of terms = args['Rabi_shape']
    if args['PM_shape'] == 1:
        phase_mod_right = np.pi * 0.5 * (-np.sin(args['Delta'] * t + args['phi_right']) + 1)
        phase_mod_left = -np.pi * 0.5 * (-np.sin(args['Delta'] * t + args['phi_left']) + 1)

    if args['PM_shape'] > 1:
        phase_mod_right = np.pi * 0.5 * (-sum_harmonic_wave(t, args, 1, phi=args['phi_right']) + 1)
        phase_mod_left = -np.pi * 0.5 * (-sum_harmonic_wave(t, args, 1, phi=args['phi_left']) + 1)
    
    return phase_mod_right, phase_mod_left

'''
Time-dependent phase of traveling green waves (with RWA) > to be multiplied with the Rabi frequency
'''
def wave_args(t, args, mod=True):
    return -args['detuning'] * t + args['k'] * args['z'] + args['k'] * args['v'] * t


# Compute the off-diagonal elements of the 2x2 Hamiltonian (H[1,2] and its conjugate H[2,1])
def H_12(t, args):
    T = t + args['t_elapsed']
    PM_right, PM_left = phase_mod_match(T, args)

    # Square wave Rabi freq
    # Right: red-detuned, counter-propagating: exp(i(wt - delta t + kz + kvt))
    wave1 = Rabi_freq(T, args, phi=args['phi_right']) * np.exp(1j * wave_args(T, args)) * \
    np.exp(1j * args['phasemod_right'] * PM_right)

    # Left: co-propagating: exp(i(wt + delta t - kz - kvt))
    wave2 = Rabi_freq(T, args, phi=args['phi_left']) * np.exp(-1j * wave_args(T, args)) * \
    np.exp(1j * args['phasemod_left'] * PM_left)

    return (wave1 + wave2) * args['Omega']

def H_21(t, args):
    return np.conj(H_12(t, args))

# Compute the z-derivative of H12 -> to calculate force
def H_12_deriv_z(t, args):
    T = t + args['t_elapsed']
    PM_right, PM_left = phase_mod_match(T, args)

    wave1_deriv = 1j * args['k'] * Rabi_freq(T, args, phi=args['phi_right']) * np.exp(1j * wave_args(T, args)) * \
    np.exp(1j * args['phasemod_right'] * PM_right)

    wave2_deriv = -1j * args['k'] * Rabi_freq(T, args, phi=args['phi_left']) * np.exp(-1j * wave_args(T, args)) * \
    np.exp(1j * args['phasemod_left'] * PM_left)

    return (wave1_deriv + wave2_deriv) * args['Omega']


'''
Functions copied from SSMCSolver_BCF.py
'''
# Laser chirping
def chirped_detuning(T, parameters):
    period, v_max, v_min = parameters
    f = 1 / period  # default period = 0.8 ms
    k = 2 * np.pi / 556e-9  # wavenumber
    # chirp laser so it's detuned between 300 m/s and 60 m/s
    return k * (-0.5 * (v_max - v_min) * signal.sawtooth(2 * np.pi * f * T) + 0.5 * (v_min + v_max))

def merge_two_arrays(a1, a2):
    out = [x for x in a1] + [y for y in a2]
    return out


class MCSolver():
    def __init__(self, delta=50, omega=60, Rabi_shape=0, phi=0.36*np.pi, detuning=0, PM_shape=0, beam_radius=0.004, \
        gravity=True, loading_time=1e-3, Gaussian_beam=True):
        self.args = {
                    'gamma': 2 * np.pi * 182e3, # spontaneous decay rate / natural linewidth
                    'Delta': delta * 2 * np.pi * 182e3, # modulation frequency
                    'Omega': omega * 2 * np.pi * 182e3, # Rabi frequency
                    'detuning': detuning,
                    'k': 2 * np.pi / 556e-9, # wavenumber
                    'k_Rabi' : delta * 2 * np.pi * 182e3 / 3e8,  # wave number for Rabi freq modulation
                    'Rabi_shape' : Rabi_shape, # Set the amplitude modulation (see Rabi_freq function above)
                    'v': 0, # atom's velocity vz
                    'z': 0, # atom's position z
                    'phi_left': phi, # phase of laser beams moving towards the atom (0.36pi by default)
                    'phi_right': 0, # phase of laser beams moving in the same direction as the atom (zero by default) 
                    'phasemod_left': 0, # 0: no phase mod, 1,-1: with phase mod
                    'phasemod_right': 0, # 0: no phase mod, 1,-1: with phase mod
                    'duty': 0.5, # square wave duty cycle (default = 50%)
                    't_elapsed': 0, # time elapsed
                    'hbar' : 6.63e-34 / (2 * np.pi), # hbar = h / 2pi
                    'mass' : 174 * 1.67e-27, # mass of Yb-174 atom
                    'tau' : 1 / 2 * np.pi * 182e3, # atom's lifetime = 1/gamma
                    'F_rad': (6.63e-34 / 556e-9) * np.pi * 182e3, # max radiative force h_bar * k * gamma / 2
                    'PM_shape': PM_shape, # shape of phase modulation (see phase_mod_match function above)
                    'non_neg' : False # False= [-1,1] AM envelope, True= [0,1] AM envelope
        }
        # Atom's properties
        self.v_3D = np.array([0, 0, 0])  # atom's velocity 3D
        self.r_3D = np.array([0, 0, 0])  # Random starting position
        self.chirp = None
        self.t_elapsed = 0
        # Quantities for qutip mesolve
        self.H0 = Qobj([[0, 0], [0, 0]])
        self.H = [self.H0, [Qobj([[0, 1], [0, 0]]), H_12],
            [Qobj([[0, 0], [1, 0]]), H_21]]
        self.rho0 = ket2dm(basis(2, 0))  # Start in ground state
        self.c_op = np.sqrt(self.args['gamma']) * Qobj([[0, 1], [0, 0]])  # Collapse operator
        self.tlist = np.linspace(0, 10e-6, 100)  # tlist for each cycle (10 us)
        self.force = None  # Force as a function of time
        self.p_ex = None  # Excitation population
        self.p_ex_mean = None  # Average excitation prob
        self.coherence = None # Coherence
        self.loading_time = loading_time  # loading time default = 1 ms
        # Gaussian beam properties
        self.Gaussian_beam = Gaussian_beam
        self.Omega_0 = self.args['Omega']  # maximum Rabi frequency (set)
        self.beam_radius = beam_radius  # beam radius
        self.gravity = gravity  # gravity taken into account?

    def SetChirping(self, period, v_max, v_min):
        self.chirp = [period, v_max, v_min]

    # Gaussian beam profile
    def Omega_Gaussian(self):
        r = np.sqrt(self.r_3D[0]**2 + self.r_3D[1]**2)
        return self.Omega_0 * np.exp(- r**2 / self.beam_radius**2)

    def UpdateArgs(self):
        if self.chirp is not None:
            self.args['detuning'] = chirped_detuning(self.t_elapsed, self.chirp)

        # Update arguments from atom's variables
        self.args['v'] = self.v_3D[2]
        self.args['z'] = self.r_3D[2]
        self.args['t_elapsed'] = self.t_elapsed

        if self.Gaussian_beam:
            self.args['Omega'] = self.Omega_Gaussian()


    def ResetArgs_static(self):
        self.args['t_elapsed'] = 0
        self.rho0 = ket2dm(basis(2, 0))  # Start in ground state


    def numerical_deriv_z(self, bound_size=1e-5, zlist_length=11):
        z0 = copy.copy(self.args['z'])
        z_bound = bound_size * np.pi / self.args['k_Rabi']
        zlist = np.linspace(z0 - z_bound, z0 + z_bound, zlist_length)
        self.args['z'] = zlist
        deriv = np.zeros(np.shape(self.tlist), dtype=complex)

        for i in range(len(self.tlist)):
            t = self.args['t_elapsed'] + self.tlist[i]
            deriv_list = np.diff(H_12(t, self.args)) / np.diff(zlist)
            ind = int((zlist_length - 1) / 2)
            deriv[i] = np.mean([deriv_list[ind - 1], deriv_list[ind]])
    
        self.args['z'] = z0
        return deriv

    def SolveStimulatedForce_num(self):
        '''
        (NUMERICAL) For 10 us, solve for stimulated force(t) and average excitation prob
        '''
        # Update args before solving
        self.UpdateArgs()

        # Solve Master equation using current atom's parameters
        options = Odeoptions(nsteps=4000)
        output = mesolve(self.H, self.rho0, self.tlist, c_ops=[self.c_op], args=self.args,
            options=options)
        # Calculate force(t)
        self.force = np.zeros_like(self.tlist)
        self.coherence = np.zeros(np.shape(self.tlist), dtype=complex)

        # Calculate partial derivative dH_12/dz
        deriv = self.numerical_deriv_z(bound_size=1e-14, zlist_length=11)

        for i in range(len(self.tlist)):
            self.force[i] = (4 / (self.args['k'] * self.args['gamma'])) * np.real(output.states[i][1, 0] * deriv[i])
            self.coherence[i] = np.imag(output.states[i][1, 0])

        self.rho0 = output.states[-1]  # Update states for next cycle
        # Calculate average excitation probability
        self.p_ex = expect(num(2), output.states)
        self.p_ex_mean = np.mean(self.p_ex)


    def SolveStimulatedForce(self):
        '''
        (EXACT) For 10 us, solve for stimulated force(t) and average excitation prob
        '''
        # Update args before solving
        self.UpdateArgs()

        # Solve Master equation using current atom's parameters
        options = Odeoptions(nsteps=4000)
        output = mesolve(self.H, self.rho0, self.tlist, c_ops=[self.c_op], args=self.args,
            options=options)
        # Calculate force(t)
        self.force = np.zeros_like(self.tlist)
        self.coherence = np.zeros(np.shape(self.tlist), dtype=complex)

        # Calculate partial derivative dH_12/dz
        deriv = H_12_deriv_z(self.tlist, self.args)

        for i in range(len(self.tlist)):
            self.force[i] = (4 / (self.args['k'] * self.args['gamma'])) * np.real(output.states[i][1, 0] * deriv[i])
            self.coherence[i] = np.imag(output.states[i][1, 0])

        self.rho0 = output.states[-1]  # Update states for next cycle
        # Calculate average excitation probability
        self.p_ex = expect(num(2), output.states)
        self.p_ex_mean = np.mean(self.p_ex)


    def PlotResults_static(self, tmin=0, tmax=2):
        self.ResetArgs_static()

        plt.figure(figsize=(6, 8))
        ax = plt.subplot(5, 1, 1)
        plt.plot(self.tlist * 1e6, Rabi_freq(self.tlist, self.args, phi=self.args['phi_right']) * self.args['Omega'] / self.args['gamma'],
            color='C3')
        plt.plot(self.tlist * 1e6, Rabi_freq(self.tlist, self.args, phi=self.args['phi_left']) * self.args['Omega'] / self.args['gamma'],
            color='C0')
        ax.set_xlim([tmin, tmax])
        ax.axes.xaxis.set_ticklabels([])
        # ax.set_xlabel('t ($\mu s$)', fontsize=12)
        ax.set_ylabel('$\Omega / \gamma$', fontsize=12)
        plt.title('$\Delta = {1:.0f}\gamma$, $\Omega = {0:.0f}\gamma$'.format(
            self.args['Omega'] / self.args['gamma'], self.args['Delta'] / self.args['gamma']), fontsize=14)

        ax = plt.subplot(5, 1, 2)
        PM_right, PM_left = phase_mod_match(self.tlist, self.args)
        plt.plot(self.tlist * 1e6, self.args['phasemod_right'] * PM_right, color='C3')
        plt.plot(self.tlist * 1e6, self.args['phasemod_left'] * PM_left, color='C0')

        ax.set_xlim([tmin, tmax])
        ax.axes.xaxis.set_ticklabels([])
        ax.set_ylabel(r'$\delta \theta(t)$', fontsize=12)

        ax = plt.subplot(5, 1, 3)
        plt.plot(self.tlist * 1e6, self.force)
        plt.plot(self.tlist * 1e6, np.mean(self.force) * np.ones_like(self.tlist), 'k--')
        ax.set_ylabel('Force ($\hbar k / 2\gamma$)', fontsize=12)
        ax.set_xlim([tmin, tmax])
        ax.axes.xaxis.set_ticklabels([])

        ax = plt.subplot(5, 1, 4)
        plt.plot(self.tlist * 1e6, self.p_ex)
        ax.set_xlim([tmin, tmax])
        ax.set_ylim([0, 1])
        ax.set_ylabel('Population', fontsize=12)
        ax.axes.xaxis.set_ticklabels([])

        ax = plt.subplot(5, 1, 5)
        plt.plot(self.tlist * 1e6, self.coherence)
        ax.set_xlim([tmin, tmax])
        ax.set_ylabel(r'Coherence $\rho_{01}$', fontsize=12)
        ax.set_xlabel('t ($\mu s$)', fontsize=12)


    def IntegrateForce_1D(self):
        '''
        Integrate F(t) for v(t) and z(t)
        Ignore any transverse heating for now
        '''
        v_array = [self.v_3D[2]]

        for ind in range(len(self.tlist) - 1):
            dv = np.trapz(self.force[:(ind + 1)], self.tlist[:(ind + 1)]) * self.args['F_rad'] / self.args['mass']
            v_array.append(self.v_3D[2] + dv)

        # Update atom's position, velocity, and time_elapsed
        dt = self.tlist[-1]
        # x += v_x * t
        self.r_3D[0] += self.v_3D[0] * dt

        g = 9.81  # acceleration due to gravity
        if self.gravity:
            # update y and v_y if there's gravity
            self.r_3D[1] += self.v_3D[1] * dt - 0.5 * g * dt**2
            self.v_3D[1] -= g * dt
        else:
            # without gravity y += v_y * t and v_y stays the same
            self.r_3D[1] += self.v_3D[1] * dt

        # Update z, v_z and move the clock forward
        self.r_3D[2] += np.trapz(v_array, self.tlist)
        self.v_3D[2] = v_array[-1]
        self.t_elapsed += dt


    def MoveAtom_NoForce(self, dt=10e-6):
        '''
        In case atom's outside the beam: atom moves without being slowed
        Update position and time but not velocity
        '''
        dr = np.array(self.v_3D) * dt
        self.r_3D = np.add(self.r_3D, dr)

        g = 9.81  # acceleration due to gravity

        if self.gravity:
            # update y and v_y if there's gravity
            self.r_3D[1] -= 0.5 * g * dt**2
            self.v_3D[1] -= g * dt

        self.t_elapsed += dt


    '''
    Code blocks for checking atom's status
    '''

    def Check_If_Passed(self):
        # Return true if atom passed the MOT region
        return self.r_3D[2] > 0.33 + 0.01

    def Check_If_Trapped(self):
        # Return true if the atom would be trapped (slow enough and close enough to the MOT)
        r_atom = np.sqrt(self.r_3D[0]**2 + self.r_3D[1]**2 + (self.r_3D[2] - 0.33)**2)
        Near_MOT = r_atom <= 0.01  # atom is within 1 cm radius MOT
        Slow = np.sqrt(self.v_3D[0]**2 + self.v_3D[1]**2 + self.v_3D[2]**2) <= 5  # with a speed below 5 m/s
        return Near_MOT and Slow

    def Check_If_InsideBeam(self):
        if self.Gaussian_beam:  # Gaussian beam: only check if z >=0 
            return self.r_3D[2] >= 0
        else:  # flat-top beam: check if z >= 0 and atom is within the beam size
            return self.r_3D[2] >= 0 and self.Check_If_InsideBeam_FlatTop()

    def Check_If_InsideVRange(self):
        # Return true if atom's velocity is inside v_c +/- 1.5 * Delta / k
        v_detuning = self.args['detuning'] / self.args['k']
        v_range = 1.5 * self.args['Delta'] / self.args['k']
        return np.abs(self.v_3D[2] - v_detuning) <= v_range

    def Check_If_InsideBeam_FlatTop(self):
        # Return true if z >= 0 and atom's inside the beam radius
        # Only use for flat-top laser intensity profile and not with the default Gaussian beam profile setting
        return np.sqrt(self.r_3D[0]**2 + self.r_3D[1]**2) <= self.beam_radius

    def Check_If_InTime(self):
        return self.t_elapsed < self.loading_time

    
    '''
    MC SIMULATION
    ==== TEST FIXED DETUNING ====
    Run MC at a fixed detuning for a specified loading time
    '''
    def SolveMC_single_fixed_test(self, v_initial, r_initial, t_initial):
        self.v_3D = copy.copy(v_initial)  # Initial velocity from argument
        self.r_3D = copy.copy(r_initial)  # Initial position
        self.t_elapsed = t_initial   # Reset the clock
        self.rho0 = ket2dm(basis(2, 0))  # Start in ground state

        self.UpdateArgs()
        v_max = self.args['detuning'] / self.args['k']

        start_time = time.time()

        # all atoms outside range
        if self.Check_If_InsideVRange() == 0:
            self.MoveAtom_NoForce(dt=self.loading_time - self.t_elapsed)
            outputs = np.concatenate((v_initial, r_initial, t_initial, self.v_3D, self.r_3D, self.loading_time), axis=None)
            return np.concatenate((outputs, 'SKIPPED', time.time() - start_time), axis=None)
        
        while self.Check_If_InTime():
            # For atoms in z >= 0 (laser present) and |v_atom - v_detuning| <= 1.5 Delta/k
            # Calculate force and integrate
            if self.Check_If_InsideBeam() and self.Check_If_InsideVRange():
                self.SolveStimulatedForce()
                self.IntegrateForce_1D()
                self.UpdateArgs()
            # without force
            else:
                self.MoveAtom_NoForce(dt=self.loading_time - self.t_elapsed)
                outputs = np.concatenate((v_initial, r_initial, t_initial, self.v_3D, self.r_3D, self.t_elapsed), axis=None)
                return np.concatenate((outputs, 'DONE', time.time() - start_time), axis=None)

        outputs = np.concatenate((v_initial, r_initial, t_initial, self.v_3D, self.r_3D, self.t_elapsed), axis=None)
        return np.concatenate((outputs, 'SLOWED', time.time() - start_time), axis=None)


    def SolveMC_parallel_fixed_test(self, v_input, z_input, t_input, max_cores=32):
        '''
        Solve MC for N atoms using multiprocessing
        '''
        import multiprocessing as mp

        start_time = time.time()
        n_cores = np.amin([mp.cpu_count(), max_cores])
        pool = mp.Pool(processes=n_cores)

        args = list(zip(v_input, z_input, t_input))
        # merge the inputs into args = [[[v1], [z1]], [[v2], [z2]], ...]

        # then use pool.starmap instead of map
        outputs = pool.starmap(self.SolveMC_single_fixed_test, args)  # list of function outputs
        print('Time elapsed: {0:.2f} sec'.format(time.time() - start_time))
        return outputs

    '''
    MC SIMULATION: single atom or parallel
    ==== TEST chirp ====
    Run MC with chirp until specified loading_time is up 
    '''
    def SolveMC_single_chirp_test(self, v_initial, r_initial, t_initial):
        self.v_3D = copy.copy(v_initial)  # Initial velocity from argument
        self.r_3D = copy.copy(r_initial)  # Initial position
        self.t_elapsed = t_initial   # Reset the clock
        self.rho0 = ket2dm(basis(2, 0))  # Start in ground state

        self.UpdateArgs()
        v_max = self.chirp[1]  # v_max = upper chirp limit
        
        start_time = time.time()

        # skip fast atoms
        if v_initial[2] > v_max + 1.5 * self.args['Delta'] / self.args['k']:
            self.MoveAtom_NoForce(dt=self.loading_time - self.t_elapsed)
            outputs = np.concatenate((v_initial, r_initial, t_initial, self.v_3D, self.r_3D, self.loading_time), axis=None)
            return np.concatenate((outputs, 'FAST', time.time() - start_time), axis=None)

        while self.Check_If_InTime():
            # For atoms in z >= 0 (laser present) and |v_atom - v_detuning| <= 1.5 Delta/k
            # Calculate force and integrate
            if self.Check_If_InsideBeam() and self.Check_If_InsideVRange():
                self.SolveStimulatedForce()
                self.IntegrateForce_1D()
                self.UpdateArgs()
            # without force
            else:
                self.MoveAtom_NoForce()
                self.UpdateArgs()

        outputs = np.concatenate((v_initial, r_initial, t_initial, self.v_3D, self.r_3D, self.t_elapsed), axis=None)
        return np.concatenate((outputs, 'PASSED', time.time() - start_time), axis=None)


    def SolveMC_parallel_chirp_test(self, v_input, z_input, t_input, max_cores=32):
        '''
        Solve MC for N atoms using multiprocessing
        '''
        import multiprocessing as mp

        start_time = time.time()
        n_cores = np.amin([mp.cpu_count(), max_cores])
        pool = mp.Pool(processes=n_cores)

        args = list(zip(v_input, z_input, t_input))
        # merge the inputs into args = [[[v1], [z1]], [[v2], [z2]], ...]

        # then use pool.starmap instead of map
        outputs = pool.starmap(self.SolveMC_single_chirp_test, args)  # list of function outputs
        print('Time elapsed: {0:.2f} sec'.format(time.time() - start_time))
        return outputs

    '''
    MC SIMULATION: single atom or parallel
    ==== Gaussian beam or flat-top beam with chirp ====
    Gaussian beam; function takes 3 arguments: starting v, starting position, starting time
    '''
    def SolveMC_single_chirp_InputAll(self, v_initial, r_initial, t_initial):
        '''
        Solve MC for a single atom (v0, r0, t0)
        '''

        self.v_3D = copy.copy(v_initial)  # Initial velocity from argument
        self.r_3D = copy.copy(r_initial)  # Initial position
        self.t_elapsed = t_initial   # Reset the clock
        self.rho0 = ket2dm(basis(2, 0))  # Start in ground state

        self.UpdateArgs()
        v_max = self.chirp[1]  # v_max = upper chirp limit
        
        # v_atom > v_max + Delta/k
        # Atoms won't be affected at all -- not trapped
        # Skip all atoms with v_z > v_max (upper chirp limit) + Delta / k (outside force's velocity range)
        start_time = time.time()

        # Skip atoms faster than v_max + 1.5 * Delta / k
        if v_initial[2] > v_max + 1.5 * self.args['Delta'] / self.args['k']:
            dt = (0.33 + 0.01 - r_initial[2]) / self.v_3D[2]  # dt = how long it takes for atom to move passed the MOT region
            self.MoveAtom_NoForce(dt=dt)
            outputs = np.concatenate((v_initial, r_initial, t_initial, self.v_3D, self.r_3D, t_initial + dt), axis=None)
            return np.concatenate((outputs, 'PASSED', time.time() - start_time), axis=None)

        # Add this check to make sure we don't miss potential atoms that can be trapped
        if self.Check_If_Trapped():
            outputs = np.concatenate((v_initial, r_initial, t_initial, self.v_3D, self.r_3D, self.t_elapsed), axis=None)
            return np.concatenate((outputs, 'TRAPPED', time.time() - start_time), axis=None)

        # Run while z < 33 + 1 cm
        while self.r_3D[2] < 0.33 + 0.01:
            # while the atom is still active, want to check if it is trapped
            # if trapped, exit the simulation and flag the result as "TRAPPED"
            if self.Check_If_Trapped():
                outputs = np.concatenate((v_initial, r_initial, t_initial, self.v_3D, self.r_3D, self.t_elapsed), axis=None)
                return np.concatenate((outputs, 'TRAPPED', time.time() - start_time), axis=None)

            # if it is not trapped and already starts moving backwards, flag as "FAILED"
            if self.v_3D[2] < -0.05:
                outputs = np.concatenate((v_initial, r_initial, t_initial, self.v_3D, self.r_3D, self.t_elapsed), axis=None)
                return np.concatenate((outputs, 'FAILED', time.time() - start_time), axis=None)

            # For atoms in z >= 0 (laser present) and |v_atom - v_detuning| <= 1.5 Delta/k
            # Calculate force and integrate
            if self.Check_If_InsideBeam() and self.Check_If_InsideVRange():
                self.SolveStimulatedForce()
                self.IntegrateForce_1D()
                self.UpdateArgs()
            # without force
            else:
                self.MoveAtom_NoForce()
                self.UpdateArgs()

        # All others
        outputs = np.concatenate((v_initial, r_initial, t_initial, self.v_3D, self.r_3D, self.t_elapsed), axis=None)
        return np.concatenate((outputs, 'PASSED', time.time() - start_time), axis=None)
        # return [v0, t0, r0, v_final, t_final, r_final]


    def SolveMC_parallel_chirp_InputAll(self, v_input, z_input, t_input, max_cores=32):
        '''
        Solve MC for N atoms using multiprocessing
        '''
        import multiprocessing as mp

        start_time = time.time()
        n_cores = np.amin([mp.cpu_count(), max_cores])
        pool = mp.Pool(processes=n_cores)

        args = list(zip(v_input, z_input, t_input))
        # merge the inputs into args = [[[v1], [z1]], [[v2], [z2]], ...]

        # then use pool.starmap instead of map
        outputs = pool.starmap(self.SolveMC_single_chirp_InputAll, args)  # list of function outputs
        print('Time elapsed: {0:.2f} sec'.format(time.time() - start_time))
        return outputs


    '''
    ==== FIXED DETUNING ====
    Gaussian beam; function takes 3 arguments: starting v, starting position, starting time
    '''
    def SolveMC_single_FixedDetuning_InputAll(self, v_initial, r_initial, t_initial):
        '''
        Solve MC for a single atom (v0, r0, t0)
        '''
        self.v_3D = copy.copy(v_initial)  # Initial velocity from argument
        self.r_3D = copy.copy(r_initial)  # Initial position
        self.t_elapsed = t_initial   # Reset the clock
        self.rho0 = ket2dm(basis(2, 0))  # Start in ground state

        self.UpdateArgs()

        # v_atom > v_max + Delta/k
        # Atoms won't be affected at all -- not trapped
        # Skip all atoms with v_z > v_max (upper chirp limit) + Delta / k (outside force's velocity range)
        start_time = time.time()
        v_max = self.args['detuning'] / self.args['k']

        # Ignore atoms that come out later than loading time "ON_HOLD"
        if self.t_elapsed >= self.loading_time:
            outputs = np.concatenate((v_initial, r_initial, t_initial, self.v_3D, self.r_3D, self.t_elapsed), axis=None)
            return np.concatenate((outputs, 'ON_HOLD', time.time() - start_time), axis=None)

        # Ignore atoms that are outside 1.5 Delta/k
        if self.Check_If_InsideVRange() == 0:
            # dt is always positive because we already rule out cases where t_elapsed >= loading_time
            dt = self.loading_time - self.t_elapsed
            self.MoveAtom_NoForce(dt=dt)
            outputs = np.concatenate((v_initial, r_initial, t_initial, self.v_3D, self.r_3D, self.t_elapsed), axis=None)
            return np.concatenate((outputs, 'SKIPPED', time.time() - start_time), axis=None)

        # Only treat atoms with t_elapsed < loading_time
        while self.Check_If_InTime():
            # while the atom is still active, want to check if it is trapped
            # if trapped, exit the simulation and flag the result as "TRAPPED"
            if self.Check_If_Trapped():
                outputs = np.concatenate((v_initial, r_initial, t_initial, self.v_3D, self.r_3D, self.t_elapsed), axis=None)
                return np.concatenate((outputs, 'TRAPPED', time.time() - start_time), axis=None)

            # if it is not trapped and already starts moving backwards, flag as "FAILED"
            if self.v_3D[2] < -0.05:
                outputs = np.concatenate((v_initial, r_initial, t_initial, self.v_3D, self.r_3D, self.t_elapsed), axis=None)
                return np.concatenate((outputs, 'FAILED', time.time() - start_time), axis=None)

            # For atoms with |v_atom - v_detuning| <= 1.5 Delta/k
            # Calculate force and integrate
            if self.Check_If_InsideBeam() and self.Check_If_InsideVRange():
                self.SolveStimulatedForce()
                self.IntegrateForce_1D()
                self.UpdateArgs()
            # without force (outside the fixed detuning range)
            else:
                dt = self.loading_time - self.t_elapsed
                # dt is always positive because of the "while" condition
                self.MoveAtom_NoForce(dt=dt)
                self.UpdateArgs()
                outputs = np.concatenate((v_initial, r_initial, t_initial, self.v_3D, self.r_3D, self.t_elapsed), axis=None)
                return np.concatenate((outputs, 'DONE', time.time() - start_time), axis=None)

        # For atoms that are done
        outputs = np.concatenate((v_initial, r_initial, t_initial, self.v_3D, self.r_3D, self.t_elapsed), axis=None)
        return np.concatenate((outputs, 'DONE', time.time() - start_time), axis=None)


    def SolveMC_parallel_FixedDetuning_InputAll(self, v_input, z_input, t_input, max_cores=32):
        '''
        Solve MC for N atoms using multiprocessing
        '''
        import multiprocessing as mp

        start_time = time.time()
        n_cores = np.amin([mp.cpu_count(), max_cores])
        pool = mp.Pool(processes=n_cores)

        args = list(zip(v_input, z_input, t_input))
        # merge the inputs into args = [[[v1], [z1]], [[v2], [z2]], ...]

        # then use pool.starmap instead of map
        outputs = pool.starmap(self.SolveMC_single_FixedDetuning_InputAll, args)  # list of function outputs
        print('Time elapsed: {0:.2f} sec'.format(time.time() - start_time))
        return outputs






