import numpy as np
from qutip import *
from scipy import signal
import RandomInitialize as RI
import time
import copy


# Laser chirping
def chirped_detuning(T, parameters):
    period, v_max, v_min = parameters
    f = 1 / period  # default period = 0.8 ms
    k = 2 * np.pi / 556e-9  # wavenumber
    # chirp laser so it's detuned between 300 m/s and 60 m/s
    return k * (-0.5 * (v_max - v_min) * signal.sawtooth(2 * np.pi * f * T) + 0.5 * (v_min + v_max))


# Four plane waves (with RTA)
def H_12_four(t, args):
    T = t + args['t_elapsed']
    # Left (co-prop)
    W1 = np.exp(1j * (args['Delta'] * T + args['detuning'] * T - args['k'] * args['v'] * T
        - args['k'] * args['z'] + args['phi_left']))
    W2 = np.exp(1j * (-args['Delta'] * T + args['detuning'] * T - args['k'] * args['v'] * T
        - args['k'] * args['z'] - args['phi_left']))
    # Right (counter-prop)
    W3 = np.exp(1j * (args['Delta'] * T - args['detuning'] * T + args['k'] * args['v'] * T
        + args['k'] * args['z'] + args['phi_right']))
    W4 = np.exp(1j * (-args['Delta'] * T - args['detuning'] * T + args['k'] * args['v'] * T
        + args['k'] * args['z'] - args['phi_right']))

    H12 = (W1 + W2 + W3 + W4) * args['Omega'] / 2
    return np.conj(H12) 

def H_21_four(t, args):
    return np.conj(H_12_four(t, args))

def H12_derivative_z_4(t, args):
    T = t + args['t_elapsed']

    # Left
    W1 = 1j * args['k'] * np.exp(1j * (args['Delta'] * T + args['detuning'] * T
        - args['k'] * args['v'] * T - args['k'] * args['z'] + args['phi_left']))
    W2 = 1j * args['k'] * np.exp(1j * (-args['Delta'] * T + args['detuning'] * T
        - args['k'] * args['v'] * T - args['k'] * args['z'] - args['phi_left']))
    # Right
    W3 = -1j * args['k'] * np.exp(1j * (args['Delta'] * T - args['detuning'] * T
        + args['k'] * args['v'] * T + args['k'] * args['z'] + args['phi_right']))
    W4 = -1j * args['k'] * np.exp(1j * (-args['Delta'] * T - args['detuning'] * T
        + args['k'] * args['v'] * T + args['k'] * args['z'] - args['phi_right']))
    H12_deriv = (W1 + W2 + W3 + W4) * args['Omega'] / 2

    return H12_deriv

def spont_decay(p_ex, T, tau):
    p_decay = p_ex * (1 - np.exp(- T / tau))
    return np.random.rand() <= p_decay

def merge_two_arrays(a1, a2):
    out = [x for x in a1] + [y for y in a2]
    return out


# Define new class here
class MCSolver():
    def __init__(self, delta=50, omega=60, phi=-np.pi/4, beam_radius=0.004, detuning=0, gravity=True, loading_time=1e-3):
        # Dictionary "args" contains parameters that need to be fed into QuTiP mesolve function
        self.args = {
                    'gamma': 2 * np.pi * 182e3, # spontaneous decay rate / natural linewidth
                    'Delta': delta * 2 * np.pi * 182e3, # bichromatic detuning
                    'Omega': omega * 2 * np.pi * 182e3, # Rabi frequency
                    'detuning': detuning, # laser detuning from atomic resonance w0
                    'k': 2 * np.pi / 556e-9, # wavenumber
                    'v': 0, # atom's velocity vz
                    'z': 0, # atom's position z
                    'phi_left': phi, # phase of laser beams moving towards the atom (-pi/4 by default)
                    'phi_right': 0, # phase of laser beams moving in the same direction as the atom (zero by default)
                    't_elapsed': 0, # time elapsed 
                    'hbar' : 6.63e-34 / (2 * np.pi), # hbar = h / 2pi
                    'mass' : 174 * 1.67e-27, # mass of Yb-174 atom
                    'tau' : 1 / 2 * np.pi * 182e3 # atom's lifetime = 1/gamma
        }

        # Atom's properties
        self.v_3D = np.array([0, 0, 0])  # atom's velocity 3D
        self.r_3D = RI.rand_position().flatten()  # Random starting position
        self.chirp = None
        self.t_elapsed = 0

        # Quantities for qutip mesolve
        self.H0 = Qobj([[0, 0], [0, 0]])
        self.H = [self.H0, [Qobj([[0, 1], [0, 0]]), H_12_four],
            [Qobj([[0, 0], [1, 0]]), H_21_four]]
        self.rho0 = ket2dm(basis(2, 0))  # Start in ground state
        self.c_op = np.sqrt(self.args['gamma']) * Qobj([[0, 1], [0, 0]])  # Collapse operator
        self.tlist = np.linspace(0, 10e-6, 100)  # tlist for each cycle (10 us)
        self.force = None  # Force as a function of time
        self.p_ex = None  # Excitation population
        self.p_ex_mean = None  # Average excitation prob
        self.loading_time = loading_time  # loading time default = 1ms
        self.chirp_successful = False
        self.Gaussian_beam = False
        self.Omega_0 = self.args['Omega']  # maximum Rabi frequency (set)
        self.beam_radius = beam_radius  # beam radius 4mm by default
        self.gravity = gravity  # gravity taken into account?
        self.F_rad = self.args['hbar'] * self.args['k'] * self.args['gamma'] / 2


    # Set up class attribute self.chirp 
    def SetChirping(self, period, v_max, v_min):
        self.chirp = [period, v_max, v_min]

    # Gaussian beam profile
    def Omega_Gaussian(self):
        r = np.sqrt(self.r_3D[0]**2 + self.r_3D[1]**2)
        return self.Omega_0 * np.exp(- r**2 / self.beam_radius**2)

    def UpdateArgs(self):
        # Update arguments from atom's variables
        if self.chirp is not None:
            self.args['detuning'] = chirped_detuning(self.t_elapsed, self.chirp)

        self.args['v'] = self.v_3D[2]
        self.args['z'] = self.r_3D[2]
        self.args['t_elapsed'] = self.t_elapsed

        if self.Gaussian_beam:
            self.args['Omega'] = self.Omega_Gaussian()


    def SolveStimulatedForce(self):
        '''
        For 10 us, solve for stimulated force(t) and average excitation prob
        '''
        # Update args before solving
        self.UpdateArgs()

        # Solve Master equation using current atom's parameters
        options = Odeoptions(nsteps=4000)
        output = mesolve(self.H, self.rho0, self.tlist, c_ops=[self.c_op], args=self.args,
            options=options)
        # Calculate force(t)
        self.force = np.zeros_like(self.tlist)
        for n in np.arange(len(self.tlist)):
            self.force[n] = (4 / (self.args['k'] * self.args['gamma'])) * np.real(
                H12_derivative_z_4(self.tlist, self.args)[n] * output.states[n][1, 0])
        self.rho0 = output.states[-1]  # Update states for next cycle
        # Calculate average excitation probability
        self.p_ex = expect(num(2), output.states)
        self.p_ex_mean = np.mean(self.p_ex)

    def IntegrateForce_1D(self):
        '''
        Integrate F(t) for v(t) and z(t)
        Ignore any transverse heating for now
        '''
        v_array = [self.v_3D[2]]

        for ind in range(len(self.tlist) - 1):
            dv = np.trapz(self.force[:(ind + 1)], self.tlist[:(ind + 1)]) * self.F_rad / self.args['mass']
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

    def MoveAtom_NoForce(self, dt=None):
        '''
        In case atom's outside the beam: atom moves without being slowed
        Update position and time but not velocity
        '''
        if dt is None:
            # if dt is not set explicitly, use dt = 10us
            dt = self.tlist[-1]

        dr = np.array(self.v_3D) * dt
        self.r_3D = np.add(self.r_3D, dr)

        g = 9.81  # acceleration due to gravity

        if self.gravity:
            # update y and v_y if there's gravity
            self.r_3D[1] -= 0.5 * g * dt**2
            self.v_3D[1] -= g * dt

        self.t_elapsed += dt

    def SpontaneousDecays(self):
        dv_max = self.args['hbar'] * self.args['k'] / self.args['mass']
        # How many lifetimes in one interval
        N = int(np.round(self.tlist[-1] / self.args['tau']))

        # Arrays v and t to store v(t)
        v = np.zeros((N, 3))
        t = np.zeros(N)

        T0 = self.args['tau']
        t_last_decay = 0

        for i in range(N):
            start_idx = int(i * len(self.tlist) / N)
            end_idx = int((i + 1) * len(self.tlist) / N) - 1
            t[i] = self.t_elapsed + self.tlist[end_idx]

        # Velocity change from stimulated force (after one lifetime)
            dv = [0, 0, (self.args['hbar'] * self.args['k'] * self.args['gamma'] / 2) *
                np.trapz(self.force[start_idx:end_idx],
                self.tlist[start_idx:end_idx]) / self.args['mass']]

            if spont_decay(self.p_ex_mean, t_last_decay + T0, self.args['tau']):
                dv += RI.rand_direction().flatten() * dv_max
                t_last_decay = 0
            else:
                t_last_decay += T0

            if i == 0:
                v[i, :] = np.add(self.v_3D, dv)
            else:
                v[i, :] = np.add(v[i-1, :], dv)
                # Update velocity next line

        # Update atom's velocity and position for after 30*tau interval
        self.v_3D = v[-1, :]
        self.r_3D += np.array([np.trapz(v[:, 0], t), np.trapz(v[:, 1], t), np.trapz(v[:, 2], t)])
        # Update time_elapsed
        self.t_elapsed += self.tlist[-1]

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
        # Return true if atom's in the slowing region z >= 0
        # In some cases we may start atoms with z0 < 0, where they should not experience any force yet
        return self.r_3D[2] >= 0

    def Check_If_InsideVRange(self):
        # Return true if atom's velocity is inside v_c +/- 1.5 * Delta / k
        v_detuning = self.args['detuning'] / self.args['k']
        v_range = 1.5 * self.args['Delta'] / self.args['k']
        return np.abs(self.v_3D[2] - v_detuning) <= v_range

    def Check_If_InsideBeam_FlatTop(self):
        # Return true if z >= 0 and atom's inside the beam radius
        # Only use for flat-top laser intensity profile and not with the default Gaussian beam profile setting
        return np.sqrt(self.r_3D[0]**2 + self.r_3D[1]**2) <= self.beam_radius and self.r_3D[2] >= 0

    '''
    UPGRADE: Gaussian beam; function takes 3 arguments: starting v, starting position, starting time
    '''
    def SolveMC_single_GaussianBeam_InputAll(self, v_initial, r_initial, t_initial):
        '''
        Solve MC for a single atom (v0, r0, t0)
        '''

        self.Gaussian_beam = True
        self.v_3D = copy.copy(v_initial)  # Initial velocity from argument
        self.r_3D = copy.copy(r_initial)  # Initial position
        self.t_elapsed = t_initial   # Reset the clock
        self.rho0 = ket2dm(basis(2, 0))  # Start in ground state

        self.UpdateArgs()

        if self.chirp is not None:
            self.loading_time = self.chirp[0]  # replace loading time with chirp period
            v_max = self.chirp[1]  # v_max = upper chirp limit
        else:
            v_max = self.args['detuning'] / self.args['k']

        # v_atom > v_max + Delta/k
        # Atoms won't be affected at all -- not trapped
        # Skip all atoms with v_z > v_max (upper chirp limit) + Delta / k (outside force's velocity range)
        start_time = time.time()

        if v_initial[2] > v_max + self.args['Delta'] / self.args['k']:
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


    '''
    FLAT_TOP_VARIATION of final code: flat top; function takes 3 arguments: starting v, starting position, starting time
    '''
    def SolveMC_single_FlatTop_InputAll(self, v_initial, r_initial, t_initial):
        '''
        Solve MC for a single atom (v0, r0, t0)
        '''

        self.v_3D = copy.copy(v_initial)  # Initial velocity from argument
        self.r_3D = copy.copy(r_initial)  # Initial position
        self.t_elapsed = t_initial   # Reset the clock
        self.rho0 = ket2dm(basis(2, 0))  # Start in ground state

        self.UpdateArgs()

        if self.chirp is not None:
            self.loading_time = self.chirp[0]  # replace loading time with chirp period
            v_max = self.chirp[1]  # v_max = upper chirp limit
        else:
            v_max = self.args['detuning'] / self.args['k']

        # v_atom > v_max + Delta/k
        # Atoms won't be affected at all -- not trapped
        # Skip all atoms with v_z > v_max (upper chirp limit) + Delta / k (outside force's velocity range)
        start_time = time.time()

        if v_initial[2] > v_max + self.args['Delta'] / self.args['k']:
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
            if self.Check_If_InsideBeam_FlatTop() and self.Check_If_InsideVRange():
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

    '''
    ==== FIXED DETUNING ====
    UPGRADE 6: Gaussian beam; function takes 3 arguments: starting v, starting position, starting time
    '''
    def SolveMC_single_FixedDetuning_InputAll(self, v_initial, r_initial, t_initial):
        '''
        Solve MC for a single atom (v0, r0, t0)
        '''
        self.Gaussian_beam = True
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

        # Ignore fast atoms
        if v_initial[2] > v_max + 1.5 * self.args['Delta'] / self.args['k']:
            dt = self.loading_time - self.t_elapsed
            self.MoveAtom_NoForce(dt=dt)
            outputs = np.concatenate((v_initial, r_initial, t_initial, self.v_3D, self.r_3D, self.t_elapsed), axis=None)
            return np.concatenate((outputs, 'FAST', time.time() - start_time), axis=None)

        # Only treat atoms with t < loading_time
        while self.t_elapsed < self.loading_time:
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
            if self.Check_If_InsideVRange():
                self.SolveStimulatedForce()
                self.IntegrateForce_1D()
                self.UpdateArgs()
            # without force (outside the fixed detuning range)
            else:
                dt = self.loading_time - self.t_elapsed
                self.MoveAtom_NoForce(dt=dt)
                outputs = np.concatenate((v_initial, r_initial, t_initial, self.v_3D, self.r_3D, self.t_elapsed), axis=None)
                return np.concatenate((outputs, 'DONE', time.time() - start_time), axis=None)

        # For atoms that are done
        outputs = np.concatenate((v_initial, r_initial, t_initial, self.v_3D, self.r_3D, self.t_elapsed), axis=None)
        return np.concatenate((outputs, 'DONE', time.time() - start_time), axis=None)


    def SolveMC_parallel_InputAll(self, v_input, z_input, t_input, max_cores=32):
        '''
        Solve MC for N atoms using multiprocessing
        max_cores = 32 for running things on Sherlock
        '''
        import multiprocessing as mp

        start_time = time.time()
        n_cores = np.amin([mp.cpu_count(), max_cores])
        pool = mp.Pool(processes=n_cores)

        args = list(zip(v_input, z_input, t_input))
        # merge the inputs into args = [[[v1], [z1], [t1]], [[v2], [z2], [t2]], ...]

        # then use pool.starmap instead of map
        # can change the function in starmap() to use other single functions in this class
        outputs = pool.starmap(self.SolveMC_single_GaussianBeam_InputAll, args)  # list of function outputs
        print('Time elapsed: {0:.2f} sec'.format(time.time() - start_time))
        return outputs


    def SolveMC_single_track(self, v_initial):
        '''
        Solve MC for a single atom
        '''
        self.v_3D = v_initial  # Initial velocity from argument
        self.r_3D = RI.rand_position().flatten()  # Initial position
        self.t_elapsed = 0   # Reset the clock
        self.rho0 = ket2dm(basis(2, 0))  # Start in ground state

        N_cycle = 31

        for cycle in range(N_cycle):
            self.SolveStimulatedForce()
            self.SpontaneousDecays()

            if self.Check_If_Done():
                return merge_two_arrays(v_initial, self.v_3D), self.r_3D, cycle

            # if self.Check_If_Fail():
                # return merge_two_arrays(v_initial, ['FAILED'])

        return merge_two_arrays(v_initial, self.v_3D), self.r_3D, cycle  # Return both initial and final velocities
