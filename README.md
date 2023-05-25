# IN PROGRESS - more files to be added

# Yb-slowing
Numerical calculations and Monte Carlo simulations of stimulated slowing of Yb atoms with amplitude-modulated light

## Project Overview
This repo contains Python codes developed for the work published in T. Na Narong, et al. *Physical Review A* (2021) [[1]](https://journals-aps-org.stanford.idm.oclc.org/pra/abstract/10.1103/PhysRevA.104.053117). We numerically calculate optical forces on Yb atoms to design a stimulated slowing experiment with counter-propagating bichromatic and amplitude-modulated light. We also run Monte Carlo simulations of >20,000 atoms to evaluate the slowing efficiency.

### Numerical calculation of bichromatic force (BCF) and polychromatic force (PCF)
We derived the time-dependent Hamiltonian matrix H(t) for the two-level atom, and then used the `mesolve` function from the [QuTiP](https://qutip.org/docs/4.0.2/index.html) library to solve the [Linblad Master Equation](https://qutip.org/docs/latest/guide/dynamics/dynamics-master.html) for density matrix. The stimulated force is calculated from the Hamiltonian and density matrix, then integrated to update the atom's velocity and position. 

### Monte Carlo simulations
Assuming all atoms are independent, we developed Monte Carlo simulations to predict which atoms are successfully slowed and trapped by the MOT for given input experimental parameters. The simulations account for the laser Gaussian beam profile, atomic beam divergence and gravity. <br>
Atoms' initial positions, velocities, and departure times from the oven are set by `GenerateBatchAtoms.py`. Running this file generates `v_input.npy`, `r_input.npy`, `t_input.npy`, to be loaded when running the simulation files. The input_files folder contains these files with 5000 or 20000 atoms as examples.

## Important files and dependencies
### Class files 
These files define a class `MCSolver`, which contains functions to numerically solve the Master Equation for the density matrix and the stimulated force, and functions to run Monte Carlo simulations with parallel computing. 
- `SSMCSolver_BCF.py` Bichromatic force (BCF)  from overlapping CW beams
- `SSMCSolver_square.py` Polychromatic force (PCF) from square-wave AM light
### Dependencies
- `RandomInitialize.py` defines functions to sample atoms' starting velocities (effusion distribution) and positions (uniform).
### Main simulation files
- `RunMC_local_BCF.py` Use this to test run on your local computer to make sure all files are ready.
The following files run Monte Carlo simulations from input_files. 
- `RunMC_allinput_BCF_chirp.py` runs the simulation with pre-set v,r,t inputs.

To run Monte Carlo simulations on the Sherlock cluster, create a bash submission script or a python submission script (see examples in the folder submission_files) and run them on the Sherlock cluster to submit jobs to the server. See [SHERLOCK-GUIDE.md](SHERLOCK-GUIDE.md) for details.
- `submit_BCF_chirp_single.py`
- `submit_BCF_chirp_batch.py`

## Software and Libraries
- Python 3.6
- [Numpy](https://numpy.org/)
- [QuTiP: Quantum Toolbox in Python](https://qutip.org/docs/4.0.2/index.html)
- [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) 
- [matplotlib](https://matplotlib.org/)
- [pandas](https://pandas.pydata.org/)
- [jupyter notebook](https://jupyter.org/)

## References
1. Na Narong, Tanaporn, TianMin Liu, Nikhil Raghuram, and Leo Hollberg. "Stimulated slowing of Yb atoms on the narrow 1S0→3P1 transition." Physical Review A 104, no. 5 (2021): 053117. https://doi-org/10.1103/PhysRevA.104.053117
