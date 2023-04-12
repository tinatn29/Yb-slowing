# Yb-stimulated-slowing
Numerical calculations and Monte Carlo simulations of stimulated slowing of Yb atoms with amplitude-modulated light

## Project Overview
Python codes and some simulation data used to produced results in our publication, T. Na Narong, et al. *Physical Review A* (2021) [[1]](https://journals-aps-org.stanford.idm.oclc.org/pra/abstract/10.1103/PhysRevA.104.053117), on stimulated slowing of Yb atoms with counter-propagating bichromatic and amplitude-modulated light.

### Numerical calculation of bichromatic force (BCF) and polychromatic force (PCF)
We derived the time-dependent Hamiltonian matrix H(t) for the two-level atom, and then used the `mesolve` function from the [QuTiP](https://qutip.org/docs/4.0.2/index.html) library to solve the [Linblad Master Equation](https://qutip.org/docs/latest/guide/dynamics/dynamics-master.html) for density matrix. The stimulated force is calculated from the Hamiltonian and density matrix, then integrated to update the atom's velocity and position. 

### Monte Carlo simulations
Assuming all atoms are independent, we developed Monte Carlo simulations to predict which atoms are successfully trapped by the MOT and estimate the MOT loading rate for given input experimental parameters. The simulations account for the laser Gaussian beam profile, atomic beam divergence and gravity. Atoms' initial positions, velocities, and departure times from the oven are set by `GenerateBatchAtoms.py`. Running this file generates `v_input.npy`, `r_input.npy`, `t_input.npy`, to be loaded when running the simulation files.

## Important files and dependencies
### Class files 
These files define a class `MCSolver`, which contains functions to numerically solve the Master Equation for the density matrix and the stimulated force, and functions to run Monte Carlo simulations with parallel computing. 
- `SSMCSolver_BCF.py` Bichromatic force (BCF) 
-  Polychromatic force
### Dependencies
- `SSParameters.py` defines the class `GreenLaser`, which contains the parameters for bichromatic light.
- `RandomInitialize.py` defines the class `GreenLaserSquare`, which contains the parameters for square-wave AM light.
### Job submission files
The following files can be run on a local computer or on the Sherlock computing cluster. On the local computer, ONLY RUN THIS WITH VERY FEW ATOMS (or it will take very long). Locally, type `python3 <RunMC_file.py> <arguments>` on a local terminal. 
- `RunMC...py`
- `RunMC...py`

To run Monte Carlo simulations on the Sherlock cluster, modify the following files and run them on the Sherlock cluster to submit jobs to the server. See [SHERLOCK-GUIDE.md](SHERLOCK-GUIDE.md) for details.
- `submit_chirp_python.py`
- `submit_chirp_batch_python.py`
- `submit_chirp_squareAM.py`
- `submit_chirp_batch_square.py`

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
