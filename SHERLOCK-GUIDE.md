# Guide to Sherlock
Sherlock is a computing cluster operated by the Stanford Research Computing Center for research purposes. See their full documentation [here](https://www.sherlock.stanford.edu/docs/).<br>
Monte Carlo simulations in this project utilize parallel computing (the Python multiprocessing package) and are run on Sherlock with 16-32 CPU cores.

## Preparation
1. Prepare a job file (See [this guide](https://vsoch.github.io/lessons/sherlock-jobs/) on how to create a job file using R/Python/Bash). `submit_chirp_python.py` is a Python example, which takes certain simulation parameters as arguments and submits one simulation job to Sherlock. To submit multiple jobs at once, a Python file such as `submit_chirp_batch_python.py` loops through a set of parameters/conditions we want to vary. In addition to the simulation parameters, the files specify demands on computing resources including no. of CPUs, memory usage, time.
2. [Data transfer](https://www.sherlock.stanford.edu/docs/storage/data-transfer/#transfer-protocols): tranfer .py files (programming files and job submission files) and their dependencies to your storage folder on Sherlock.
3. [Submit jobs](https://vsoch.github.io/lessons/slurm/) on Sherlock

## Job submission
1. On Terminal, ssh into Sherlock by typing `ssh <SUnetID>@login.sherlock.stanford.edu`. Type in the password when prompted and authorize the two-factor notification.
2. `cd` into the file directory
3. Load Python module by typing `ml python/3.6.1` (provided python 3.6.1 has been installed to your Sherlock directory)
4. To submit jobs via a Python submission file, type `python3 <submission_file.py> <arguments>`. You should see `Submitted batch job <job number>` if the job submission is successful.
5. To see the status(es) of your job(s), type `squeue -u <SUnetID>`
 
## Running Jupyter notebook on Sherlock
Simulation results are saved in .csv format. In the past I transferred the files from my Sherlock directory to my local computer before processing them. I recently adopted a new (better) workflow where I run Jupyter notebook on Sherlock and access the files from there, without having to move the files to my computer. After following these [instructions](https://vsoch.github.io/lessons/sherlock-jupyter/), I can now start a Jupyter notebook session from my local Terminal.

## Additional resources
- [Python on Sherlock](https://www.sherlock.stanford.edu/docs/software/using/python/): how to install Python and Python packages on Sherlock
- [Running Jupyter notebook on Sherlock](https://vsoch.github.io/lessons/sherlock-jupyter/)
