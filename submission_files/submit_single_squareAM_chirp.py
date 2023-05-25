#!/usr/bin/env python

import os
import sys


# parameters
delta = float(sys.argv[1])  # delta in gamma
Rabi_shape = int(sys.argv[2])   # 0 = square, 1 = cosine, 2+ = square Fourier series
vmax = float(sys.argv[3])  # v_max chirp in m/s
vmin = float(sys.argv[4])  # v_min chirp in m/s
chirp_period = float(sys.argv[5])  # chirp period in s
beam_radius = float(sys.argv[6])  # beam radius in m

job_file = os.path.join(os.getcwd(),f"./jobs/{Rabi_shape}_{delta:.0f}_{beam_radius * 1000:.0f}mm_{vmax:.0f}_{vmin:.0f}_{chirp_period * 1000:.0f}ms.sh")

with open(job_file, 'w') as fh:
    fh.writelines("#!/bin/bash\n")
    fh.writelines(f"#SBATCH --job-name={Rabi_shape}_{delta:.0f}_{beam_radius * 1000:.0f}mm_{vmax:.0f}_{vmin:.0f}_{chirp_period * 1000:.0f}ms.job\n")
    fh.writelines(f"#SBATCH --output=./out/{Rabi_shape}_{delta:.0f}_{beam_radius * 1000:.0f}mm_{vmax:.0f}_{vmin:.0f}_{chirp_period * 1000:.0f}ms.txt\n")
    fh.writelines(f"#SBATCH --error=./err/{Rabi_shape}_{delta:.0f}_{beam_radius * 1000:.0f}mm_{vmax:.0f}_{vmin:.0f}_{chirp_period * 1000:.0f}ms.txt\n")
    fh.writelines("#SBATCH -c 32\n")
    fh.writelines("#SBATCH -N 1\n")
    fh.writelines("#SBATCH --time=0-24:00\n")
    fh.writelines("#SBATCH --mem=64000\n")
    fh.writelines("#SBATCH --qos=normal\n")
    fh.writelines("#SBATCH -p hns\n")
    fh.writelines("ml python/3.6.1\n")
    fh.writelines(f"python3 $SCRATCH/SS_Simulation/RunMC_allinput_squareAM_chirp.py {delta} {Rabi_shape}\
        {vmax} {vmin} {chirp_period} {beam_radius}\
        {Rabi_shape}_{delta:.0f}_{beam_radius * 1000:.0f}mm_{vmax:.0f}_{vmin:.0f}_{chirp_period * 1000:.0f}ms.csv")

# os.system("sbatch %s" %job_file)

