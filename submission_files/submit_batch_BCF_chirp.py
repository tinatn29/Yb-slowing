#!/usr/bin/env python

import os
import numpy as np


deltas = np.linspace(100, 250, 7)
beam_radii = (100 * 4 / deltas) * 1e-3  # in m
upper_vs = deltas + 10
lower_v = 10
chirp_period = 1e-3  # in seconds

for i in range(len(deltas)):
    delta = deltas[i]
    beam_radius = beam_radii[i]
    upper_v = upper_vs[i]

    job_file = os.path.join(os.getcwd(),f"./jobs/{delta:.0f}_{beam_radius*1000:.1f}mm_{upper_v:.0f}_{lower_v:.0f}_{chirp_period*1000:.0f}ms.sh")

    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name={delta:.0f}_{beam_radius*1000:.1f}mm_{upper_v:.0f}_{lower_v:.0f}_{chirp_period*1000:.0f}ms.job")
        fh.writelines(f"#SBATCH --output=./out/{delta:.0f}_{beam_radius*1000:.1f}mm_{upper_v:.0f}_{lower_v:.0f}_{chirp_period*1000:.0f}ms.txt\n")
        fh.writelines(f"#SBATCH --error=./err/{delta:.0f}_{beam_radius*1000:.1f}mm_{upper_v:.0f}_{lower_v:.0f}_{chirp_period*1000:.0f}ms.txt\n")
        fh.writelines("#SBATCH -c 32\n")
        fh.writelines("#SBATCH -N 1\n")
        fh.writelines("#SBATCH --time=0-04:00\n")
        fh.writelines("#SBATCH --mem=64000\n")
        fh.writelines("#SBATCH --qos=normal\n")
        fh.writelines("#SBATCH -p hns\n")
        fh.writelines("ml python/3.6.1\n")
        fh.writelines(f"python3 $HOME/SS_Simulation/RunMC_allinput_BCF_chirp.py {delta} {delta:.0f}_{beam_radius*1000:.1f}mm_{upper_v:.0f}_{lower_v:.0f}_{chirp_period*1000:.0f}ms.csv \
            {upper_v} {lower_v} {chirp_period} {beam radius}")

    os.system("sbatch %s" %job_file)
