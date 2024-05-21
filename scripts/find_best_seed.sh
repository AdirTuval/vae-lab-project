#!/bin/csh
for seed in 102 1 32 42 19238 9283 1846729 192 8952 8123675
	do sbatch /cs/labs/yweiss/adirt/lab_project/vae-lab-project/scripts/job.csh $seed
	done