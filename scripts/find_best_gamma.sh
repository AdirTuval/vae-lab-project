#!/bin/csh
for decoder_var in 10.0 1.0 0.1 0.01 0.001 0.0001
	do sbatch /cs/labs/yweiss/adirt/lab_project/vae-lab-project/scripts/gamma_job.csh $decoder_var
	done