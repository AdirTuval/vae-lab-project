#!/bin/csh
#SBATCH --mem=20g
#SBATCH --time=1-0
#SBATCH --mail-type=END
#SBATCH --mail-user=adir.tuval
#SBATCH --error=/cs/labs/yweiss/adirt/lab_project/vae-lab-project/out_job_logs/job.%J.err 
#SBATCH --output=/cs/labs/yweiss/adirt/lab_project/vae-lab-project/out_job_logs/job.%J.out
#SBATCH --gres=gpu:1,vmem:10g
#SBATCH --killable
echo "Running IMA VAE"
source /cs/labs/yweiss/adirt/lab_project/vae-lab-project/env/bin/activate.csh
cd /cs/labs/yweiss/adirt/lab_project/vae-lab-project/
python /cs/labs/yweiss/adirt/lab_project/vae-lab-project/cli.py fit \
       -c /cs/labs/yweiss/adirt/lab_project/vae-lab-project/configs/vae.yaml --model.vae.decoder_var $1