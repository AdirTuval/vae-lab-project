#!/bin/csh
#SBATCH --mem=20g
#SBATCH --time=1-0
#SBATCH --mail-type=END
#SBATCH --mail-user=adir.tuval
#SBATCH --output=/cs/labs/yweiss/adirt/lab_project/out/ima_vae_250.out
#SBATCH --gres=gpu:1,vmem:10g
echo "Running ima_vae_250"
source /cs/labs/yweiss/adirt/lab_project/vae-lab-project/env/bin/activate.csh
cd /cs/labs/yweiss/adirt/lab_project/vae-lab-project/
python /cs/labs/yweiss/adirt/lab_project/vae-lab-project/cli.py fit \
       -c /cs/labs/yweiss/adirt/lab_project/vae-lab-project/configs/vae.yaml