#!/bin/bash
#SBATCH --job-name=analyseSimPhyl
#SBATCH --account=nimwegen
#SBATCH --mem=20G
#SBATCH --qos=1day
#SBATCH --cpus-per-task=1
#SBATCH --array=6

ml PhyML
declare -a IDs=("simSelection_1" "simSelection_2" "simSelection_3" "simSelection_4" "simSelection_5" "simSelection_6" "simSelection_7")
ID=${IDs[$SLURM_ARRAY_TASK_ID]}
echo processing $ID
python analyseGenotypesSelection.py $ID
#phyml -i $ID"/sim_"$ID".phy" -c 1 -p
#python variableColumnStats.py $ID > $ID"/snpstats_"$ID".txt"
