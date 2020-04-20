#!/bin/bash
#SBATCH --job-name=simPop
#SBATCH --account=scicore
#SBATCH --mem=16G
#SBATCH --qos=1day
#SBATCH --cpus-per-task=1
#SBATCH --partition=pascal
#SBATCH --gres=gpu:1
#SBATCH --array=0

declare -a IDs=("simSelection_1" "simSelection_2" "simSelection_3" "simSelection_4" "simSelection_5" "simSelection_6" "simSelection_7")
ID0=${IDs[0]}
echo simulating $ID0
mkdir $ID0
python selectionSim4Letters.py -I $ID0 -v --threadsperblock 20 --blocks 20 -L 4000 -s 1 

ID1=${IDs[1]}
echo simulating $ID1
mkdir $ID1
python selectionSim4Letters.py -I $ID1 -v --threadsperblock 20 --blocks 20 -L 4000 -s 0.98

ID2=${IDs[2]}
echo simulating $ID2
mkdir $ID2
python selectionSim4Letters.py -I $ID2 -v --threadsperblock 20 --blocks 20 -L 4000 -s 0.99

ID3=${IDs[3]}
echo simulating $ID3
mkdir $ID3
python selectionSim4Letters.py -I $ID3 -v --threadsperblock 20 --blocks 20 -L 4000 -s 0.999

ID4=${IDs[4]}
echo simulating $ID4
mkdir $ID4
python selectionSim4Letters.py -I $ID4 -v --threadsperblock 20 --blocks 20 -L 4000 -s 0.9999

ID5=${IDs[5]}
echo simulating $ID5
mkdir $ID5
python selectionSim4Letters.py -I $ID5 -v --threadsperblock 20 --blocks 20 -L 4000 -s 1.01

ID6=${IDs[6]}
echo simulating $ID6
mkdir $ID6
python selectionSim4Letters.py -I $ID6 -v --threadsperblock 20 --blocks 20 -L 4000 -s 1.001

