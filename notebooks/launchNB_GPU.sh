#!/bin/bash
#SBATCH --partition=pascal
##SBATCH --nodes 1
##SBATCH --ntasks-per-node 01
##SBATCH --exclusive
#SBATCH --account=scicore
#SBATCH --time 24:00:00
#SBATCH --qos=1day
#SBATCH --mem-per-cpu 16000
#SBATCH --job-name tunnel_gpu
#SBATCH --output jupyter-log-%J.txt
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

## get tunneling info
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)

## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
     Copy/Paste this in your local terminal to ssh tunnel with remote
-----------------------------------------------------------------
     ssh -N -L $ipnport:$ipnip:$ipnport saktho00@worker-nimwegen.scicore.unibas.ch
-----------------------------------------------------------------

     Then open a browser on your local machine to the following address
------------------------------------------------------------------
     https://localhost:$ipnport  (prefix w/ https:// if using password)
------------------------------------------------------------------
     "
## start an ipcluster instance and launch jupyter server
#jupyter-notebook --no-browser --port=$ipnport --ip=$ipnip --keyfile=/scicore/home/nimwegen/saktho00/.mycerts/mycert.key --certfile=/scicore/home/nimwegen/saktho00/.mycerts/mycert.pem
jupyter lab --no-browser --port=$ipnport --ip=$ipnip --keyfile=/scicore/home/nimwegen/saktho00/.mycerts/mycert.key --certfile=/scicore/home/nimwegen/saktho00/.mycerts/mycert.pem

