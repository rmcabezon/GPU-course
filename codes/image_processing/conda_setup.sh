#!/usr/bin/env bash

conda init bash

conda create -y -n gpu_course_new python=3.7
conda activate gpu_course_new

conda install -y jupyter
conda install -y jupyterlab

#conda config --add channels conda-forge
#
conda install -y -c conda-forge ocl-icd-system  # needed on linux for conda to env to find ocl-icd files living in /etc/OpenCL/vendors; see: https://documen.tician.de/pyopencl/misc.html#using-vendor-supplied-opencl-drivers-linux
conda install -y numpy
conda install -y -c conda-forge pyopencl

conda install -y numba
conda install -y cudatoolkit=11.0.221

#conda install -y -c conda-forge opencv
conda install -y scikit-image

conda deactivate
