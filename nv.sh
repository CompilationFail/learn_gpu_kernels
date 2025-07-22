# !/usr/bin/bash

# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local
# follow the guide above, use runfile (local) install type!!!!
# don't use apt install nvidia-tool-kit cause it will instal a 11.5 nvcc
# cause all the issues maybe
wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
sudo sh cuda_12.9.1_575.57.08_linux.run
echo -e "export PATH=/usr/local/cuda/bin/:$PATH" > ~/.bashrc
lspci | grep -i nvidia
nvcc --version
nvcc --list-gpu-arch
nvidia-smi
