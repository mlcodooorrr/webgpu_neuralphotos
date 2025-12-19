# set up conductor
curl --proto '=https' --tlsv1.2 -sSf https://pages.github.pie.apple.com/storage-orchestration/conductor/docs/setup-conductor.sh | bash

# opencv stuff i think
apt-get update && apt-get install -y libgl1-mesa-glx
apt-get install -y libglib2.0-0 

# Downloading gan dataset
echo "Downloading gan files and h5 file"
cd gan_project
python download_from_s3.py

# preprocess data, into memory-mapped numpy files, remove h5 file overhead
python preprocess_data.py

# Training GAN model
# TODO: add resume logic for training if on a resume run
echo "Training GAN model"
python train_gan.py --name test_run


# Helpful commands to copy and paste
# bolt task submit --interactive --tar . --config config.yaml
# watch -n 0.1 nvidia-smi
# pip install -r requirements.txt

# nsys
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install nsight-systems-2025.3.2

# helps check if im compute bound or memory bound
# nvidia-smi dmon -s u


nsys profile \
  -t cuda,nvtx,osrt,cudnn,cublas \
  --stats=true \
  --force-overwrite=true \
  -o profiles/gan_profile \
  python train_gan.py

nsys stats --report cuda_api_sum,cuda_gpu_kern_sum profiles/gan_profile.nsys-rep > profiles/full_analysis_12.txt
