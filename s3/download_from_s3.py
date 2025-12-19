# curl --proto '=https' --tlsv1.2 -sSf https://pages.github.pie.apple.com/storage-orchestration/conductor/docs/setup-conductor.sh | bash
# use to download conductor

import os
from conductor_manager import ConductorManager
import sys
import gzip

# Parameters
# -----------------------------------------------------------------------------
conductor_manager = ConductorManager()
bucket_name = "niranjan-s-personal"

# Gan Frames
# -----------------------------------------------------------------------------
import os
import subprocess

def download_gan_frames(dataset_name="dataset_preprocessed.h5"):
    local_path = "gan_frames/"

    s3_path = "s3://niranjan-s-personal/gan_frames/"

    print(f"Downloading gan frames from {s3_path} to {local_path}...")
    try:
        # Create directory if it doesn't exist
        os.makedirs(local_path, exist_ok=True)
        
        # # Download single all_frames.npy file
        # file_s3_path = os.path.join(s3_path, "all_frames.npy")
        # file_local_path = os.path.join(local_path, "all_frames.npy")

        # print(f"Downloading all_frames.npy...")
        # subprocess.run([
        #     "conductor", "s3", "cp",
        #     file_s3_path,
        #     file_local_path
        # ], check=True)

        # download dataset file
        file_s3_path = os.path.join(s3_path, dataset_name)
        file_local_path = os.path.join(local_path, dataset_name)

        print(f"Downloading {dataset_name}...")
        subprocess.run([
            "conductor", "s3", "cp",
            file_s3_path,
            file_local_path
        ], check=True)
        
        print(f"All gan frames and {dataset_name} downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading files: {e}")
        return False
    return True

def upload_to_s3(file_path, folder, name, bucket_name="niranjan-s-personal"):
    conductor_manager = ConductorManager()
    s3_key = f"{folder}/{name}"
    
    try:
        # Upload to S3
        conductor_manager.upload_file(bucket_name, file_path, s3_key)
        print(f"Uploaded {file_path} to S3 bucket {bucket_name} with key {s3_key}")
        
        # Delete local file after successful upload
        # os.remove(file_path)
        # print(f"Deleted local file {file_path}")
    except Exception as e:
        print(f"Error uploading {file_path} to S3: {e}")

if __name__ == "__main__":
    download_gan_frames(dataset_name="dataset_preprocessed.h5")
    # upload_to_s3(
    #     file_path="gan_frames/frames.npy",
    #     folder="gan_frames",
    #     name="frames.npy",
    #     bucket_name="niranjan-s-personal"
    # )