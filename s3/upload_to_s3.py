import os
from conductor_manager import ConductorManager
import sys
import gzip
import yaml
import argparse

from download_from_s3 import upload_to_s3

# Parameters
# -----------------------------------------------------------------------------
conductor_manager = ConductorManager()
bucket_name = "niranjan-s-personal"

# Load wandb name from trainer.yaml
# with open("config/trainer.yaml", "r") as f:
#     config = yaml.safe_load(f)
# wandb_name = config["wandb"]["name"]
# data_path = f"flappy_models_test/{wandb_name}"
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # # Parse command line arguments
    # parser = argparse.ArgumentParser(description='Upload files to S3')
    # parser.add_argument('--resume', type=str, default='False', help='Resume from /root/diamond/checkpoints')
    # args = parser.parse_args()
    
    # resume_mode = args.resume.lower() == 'true'
    
    # if resume_mode:
    #     # Resume mode: use checkpoints folder, use wandb_name_resume folder for resume mode
    #     base_dir = "checkpoints"
    #     local_dir = os.path.join(base_dir, "agent_versions")
    #     checkpoints_dir = base_dir
    # else:
    #     # Normal mode: find the pt file in the checkpoint directory
    #     base_dir = "outputs"
    #     date_folder = max(os.listdir(base_dir))
    #     date_path = os.path.join(base_dir, date_folder)
    #     # Get the most recent time folder
    #     time_folder = max(os.listdir(date_path))
    #     # Construct the full path
    #     local_dir = os.path.join(date_path, time_folder, "checkpoints", "agent_versions")
    #     checkpoints_dir = os.path.join(date_path, time_folder, "checkpoints")

    # # Loop through all files in the agent_versions directory
    
    # # Also upload state.pt file from checkpoints directory
