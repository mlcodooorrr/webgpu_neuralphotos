import numpy as np
import torch
import cv2
import random
from itertools import cycle
import h5py
from collections import deque
from torch.utils.data._utils import pin_memory
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', message='.*NumPy array is not writable.*')

class GAN_Dataset(Dataset):
    """Pytorch Dataset for GAN training"""

    def __init__(self, split="train", train_ratio=0.9, seed=42):

        self.metadata = np.load("gan_frames/metadata.npy", allow_pickle=True).item()
        num_frames = self.metadata["num_frames"]

        # Memory-map all arrays (read-only, no memory copy!)
        self.frames = np.load("gan_frames/frames.npy", mmap_mode='r')
        self.scale_0 = np.load("gan_frames/scale_0.npy", mmap_mode='r')
        self.scale_1 = np.load("gan_frames/scale_1.npy", mmap_mode='r')
        self.scale_2 = np.load("gan_frames/scale_2.npy", mmap_mode='r')
        self.controls = np.load("gan_frames/controls.npy", mmap_mode='r')

        # Get valid frame indices
        start_frame = 32
        all_indices = list(range(start_frame, num_frames-1))

        random.Random(seed).shuffle(all_indices)
        split_point = int(len(all_indices) * train_ratio)

        if split == "train":
            self.frame_indices = all_indices[:split_point]
        else:
            self.frame_indices = all_indices[split_point:]
        
        print(f"{split} set: {len(self.frame_indices)} samples")
    
    def __len__(self):
        """return total number of samples"""
        return len(self.frame_indices)

    def __getitem__(self, idx):
        """Load one sample - called by workers in parallel"""
        frame_idx = self.frame_indices[idx]
        sample_idx = frame_idx - 32

        # return as dict of pytorch tensors
        result = {
            'controls': torch.from_numpy(self.controls[sample_idx].copy()), 
            'past_0': torch.from_numpy(self.scale_0[sample_idx].copy()),
            'past_1': torch.from_numpy(self.scale_1[sample_idx].copy()),
            'past_2': torch.from_numpy(self.scale_2[sample_idx].copy()),
            'past_3': torch.from_numpy(self.frames[frame_idx - 4:frame_idx].copy()),
            'target': torch.from_numpy(self.frames[frame_idx].copy())
        }
        return result

def collate_fn(batch):
    """
        custom collate function to sperate inputs and targets

        input: batch = [sample1, sample2, ..., sampleB]
        where each sample is a dict from __getitem__

        output: (inputs_dict, targets_tensor)
    """

    inputs = {
        'controls': torch.stack([b['controls'] for b in batch]),
        'past_0': torch.stack([b['past_0'] for b in batch]),
        'past_1': torch.stack([b['past_1'] for b in batch]),
        'past_2': torch.stack([b['past_2'] for b in batch]),
        'past_3': torch.stack([b['past_3'] for b in batch]),
    }
    targets = torch.stack([b['target'] for b in batch])
    return inputs, targets


def distributed_data_loader(B, split, device="cuda", train_ratio=0.9, seed=42):
    """
        Create dataloader with prefetching and parallel workers

        args:
            B: Batch size
            split: train of val
            device: training device, pin_memory depends on this
            train_ratio: train/val split ratio
            seed: random seed for reproducible splits
        
        returns:
        pytorch dataloader - this time it's optimized
    """

    dataset = GAN_Dataset(split=split, train_ratio=train_ratio, seed=seed)

    return DataLoader(
        dataset,
        batch_size=B,
        shuffle=(split == "train"),         # shuffle training data
        num_workers=8,                      # 4 parallel workers loading data 
        pin_memory=False,        # pin memory for fast gpu transfer
        collate_fn=collate_fn,              # custom batching function
        persistent_workers=True,            # keep workers alive between epochs
        prefetch_factor=2                   # each worker prefetches 2 batches
    )

# Helper functions
def compute_controls(curr_transform, prev_transform, curr_gravity, curr_timestamp, prev_timestamp):
    "compute relative controls for model"
    # compute relative_transform
    rel_transform = np.linalg.inv(prev_transform) @ curr_transform
    rel_pose_3x4 = rel_transform[:3, :] # (3, 4)
    res_pose_flattened = rel_pose_3x4.flatten()

    # compute gravity vector
    gx, gy, gz = curr_gravity[0], curr_gravity[1], curr_gravity[2]

    # compute roll (rotation around forward axis, left/right tilt)
    roll = np.arctan2(gx, gy)

    # compute pitch (rotation around side axis, up/down tilt)
    pitch = np.arctan2(gz, np.sqrt(gx**2 + gy**2))

    rel_timestamp = curr_timestamp - prev_timestamp

    # TODO: when dealing with augmented data, create logic to handle this
    valid_flag = 1.0
    
    controls = np.concatenate([
        res_pose_flattened,
        [roll, pitch],
        [rel_timestamp],
        [valid_flag]
    ], dtype=np.float32)

    return controls

def generate_noise():
    # create noise tensors at same spatial scales as memory buffers -> 4 U(0,1) single channel noise tensors at each spatial scale
    noise_0 = np.random.uniform(0,1, size=(1, 4, 3)).astype(np.float32)
    noise_1 = np.random.uniform(0,1, size=(1, 16, 12)).astype(np.float32)
    noise_2 = np.random.uniform(0,1, size=(1, 64, 48)).astype(np.float32)
    noise_3 = np.random.uniform(0,1, size=(1, 256, 192)).astype(np.float32)

    return noise_0, noise_1, noise_2, noise_3