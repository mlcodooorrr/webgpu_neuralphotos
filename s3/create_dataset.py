import cv2
import os
import numpy as np
from tqdm import tqdm
import json
import h5py


# split the video in to frames
# normalize the frames
# save frames into a folder
# name frames as frame_00000.jpg, frame_00001.jpg, frame_00002.jpg, etc. do it based on length of frames
# {
#     "frame_number": 
#     "past_32_frames":
#     "previous_timestep"
#     "current_timestep"
#     "valid_bit"
#     "previous_gravity"
#     "curr_gravity"
#     "previous_transform"
#     "curr_transform"
# }

# create dataset:
# calculate relative poses etc.
# generate fresh noise per batch

WIDTH = 192
HEIGHT = 256
def normalize_video_frames(video_file, frames_file) -> None:
    # Open the video file
    cap = cv2.VideoCapture(video_file)
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get the frame rate
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    # Get the width and height of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Total frames: {total_frames}")
    print(f"Frame rate: {frame_rate}")
    print(f"Width: {width}")
    print(f"Height: {height}")

    # Create a folder to save the frames
    os.makedirs("frames", exist_ok=True)

    # shape: (num_frames, channels, height, width)
    all_frames = np.zeros((total_frames, 3, HEIGHT, WIDTH), dtype=np.float32)

    for i in tqdm(range(total_frames), desc="Extracting frames, normalizing frames [-1,1] into float32 tensor (num_channels, height, width)"):
        ret, frame = cap.read()
        if not ret:
            break

        # convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # resize to width=192, height=256
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        
        # normalize to [-1, 1]
        frame = frame.astype(np.float32)
        frame = (frame - 127.5) / 127.5

        # transpose to channels, height, width
        frame = frame.transpose(2, 0, 1)

        all_frames[i] = frame

    # save all frames in a single npy file
    np.save(frames_file, all_frames)
    # Close the video file
    cap.release()

def load_poses(poses_file) -> None:
    with open(poses_file, 'r') as f:
        data = json.load(f)
    # Iterate through the list of pose dictionaries, extract each element for each key, and collect them in lists
    poses_list = []
    timestamp_list = []
    gravity_list = []

    for entry in data["poses"]:
        poses_list.append(entry["transform"])
        timestamp_list.append(entry["timestamp"])
        gravity_list.append(entry["gravity"])
    return poses_list, timestamp_list, gravity_list

def create_h5_dataset(frames_file, poses_file, dataset_file) -> None:
    frames = np.load(frames_file)
    poses, timestamps, gravity = load_poses(poses_file)

    frames = frames.astype(np.float16)  # Half the size!
    poses = np.array(poses, dtype=np.float32)        # (28031, 4, 4)
    gravity = np.array(gravity, dtype=np.float32)    # (28031, 3)
    timestamps = np.array(timestamps, dtype=np.float32)  # (28031,)

    print(f"Frames: {len(frames)}, Poses: {len(poses)}, Gravity: {len(gravity)}, Timestamps: {len(timestamps)}")
    print("creating dataset...")

    with h5py.File(dataset_file, "w") as f:
        f.create_dataset("frames", data=frames)
        f.create_dataset("poses", data=poses)
        f.create_dataset("gravity", data=gravity)
        f.create_dataset("timestamps", data=timestamps)

        f.attrs["num_frames"] = len(frames)
        f.attrs['fps'] = 30

if __name__ == "__main__":
    video_file = "data/video_1762319787.140873.mp4"
    poses_file = "gan_frames/poses_1762320721.606367.json"
    frames_file = "gan_frames/all_frames.npy"
    dataset_file = "gan_frames/dataset_2.h5"

    # normalize_video_frames(video_file, frames_file)
    create_h5_dataset(frames_file, poses_file, dataset_file)

