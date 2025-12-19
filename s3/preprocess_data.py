import h5py
import numpy as np
import cv2
from tqdm import tqdm
from dataloader import compute_controls

def preprocess_all_frames():
    """Pre-compute all multi-scale buffers and save to new HDF5 file"""
    
    # Open original dataset
    f_in = h5py.File("gan_frames/dataset_2.h5", "r")
    num_frames = f_in.attrs["num_frames"]
    
    # Create new dataset for preprocessed data
    f_out = h5py.File("gan_frames/dataset_2_preprocessed.h5", "w")
    
    # Copy original frames
    print("Copying original frames...")
    f_out.create_dataset("frames", data=f_in["frames"][:], dtype=np.float16)
    f_out.create_dataset("poses", data=f_in["poses"][:]) # Keep float32
    f_out.create_dataset("gravity", data=f_in["gravity"][:]) # Keep float32
    f_out.create_dataset("timestamps", data=f_in["timestamps"][:])  # Keep float32
    f_out.attrs["num_frames"] = num_frames  
    f_out.attrs["fps"] = f_in.attrs["fps"]
    
    # Pre-compute resized buffers for each valid index
    start_frame = 32
    valid_indices = list(range(start_frame, num_frames - 1))
    
    print(f"Pre-computing multi-scale buffers for {len(valid_indices)} samples...")
    
    # Create datasets for pre-computed scales
    # Scale 0: (num_samples, 32, 3, 4, 3)
    # Scale 1: (num_samples, 16, 3, 16, 12)
    # Scale 2: (num_samples, 8, 3, 64, 48)
    # Scale 3 doesn't need resizing
    
    scale_0_data = f_out.create_dataset(
        "scale_0", 
        shape=(len(valid_indices), 32, 3, 4, 3),
        dtype=np.float16,
        chunks=(1, 32, 3, 4, 3),
    )
    
    scale_1_data = f_out.create_dataset(
        "scale_1",
        shape=(len(valid_indices), 16, 3, 16, 12),
        dtype=np.float16,
        chunks=(1, 16, 3, 16, 12),
    )
    
    scale_2_data = f_out.create_dataset(
        "scale_2",
        shape=(len(valid_indices), 8, 3, 64, 48),
        dtype=np.float16,
        chunks=(1, 8, 3, 64, 48),
    )
    
    # Process each sample
    for sample_idx, frame_idx in enumerate(tqdm(valid_indices)):
        # Load past 32 frames
        past_frames = f_in["frames"][frame_idx - 32:frame_idx]
        
        # Compute scale 0 (32 frames -> 4x3)
        scale_0 = past_frames[0:32]
        resized_0 = []
        for frame in scale_0:
            frame_float32 = frame.astype(np.float32)
            frame_hwc = frame_float32.transpose(1, 2, 0)
            frame_resized = cv2.resize(frame_hwc, (3, 4), interpolation=cv2.INTER_AREA)
            frame_chw = frame_resized.transpose(2, 0, 1)
            resized_0.append(frame_chw)
        scale_0_data[sample_idx] = np.array(resized_0, dtype=np.float16)
        
        # Compute scale 1 (16 frames -> 16x12)
        scale_1 = past_frames[16:32]
        resized_1 = []
        for frame in scale_1:
            frame_float32 = frame.astype(np.float32)
            frame_hwc = frame_float32.transpose(1, 2, 0)
            frame_resized = cv2.resize(frame_hwc, (12, 16), interpolation=cv2.INTER_AREA)
            frame_chw = frame_resized.transpose(2, 0, 1)
            resized_1.append(frame_chw)
        scale_1_data[sample_idx] = np.array(resized_1, dtype=np.float16)
        
        # Compute scale 2 (8 frames -> 64x48)
        scale_2 = past_frames[24:32]
        resized_2 = []
        for frame in scale_2:
            frame_float32 = frame.astype(np.float32)
            frame_hwc = frame_float32.transpose(1, 2, 0)
            frame_resized = cv2.resize(frame_hwc, (48, 64), interpolation=cv2.INTER_AREA)
            frame_chw = frame_resized.transpose(2, 0, 1)
            resized_2.append(frame_chw)
        scale_2_data[sample_idx] = np.array(resized_2, dtype=np.float16)
    
    f_in.close()
    f_out.close()
    
    print(f"✅ Preprocessing complete! Saved to gan_frames/dataset_2_preprocessed.h5")

def convert_hdf5_to_npy():
    """Convert preprocessed HDF5 to memory-mapped numpy files"""
    
    print("Opening HDF5 file...")
    f = h5py.File("gan_frames/dataset_preprocessed.h5", "r")
    # f = h5py.File("gan_frames/dataset_2_preprocessed.h5", "r")
    
    # Save each dataset as separate .npy file
    print("Converting frames...")
    np.save("gan_frames/frames.npy", f["frames"][:])
    
    print("Converting scale_0...")
    np.save("gan_frames/scale_0.npy", f["scale_0"][:])
    
    print("Converting scale_1...")
    np.save("gan_frames/scale_1.npy", f["scale_1"][:])
    
    print("Converting scale_2...")
    np.save("gan_frames/scale_2.npy", f["scale_2"][:])
    
    print("Converting poses...")
    np.save("gan_frames/poses.npy", f["poses"][:])
    
    print("Converting gravity...")
    np.save("gan_frames/gravity.npy", f["gravity"][:])
    
    print("Converting timestamps...")
    np.save("gan_frames/timestamps.npy", f["timestamps"][:])

    # PRE-COMPUTE ALL CONTROLS
    print("Pre-computing controls...")
    num_frames = f.attrs["num_frames"]
    start_frame = 32
    valid_indices = list(range(start_frame, num_frames - 1))

    all_controls = []
    for frame_idx in tqdm(valid_indices, desc="Computing controls"):
        curr_transform = f["poses"][frame_idx]
        prev_transform = f["poses"][frame_idx - 1]
        curr_gravity = f["gravity"][frame_idx]
        curr_timestamp = f["timestamps"][frame_idx]
        prev_timestamp = f["timestamps"][frame_idx - 1]
        
        controls = compute_controls(
            curr_transform, prev_transform, curr_gravity,
            curr_timestamp, prev_timestamp
        )
        all_controls.append(controls)
    np.save("gan_frames/controls.npy", np.array(all_controls, dtype=np.float32))
    
    # Save metadata as separate file
    metadata = {
        'num_frames': int(f.attrs["num_frames"]),
        'fps': float(f.attrs["fps"])
    }
    np.save("gan_frames/metadata.npy", metadata)
    
    f.close()
    print("✅ Conversion complete!")

if __name__ == "__main__":
    # preprocess_all_frames()
    convert_hdf5_to_npy()