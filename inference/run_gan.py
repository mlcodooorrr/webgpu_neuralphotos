import torch
import cv2
import numpy as np
from collections import deque
import pygame
from gan import GAN
import os
from datetime import datetime

# ==================== CONFIGURATION ====================
class InferenceConfig:
    device = "cpu"  # IMPORTANT: Use same device as training! (cuda/cpu/mps)
    checkpoint_path = "epoch_99_v2.pth"  # or checkpoints_l1/latest.pth
    
    # Input source - CHOOSE ONE:
    # Option 1: Use .npy file
    # frames_npy_path = "../data/frames/all_frames.npy"
    frames_npy_path = "../gan_frames/frames/all_frames.npy"
    use_npy = True
        
    # Starting frame index
    start_frame_idx = 8000  # Which frame to start from (need at least 32 frames before this)
    
    # Display settings
    display_width = 192 * 3  # Scale up 3x for visibility
    display_height = 256 * 3
    fps = 30
    
    # Movement settings - START STATIONARY
    move_speed = 0.0  # Set to 0 to start stationary
    rotation_speed = 0.0  # Set to 0 to start stationary
    camera_rotation_speed = 0.0  # Set to 0 to start stationary
    
    # Speed increments (for adjusting with +/- keys)
    move_speed_increment = 0.02
    rotation_speed_increment = 0.01

    # Frame saving settings
    save_frames = True
    save_dir = "inference_frames"  # Will create timestamped subdirectory
    save_interval = 1  # Save every N frames (1 = save all frames)


class GameState:
    """Manages the game state (position, orientation, gravity, etc.)"""
    
    def __init__(self):
        # Position and orientation
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.yaw = 0.0  # Rotation around vertical axis
        self.pitch = 0.0  # Camera up/down
        self.roll = 0.0  # Camera tilt (optional)
        
        # Gravity vector (always points down)
        self.gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        
        # Transform matrix (4x4)
        self.transform = np.eye(4, dtype=np.float32)
        
        # Previous state for computing controls
        self.prev_transform = np.eye(4, dtype=np.float32)
        self.prev_gravity = self.gravity.copy()
        self.prev_timestamp = 0.0
        self.timestamp = 0.0
        
    def update_transform(self):
        """Compute 4x4 transform matrix from position and rotation"""
        # Rotation matrix from yaw and pitch
        cy = np.cos(self.yaw)
        sy = np.sin(self.yaw)
        cp = np.cos(self.pitch)
        sp = np.sin(self.pitch)
        
        # Combined rotation matrix (yaw then pitch)
        R = np.array([
            [cy, -sy * cp, sy * sp],
            [sy, cy * cp, -cy * sp],
            [0, sp, cp]
        ], dtype=np.float32)
        
        # Build 4x4 transform
        self.transform = np.eye(4, dtype=np.float32)
        self.transform[:3, :3] = R
        self.transform[:3, 3] = self.position
        
    def compute_controls(self):
        """Compute control vector from current and previous state"""
        rel_transform = np.linalg.inv(self.prev_transform) @ self.transform
        rel_pose_3x4 = rel_transform[:3, :]
        res_pose_flattened = rel_pose_3x4.flatten()
        
        # 🔧 NEW: Add small noise to match training data distribution
        # Training data has small jitter even when "standing still"
        noise_level = 0.001  # Based on training std ~0.0002-0.0072
        res_pose_flattened = res_pose_flattened + np.random.randn(12).astype(np.float32) * noise_level
        
        # Clamp diagonal values to stay near 1.0
        res_pose_flattened[0] = np.clip(res_pose_flattened[0], 0.99, 1.01)
        res_pose_flattened[4] = np.clip(res_pose_flattened[4], 0.99, 1.01)
        res_pose_flattened[8] = np.clip(res_pose_flattened[8], 0.99, 1.01)
        
        # Gravity
        gx, gy, gz = self.gravity
        roll = np.arctan2(gx, gy)
        pitch = np.arctan2(gz, np.sqrt(gx**2 + gy**2))
        
        rel_timestamp = self.timestamp - self.prev_timestamp
        valid_flag = 1.0
        
        controls = np.concatenate([
            res_pose_flattened,
            [roll, pitch],
            [rel_timestamp],
            [valid_flag]
        ]).astype(np.float32)
        
        return controls
    
    def move_forward(self, speed):
        """Move in the direction we're facing"""
        direction = np.array([
            np.cos(self.yaw) * np.cos(self.pitch),
            np.sin(self.yaw) * np.cos(self.pitch),
            np.sin(self.pitch)
        ])
        self.position += direction * speed
        
    def move_backward(self, speed):
        """Move backward"""
        self.move_forward(-speed)
        
    def strafe_left(self, speed):
        """Move left (perpendicular to facing direction)"""
        direction = np.array([
            np.cos(self.yaw - np.pi/2),
            np.sin(self.yaw - np.pi/2),
            0.0
        ])
        self.position += direction * speed
        
    def strafe_right(self, speed):
        """Move right"""
        self.strafe_left(-speed)
        
    def rotate_yaw(self, delta):
        """Rotate camera left/right"""
        self.yaw += delta
        # Keep in [-pi, pi] range
        self.yaw = np.arctan2(np.sin(self.yaw), np.cos(self.yaw))
        
    def rotate_pitch(self, delta):
        """Rotate camera up/down"""
        self.pitch += delta
        # Clamp to avoid flipping
        self.pitch = np.clip(self.pitch, -np.pi/2 + 0.1, np.pi/2 - 0.1)
        
    def save_previous_state(self):
        """Save current state as previous for next frame"""
        self.prev_transform = self.transform.copy()
        self.prev_gravity = self.gravity.copy()
        self.prev_timestamp = self.timestamp


class MemoryBuffer:
    """Maintains the 32-frame history needed for the model"""
    
    def __init__(self, initial_frames=None):
        """
        Initialize with real frames from video/npy
        
        Args:
            initial_frames: numpy array of shape (32, 3, 256, 192) in [-1, 1] range
        """
        self.frames = deque(maxlen=32)
        
        if initial_frames is not None:
            # Load from provided frames
            assert initial_frames.shape == (32, 3, 256, 192), \
                f"Expected (32, 3, 256, 192), got {initial_frames.shape}"
            for i in range(32):
                self.frames.append(initial_frames[i])
            print(f"✅ Initialized memory buffer with {len(self.frames)} real frames")
        else:
            # Fallback: black frames
            for _ in range(32):
                black_frame = -np.ones((3, 256, 192), dtype=np.float32)
                self.frames.append(black_frame)
            print("⚠️ Initialized memory buffer with black frames")
    
    def add_frame(self, frame):
        """Add new frame to history"""
        # frame should be (3, 256, 192) in [-1, 1] range
        self.frames.append(frame)
    
    def get_multiscale_buffers(self):
        """Create multi-scale memory buffers like in training"""
        frames_array = np.array(list(self.frames))  # (32, 3, 256, 192)
        
        scale_0 = frames_array[0:32]   # All 32 frames
        scale_1 = frames_array[16:32]  # Last 16 frames
        scale_2 = frames_array[24:32]  # Last 8 frames
        past_3 = frames_array[28:32]   # Last 4 frames (no resize needed)
        
        # Resize scale_0 to (32, 3, 4, 3)
        resized_0 = []
        for frame in scale_0:
            frame_hwc = frame.transpose(1, 2, 0)  # (H, W, C)
            frame_resized = cv2.resize(frame_hwc, (3, 4), interpolation=cv2.INTER_AREA)
            frame_chw = frame_resized.transpose(2, 0, 1)  # (C, H, W)
            resized_0.append(frame_chw)
        past_0 = np.array(resized_0)
        
        # Resize scale_1 to (16, 3, 16, 12)
        resized_1 = []
        for frame in scale_1:
            frame_hwc = frame.transpose(1, 2, 0)
            frame_resized = cv2.resize(frame_hwc, (12, 16), interpolation=cv2.INTER_AREA)
            frame_chw = frame_resized.transpose(2, 0, 1)
            resized_1.append(frame_chw)
        past_1 = np.array(resized_1)
        
        # Resize scale_2 to (8, 3, 64, 48)
        resized_2 = []
        for frame in scale_2:
            frame_hwc = frame.transpose(1, 2, 0)
            frame_resized = cv2.resize(frame_hwc, (48, 64), interpolation=cv2.INTER_AREA)
            frame_chw = frame_resized.transpose(2, 0, 1)
            resized_2.append(frame_chw)
        past_2 = np.array(resized_2)
        
        return past_0, past_1, past_2, past_3


def load_frames_from_npy(npy_path, start_idx):
    """
    Load 32 frames from .npy file
    
    Args:
        npy_path: Path to .npy file with all frames
        start_idx: Index to start from (must be >= 32)
    
    Returns:
        numpy array of shape (32, 3, 256, 192) in [-1, 1] range
    """
    print(f"Loading frames from {npy_path}...")
    all_frames = np.load(npy_path)  # Shape: (num_frames, 3, height, width)
    
    print(f"  Total frames available: {len(all_frames)}")
    print(f"  Frame shape: {all_frames.shape}")
    
    # Check bounds
    if start_idx < 32:
        print(f"⚠️ start_idx {start_idx} < 32, using idx 32 instead")
        start_idx = 32
    
    if start_idx >= len(all_frames):
        print(f"⚠️ start_idx {start_idx} >= total frames, using last 32 frames")
        start_idx = len(all_frames) - 1
    
    # Extract 32 frames ending at start_idx
    frames = all_frames[start_idx - 32:start_idx].copy()  # (32, 3, H, W)
    
    print(f"✅ Loaded 32 frames from index {start_idx - 32} to {start_idx}")
    print(f"  Frame range: [{frames.min():.3f}, {frames.max():.3f}]")
    print(f"  Frame mean: {frames.mean():.3f}")
    
    return frames.astype(np.float32)


def load_generator(checkpoint_path, device):
    """Load trained generator from checkpoint"""
    print(f"Loading generator from {checkpoint_path}...")
    
    G = GAN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(checkpoint['generator_state_dict'])
    G.eval()
    
    print("✅ Generator loaded successfully")
    return G


def denormalize_frame(frame):
    """Convert from [-1, 1] to [0, 255] uint8 for display"""
    # frame is (3, H, W)
    frame = (frame + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    frame = np.clip(frame, 0, 1)
    frame = (frame * 255).astype(np.uint8)
    frame = frame.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB -> BGR for OpenCV
    return frame


def generate_noise():
    """Generate noise tensors for the model"""
    noise_0 = np.random.uniform(0, 1, size=(1, 4, 3)).astype(np.float32)
    noise_1 = np.random.uniform(0, 1, size=(1, 16, 12)).astype(np.float32)
    noise_2 = np.random.uniform(0, 1, size=(1, 64, 48)).astype(np.float32)
    noise_3 = np.random.uniform(0, 1, size=(1, 256, 192)).astype(np.float32)
    return noise_0, noise_1, noise_2, noise_3


def save_frame_to_disk(frame, frame_count, save_dir, stats=None):
    """
    Save generated frame to disk with diagnostics

    Args:
        frame: numpy array (3, 256, 192) in [-1, 1] range
        frame_count: current frame number
        save_dir: directory to save to
        stats: optional dict with frame statistics
    """
    # Save raw frame (normalized)
    raw_path = os.path.join(save_dir, f"frame_{frame_count:05d}_raw.npy")
    np.save(raw_path, frame)

    # Save denormalized image for visualization
    frame_hwc = frame.transpose(1, 2, 0)  # (H, W, C)
    frame_denorm = (frame_hwc + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    frame_denorm = np.clip(frame_denorm, 0, 1)
    frame_uint8 = (frame_denorm * 255).astype(np.uint8)

    img_path = os.path.join(save_dir, f"frame_{frame_count:05d}.png")
    cv2.imwrite(img_path, cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR))

    # Save statistics if provided
    if stats is not None:
        stats_path = os.path.join(save_dir, f"frame_{frame_count:05d}_stats.txt")
        with open(stats_path, 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")


def run_inference():
    """Main inference loop with interactive controls"""
    config = InferenceConfig()

    # Create save directory if needed
    save_dir = None
    if config.save_frames:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(config.save_dir, f"run_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"💾 Saving frames to: {save_dir}")

    # Initialize pygame for keyboard input
    pygame.init()
    screen = pygame.display.set_mode((config.display_width, config.display_height))
    pygame.display.set_caption("GAN Game Inference - WASD: Move, Arrows: Look, SPACE: Pause, ESC: Quit")
    clock = pygame.time.Clock()
    
    # Load model
    device = torch.device(config.device)
    G = load_generator(config.checkpoint_path, device)
    
    # Load initial frames
    print("\n" + "="*70)
    print("📁 LOADING INITIAL FRAMES")
    print("="*70)
    
    initial_frames = load_frames_from_npy(
        config.frames_npy_path,
        config.start_frame_idx
    )
    
    # STORE initial frames for reset functionality
    initial_frames_backup = initial_frames.copy()

    # Save initial frames if saving is enabled
    if config.save_frames and save_dir is not None:
        print("💾 Saving initial frames for comparison...")
        initial_frames_dir = os.path.join(save_dir, "initial_frames")
        os.makedirs(initial_frames_dir, exist_ok=True)
        for i, frame in enumerate(initial_frames):
            frame_stats = {
                'frame_type': 'initial',
                'index': i,
                'min': f'{frame.min():.6f}',
                'max': f'{frame.max():.6f}',
                'mean': f'{frame.mean():.6f}',
                'std': f'{frame.std():.6f}'
            }
            save_frame_to_disk(frame, i, initial_frames_dir, stats=frame_stats)
        print(f"✅ Saved {len(initial_frames)} initial frames to {initial_frames_dir}")

    # Initialize game state and memory
    state = GameState()
    memory = MemoryBuffer(initial_frames=initial_frames)
    
    # Speed settings (can be adjusted at runtime)
    move_speed = 0.05  # Default movement speed when keys are pressed
    camera_rotation_speed = 0.03  # Default rotation speed when keys are pressed
    
    # Pause state
    paused = False
    
    print("\n" + "="*70)
    print("🎮 CONTROLS:")
    print("  W/S: Move forward/backward")
    print("  A/D: Strafe left/right")
    print("  Arrow Keys: Look around (up/down/left/right)")
    print("  SPACE: Pause/Unpause (stops generation)")
    print("  +/-: Increase/decrease movement speed")
    print("  [/]: Increase/decrease rotation speed")
    print("  R: Reset position and orientation")
    print("  BACKSPACE: Reset to initial frames (full reset)")
    print("  ESC: Quit")
    print("="*70)
    print(f"\n📊 Move speed: {move_speed:.3f}, Rotation speed: {camera_rotation_speed:.3f}")
    print(f"💡 Generator runs continuously, but only moves when you press keys!\n")
    
    running = True
    frame_count = 0
    
    while running:
        # Handle input
        keys = pygame.key.get_pressed()
        
        # Check for quit and special keys
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    
                # Toggle pause with SPACE
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    if paused:
                        print("⏸️  PAUSED (generation stopped)")
                    else:
                        print("▶️  RESUMED (generation running)")
                
                # Adjust movement speed with +/-
                if event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    move_speed += 0.01
                    print(f"📊 Move speed: {move_speed:.3f}")
                if event.key == pygame.K_MINUS:
                    move_speed = max(0.01, move_speed - 0.01)
                    print(f"📊 Move speed: {move_speed:.3f}")
                
                # Adjust rotation speed with [/]
                if event.key == pygame.K_RIGHTBRACKET:
                    camera_rotation_speed += 0.01
                    print(f"📊 Rotation speed: {camera_rotation_speed:.3f}")
                if event.key == pygame.K_LEFTBRACKET:
                    camera_rotation_speed = max(0.01, camera_rotation_speed - 0.01)
                    print(f"📊 Rotation speed: {camera_rotation_speed:.3f}")
                
                # Reset position with R (keep memory)
                if event.key == pygame.K_r:
                    state.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    state.yaw = 0.0
                    state.pitch = 0.0
                    print("🔄 Reset position and orientation (memory preserved)")
                
                # FULL RESET with BACKSPACE (reset memory to initial frames)
                if event.key == pygame.K_BACKSPACE:
                    # Reset game state
                    state.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    state.yaw = 0.0
                    state.pitch = 0.0
                    state.timestamp = 0.0
                    state.prev_timestamp = 0.0
                    state.transform = np.eye(4, dtype=np.float32)
                    state.prev_transform = np.eye(4, dtype=np.float32)
                    
                    # Reset memory buffer to initial frames
                    memory = MemoryBuffer(initial_frames=initial_frames_backup.copy())
                    
                    # Reset frame counter
                    frame_count = 0
                    
                    print("🔄 FULL RESET - Back to initial frames!")
        
        # Track if movement input was given this frame
        movement_input = False
        
        # Only process movement if not paused
        if not paused:
            # Movement (WASD) - only update state if keys pressed
            if keys[pygame.K_w]:
                state.move_forward(move_speed)
                movement_input = True
            if keys[pygame.K_s]:
                state.move_backward(move_speed)
                movement_input = True
            if keys[pygame.K_a]:
                state.strafe_left(move_speed)
                movement_input = True
            if keys[pygame.K_d]:
                state.strafe_right(move_speed)
                movement_input = True
            
            # Camera rotation (Arrow keys)
            if keys[pygame.K_LEFT]:
                state.rotate_yaw(camera_rotation_speed)
                movement_input = True
            if keys[pygame.K_RIGHT]:
                state.rotate_yaw(-camera_rotation_speed)
                movement_input = True
            if keys[pygame.K_UP]:
                state.rotate_pitch(camera_rotation_speed)
                movement_input = True
            if keys[pygame.K_DOWN]:
                state.rotate_pitch(-camera_rotation_speed)
                movement_input = True
            
            # ALWAYS update timestamp and generate (even if standing still)
            state.timestamp += 1.0 / config.fps
            
            # Update transform matrix (may be same as previous if no movement)
            state.update_transform()

            # Compute controls (will show zero movement if standing still)
            controls = state.compute_controls()
            
            # Get memory buffers
            past_0, past_1, past_2, past_3 = memory.get_multiscale_buffers()
            
            # Generate noise
            noise_0, noise_1, noise_2, noise_3 = generate_noise()
            
            # Convert to tensors
            controls_tensor = torch.from_numpy(controls).unsqueeze(0).to(device)
            past_0_tensor = torch.from_numpy(past_0).unsqueeze(0).to(device)
            past_1_tensor = torch.from_numpy(past_1).unsqueeze(0).to(device)
            past_2_tensor = torch.from_numpy(past_2).unsqueeze(0).to(device)
            past_3_tensor = torch.from_numpy(past_3).unsqueeze(0).to(device)
            noise_0_tensor = torch.from_numpy(noise_0).unsqueeze(0).to(device)
            noise_1_tensor = torch.from_numpy(noise_1).unsqueeze(0).to(device)
            noise_2_tensor = torch.from_numpy(noise_2).unsqueeze(0).to(device)
            noise_3_tensor = torch.from_numpy(noise_3).unsqueeze(0).to(device)
            
            # ALWAYS generate frame (even when standing still)
            with torch.no_grad():
                generated_frame = G(
                    controls_tensor,
                    past_0_tensor,
                    past_1_tensor,
                    past_2_tensor,
                    past_3_tensor,
                    noise_0_tensor,
                    noise_1_tensor,
                    noise_2_tensor,
                    noise_3_tensor
                )
            
            # Convert to numpy
            generated_frame = generated_frame.squeeze(0).cpu().numpy()

            # Collect frame statistics
            frame_stats = {
                'frame_count': frame_count,
                'min': f'{generated_frame.min():.6f}',
                'max': f'{generated_frame.max():.6f}',
                'mean': f'{generated_frame.mean():.6f}',
                'std': f'{generated_frame.std():.6f}',
                'position': f'{state.position}',
                'yaw': f'{np.degrees(state.yaw):.2f}',
                'pitch': f'{np.degrees(state.pitch):.2f}',
                'moving': str(movement_input)
            }

            # Print diagnostics
            print(f"Frame {frame_count} - Range: [{generated_frame.min():.3f}, {generated_frame.max():.3f}], "
                  f"Mean: {generated_frame.mean():.3f}, Std: {generated_frame.std():.3f}")

            # Save frame to disk if enabled
            if config.save_frames and save_dir is not None:
                if frame_count % config.save_interval == 0:
                    save_frame_to_disk(generated_frame, frame_count, save_dir, stats=frame_stats)

            # ALWAYS add to memory (even if standing still)
            memory.add_frame(generated_frame)
            state.save_previous_state()

            frame_count += 1
            
            # Print status
            if frame_count % 30 == 0:
                status_msg = "MOVING" if movement_input else "STANDING STILL"
                print(f"Frame {frame_count} [{status_msg}] | Pos: ({state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f}) | Yaw: {np.degrees(state.yaw):.1f}°")
        else:
            # When paused, use the last frame from memory
            generated_frame = np.array(list(memory.frames)[-1])
        
        # Denormalize for display
        display_frame = denormalize_frame(generated_frame)
        
        # Resize for display
        display_frame = cv2.resize(
            display_frame, 
            (config.display_width, config.display_height),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Convert to pygame surface
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        display_frame_rgb = np.rot90(display_frame_rgb)
        display_frame_rgb = np.flipud(display_frame_rgb)
        surface = pygame.surfarray.make_surface(display_frame_rgb)
        
        # Draw to screen
        screen.blit(surface, (0, 0))
        
        # Add info overlay
        font = pygame.font.Font(None, 24)
        
        # Show current state
        if paused:
            status = "⏸️  PAUSED"
            status_color = (255, 255, 0)  # Yellow
        elif movement_input:
            status = "🎮 MOVING"
            status_color = (0, 255, 0)  # Green
        else:
            status = "🧍 STANDING STILL"
            status_color = (200, 200, 200)  # Gray
        
        info_text = [
            status,
            f"Frames Generated: {frame_count}",
            f"Pos: ({state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f})",
            f"Yaw: {np.degrees(state.yaw):.1f}° Pitch: {np.degrees(state.pitch):.1f}°",
            f"Move Speed: {move_speed:.3f} | Rot Speed: {camera_rotation_speed:.3f}",
            f"Display FPS: {clock.get_fps():.1f}",
            f"Press BACKSPACE to reset"
        ]
        
        y_offset = 10
        for i, text in enumerate(info_text):
            # Use status color for first line, white for rest
            color = status_color if i == 0 else (255, 255, 255)
            
            text_surface = font.render(text, True, color)
            # Add black background for readability
            bg_rect = text_surface.get_rect()
            bg_rect.topleft = (10, y_offset)
            bg_surface = pygame.Surface((bg_rect.width + 10, bg_rect.height + 4))
            bg_surface.fill((0, 0, 0))
            bg_surface.set_alpha(128)
            screen.blit(bg_surface, (5, y_offset - 2))
            screen.blit(text_surface, (10, y_offset))
            y_offset += 25
        
        pygame.display.flip()
        clock.tick(config.fps)
    
    pygame.quit()
    print(f"\n✅ Inference completed - Generated {frame_count} frames")

    if config.save_frames and save_dir is not None:
        print(f"💾 Frames saved to: {save_dir}")
        print(f"   - Initial frames: {os.path.join(save_dir, 'initial_frames')}")
        print(f"   - Generated frames: {save_dir}")
        print(f"   - Total frames saved: {(frame_count // config.save_interval) + 32}")


if __name__ == "__main__":
    run_inference()