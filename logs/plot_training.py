import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_training_curves(name='', log_file='training_log.json', show_batches=True, max_loss=None):
    """
    Plot training curves with better handling of NaN/Inf values
    
    Args:
        log_file: path to training log JSON
        show_batches: if True, plot batch-level data, else epoch-level
        max_loss: if set, clip losses above this value for better visualization
    """
    log_file = f"{name}/{log_file}"

    # Load log
    with open(log_file, 'r') as f:
        history = json.load(f)
        
    plot_batch_curves(history, name=name, max_loss=max_loss)

def plot_batch_curves(history, name, max_loss=None):
    """Plot batch-level training curves"""
    batch_data = history['train_batches']
    
    if not batch_data:
        print("No batch data found!")
        return
    
    # Extract data
    global_steps = [d['global_step'] for d in batch_data]
    d_loss = [d['d_loss'] for d in batch_data]
    g_adv = [d['g_adv_loss'] for d in batch_data]
    g_l1 = [d['g_l1_loss'] for d in batch_data]
    g_total = [d['g_total_loss'] for d in batch_data]
    
    # Check if gradient penalty is logged
    has_gp = 'd_gp' in batch_data[0]
    if has_gp:
        d_gp = [d.get('d_gp', 0) for d in batch_data]
    
    # Create figure
    nrows = 3 if has_gp else 2
    fig, axes = plt.subplots(nrows, 2, figsize=(16, 5*nrows))
    fig.suptitle('GAN Training Curves (Batch Level)', fontsize=16)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Plot 1: Discriminator Loss
    axes[0].plot(global_steps, d_loss, alpha=0.3, linewidth=0.5, color='blue')
    # Add moving average
    window = 2
    if len(d_loss) > window:
        valid_mask = ~np.isnan(d_loss)
        if np.sum(valid_mask) > window:
            moving_avg = np.convolve(np.where(valid_mask, d_loss, 0), 
                                    np.ones(window)/window, mode='valid')
            axes[0].plot(global_steps[window-1:window-1+len(moving_avg)], 
                        moving_avg, linewidth=2, label='Moving Avg (2)', color='darkblue')
    axes[0].set_xlabel('Global Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Discriminator Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)  # Force start at 0
    
    # Plot 2: Generator Adversarial Loss
    axes[1].plot(global_steps, g_adv, alpha=0.3, linewidth=0.5, color='green')
    if len(g_adv) > window:
        valid_mask = ~np.isnan(g_adv)
        if np.sum(valid_mask) > window:
            moving_avg = np.convolve(np.where(valid_mask, g_adv, 0), 
                                    np.ones(window)/window, mode='valid')
            axes[1].plot(global_steps[window-1:window-1+len(moving_avg)], 
                        moving_avg, linewidth=2, label='Moving Avg (2)', color='darkgreen')
    axes[1].set_xlabel('Global Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Generator Adversarial Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Generator L1 Loss
    axes[2].plot(global_steps, g_l1, alpha=0.3, linewidth=0.5, color='orange')
    if len(g_l1) > window:
        valid_mask = ~np.isnan(g_l1)
        if np.sum(valid_mask) > window:
            moving_avg = np.convolve(np.where(valid_mask, g_l1, 0), 
                                    np.ones(window)/window, mode='valid')
            axes[2].plot(global_steps[window-1:window-1+len(moving_avg)], 
                        moving_avg, linewidth=2, label='Moving Avg (2)', color='darkorange')
    axes[2].set_xlabel('Global Step')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Generator L1 Reconstruction Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(bottom=0)
    
    # Plot 4: Generator Total Loss
    axes[3].plot(global_steps, g_total, alpha=0.3, linewidth=0.5, color='red')
    if len(g_total) > window:
        valid_mask = ~np.isnan(g_total)
        if np.sum(valid_mask) > window:
            moving_avg = np.convolve(np.where(valid_mask, g_total, 0), 
                                    np.ones(window)/window, mode='valid')
            axes[3].plot(global_steps[window-1:window-1+len(moving_avg)], 
                        moving_avg, linewidth=2, label='Moving Avg (2)', color='darkred')
    axes[3].set_xlabel('Global Step')
    axes[3].set_ylabel('Loss')
    axes[3].set_title('Generator Total Loss')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # Plot 5: Gradient Penalty (if available)
    if has_gp:
        axes[4].plot(global_steps, d_gp, alpha=0.3, linewidth=0.5, color='purple')
        if len(d_gp) > window:
            valid_mask = ~np.isnan(d_gp)
            if np.sum(valid_mask) > window:
                moving_avg = np.convolve(np.where(valid_mask, d_gp, 0), 
                                        np.ones(window)/window, mode='valid')
                axes[4].plot(global_steps[window-1:window-1+len(moving_avg)], 
                            moving_avg, linewidth=2, label='Moving Avg (2)', color='indigo')
        axes[4].set_xlabel('Global Step')
        axes[4].set_ylabel('Penalty')
        axes[4].set_title('Gradient Penalty')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
        axes[4].set_ylim(bottom=0)
        
        # Plot 6: Loss stability indicator (std deviation over window)
        window_std = 100
        if len(d_loss) > window_std:
            d_std = []
            g_std = []
            for i in range(window_std, len(d_loss)):
                d_window = d_loss[i-window_std:i]
                g_window = g_total[i-window_std:i]
                d_std.append(np.nanstd(d_window))
                g_std.append(np.nanstd(g_window))
            
            steps_for_std = global_steps[window_std:]
            axes[5].plot(steps_for_std, d_std, label='D Loss Std', alpha=0.7)
            axes[5].plot(steps_for_std, g_std, label='G Loss Std', alpha=0.7)
            axes[5].set_xlabel('Global Step')
            axes[5].set_ylabel('Std Dev')
            axes[5].set_title(f'Training Stability (Std over {window_std} batches)')
            axes[5].legend()
            axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save with timestamp to avoid overwriting
    filename = f'{name}/training_curves_batch.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"📊 Saved batch-level plot to {filename}")
    plt.show()

if __name__ == "__main__":
    # Plot with max loss clipping to handle extreme values
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="", required=True)
    args = parser.parse_args()

    file_dir = f"training_runs/{args.name}"
    log_file = "training_log.json"
    plot_training_curves(name=file_dir, log_file=log_file, show_batches=True, max_loss=1000)
