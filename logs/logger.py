import json
import os
from datetime import datetime

class TrainingLogger:
    """Logs training metrics at both batch and epoch level"""
    def __init__(self, log_file):
        self.log_file = log_file
        
        # Default structure
        default_history = {
            'train_epochs': [],
            'val_epochs': [],
            'train_batches': [],
            'config': {},
            'start_time': datetime.now().isoformat()
        }
        
        # Load existing log if it exists
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    loaded_history = json.load(f)
                # Start with defaults, then update with loaded data
                self.history = default_history.copy()
                self.history.update(loaded_history)
                # Ensure all required keys exist
                for key in default_history:
                    if key not in self.history:
                        self.history[key] = default_history[key]
                print(f"📂 Loaded existing log from {log_file}")
            except Exception as e:
                print(f"⚠️  Could not load {log_file}: {e}, starting fresh")
                self.history = default_history
        else:
            self.history = default_history
    
    def log_config(self, config):
        """Save training configuration"""
        self.history['config'] = {
            'batch_size': config.batch_size,
            'lr_g': config.lr_g,
            # 'lr_d': config.lr_d,
            # 'lambda_l1': config.lambda_l1,
            # 'loss_type': config.loss_type,
            'num_epochs': config.num_epochs,
            'batches_per_epoch': config.batches_per_epoch,
        }
        self.save()
    
    def log_batch(self, epoch, batch_idx, metrics):
        """Log metrics for a single batch"""
        batches_per_epoch = self.history['config'].get('batches_per_epoch', 1000)
        entry = {
            'epoch': epoch,
            'batch': batch_idx,
            'global_step': epoch * batches_per_epoch + batch_idx,
            **metrics
        }
        self.history['train_batches'].append(entry)
        
        # Only save every 10 batches to avoid slowdown
        if batch_idx % 10 == 0:
            self.save()
    
    def log_train_epoch(self, epoch, metrics):
        """Log training metrics for an epoch"""
        entry = {'epoch': epoch, **metrics}
        self.history['train_epochs'].append(entry)
        self.save()
    
    def log_val(self, epoch, metrics):
        """Log validation metrics for an epoch"""
        entry = {'epoch': epoch, **metrics}
        self.history['val_epochs'].append(entry)
        self.save()
    
    def save(self):
        """Save log to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)