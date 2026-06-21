import torch
import random
import numpy as np
import os
import json
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_results(history, test_metrics, config):
    os.makedirs(config['experiment']['output_dir'], exist_ok=True)
    
    # Save metrics
    with open(f"{config['experiment']['output_dir']}/metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].legend()
    axes[0].set_title("Loss")
    axes[0].grid(True)
    
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].legend()
    axes[1].set_title("Accuracy")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{config['experiment']['output_dir']}/training_history.png", dpi=300)
    plt.close()
