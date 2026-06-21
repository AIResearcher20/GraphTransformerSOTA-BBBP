#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from src.model import create_model
from src.dataset import load_dataset, scaffold_split, create_loaders
from src.train import train, evaluate
from src.utils import set_seed, save_results

def main():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config['experiment']['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {device}")
    
    # Load dataset
    print(f"📦 Loading {config['data']['dataset_name']}...")
    dataset = load_dataset(config['data']['dataset_name'], config['data']['root'])
    
    # Scaffold split
    train_idx, val_idx, test_idx = scaffold_split(
        dataset,
        train_ratio=1 - config['data']['val_size'] - config['data']['test_size'],
        val_ratio=config['data']['val_size']
    )
    
    # DataLoaders
    train_loader, val_loader, test_loader = create_loaders(
        dataset, train_idx, val_idx, test_idx,
        config['data']['batch_size']
    )
    
    print(f"📊 Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Model
    model = create_model(config).to(device)
    
    # Train
    print("\n🚀 Starting training...")
    model, history = train(model, train_loader, val_loader, config, device)
    
    # Evaluate
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, test_f1, test_auc = evaluate(model, test_loader, device, criterion)
    
    print(f"\n📊 Final Test Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1-Score: {test_f1:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    
    # Save results
    metrics = {"accuracy": test_acc, "f1": test_f1, "auc": test_auc}
    save_results(history, metrics, config)

if __name__ == "__main__":
    main()
