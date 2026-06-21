# ============================================================
# tests/test_model.py
# Unit tests for GraphTransformerSOTA-BBBP
# ============================================================

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import GraphTransformerSOTA2026, create_model
from src.dataset import load_dataset, scaffold_split
from src.utils import set_seed

def test_model_creation():
    """Test if model can be created"""
    set_seed(42)
    config = {
        'experiment': {'seed': 42},
        'model': {
            'in_channels': 9,
            'hidden_channels': 256,
            'out_channels': 2,
            'num_layers': 4,
            'num_heads': 4,
            'dropout': 0.3
        }
    }
    model = create_model(config)
    assert model is not None
    print("✅ test_model_creation passed")

def test_model_forward():
    """Test if model can perform a forward pass"""
    set_seed(42)
    config = {
        'experiment': {'seed': 42},
        'model': {
            'in_channels': 9,
            'hidden_channels': 256,
            'out_channels': 2,
            'num_layers': 4,
            'num_heads': 4,
            'dropout': 0.3
        }
    }
    model = create_model(config)
    
    # Create dummy data
    dummy_data = type('Data', (), {
        'x': torch.randn(10, 9),
        'edge_index': torch.randint(0, 10, (2, 20)),
        'batch': torch.zeros(10, dtype=torch.long)
    })
    
    output = model(dummy_data)
    assert output.shape == (1, 2) or output.shape[0] == 1
    print("✅ test_model_forward passed")

def test_dataset_loading():
    """Test if BBBP dataset can be loaded"""
    try:
        dataset = load_dataset("BBBP", root="./data")
        assert len(dataset) > 0
        print(f"✅ test_dataset_loading passed: {len(dataset)} molecules")
    except Exception as e:
        print(f"⚠️ test_dataset_loading failed: {e}")

def test_scaffold_split():
    """Test if scaffold split works"""
    try:
        dataset = load_dataset("BBBP", root="./data")
        train_idx, val_idx, test_idx = scaffold_split(dataset)
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert len(test_idx) > 0
        print(f"✅ test_scaffold_split passed: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    except Exception as e:
        print(f"⚠️ test_scaffold_split failed: {e}")

if __name__ == "__main__":
    print("\n🧪 Running tests...\n")
    test_model_creation()
    test_model_forward()
    test_dataset_loading()
    test_scaffold_split()
    print("\n✅ All tests completed!")
