import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def load_dataset(dataset_name="BBBP", root="./data"):
    dataset = MoleculeNet(root=root, name=dataset_name)
    return dataset

def scaffold_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)
    
    # Extract SMILES
    smiles_list = []
    valid_indices = []
    for i, data in enumerate(dataset):
        if hasattr(data, 'smiles') and data.smiles is not None:
            smiles_list.append(data.smiles)
            valid_indices.append(i)
    
    # Group by scaffold
    scaffold_map = defaultdict(list)
    for idx, smi in zip(valid_indices, smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            if scaffold is not None:
                scaffold_map[scaffold].append(idx)
    
    groups = list(scaffold_map.values())
    random.shuffle(groups)
    
    train_idx, val_idx, test_idx = [], [], []
    for g in groups:
        if len(train_idx) < train_ratio * len(valid_indices):
            train_idx.extend(g)
        elif len(val_idx) < val_ratio * len(valid_indices):
            val_idx.extend(g)
        else:
            test_idx.extend(g)
    
    return train_idx, val_idx, test_idx

def create_loaders(dataset, train_idx, val_idx, test_idx, batch_size=32):
    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]
    test_dataset = [dataset[i] for i in test_idx]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
