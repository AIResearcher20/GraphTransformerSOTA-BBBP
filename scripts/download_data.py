import os
from torch_geometric.datasets import MoleculeNet

def main():
    print("📦 Downloading BBBP dataset...")
    dataset = MoleculeNet(root="./data", name="BBBP")
    print(f"✅ Downloaded {len(dataset)} molecules")
    
if __name__ == "__main__":
    main()
