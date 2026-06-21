# 🧠 GraphTransformerSOTA2026: A Hybrid GIN-Transformer Architecture for Blood-Brain Barrier Penetration Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch_Geometric-3.0+-3C2179?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/RDKit-2022.09-4A8B9E?style=for-the-badge&logo=rdkit&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logo=opensource&logoColor=white" />
  <img src="https://img.shields.io/badge/Reproducible-Yes-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Paper-ChemRxiv-ff69b4?style=for-the-badge" />
</p>

---

## 📌 **Overview**

> **GraphTransformerSOTA2026** is a cutting-edge hybrid Graph Neural Network architecture that synergistically combines the power of **Graph Isomorphism Networks (GIN)** with **Transformer self-attention** through a **learnable gating mechanism**. Designed for molecular property prediction, it achieves state-of-the-art performance on the **BBBP** (Blood-Brain Barrier Penetration) dataset and demonstrates exceptional transfer learning capabilities.

---

## 🏆 **Key Achievements**

| Task | Dataset | Performance |
|------|---------|-------------|
| **BBBP Prediction** | BBBP (2,039 molecules) | **83.67%** Accuracy |
| **Transfer Learning** | Tox21 | **97.03%** Accuracy |
| **Baseline Comparison** | BBBP | Outperforms GCN, GAT, GIN |
| **Reproducibility** | All | ✅ Fully reproducible |

---

## 🧬 **Why GraphTransformerSOTA2026?**

### 🧩 **Architecture Highlights**

| Component | Description |
|-----------|-------------|
| **Node Encoder** | Projects atomic features (9 per atom) to 256D embedding |
| **GIN Branch** | Captures local molecular structure via message passing |
| **Transformer Branch** | Learns global molecular context via multi-head self-attention |
| **Learnable Gate** | Adaptively fuses local and global representations |
| **Classifier** | MLP with hidden layers 256 → 128 → 2 (binary) |

### 📊 **Comparison with Baselines**

| Model | Accuracy | 5-Fold CV |
|-------|----------|-----------|
| **GraphTransformerSOTA2026 (Ours)** | **83.67%** | **84.12%** |
| Graph Attention Network (GAT) | 82.45% | 82.86% |
| Graph Isomorphism Network (GIN) | 82.04% | 82.45% |
| Graph Convolutional Network (GCN) | 80.00% | 79.18% |

---

## 🚀 **Transfer Learning Excellence**

Our model demonstrates remarkable **transfer learning** capabilities:

| Dataset | Pre-trained | Fine-tuned |
|---------|-------------|------------|
| Tox21 (14.57%) | → | **97.03%** |

> ✨ *"Fine-tuning on Tox21 with just 20 epochs achieved 97.03% accuracy, validating the model's strong generalization and adaptability."*

---

## 🛠️ **Tech Stack**

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch_Geometric-3.0+-3C2179?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/RDKit-2022.09-4A8B9E?style=flat-square&logo=rdkit&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-1.24+-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-2.0+-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-3.7+-11557C?style=flat-square&logo=matplotlib&logoColor=white" />
  <img src="https://img.shields.io/badge/Seaborn-0.12+-3776AB?style=flat-square&logo=seaborn&logoColor=white" />
  <img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white" />
</p>

---

## 📁 **Project Structure**

```

bbbp-graph-transformer/
│
├── README.md                    # 📄 This file
├── LICENSE                      # 📜 MIT License
├── requirements.txt             # 📦 Dependencies
├── .gitignore                   # 🚫 Ignored files
├── config.yaml                  # ⚙️ Master configuration
│
├── src/                         # 🧠 Source code
│   ├── init.py
│   ├── model.py                 # GraphTransformerSOTA2026
│   ├── dataset.py               # Data loading & scaffold split
│   ├── train.py                 # Training & evaluation loops
│   └── utils.py                 # Utilities
│
├── scripts/                     # 🚀 Executable scripts
│   ├── download_data.py
│   └── run_experiment.py
│
├── notebooks/                   # 📓 Jupyter/Colab notebooks
│   └── bbbp_training.ipynb
│
├── results/                     # 📊 Saved outputs
│   ├── best_model.pt
│   ├── training_history.png
│   ├── baseline_comparison.csv
│   └── tox21_result.txt
│
├── paper/                       # 📄 Publication materials
│   ├── paper_full.tex
│   ├── article_with_figs.pdf
│   ├── figure1_comparison.png
│   └── figure2_transfer.png
│
└── data/                        # 🗃️ Dataset (auto-downloaded)
└── .gitkeep

```

---

## 🔬 **Methodology**

### 1. Dataset & Preprocessing
- **BBBP Dataset:** 2,039 molecules from MoleculeNet
- **Scaffold Split:** 70/15/15 (Train/Val/Test) to prevent data leakage
- **Node Features:** 9 atomic properties (atomic number, mass, degree, etc.)

### 2. Model Architecture

#### GraphTransformerBlock
```python
h = self.attn(x, edge_index)
x = self.norm1(x + dropout(h))
h = self.ffn(x)
x = self.norm2(x + dropout(h))
```

Gated Fusion

```python
x = g * x_gnn + (1 - g) * x_trans
```

where g is a learnable parameter initialized to 0.5.

3. Training Protocol

Parameter Value
Optimizer Adam
Learning Rate 0.001
Weight Decay 5e-4
Batch Size 32
Epochs 50 (with early stopping)
Patience 10
Gradient Clipping 1.0
Scheduler ReduceLROnPlateau

---

📊 Results & Visualizations

Figure 1: Model Comparison on BBBP

<p align="center">
  <img src="paper/figure1_comparison.png" width="700">
</p>

Figure 2: Transfer Learning to Tox21

<p align="center">
  <img src="paper/figure2_transfer.png" width="700">
</p>

Training History

<p align="center">
  <img src="results/training_history.png" width="700">
</p>

---

🚀 Quick Start

Option 1: Run on Google Colab

https://colab.research.google.com/assets/colab-badge.svg

Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/yourusername/bbbp-graph-transformer.git
cd bbbp-graph-transformer

# Install dependencies
pip install -r requirements.txt

# Download data
python scripts/download_data.py

# Run experiment
python scripts/run_experiment.py
```

Option 3: Docker

```bash
docker build -t bbbp-graph-transformer .
docker run --gpus all -v $(pwd)/results:/app/results bbbp-graph-transformer
```

---

📄 Citation

If you use this work, please cite:

```bibtex
@software{karimi2026graphtransformer,
  author = {Karimi, Vania},
  title = {GraphTransformerSOTA2026: Hybrid GIN-Transformer for BBB Prediction},
  year = {2026},
  url = {https://github.com/yourusername/bbbp-graph-transformer}
}
```

---

🧠 Research Impact

· ✅ State-of-the-Art on BBBP (83.67% accuracy)
· ✅ Exceptional Transfer Learning (97.03% on Tox21)
· ✅ Scaffold Split prevents data leakage
· ✅ Fully Reproducible (fixed seed, config-driven)
· ✅ Open-Source for community use
· ✅ Published on ChemRxiv

---

🔭 Future Directions

· Extension to 3D molecular conformations
· Multi-task learning across multiple properties
· Integration with generative models for de novo drug design
· Deployment as a web service (Streamlit/FastAPI)


Research Interests:

· 🧠 Medical Artificial Intelligence
· 🔬 Graph Neural Networks
· 💊 Drug Discovery
· 🧬 Computational Biology
· 🤖 Explainable AI

---

📜 License

This project is licensed under the MIT License – see the LICENSE file for details.

---

<p align="center">
  <b>⭐ If you find this project useful, please consider giving it a star!</b>
</p>

<p align="center">
  <i>Built with ❤️ for the scientific community</i>
</p>
```

