# ECoG-Based Finger Flexion Decoding via Multi-Level Optimized XGBoost

This repository contains the implementation of a Brain-Computer Interface (BCI) pipeline designed to decode continuous finger flexion from Electrocorticography (ECoG) signals. Developed as part of the graduate curriculum for EE 608 at Stevens Institute of Technology, this project evaluates advanced machine learning and deep learning methodologies to translate neural activity into motor outputs.

## 📌 Project Overview
Decoding specific motor intentions from cortical surfaces is a foundational challenge in neural engineering. This project utilizes electrophysiological data to predict finger movements, benchmarking a highly optimized gradient boosting framework against state-of-the-art neural networks.

### Key Highlights:
* **Core Model:** XGBoost enhanced with a custom multi-level optimization pipeline.
* **Deep Learning Baselines:** 1D Convolutional Neural Networks (Conv1D) and EEGNet architectures optimized for temporal neural data.
* **Application:** Precise time-series decoding of discrete/continuous finger flexion states.

---

## 🛠️ Repository Structure

```text
├── notebooks/
│   ├── conv1d_baseline.ipynb        # Time-series deep learning baseline
│   ├── eegnet_bci.ipynb             # EEGNet architecture deployment
│   └── xgboost_optimization.ipynb  # Multi-level optimized XGBoost pipeline
├── .gitignore                       # Ensures large files and checkpoints are untracked
├── LICENSE                          # MIT License
└── README.md                        # Project documentation
