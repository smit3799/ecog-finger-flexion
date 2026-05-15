# Machine Learning Optimization for Finger Movement Classification Using XGBoost

**Course:** EE 608 – Applied Modeling and Optimization  
**Instructor:** Prof. K. P. Subbalakshmi  
**Team Members:** Jiajun Zhang & Smit Desai  
**Institution:** Stevens Institute of Technology

---

## 📌 Project Overview
This project addresses the challenge of decoding human motor intentions from electrophysiological signals to control assistive robotics. Specifically, we utilize Electrocorticography (ECoG) signals to classify finger movement states—including resting, opening, and closing of various finger groups—across 9 distinct movement classes.

Due to the inherent noise in brain signals and the non-convex nature of the problem, we developed a multi-level optimization pipeline to improve upon baseline performance.

---

## 🛠️ Optimization Methodology
The objective is to maximize the average cross-validation Macro-F1 score, selected to reward performance on minority classes in an imbalanced 9-class problem.

$$\max_{w, \alpha, \theta} \text{CV-MacroF1}(w, \alpha, \theta)$$

### Stage 1: Data Pipeline Optimization (Grid Search)
We optimized signal windowing and movement thresholds to ensure high-quality feature extraction.
* **Parameters:** Window size ($w$) and threshold ($\alpha$).
* **Feasible Sets:** $w \in \{0.20, 0.25, 0.30, 0.50\}$, $\alpha \in \{0.15, 0.20, 0.25, 0.30, 0.35\}$.

### Stage 2: Hyperparameter Tuning (Bayesian Optimization)
Using the optimal window and threshold found in Stage 1, we optimized the XGBoost model's complexity and generalization capabilities.
* **Toolbox:** Optuna for Bayesian search.
* **Parameters ($\theta$):** `max_depth`, `learning_rate`, `n_estimators`, `gamma`, `reg_alpha`, and `reg_lambda`.

---

## 📈 Performance & Validation Results
The model was trained on Subject 1 data (62 ECoG channels) with 5-fold Stratified Cross-Validation.

| Stage | Window ($w$) | $\alpha$ | Train Wins | CV Acc | CV Macro-F1 | Test Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Manual Baseline | 0.25s | 0.25 | 1600 | 0.459 | 0.28 | 0.486 |
| Optuna (Single Val) | 0.25s | 0.25 | 1600 | 0.500 | — | 0.484 |
| CV-Optuna + Weights | 0.25s | 0.25 | 1600 | 0.519 | 0.34 | 0.476 |
| **Joint Optimization** | **0.5s** | **0.2** | **800** | **0.521** | **0.436** | **0.433** |

> **Note:** The results above reflect the iterative improvement from systematic tuning.

---
## 🚀 Usage (Google Colab)
This project is designed to be executed in **Google Colab** following a two-part sequence to ensure joint optimization of data and model parameters.

### Part 1: Signal Parameter Optimization
1. Open Google Colab and upload the notebook for **Part 1** (Grid Search).
2. Upload the two ECoG data files from the repository's `data/` folder to the local Colab session.
3. Select **"Run all"** to perform the initial grid search.
4. Note the optimal window size and threshold values obtained from the output.

### Part 2: Model Hyperparameter Tuning
1. Open Google Colab and upload the notebook for **Part 2** (Bayesian Optimization).
2. Upload the same two data files to the Colab session.
3. **Configure Parameters:** Replace the default parameters in Part 2 with the optimal window size and threshold values obtained from Part 1.
4. Select **"Run all"** to find the final optimized XGBoost hyperparameters and evaluate performance.

---

## ⚙️ Project Structure & Setup
### Dependencies
* **Data Processing:** NumPy, Pandas.
* **ML & Evaluation:** Scikit-learn, XGBoost.
* **Optimization:** Optuna / Scikit-optimize.

### Usage
1.  **Preprocessing:** Feature extraction (9 features × 62 channels) generates 558 statistical features per window.
2.  **Training:** Execute the multi-stage optimization scripts to find the best configuration for your local data path.
3.  **Validation:** Held-out test sets must remain unseen during the optimization trials.

---

## 🔮 Future Work
* **Bigger Data:** Integrate multi-subject data and the Stanford ECoG Library.
* **Architecture Evolution:** Explore LSTM or Transformer models for better temporal context and frequency-domain features.
* **Hardware Integration:** Prototype real-time robotic control via Arduino.

---

## 📄 License
This project is licensed under the MIT License.
