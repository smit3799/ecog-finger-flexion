# Machine Learning Optimization for Finger Movement Classification Using XGBoost

**Course:** EE 608 – Applied Modeling and Optimization  
[cite_start]**Instructor:** Prof. K. P. Subbalakshmi [cite: 11]  
[cite_start]**Team Members:** Jiajun Zhang & Smit Desai [cite: 7, 8]  
[cite_start]**Institution:** Stevens Institute of Technology [cite: 1, 2]

---

## 📌 Project Overview
[cite_start]This project addresses the challenge of decoding human motor intentions from electrophysiological signals to control assistive robotics[cite: 21, 22]. [cite_start]Specifically, we utilize Electrocorticography (ECoG) signals to classify finger movement states—including resting, opening, and closing of various finger groups—across 9 distinct movement classes[cite: 25, 26, 39].

[cite_start]Due to the inherent noise in brain signals and the non-convex nature of the problem, we developed a multi-level optimization pipeline to improve upon baseline performance[cite: 24, 55, 57].

---

## 🛠️ Optimization Methodology
[cite_start]The objective is to maximize the average cross-validation Macro-F1 score, selected to reward performance on minority classes in an imbalanced 9-class problem[cite: 29, 36, 39].

$$\max_{w, \alpha, \theta} \text{CV-MacroF1}(w, \alpha, \theta)$$

### Stage 1: Data Pipeline Optimization (Grid Search)
[cite_start]We optimized signal windowing and movement thresholds to ensure high-quality feature extraction[cite: 41, 47, 78].
* [cite_start]**Parameters:** Window size ($w$) and threshold ($\alpha$)[cite: 47, 50].
* [cite_start]**Feasible Sets:** $w \in \{0.20, 0.25, 0.30, 0.50\}$, $\alpha \in \{0.15, 0.20, 0.25, 0.30, 0.35\}$[cite: 50, 85].

### Stage 2: Hyperparameter Tuning (Bayesian Optimization)
[cite_start]Using the optimal window and threshold found in Stage 1, we optimized the XGBoost model's complexity and generalization capabilities[cite: 42, 49, 80].
* [cite_start]**Toolbox:** Optuna for Bayesian search[cite: 250].
* [cite_start]**Parameters ($\theta$):** `max_depth`, `learning_rate`, `n_estimators`, `gamma`, `reg_alpha`, and `reg_lambda`[cite: 52, 117].

---

## 📈 Performance & Validation Results
[cite_start]The model was trained on Subject 1 data (62 ECoG channels) with 5-fold Stratified Cross-Validation[cite: 265, 266, 275].

| Stage | Window ($w$) | $\alpha$ | Train Wins | CV Acc | CV Macro-F1 | Test Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Manual Baseline | 0.25s | 0.25 | 1600 | 0.459 | 0.28 | 0.486 |
| Optuna (Single Val) | 0.25s | 0.25 | 1600 | 0.500 | — | 0.484 |
| CV-Optuna + Weights | 0.25s | 0.25 | 1600 | 0.519 | 0.34 | 0.476 |
| **Joint Optimization** | **0.5s** | **0.2** | **800** | **0.521** | **0.436** | **0.433** |

> [cite_start]**Note:** The results above reflect the iterative improvement from systematic tuning[cite: 318].

---

## ⚙️ Project Structure & Setup
### Dependencies
* [cite_start]**Data Processing:** NumPy, Pandas[cite: 246].
* [cite_start]**ML & Evaluation:** Scikit-learn, XGBoost[cite: 247, 248].
* [cite_start]**Optimization:** Optuna / Scikit-optimize[cite: 250].

### Usage
1.  [cite_start]**Preprocessing:** Feature extraction (9 features × 62 channels) generates 558 statistical features per window[cite: 271].
2.  **Training:** Execute the multi-stage optimization scripts to find the best configuration for your local data path.
3.  [cite_start]**Validation:** Held-out test sets must remain unseen during the optimization trials[cite: 274, 275].

---

## 🔮 Future Work
* [cite_start]**Bigger Data:** Integrate multi-subject data and the Stanford ECoG Library[cite: 335, 337].
* [cite_start]**Architecture Evolution:** Explore LSTM or Transformer models for better temporal context and frequency-domain features[cite: 338].
* [cite_start]**Hardware Integration:** Prototype real-time robotic control via Arduino[cite: 339].

---

## 📄 License
This project is licensed under the MIT License.
