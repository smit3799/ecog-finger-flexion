Machine Learning Optimization for Finger Movement Classification Using XGBoost

Course: EE 608 – Applied Modeling and Optimization   


Instructor: Prof. K. P. Subbalakshmi   


Team Members: Jiajun Zhang & Smit Desai   


Institution: Stevens Institute of Technology

📌 Project OverviewThis project addresses the challenge of decoding human motor intentions from electrophysiological signals to control assistive robotics. Specifically, we utilize Electrocorticography (ECoG) signals to classify finger movement states—including resting, opening, and closing of various finger groups—across 9 distinct movement classes.  Due to the inherent noise in brain signals and the non-convex nature of the problem, we developed a multi-level optimization pipeline to improve upon baseline performance.

🛠️ Optimization MethodologyThe objective is to maximize the average cross-validation Macro-F1 score, selected to reward performance on minority classes in an imbalanced 9-class problem.  $$\max_{w, \alpha, \theta} \text{CV-MacroF1}(w, \alpha, \theta)$$Stage 1: Data Pipeline Optimization (Grid Search)We optimized signal windowing and movement thresholds to ensure high-quality feature extraction.  Parameters: Window size ($w$) and threshold ($\alpha$).  Feasible Sets: $w \in \{0.20, 0.25, 0.30, 0.50\}$, $\alpha \in \{0.15, 0.20, 0.25, 0.30, 0.35\}$.  

Stage 2: Hyperparameter Tuning (Bayesian Optimization)Using the optimal window and threshold found in Stage 1, we optimized the XGBoost model's complexity and generalization capabilities.  Toolbox: Optuna for Bayesian search.  Parameters ($\theta$): max_depth, learning_rate, n_estimators, gamma, reg_alpha, and reg_lambda. 

📈 Performance & Validation ResultsThe model was trained on Subject 1 data (62 ECoG channels) with 5-fold Stratified Cross-Validation to ensure generalization.  StageWindow (w)αTrain WinsCV AccCV Macro-F1Test AccManual Baseline 0.25s0.2516000.4590.280.486Optuna (Single Val) 0.25s0.2516000.500—0.484CV-Optuna + Weights 0.25s0.2516000.5190.340.476Joint Optimization 0.5s0.2800

Key FindingsData vs. Model: Data pipeline tuning (window size and threshold) provided the most significant performance gains compared to hyperparameter tuning alone.  The Ceiling: Multiple optimization strategies converged around ~52% accuracy, suggesting a "data information ceiling" for single-subject decoding using statistical features.  Metric Shift: Applying sample_weight='balanced' and optimizing for Macro-F1 improved the model's ability to catch rare finger actions, which is critical for BCI safety and utility. 

⚙️ Project Structure & Setup
Dependencies

Data Processing: NumPy, Pandas   


ML & Evaluation: Scikit-learn, XGBoost   


Optimization: Optuna   

Usage

Preprocessing: Feature extraction (9 features × 62 channels) generates 558 features per window.  


Training: Execute the multi-stage optimization scripts to find the best configuration for your local data path.  


Validation: Held-out test sets must remain unseen during the optimization trials.

🔮 Future Work

Bigger Data: Integrate multi-subject data and the Stanford ECoG Library.  


Architecture Evolution: Explore LSTM or Transformer models for better temporal context and frequency-domain features.  


Hardware Integration: Prototype real-time robotic control via Arduino.

📄 License
This project is licensed under the MIT License.
