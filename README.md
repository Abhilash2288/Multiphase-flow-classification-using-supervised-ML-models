# Multiphase Flow Regime Classification in Horizontal Pipes

This repository contains code, data, and results for the paper:

> **Classification of Multiphase Flow Regime in a Horizontal Pipe Using Supervised Machine Learning and Bayesian Optimization**  
> *Abhilash Ravichandran, Namburi Pooja Sai Venkata Sri Harika, Sai Manikiran Garimella, Mohan Anand*  
> Presented at the 2025 International Conference on Data Science, Agents, and Artificial Intelligence (ICDSAAI 2025).

---

## 📋 Overview

Multiphase flow refers to simultaneous flow of gas, liquid, and/or solid phases.  
Accurate classification of flow regimes is crucial in industries like oil & gas, chemical processing, and energy.  
This work applies **supervised machine learning** models enhanced with **Bayesian Optimization** for hyperparameter tuning to classify **three-phase (air–water–oil) flow regimes in horizontal pipes**.

---

## 🔷 Features

✅ Supervised ML models:
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- XGBoost
- CatBoost
- LightGBM
- Hybrid Stacking Model

✅ Bayesian Optimization for hyperparameter tuning.  
✅ Handles imbalanced dataset using class weights.  
✅ Evaluation with stratified 10-fold cross-validation.

---

## 🔷 Dataset

- Experimental dataset of 381 samples.
- Inputs:  
  - Superficial gas velocity
  - Superficial liquid velocity
  - Water cut
- Output:
  - Flow regime (6 classes: SL, SW, EB, AN, DB, SS)

---

