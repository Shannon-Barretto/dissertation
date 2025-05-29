# Dissertation: Generation and Evaluation of High-Fidelity Synthetic Biomarker Data for Alzheimer’s Disease Research

## Overview

This repository contains the code, data, and documentation associated with my Bachelor's dissertation titled "Generation and Evaluation of High-Fidelity Synthetic Biomarker Data for Alzheimer’s Disease Research". 
The research focuses on generating high-fidelity synthetic datasets for Alzheimer’s Disease and Mild Cognitive Impairment biomarkers using advanced generative models, aiming to develop and validate a novel, extensible evaluation framework to rigorously benchmark synthetic-data generators in biomedical contexts and accelerate reliable, reproducible advances in AD/MCI diagnostics and predictive modeling.


## Abstract

Research on Alzheimer’s Disease (AD) is constrained by limited and heterogeneous clinical datasets, exacerbated by strict privacy regulations. This project explores the use of generative models—CTGAN and Gaussian Copula—to produce high-fidelity synthetic biomarker data derived from the ADNI cohort. A novel evaluation framework is introduced, combining divergence metrics, correlation analysis, and statistical tests to rigorously benchmark synthetic data quality. Bayesian optimization was used to fine-tune each model, and results show that Gaussian Copula outperforms CTGAN across multiple metrics, including JS divergence and correlation preservation. This research provides reproducible tools and evaluation criteria that support the scalable generation of synthetic biomedical datasets, accelerating advances in AD/MCI diagnostics.


## Project Structure

```bash
├── code/           # Jupyter notebooks for analysis
├── data/           # Raw datasets
└── README.md       # Project overview
```


## Results


## ## Technologies Used

- Programming Language: Python 3.10
- **Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `seaborn` – for data manipulation and visualization
  - `scipy`, `scikit-learn` – for statistical analysis and machine learning utilities
  - `ctgan` – for synthetic tabular data generation using GANs
  - `hyperopt` – for Bayesian hyperparameter optimization
- Tools: Jupyter Notebook, Git


## License
