# Dissertation: Generation and Evaluation of High-Fidelity Synthetic Biomarker Data for Alzheimer’s Disease Research

## Overview

This repository contains the code, data, and documentation associated with my Bachelor's dissertation titled `Generation and Evaluation of High-Fidelity Synthetic Biomarker Data for Alzheimer’s Disease Research`. 
The research focuses on generating high-fidelity synthetic datasets for Alzheimer’s Disease and Mild Cognitive Impairment biomarkers using advanced generative models, aiming to develop and validate a novel, extensible evaluation framework to rigorously benchmark synthetic-data generators in biomedical contexts and accelerate reliable, reproducible advances in AD/MCI diagnostics and predictive modeling.


## Abstract

Research on Alzheimer’s Disease (AD) is constrained by limited and heterogeneous clinical datasets, exacerbated by strict privacy regulations. This project explores the use of generative models, CTGAN and Gaussian Copula, to produce high-fidelity synthetic biomarker data derived from the ADNI cohort. A novel evaluation framework is introduced, combining divergence metrics, correlation analysis, and statistical tests to rigorously benchmark synthetic data quality. Bayesian optimization was used to fine-tune each model, and results show that Gaussian Copula outperforms CTGAN across multiple metrics, including JS divergence and correlation preservation. This research provides reproducible tools and evaluation criteria that support the scalable generation of synthetic biomedical datasets, accelerating advances in AD/MCI diagnostics.


## Project Structure

```bash
├── code/           # Jupyter notebooks for analysis
├── data/           # Raw datasets
└── README.md       # Project overview
```


## Results

### Performance difference between CTGAN and Gaussain Copula

![image](https://github.com/user-attachments/assets/53f5dde5-0a12-43b5-b9f4-973e70ac07d4)

![image](https://github.com/user-attachments/assets/77530c01-07fa-425e-b878-3660b05fce3c)


`Note on x-axis presentation`:<br>
In the above figures, jsd is the JS divergence, corr is correaltion difference and summary_stats is the statistical summary.

`Note on y-axis presentation`:<br>
The y-axis in the above figures represents the inverse of each metric value (1/metric). This inversion was implemented because lower original metric values indicate better performance (smaller
differences between synthetic and original data distributions). By displaying the inverse, higher bars
now represent better model performance, making visual comparison more intuitive - taller bars mean
better performance. For example, a JS divergence of 0.02 (very good) would appear as 1/0.02 = 50
on our plots, while a poorer divergence of 0.1 would appear as only 1/0.1 = 10. This transformation
preserves the relative performance differences while making the visualization more aligned with conventional interpretations where ''higher is better.”

From the figure titled `Gaussian Copula vs CTGAN: Performance Comparison Across Biomarkers`, we can conclude the Gaussian Copula outperformed the CTGAN model
across all biomarkers. There is a stark improvement in the replication of the original’s data IQR in the
synthetic data. What is interesting to see is CTGAN managed to get a better correlation for PTAU
compared to Gaussian Copula.

We took the mean of each metric across all biomarkers, and compared the performance
between Gaussian Copula and CTGAN. This can be seen in the figure titled `Gaussian Copula vs CTGAN Metrics`.  Gaussian Copula clearly dominates.


### Bayesian Optimzation results

![image](https://github.com/user-attachments/assets/eb1c160c-101d-42e5-919b-a92f23b8d62b)

The rapid convergence evident in above right figure, quantitatively demonstrates the superior optimization landscape of parametric copula models for this application domain. In the left figure, the loss oscillates between
≈ 0.5−5.5 at initial iterations, and starts to converge after ≈ 100 iterations, with loss oscillating between
≈ 0.5 − 1.5 for most biomarkers. In the right figure, at the initial iterations itself, the loss oscillates between
≈ 0.1−2.5, and within 25 iterations, the loss is tending to converge and oscillate between ≈ 1.75−0.75.
A key insight from the right figure is that PLASMAPTAU181 (purple), for majority of the iteration achieved
the lowest loss, suggesting it has statistical properties that are more amenable to accurate modeling
with the Gaussian Copula approach.

## Technologies Used

- Programming Language: Python 3.10
- **Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `seaborn` – for data manipulation and visualization
  - `scipy`, `scikit-learn` – for statistical analysis and machine learning utilities
  - `ctgan` – for synthetic tabular data generation using GANs
  - `hyperopt` – for Bayesian hyperparameter optimization
- Tools: Jupyter Notebook, Git


## License
This project is licensed under the [MIT License](LICENSE).
