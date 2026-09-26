# Quantum Kernel Classification

An experimental study of quantum kernel methods for binary classification and their comparison with classical kernel methods.

---

## Overview

This project investigates the application of kernel methods in Quantum Machine Learning (QML) for binary classification.

The main objective is to experimentally compare classical kernel-based Support Vector Machines (SVMs) with quantum kernel methods under a common experimental framework.

The project currently focuses on one quantum approache:

1. **Data-dependent EfficientSU2-based Quantum Kernel**

This quantum method is evaluated against several classical SVM kernels:

- Linear
- RBF
- Polynomial
- Sigmoid

The purpose of the project is not to assume or demonstrate quantum advantage in advance. Instead, the goal is to investigate how different kernel and quantum encoding choices affect classification performance and computational characteristics.

---

## Research Questions

The main research question is:

> **How do quantum kernel methods compare with classical kernel methods for binary classification?**

The project also investigates the following questions:

### RQ1 — Classical vs Quantum Kernels

How does the performance of fidelity-based quantum kernels compare with commonly used classical SVM kernels?

### RQ2 — Quantum Data Encoding

How does the choice of quantum data encoding affect the performance of a fidelity-based quantum kernel?

### RQ3 — Parameterized Quantum Encoding

How does the structure and depth of a parameterized quantum encoding circuit affect classification performance and computational cost?

Further research questions may be introduced as the experimental results reveal new patterns or limitations.

---

## Dataset

The current experiments use the **Breast Cancer Wisconsin Diagnostic dataset** available through `scikit-learn`.

The dataset contains:

- **569 samples**
- **30 numerical features**
- **2 classes**

The two classes represent:

- Malignant
- Benign

All 30 available features are currently used in the experiments.

The dataset is loaded using:

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
```

---

## Experimental Methodology

The project follows a comparative experimental methodology.

The general pipeline is:

```text
Dataset
   │
   ▼
Preprocessing
   │
   ├───────────────┐
   │               │
   ▼               ▼
Classical       Quantum
Kernels         Encoding
   │               │
   ▼               ▼
SVM             Quantum Kernel
   │               │
   └───────┬───────┘
           ▼
       Classification
           │
           ▼
       Evaluation
```

The experiments are performed using cross-validation in order to obtain more reliable performance estimates.

The current experimental setup uses **5-fold cross-validation**.

---

## Classical Baselines

Four classical SVM kernels are currently used as baselines.

### 1. Linear Kernel

```python
SVC(
    kernel="linear",
    C=1.0
)
```

### 2. RBF Kernel

```python
SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)
```

### 3. Polynomial Kernel

```python
SVC(
    kernel="poly",
    degree=3,
    C=1.0,
    gamma="scale",
    coef0=1.0
)
```

### 4. Sigmoid Kernel

```python
SVC(
    kernel="sigmoid",
    C=1.0,
    gamma="scale",
    coef0=0.0
)
```

These models provide classical reference points for evaluating the quantum approaches.

---

## Quantum Kernel Methods

### 1. EfficientSU2-based Quantum Kernel

The first quantum approach investigates a parameterized quantum circuit based on `EfficientSU2`.


The general structure is:

```text
Classical features
       │
       ▼
Parameter mapping
       │
       ▼
EfficientSU2
       │
       ▼
Quantum state
       │
       ▼
Fidelity kernel
       │
       ▼
SVM classifier
```
---

## Quantum Kernel Construction

The quantum kernel is used as a similarity function between pairs of samples.

For training samples:

```text
x₁, x₂, ..., xₙ
```

a kernel matrix is constructed:

```text
K =
[ K(x₁,x₁)  K(x₁,x₂)  ... ]
[ K(x₂,x₁)  K(x₂,x₂)  ... ]
[    ...       ...      ... ]
```

This matrix is then supplied to a classical SVM classifier.

Therefore, the quantum circuit provides the kernel values, while the final classification step remains classical.

The architecture can be summarized as:

```text
Classical Dataset
       │
       ▼
Quantum Encoding
       │
       ▼
Quantum Kernel Matrix
       │
       ▼
Classical SVM
       │
       ▼
Prediction
```

---

## Evaluation Metrics

Classification performance is evaluated using both overall and class-specific metrics.

For every model, the following metrics are calculated:

### Overall Metrics

- Accuracy
- Macro F1-score
- Balanced Accuracy

### Malignant Class

- Precision
- Recall
- F1-score

### Benign Class

- Precision
- Recall
- F1-score

The results table therefore follows this structure:

| Model | Accuracy | Malignant Precision | Malignant Recall | Malignant F1 | Benign Precision | Benign Recall | Benign F1 | Macro F1 | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear SVM | 0.9525 | 0.9471 | 0.9243 | 0.9353 | 0.9561 | 0.9691 | 0.9625 | 0.9489 | 0.9467 |
| RBF SVM | 0.9210 | 0.9681 | 0.8160 | 0.8830 | 0.9020 | 0.9831 | 0.9402 | 0.9116 | 0.8995 |
| Polynomial SVM | 0.9122 | 0.9613 | 0.7971 | 0.8693 | 0.8922 | 0.9803 | 0.9337 | 0.9015 | 0.8887 |
| Sigmoid SVM | 0.4497 | 0.2064 | 0.1652 | 0.1804 | 0.5531 | 0.6194 | 0.5827 | 0.3816 | 0.3923 |
| EfficientSU2 Quantum Kernel | 0.9051 | 0.9333 | 0.8020 | 0.8613 | 0.8929 | 0.9663 | 0.9278 | 0.8946 | 0.8841 |

---

## Experimental Variables

The project is designed to allow different experimental variables to be investigated independently.

Current and planned variables include:

### Classical Models

- Kernel type
- SVM hyperparameters

### Quantum Models

- Encoding method
- Number of qubits
- Circuit depth
- Number of repetitions (`reps`)
- Parameter mapping
- Entanglement structure

### Computational Characteristics

- Kernel computation time
- Training time
- Circuit depth
- Number and type of gates
- Number of qubits
- Number of shots

These measurements can help distinguish classification performance from computational cost.

---

## Cross-Validation

The current experiments use **5-fold cross-validation**.

The dataset is divided into five folds. In each iteration:

```text
4 folds → training
1 fold  → testing
```

The process is repeated five times so that every sample is used as part of a test set.

The final performance is obtained by aggregating the results across the folds.

Stratification is used for classification experiments so that the class distribution is approximately preserved across folds.

---

## Reproducibility

The experiments are implemented in Python using Qiskit and Qiskit Aer.

Current development environment:

```text
Python       3.13.7
Qiskit       2.5.2
```

The project currently does not depend on `qiskit-machine-learning`.

The quantum kernels are implemented directly using Qiskit circuits rather than relying on the Qiskit Machine Learning package.

---

## Project Structure

The repository is organized as follows:

```text
quantum-kernel-classification/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── notebooks/
│   ├── classical_kernels.ipynb
│   └── quantum_efficient_su2.ipynb
│
├── src/
│   ├── preprocessing.py
│   └── metrics.py
│
├── results/
│   ├── classical_results.csv
│   └── quantum_results.csv
│   └── combined_results.csv
│
└── report/
    └── preliminary_report.pdf
```

The exact filenames may evolve as the project develops.

---

## Current Status

### Completed

- [x] Dataset preparation
- [x] Classical SVM baseline experiments
- [x] Linear kernel
- [x] RBF kernel
- [x] Polynomial kernel
- [x] Sigmoid kernel
- [x] Initial EfficientSU2-based quantum kernel
- [x] 5-fold cross-validation framework
- [x] Separate evaluation of Malignant and Benign classes
- [x] Macro F1-score
- [x] Balanced Accuracy
- [x] Project source-code structure
- [x] Results directory
- [x] Experimental notebooks
- [x] Measuring circuit depth and gate counts

### Planned

- [ ] Comparing classical and quantum methods
- [ ] Investigating different quantum encoding strategies
- [ ] Studying the effect of circuit depth
- [ ] Investigating different numbers of shots
- [ ] Noisy quantum simulation
- [ ] Additional datasets
- [ ] Investigation of trainable quantum kernels
- [ ] Potential experiments on real quantum hardware
- [ ] Measuring computational cost

---

## Limitations

The current project has several important limitations.

### 1. No Execution

The current quantum experiments are performed using Statevectors with
zero noise.

Therefore, the results do not yet represent performance on physical quantum hardware.

### 2. Computational Cost

Quantum kernel methods require evaluating similarities between pairs of samples.

For a dataset with `n` samples, constructing the full kernel matrix can require approximately:

```text
O(n²)
```

kernel evaluations.

This can become computationally expensive as the dataset grows.

### 3. Sampling Noise

When measurement-based kernel estimation is used, finite numbers of shots introduce statistical sampling noise.

The estimated kernel value therefore depends on the number of shots.

### 4. Dataset Dependence

Results obtained on a single dataset cannot establish general superiority of quantum kernels.

Additional datasets are required to investigate whether observed behavior generalizes.

### 5. Quantum Advantage

Higher classification accuracy alone does not demonstrate quantum advantage.

A meaningful claim of quantum advantage would require consideration of factors such as:

- computational complexity
- data-loading cost
- circuit depth
- number of qubits
- hardware noise
- classical computational cost
- scaling behavior

Therefore, this project does not assume that a quantum method should outperform classical methods.

---

## Future Work

Several extensions are planned for future versions of the project.

### Noise Experiments

Introduce realistic noise models using Qiskit Aer and investigate how quantum kernel performance changes under noise.

### Circuit Depth

Study the relationship between parameterized circuit depth and:

- classification performance
- kernel properties
- circuit complexity
- simulation cost

### Shot Count

Compare different numbers of measurement shots, for example:

```text
256
1024
4096
```

to study the effect of sampling noise.

### Multiple Datasets

Evaluate the methods on additional binary classification datasets to determine whether the observed results generalize.

### Kernel Analysis

Investigate properties of the resulting kernel matrices, including:

- eigenvalue spectra
- positive semidefiniteness
- kernel similarity
- kernel-target alignment

### Trainable Quantum Kernels

Investigate whether optimizing the parameters of a quantum feature map can improve the resulting kernel.

### Real Quantum Hardware

If access to suitable quantum hardware becomes available, compare simulated results with experiments on real quantum processors.

---



## References

The project builds on concepts from the following areas:

1. Support Vector Machines and kernel methods
2. Quantum feature maps
3. Quantum kernel methods
4. Fidelity-based quantum kernels
5. Parameterized quantum circuits
6. Quantum Machine Learning

Relevant academic references will be added to the final research report and citation file.

---

## Author

**Hassan Pourhoseini**

B.Sc. Student in Computer Engineering  
University of Tehran

Research interests:

- Quantum Machine Learning
- Quantum Computing
- Quantum Information
- Machine Learning
- Quantum Algorithms

---

## Project Status

**Current version:** `v0.1-preliminary`

This repository represents an ongoing independent research project. The experimental methodology and research questions may evolve as new results are obtained.

The current version should be considered a preliminary research stage rather than a final publication.