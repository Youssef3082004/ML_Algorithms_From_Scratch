# Machine Learning Algorithms From Scratch

A growing collection of fundamental Machine Learning algorithms implemented completely from scratch using **`Python`** and **`numpy`**. This repository serves as an educational resource to understand the inner workings of various ML models without relying on high-level frameworks like scikit-learn for the core logic.

## 📑 Table of Contents
- [Machine Learning Algorithms From Scratch](#machine-learning-algorithms-from-scratch)
  - [📑 Table of Contents](#-table-of-contents)
  - [🧠 Implemented Algorithms](#-implemented-algorithms)
    - [Supervised Learning](#supervised-learning)
    - [Unsupervised Learning](#unsupervised-learning)
    - [Cost Functions](#cost-functions)
  - [🛠️ Utilities \& Metrics](#️-utilities--metrics)
  - [📂 Project Structure](#-project-structure)
  - [⚙️ Installation](#️-installation)
  - [🗺️ Roadmap](#️-roadmap)

## 🧠 Implemented Algorithms

The repository is organized by machine learning categories. As new algorithms are added, they will be tracked here.

### Supervised Learning
**Classification**
- [x] **K-Nearest Neighbors (KNN)**: Custom implementation using Euclidean distance to find the *k* closest data points.
- [x] **Logistic Regression**: Custom implementation using Gradient Descent with *sigmoid activation function*. Supports binary classification with customizable learning rate and iterations.
- [x] **Adaline (Adaptive Linear Neuron)**: Custom implementation using Gradient Descent with *linear activation function* for weight updates and *threshold function* for final predictions. Supports binary classification with customizable learning rate and iterations.
- [x] **Perceptron**: Custom implementation using the Perceptron learning rule with *step activation function* for weight updates and predictions. Supports binary classification with customizable learning rate and iterations.
- [x] **Support Vector Machine (SVM)**: Custom implementation using *Stochastic Gradient Descent (SGD)* optimizing the hinge loss function with L2 regularization. Supports binary classification with customizable learning rate, lambda parameter, and iterations.
- [x] **Naive Bayes**: Custom implementation using *Bayes' theorem* with the *Gaussian likelihood* assumption for continuous features. Estimates class-conditional probabilities from training data (mean and variance per feature per class) and predicts via *Maximum A Posteriori (MAP)* estimation. Supports multi-class classification with customizable prior probabilities.
- [x] **Decision Tree**: Custom implementation using a recursive binary splitting approach. It utilizes impurity measures like **Gini Impurity** or **Entropy (Information Gain)** to select the best feature and threshold at each node. Supports multi-class classification with customizable constraints such as *maximum depth*, *minimum samples per split*, and *minimum impurity decrease* to prevent overfitting.

**Regression**
- [x] **Linear Regression**: Implemented via Gradient Descent with customizable learning rates and epochs. Includes a built-in visualization script.

### Unsupervised Learning
*(Algorithms coming soon)*

### Cost Functions 
- **SVM:** 
$$J(\mathbf{w}) = \lambda ||\mathbf{w}||^2 + \frac{1}{n}\sum_{i=1}^n \max(0, 1 - y_i(\mathbf{w} \cdot \mathbf{x}_i))$$

-  **Entropy**
    $$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$
-  **Information Gain:**
    $$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$



## 🛠️ Utilities & Metrics

To support the algorithms, custom evaluation and calculation modules are maintained in the `Metrices/` directory:
- **Distances**: Euclidean distance calculations.
- **Evaluation**: Measurement functions including Accuracy, Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

## 📂 Project Structure

The project follows a modular structure, making it easy to add new algorithms and metrics:

```text
ML_Algorithms_From_Scratch/
│
├── Metrices/                  # Reusable math and evaluation modules
│   ├── Distance.py            
│   └── Evaluation.py 
├── utils/         
│   └── DescionTreeHelper.py 
├── KNN.py                     # Algorithm implementations 
├── LinearRegression.py 
├── LogisticRegression.py  
├── Adaline.py
├── Perceptron.py  
├── SVM.py 
├── NaiveBayes.py     
├── DecisionTree.py     
└── ...

```

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Youssef3082004 ML_Algorithms_From_Scratch.git
cd ML_Algorithms_From_Scratch
```


2. Install the required dependencies:
```bash
pip install numpy matplotlib scikit-learn

```


*(Note: `scikit-learn` is only used to generate dummy datasets for testing).*



## 🗺️ Roadmap

Future updates will expand the repository with more algorithms. Planned additions include:


* [x] K Nearest Neighbors
* [x] Linear Regression
* [x] Logistic Regression
* [x] Adaline
* [x] Perceptron
* [x] SVM
* [x] Naive Bayes
* [x] Decision Tree
* [ ] Random Forest
* [ ] AdaBoost
* [ ] K-Means
* [ ] Principal Component Analysis (PCA)
* [ ] Linear Discriminant Analysis (LDA)

---
<h3 align="center">A repository of AI & ML algorithms implemented from Scratch</h3>