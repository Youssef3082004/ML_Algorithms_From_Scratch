# Machine Learning Algorithms From Scratch

A growing collection of fundamental Machine Learning algorithms implemented completely from scratch using Python and `numpy`. This repository serves as an educational resource to understand the inner workings of various ML models without relying on high-level frameworks like scikit-learn for the core logic.

## 📑 Table of Contents
- [Machine Learning Algorithms From Scratch](#machine-learning-algorithms-from-scratch)
  - [📑 Table of Contents](#-table-of-contents)
  - [🧠 Implemented Algorithms](#-implemented-algorithms)
    - [Supervised Learning](#supervised-learning)
    - [Unsupervised Learning](#unsupervised-learning)
  - [🛠️ Utilities \& Metrics](#️-utilities--metrics)
  - [📂 Project Structure](#-project-structure)
  - [⚙️ Installation](#️-installation)
  - [🚀 Usage Examples](#-usage-examples)
    - [Classification (e.g., KNN)](#classification-eg-knn)
    - [Regression (e.g., Linear Regression)](#regression-eg-linear-regression)
  - [🗺️ Roadmap](#️-roadmap)

## 🧠 Implemented Algorithms

The repository is organized by machine learning categories. As new algorithms are added, they will be tracked here.

### Supervised Learning
**Classification**
- [x] **K-Nearest Neighbors (KNN)**: Custom implementation using Euclidean distance to find the *k* closest data points.

**Regression**
- [x] **Linear Regression**: Implemented via Gradient Descent with customizable learning rates and epochs. Includes a built-in visualization script.

### Unsupervised Learning
*(Algorithms coming soon)*

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
│
├── KNN.py                     # Algorithm implementations 
├── LinearRegression.py        
└── ...

```

## ⚙️ Installation

1. Clone the repository:
```bash
git clone [https://github.com/Youssef3082004/ML_Algorithms_From_Scratch.git](https://github.com/Youssef3082004/ML_Algorithms_From_Scratch.git)
cd ML_Algorithms_From_Scratch

```


2. Install the required dependencies:
```bash
pip install numpy matplotlib scikit-learn

```


*(Note: `scikit-learn` is only used to generate dummy datasets for testing).*

## 🚀 Usage Examples

All algorithms share a consistent, scikit-learn-style API (`.fit()` and `.predict()`), making it simple to swap models in and out.

### Classification (e.g., KNN)

```python
from KNN import KNN

# Initialize and fit
model = KNN(k=5)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

```

### Regression (e.g., Linear Regression)

```python
from LinearRegression import LinearRegression

# Initialize and fit
regressor = LinearRegression(Learning_rate=0.01, iterations=1000)
regressor.fit(X_train, y_train)

# Predict
predictions = regressor.predict(X_test)

```

*(Run `python LinearRegression.py` directly to see a demonstration and a matplotlib visualization of the regression line).*

## 🗺️ Roadmap

Future updates will expand the repository with more algorithms. Planned additions include:


* [x] K Nearest Neighbors
* [x] Linear Regression
* [ ] Logistic Regression
* [ ] Naive Bayes
* [ ] Perceptron
* [ ] SVM
* [ ] Decision Tree
* [ ] Random Forest
* [ ] Principal Component Analysis (PCA)
* [ ] K-Means
* [ ] AdaBoost
* [ ] Linear Discriminant Analysis (LDA)

