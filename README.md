# Machine Learning Implementation Tasks

## 📋 Table of Contents
- [Overview](#overview)
- [Tasks Completed](#tasks-completed)
  - [Task 1: Decision Tree](#task-1-decision-tree)
  - [Task 2: K-Nearest Neighbors (KNN)](#task-2-k-nearest-neighbors-knn)
  - [Task 3: K-Means Clustering](#task-3-k-means-clustering)
  - [Task 4: Linear Regression](#task-4-linear-regression)
  - [Task 5: Naive Bayes Classifier](#task-5-naive-bayes-classifier)
  - [Machine Learning Fundamentals](#machine-learning-fundamentals)
    - [PCA on KNN (Digit Recognition)](#pca-on-knn-digit-recognition)
    - [POS Tagging using Spacy](#pos-tagging-using-spacy)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)

---

## 🎯 Overview

This repository contains a comprehensive collection of Machine Learning implementation tasks completed as part of the Information Technology course. Each task demonstrates the practical application of various ML algorithms, including supervised learning (classification and regression), unsupervised learning (clustering), and advanced techniques like dimensionality reduction using PCA.

The implementations showcase hands-on experience with popular Python libraries such as scikit-learn, pandas, numpy, and matplotlib, along with model evaluation, data preprocessing, and visualization techniques.

---

## ✅ Tasks Completed

### Task 1: Decision Tree
**Objective:** Implement a Decision Tree Classifier for breast cancer classification

**Description:**  
This task implements a Decision Tree Classifier to predict breast cancer diagnosis (malignant or benign) using the breast cancer dataset from scikit-learn. The implementation includes:
- Loading and preprocessing the breast cancer dataset
- Feature correlation analysis
- Stratified K-Fold cross-validation
- Model training and evaluation using accuracy metrics
- Visualization of confusion matrix and decision boundaries

**Key Features:**
- Binary classification (malignant vs benign)
- Feature importance analysis
- Performance metrics: accuracy, precision, recall, F1-score
- Confusion matrix visualization using seaborn

**Dataset:** Breast Cancer Wisconsin Dataset (scikit-learn)  
**Algorithm:** Decision Tree Classifier  
**Evaluation:** Accuracy score, Classification report, Confusion matrix

---

### Task 2: K-Nearest Neighbors (KNN)
**Objective:** Diabetes prediction using K-Nearest Neighbors algorithm

**Description:**  
This task implements the KNN algorithm to predict diabetes outcomes based on medical diagnostic measurements. The implementation covers:
- Data exploration and statistical analysis
- Handling missing or zero values in medical features
- Feature scaling using StandardScaler
- Finding optimal K value through experimentation
- Model training and testing with performance evaluation

**Key Features:**
- Multi-feature analysis (glucose, blood pressure, BMI, age, etc.)
- Data standardization for improved KNN performance
- Hyperparameter tuning (K optimization)
- Comprehensive classification metrics

**Dataset:** Pima Indians Diabetes Dataset  
**Algorithm:** K-Nearest Neighbors (KNN)  
**Evaluation:** Accuracy, Confusion matrix, Classification report

---

### Task 3: K-Means Clustering
**Objective:** Social media engagement analysis using K-Means clustering

**Description:**  
This task applies K-Means clustering algorithm to segment social media posts based on user engagement metrics. The implementation includes:
- Loading and exploring Facebook Live dataset
- Feature selection (reactions, comments, shares, likes, loves, wows)
- Data standardization and preprocessing
- Determining optimal number of clusters using Elbow method
- Cluster visualization and interpretation

**Key Features:**
- Unsupervised learning approach for pattern discovery
- Elbow method for optimal cluster selection
- Multi-dimensional feature analysis
- Cluster profiling and insights extraction

**Dataset:** Facebook Live Dataset  
**Algorithm:** K-Means Clustering  
**Evaluation:** Within-cluster sum of squares (WCSS), Silhouette analysis

---

### Task 4: Linear Regression
**Objective:** Predict per capita income using Linear Regression

**Description:**  
This task implements Simple Linear Regression to predict Canada's per capita income based on historical data. The implementation demonstrates:
- Single-feature regression modeling
- Train-test split methodology
- Model fitting and coefficient analysis
- Prediction and visualization of regression line
- Model evaluation using R² score and Mean Squared Error

**Key Features:**
- Time-series analysis approach
- Trend line visualization
- Performance metrics: R², MSE, RMSE
- Prediction vs actual value comparison

**Dataset:** Canada Per Capita Income Dataset  
**Algorithm:** Simple Linear Regression  
**Evaluation:** R² Score, Mean Squared Error (MSE), Root Mean Squared Error (RMSE)

---

### Task 5: Naive Bayes Classifier
**Objective:** Breast cancer classification using Naive Bayes algorithm

**Description:**  
This task implements the Naive Bayes classifier for breast cancer diagnosis prediction. The implementation includes:
- Comprehensive exploratory data analysis (EDA)
- Feature distribution analysis using histogram visualizations
- Data preprocessing and feature selection
- Gaussian Naive Bayes model training
- Performance evaluation with classification metrics

**Key Features:**
- Probabilistic classification approach
- Feature-wise distribution analysis
- Interactive visualizations using Plotly
- Analysis of features like area_mean, radius_mean, perimeter_mean
- Diagnosis prediction (Malignant vs Benign)

**Dataset:** Breast Cancer Dataset (CSV)  
**Algorithm:** Gaussian Naive Bayes  
**Evaluation:** Accuracy score, Confusion matrix, Classification report

---

## 🔬 Machine Learning Fundamentals

### PCA on KNN (Digit Recognition)
**Objective:** Dimensionality reduction using PCA for efficient digit classification

**Description:**  
This advanced implementation demonstrates the application of Principal Component Analysis (PCA) for dimensionality reduction in handwritten digit recognition. The project showcases how PCA can significantly reduce computational complexity while maintaining model accuracy.

**Implementation Details:**
- **Original Dataset:** MNIST handwritten digits (28×28 pixels = 784 features)
- **Data Preprocessing:** StandardScaler for feature normalization
- **Dimensionality Reduction:** PCA to reduce 784 features to 3 principal components
- **Classification:** K-Nearest Neighbors on reduced feature space
- **Visualization:** 3D scatter plots showing digit clusters in PCA space

**Key Achievements:**
- Reduced features from 784 to 3 (99.6% reduction)
- Maintained competitive accuracy with minimal features
- Significantly decreased prediction time from 9+ seconds to milliseconds
- Visual representation of high-dimensional data in 3D space

**Performance Comparison:**
- **Without PCA:** 784 features, ~9.16 seconds prediction time
- **With PCA:** 3 features, <1 second prediction time
- Accuracy maintained with substantial computational efficiency gain

#### 3D Visualization of PCA-Reduced Feature Space

**View 1: Digits PCA Embedding (Angle 25°/35°)**

![PCA 3D View 1](Machine%20Learning%20Fundamentals/PCA_On_KNN/assets/pca_knn_3d_view1.png)

*The first 3D visualization shows the digit dataset projected onto three principal components (PC1, PC2, PC3). Each color represents a different digit (0-9), demonstrating how PCA preserves the separability of different digit classes even in reduced 3D space. The clustering of similar colors indicates that digits with similar features are grouped together in the principal component space.*

**View 2: Digits PCA Embedding (Angle 10°/120°)**

![PCA 3D View 2](Machine%20Learning%20Fundamentals/PCA_On_KNN/assets/pca_knn_3d_view2.png)

*The second view provides an alternative perspective of the same PCA-reduced feature space. This angle better reveals the separation between certain digit clusters, particularly showing how digits like 0, 1, and 2 (represented by blue, orange, and green) occupy distinct regions in the 3D space. The visualization demonstrates that despite reducing from 784 dimensions to just 3, the essential discriminative information is preserved.*

**Key Insights from Visualizations:**
- Clear separation between digit classes in 3D space
- Some digits naturally cluster tightly (e.g., digit 1)
- Certain digits show more variance (e.g., digits 4, 7, 9)
- The three principal components capture most of the variance needed for classification
- Different viewing angles reveal different aspects of the data distribution

**Dataset:** MNIST Handwritten Digits (42,000 samples)  
**Techniques:** PCA, KNN, Feature Scaling  
**Result:** Dramatic computational efficiency improvement with preserved accuracy

---

### POS Tagging using Spacy
**Objective:** Natural Language Processing - Part-of-Speech tagging

**Description:**  
This task implements Part-of-Speech (POS) tagging using the Spacy NLP library. The implementation demonstrates:
- Loading pre-trained language models (en_core_web_sm)
- Automatic model download if not present
- Text tokenization and processing
- POS tag extraction for each token
- Understanding linguistic structure of sentences

**Key Features:**
- Automated NLP pipeline using Spacy
- Token-level POS identification (NOUN, VERB, PROPN, etc.)
- Efficient processing of natural language text
- Foundation for advanced NLP tasks

**Example Analysis:**
```python
Text: "I love programming in Python"
Tokens: ['I', 'love', 'programming', 'in', 'Python']
POS Tags: ['PRON', 'VERB', 'VERB', 'ADP', 'PROPN']
```

**Library:** Spacy (en_core_web_sm model)  
**Application:** Text analysis, linguistic feature extraction

---

## 🛠️ Technologies Used

### Programming Language
- Python 3.x

### Core Libraries
- **NumPy:** Numerical computing and array operations
- **Pandas:** Data manipulation and analysis
- **Matplotlib:** Data visualization and plotting
- **Seaborn:** Statistical data visualization
- **Plotly Express:** Interactive visualizations

### Machine Learning
- **Scikit-learn:**
  - Model implementations (DecisionTree, KNN, KMeans, LinearRegression, NaiveBayes)
  - Preprocessing tools (StandardScaler, train_test_split)
  - Metrics (accuracy_score, confusion_matrix, classification_report)
  - Dimensionality reduction (PCA)

### Natural Language Processing
- **Spacy:** NLP library for POS tagging and text processing

### Development Tools
- Jupyter Notebook for interactive development
- Git for version control

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Jupyter Notebook

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/NoumanZahid-85/Machine-Learning-Labs.git
cd ML_Implementation_Tasks
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages:**
```bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn spacy
```

4. **Download Spacy language model:**
```bash
python -m spacy download en_core_web_sm
```

5. **Launch Jupyter Notebook:**
```bash
jupyter notebook
```

6. **Navigate to any task folder and open the respective .ipynb file**

---

## 📊 Key Learning Outcomes

Through these implementations, the following concepts and skills were developed:

1. **Supervised Learning:**
   - Classification algorithms (Decision Tree, KNN, Naive Bayes)
   - Regression techniques (Linear Regression)
   - Model evaluation and validation

2. **Unsupervised Learning:**
   - Clustering algorithms (K-Means)
   - Pattern recognition and data segmentation

3. **Dimensionality Reduction:**
   - Principal Component Analysis (PCA)
   - Feature extraction and selection
   - Computational efficiency optimization

4. **Data Preprocessing:**
   - Feature scaling and normalization
   - Handling missing values
   - Train-test splitting

5. **Model Evaluation:**
   - Accuracy metrics
   - Confusion matrices
   - Classification reports
   - Cross-validation techniques

6. **Data Visualization:**
   - 2D and 3D plotting
   - Statistical visualizations
   - Interactive charts

7. **Natural Language Processing:**
   - Text tokenization
   - POS tagging
   - NLP pipeline implementation

---

## 📈 Performance Summary

| Task | Algorithm | Dataset | Accuracy/Metric |
|------|-----------|---------|-----------------|
| Task 1 | Decision Tree | Breast Cancer | High Classification Accuracy |
| Task 2 | KNN | Diabetes | Optimized with K-tuning |
| Task 3 | K-Means | Facebook Live | Optimal Cluster Segmentation |
| Task 4 | Linear Regression | Canada Income | R² Score & Low MSE |
| Task 5 | Naive Bayes | Breast Cancer | Probabilistic Classification |
| PCA+KNN | Dimensionality Reduction | MNIST Digits | 99.6% Feature Reduction, Fast Prediction |
| POS Tagging | Spacy NLP | Text Data | Accurate Token Classification |

---

## 📄 License

This project is created for educational purposes as part of the Machine Learning course curriculum.

