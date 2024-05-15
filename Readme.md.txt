# Mushroom Classification Project

This project explores different machine learning algorithms to classify mushrooms into edible or poisonous categories based on various features.

## Dataset

The dataset used in this project is the Mushroom Classification dataset from the UCI Machine Learning Repository. The dataset contains various features such as cap shape, cap color, gill color, etc., and the target variable indicating whether the mushroom is edible or poisonous.

## Algorithms Explored

1. **Support Vector Machine (SVM) Classifier:**
   - SVM is a supervised machine learning algorithm used for classification and regression tasks.
   - We used the radial basis function (RBF) kernel for this SVM.
   - Trained the SVM model using the training set and predicted the class labels for the test set.

2. **Random Forest Classifier:**
   - Random Forest is an ensemble learning method used for classification and regression tasks.
   - It operates by constructing a multitude of decision trees at training time.
   - Trained the Random Forest model using the training set and predicted the class labels for the test set.

3. **K-Nearest Neighbors (KNN) Classifier:**
   - KNN is a non-parametric and lazy learning algorithm used for classification and regression tasks.
   - We set the number of nearest neighbors to 73.
   - Trained the KNN model using the training set and predicted the class labels for both the training and testing sets.

4. **Decision Tree Classifier:**
   - Decision Trees are a supervised learning method used for classification and regression tasks.
   - It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed.
   - Trained the Decision Tree model using the training set and predicted the class labels for the test set.

5. **Naive Bayes Classifier:**
   - Naive Bayes is a family of probabilistic algorithms based on Bayes' Theorem.
   - It assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.
   - Trained the Naive Bayes model using the training set and predicted the class labels for the test set.

6. **Logistic Regression Classifier:**
   - Logistic Regression is a linear model used for binary classification.
   - Trained the Logistic Regression model using the training set and predicted the class labels for the test set.

## Results

| Algorithm                   | Accuracy | Precision | Recall | F1 Score | ROC AUC Score |
|-----------------------------|----------|-----------|--------|----------|---------------|
| SVM                         |   90.8   |    91.6   |  91.5  |   91.5   |     90.7      |
| Random Forest               |  99.03   |    99.1   |  99.1  |   99.1   |     99.03     |
| KNN                         |   98.9   |    98.8   |  99.2  |   99.04  |     99.6      |
| Decision Tree               |   97.8   |    97.8   |  98.2  |   98.01  |     97.7      |
| Naive Bayes                 |   63.7   |    62.2   |  69.7  |   65.7   |     59.4      |
| Logistic Regression         |   64.1   |    57.2   |   57.1 |   57.1   |       -       |




### Setup on Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Click on `File` > `Upload notebook...` and upload the provided notebook file (`mushroom_classification.ipynb`).
3. Upload the dataset `mushroom_cleaned.csv` in the same directory.

### Running the Notebook

1. Open the notebook `mushroom_classification.ipynb` on Google Colab.
2. Follow the instructions given in the notebook to execute each code cell.
3. View the results and analyze the performance of each machine learning algorithm.
