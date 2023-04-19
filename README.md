# Fraud Detection using Random Forest and Gaussian Anomaly Detection

This project is about fraud detection using machine learning algorithms. The primary focus is on two algorithms, Random Forest and Gaussian Anomaly Detection.

## Project Dependencies
- numpy
- pandas
- scipy
- sklearn

## Dataset
The dataset used in this project is `transactions1.csv`. It is used for training the machine learning model for fraud detection.

## Workflow
1. Reading the `transactions1.csv` dataset using pandas.
2. Determining the count of fraudulent and non-fraudulent transactions.
3. Estimating Gaussian parameters for non-fraudulent transactions.
4. Evaluating the probabilities for non-fraudulent and fraudulent transactions.
5. Determining the best threshold.
6. Using the Random Forest Classifier to identify feature importance.
7. Dropping features with low importance.
8. Preparing training, cross-validation, and testing datasets.
9. Training the Random Forest model using the training dataset.
10. Evaluating the model on the cross-validation dataset.
11. Determining the best threshold using F1 score.
12. Evaluating the model on the testing dataset using the best threshold.

## How to Run the Project
1. Install the required dependencies.
2. Download the `transactions1.csv` dataset.
3. Run the project in your Python environment.

## Result
The project outputs the best F1 score, recall, and precision score for the cross-validation dataset. It also outputs the feature importance of the Random Forest Classifier. The final result is the best F1 score, recall, and precision score for the testing dataset using the best threshold.
