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
1. Load dataset containing transaction records.
2. Get the count of each class (normal vs. fraudulent transactions).
3. Define a function to estimate Gaussian parameters (mean and variance) based on the training data.
4. Define a function to evaluate the multivariate Gaussian distribution for calculating the probability of each data point being normal or anomalous.
5. Train a random forest classifier to select the most important features in the dataset.
6. Drop features with low importance based on feature importance scores obtained from the random forest classifier.
7. Compress and save the modified dataset for future use.
8. Remove "Amount" and "Time" columns which are not relevant for detecting fraud.
9. Split dataset into normal and anomalous transactions.
10. Split anomalous transactions into cross-validation and test sets.
11. Split normal transactions into train, cross-validation, and test sets.
12. Combine cross-validation and test sets of normal and anomalous transactions.
13. Separate features and labels from cross-validation and test sets.
14. Convert labels to numpy arrays.
15. Estimate Gaussian parameters for normal samples in the training set.
16. Evaluate probabilities of training set with estimated parameters.
17. Evaluate probabilities of cross-validation set with estimated parameters.
18. Find the best threshold value based on F1 score of cross-validation set.
19. Loop over different epsilon values to find the best threshold.
20. Initialize variables to calculate precision, recall, and F1 score.
21. Loop over all validation examples.
22. If probability is less than current threshold, classify as anomaly.
23. Increment true positives, false positives, and false negatives as appropriate.
24. Calculate precision, recall, and F1 score based on the values obtained in step 
25. Keep track of best F1 score and corresponding threshold value.
26. Output best threshold value and F1 score.
27. Evaluate performance of model on test set using best threshold value obtained in step 
28. Output precision, recall, and F1 score on the test set.

## How to Run the Project
1. Install the required dependencies.
2. Download the `transactions1.csv` dataset.
3. Run the project in your Python environment.

## Result
The project outputs the best F1 score, recall, and precision score for the cross-validation dataset. It also outputs the feature importance of the Random Forest Classifier. The final result is the best F1 score, recall, and precision score for the testing dataset using the best threshold.
