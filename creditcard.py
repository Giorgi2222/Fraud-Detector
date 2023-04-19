import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score

# Load the dataset
train_df = pd.read_csv(r"C:\Users\Giorgi\Desktop\Fraud_Detector\transactions1.csv")

# Get the count of each class
x = train_df['Class'].value_counts().index
y = train_df['Class'].value_counts().values

# Function to estimate Gaussian parameters
def estimate_gaussian(X): 
    m, n = X.shape
    mu = np.mean(X, axis=0)
    var = np.zeros(n,)
    for i in range(n):
        var[i] = np.sum((X[:, i] - mu[i])**2) / m        
    return mu, var

# Function to evaluate the multivariate Gaussian distribution
def multivariate_gaussian(X, mu, sigma):
    p = multivariate_normal(mean=mu, cov=sigma)
    return p.pdf(X)

# Get the names of the features
v_features = train_df.iloc[:,1:29].columns

# Train a random forest classifier to select the most important features
rnd_clf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
rnd_clf.fit(train_df.iloc[:,1:29], train_df.iloc[:,30])

# Drop the features with low importance
for name, importance in zip(train_df.iloc[:,1:29].columns, rnd_clf.feature_importances_):
    if importance < 0.020:
        train_df.drop([name], axis=1, inplace=True)

# Compress and save the modified dataset
compression_opts = dict(method='zip', archive_name='out.csv')  
train_df.to_csv('out.zip', index=False, compression=compression_opts)

# Drop the columns "Amount" and "Time"
train_df.drop(labels=["Amount", "Time"], axis=1, inplace=True)

# Split the dataset into normal and anomalous transactions
train_strip_v1 = train_df[train_df["Class"] == 1]
train_strip_v0 = train_df[train_df["Class"] == 0]

# Get the lengths of the normal and anomalous transactions
Normal_len = len(train_strip_v0)
Anomolous_len = len(train_strip_v1)

# Split the anomalous transactions into cross-validation and test sets
start_mid = Anomolous_len // 2
start_midway = start_mid + 1

train_cv_v1  = train_strip_v1 [: start_mid]
train_test_v1 = train_strip_v1 [start_midway:Anomolous_len]

# Split the normal transactions into train, cross-validation, and test sets
start_mid = (Normal_len * 60) // 100
start_midway = start_mid + 1

cv_mid = (Normal_len * 80) // 100
cv_midway = cv_mid + 1

X_train = train_strip_v0 [:start_mid]
train_cv = train_strip_v0 [start_midway:cv_mid]
train_test  = train_strip_v0 [cv_midway:Normal_len]

# Combine the cross-validation and test sets of the normal and anomalous transactions
train_cv = pd.concat([train_cv, train_cv_v1], axis=0)
train_test = pd.concat([train_test, train_test_v1], axis=0)

# Separate the features and labels from the cross-validation and test sets
y_val = train_cv["Class"]
y_test = train_test["Class"]
X_val = train_cv.drop(labels=["Class"], axis=1)
X_test = train_test.drop(labels=["Class"], axis=1)

# Convert labels to numpy arrays
y_val = y_val.to_numpy()

# Estimate the Gaussian parameters for normal samples in the training set
mu, sigma = estimate_gaussian(X_train.to_numpy())

# Evaluate the probabilities of the training set with estimated parameters
p = multivariate_gaussian(X_train, mu, sigma)

# Evaluate the probabilities of the cross validation set with estimated parameters
p_val = multivariate_gaussian(X_val, mu, sigma)

# Find the best threshold value based on F1 score of cross validation set
best_epsilon = 0
best_F1 = 0
step_size = (np.max(p_val) - np.min(p_val)) / len(p_val)

# Loop over different epsilon values to find the best threshold
m = y_val.shape[0] # number of validation examples
for epsilon in (1e-15, 1e-20, 1e-21, 1e-22, 1e-23, 1e-24, 1e-25, 1e-26, 1e-27, 1e-28, 1e-29, 1e-30, 1e-35, 1e-40, 1e-45, 1e-50, 1e-55):
    # Initialize variables to calculate precision, recall, and F1 score
    prec = 0
    rec = 0
    F1 = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Loop over all validation examples
    for i in range(m):
        # If probability is less than the current threshold, classify as anomaly
        if p_val[i] < epsilon:
            # If actual label is positive, increment true positives
            if y_val[i]:
                true_positives += 1
            # Otherwise, increment false positives
            else:
                false_positives += 1
        # Otherwise, classify as normal
        elif y_val[i]:
            # If actual label is positive but predicted as normal, increment false negatives
            false_negatives += 1

    # Calculate precision, recall, and F1 score
    try:
        prec = true_positives / (true_positives + false_positives)
        rec = true_positives / (true_positives + false_negatives)
        F1 = (2 * prec * rec)/(prec + rec)
    except ZeroDivisionError:
        print('Division by zero')
    
    # Update best F1 score and corresponding epsilon if necessary
    if F1 > best_F1:
        best_F1 = F1
        best_epsilon = epsilon

# Use best epsilon to make final predictions on test set
predictions = (p_val < best_epsilon)
Recall = recall_score(y_val, predictions, average = "binary")    
Precision = precision_score(y_val, predictions, average = "binary")
F1score = f1_score(y_val, predictions, average = "binary")

# Print evaluation metrics
print ('F1 score, Recall and Precision for Cross Validation dataset')
print ('Best F1 Score %f' %F1score)
print ('Best Recall Score %f' %Recall)
print ('Best Precision Score %f' %Precision)
print('Best epsilon found using cross-validation: %e'% best_epsilon)
print()
print('# Anomalies found: %d'% sum(p < best_epsilon))







