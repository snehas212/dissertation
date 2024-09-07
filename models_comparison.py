import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import logging
import joblib
import os
import time

# Create results folder if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Set up logging
logging.basicConfig(filename='results/final_model_comparison.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Load datasets
train = pd.read_csv('/home/sneha/Sentiment Analysis/Datasets/drugs_review/drugsComTrain_raw.tsv', delimiter='\t')
test = pd.read_csv('/home/sneha/Sentiment Analysis/Datasets/drugs_review/drugsComTest_raw.tsv', delimiter='\t')

# Preprocessing

print("Train dataset shape: ", train.shape)
print("Test dataset shape: ", test.shape)

# Save a plot of the rating distribution in the training set
plt.figure(figsize=(10, 6))
sns.countplot(train['rating'])
plt.title('Rating Distribution in Training Set')
plt.savefig('results/rating_distribution_train.png')
plt.show()

# Checking missing values in training and testing datasets
print("Missing values in training data:", train.isnull().sum())
print("Missing values in testing data:", test.isnull().sum())

# Convert text to vectors
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
X_train = tfidf.fit_transform(train['review'].fillna(' ')).toarray()
X_test = tfidf.transform(test['review'].fillna(' ')).toarray()

# Create sentiment from ratings
train['sentiment'] = np.where(train['rating'] > 5, 1, 0)
test['sentiment'] = np.where(test['rating'] > 5, 1, 0)
y_train = train['sentiment']
y_test = test['sentiment']

# Model training and evaluation
models = {
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': MultinomialNB(),
    'ANN': MLPClassifier(max_iter=300),
    'LGBM': LGBMClassifier()
}

results = {}

for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
    pr_auc = auc(recall, precision)
    results[name] = (accuracy, train_time, roc_auc, pr_auc)

    # Log results
    logging.info(f'{name} Model: \nAccuracy: {accuracy}\nROC AUC: {roc_auc}\nPR AUC: {pr_auc}\nTrain Time: {train_time}s\n')

    # Confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {name}')
    plt.savefig(f'results/{name}_confusion_matrix.png')
    plt.close()

    # Save the model and vectorizer if it's Random Forest
    if name == 'Random Forest':
        joblib.dump(model, 'results/random_forest_model.pkl')
        joblib.dump(tfidf, 'results/tfidf_vectorizer.pkl')

# Generate and save comparison plots
metrics = ['Accuracy', 'Train Time', 'ROC AUC', 'PR AUC']
for metric in metrics:
    plt.figure(figsize=(10, 6))
    values = [results[model][metrics.index(metric)] for model in models]
    sns.barplot(x=list(models.keys()), y=values, palette='viridis')
    plt.title(f'{metric} Comparison')
    plt.ylabel(metric)
    plt.savefig(f'results/{metric.lower()}_comparison.png')
    plt.close()

# Save vectorizer and all model metrics to the log
logging.info(f"Vectorizer and all models' metrics have been logged and saved.")
