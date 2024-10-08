import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import logging
import joblib
import os

# Setup
if not os.path.exists('results'):
    os.makedirs('results')

logging.basicConfig(filename='results/FPR_FNR_final_model_comparison.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Load datasets
train = pd.read_csv('/home/sneha/Sentiment Analysis/Datasets/drugs_review/drugsComTrain_raw.tsv', delimiter='\t')
test = pd.read_csv('/home/sneha/Sentiment Analysis/Datasets/drugs_review/drugsComTest_raw.tsv', delimiter='\t')

# Preprocessing
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
X_train = tfidf.fit_transform(train['review'].fillna(' ')).toarray()
X_test = tfidf.transform(test['review'].fillna(' ')).toarray()

train['sentiment'] = np.where(train['rating'] > 5, 1, 0)
test['sentiment'] = np.where(test['rating'] > 5, 1, 0)
y_train = train['sentiment']
y_test = test['sentiment']

# Modeling
models = {
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': MultinomialNB(),
    'ANN': MLPClassifier(max_iter=300),
    'Gradient Boosting': GradientBoostingClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    # Logging results
    logging.info(f"Model: {name}, FPR: {fpr}, FNR: {fnr}")

    # Plot for FPR
    plt.figure(figsize=(6, 4))
    sns.barplot(x=[name], y=[fpr])
    plt.title(f'False Positive Rate for {name}')
    plt.ylabel('False Positive Rate')
    plt.savefig(f'results/{name}_FPR.png')
    plt.close()

    # Plot for FNR
    plt.figure(figsize=(6, 4))
    sns.barplot(x=[name], y=[fnr])
    plt.title(f'False Negative Rate for {name}')
    plt.ylabel('False Negative Rate')
    plt.savefig(f'results/{name}_FNR.png')
    plt.close()

print("FPR and FNR plots saved in the results folder.")
