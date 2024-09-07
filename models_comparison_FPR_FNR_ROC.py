import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import logging
import joblib
import os

# Setup
if not os.path.exists('results'):
    os.makedirs('results')

logging.basicConfig(filename='results/final_fpr_fnr__model_comparison.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

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

# Result analysis
results_data = []
roc_data = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    fp, fn = confusion_matrix(y_test, predictions).ravel()[1], confusion_matrix(y_test, predictions).ravel()[2]
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    
    # Store ROC data for plot
    roc_data.append((name, fpr, tpr, roc_auc))
    
    # Store results data
    results_data.append([name, fp, fn, roc_auc])

    # Logging results
    logging.info(f"Model: {name}, FP: {fp}, FN: {fn}, ROC AUC: {roc_auc}")

# Save results to CSV
results_df = pd.DataFrame(results_data, columns=['Model', 'False Positives', 'False Negatives', 'ROC AUC'])
results_df.to_csv('results/model_performance.csv', index=False)

# ROC Curve Plot
plt.figure(figsize=(10, 8))
for name, fpr, tpr, roc_auc in roc_data:
    plt.plot(fpr, tpr, label=f'{name} (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Various Models')
plt.legend(loc="lower right")
plt.savefig('results/fpr_fnr_roc_curves.png')
plt.close()

print("Analysis complete. Results and plots saved.")
