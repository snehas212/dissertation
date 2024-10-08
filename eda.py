import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the datasets
train = pd.read_csv('/home/sneha/Sentiment Analysis/Datasets/drugs_review/drugsComTrain_raw.tsv', delimiter='\t')
test = pd.read_csv('/home/sneha/Sentiment Analysis/Datasets/drugs_review/drugsComTest_raw.tsv', delimiter='\t')

# Combine the training and test datasets for a complete EDA
data = pd.concat([train, test], ignore_index=True)

# Initial Data Examination
print(data.describe())
print(data.info())
print(data.columns)

# Data Cleaning
# Check for missing values
print(data.isnull().sum())

# Handling missing values - example: fill missing with the median or mode
data['rating'] = data['rating'].fillna(data['rating'].median())

# Remove duplicates
data.drop_duplicates(inplace=True)

# Data Visualization
# Distribution of ratings
plt.figure(figsize=(10, 6))
sns.countplot(x='rating', data=data)
plt.title('Distribution of Ratings')
plt.show()

# Word Cloud for Review Text
text = ' '.join(review for review in data.review.dropna())
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Feature Engineering
# Text processing - here we just show tokenization, more can be done as per the NLP preprocessing
data['processed_text'] = data['review'].str.lower().str.replace('[^\w\s]', '')

# Convert categorical variables using one-hot encoding
data = pd.get_dummies(data, columns=['condition'], drop_first=True)

# Outlier Detection
# Simple method using IQR
Q1 = data['rating'].quantile(0.25)
Q3 = data['rating'].quantile(0.75)
IQR = Q3 - Q1
outlier_condition = (data['rating'] < (Q1 - 1.5 * IQR)) | (data['rating'] > (Q3 + 1.5 * IQR))
print("Number of outliers in rating:", outlier_condition.sum())

# Save the clean dataset for further analysis
data.to_csv('clean_drug_reviews.csv', index=False)
