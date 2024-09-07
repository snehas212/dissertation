import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load datasets
# Load the datasets
train = pd.read_csv('/home/sneha/Sentiment Analysis/Datasets/drugs_review/drugsComTrain_raw.tsv', delimiter='\t')
test = pd.read_csv('/home/sneha/Sentiment Analysis/Datasets/drugs_review/drugsComTest_raw.tsv', delimiter='\t')

# Ensure the results directory exists
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Basic EDA: Class Distribution in Training Data
plt.figure(figsize=(8, 6))
sns.countplot(x='rating', data=train)
plt.title('Class Distribution of Ratings in Training Data')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.savefig(f'{results_dir}/class_distribution_train.png')
plt.close()

# Class Distribution in Test Data
plt.figure(figsize=(8, 6))
sns.countplot(x='rating', data=test)
plt.title('Class Distribution of Ratings in Test Data')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.savefig(f'{results_dir}/class_distribution_test.png')
plt.close()

# Analyze class distribution based on conditions
# Top 10 conditions
top_conditions = train['condition'].value_counts().head(10).index

# Filter data to include only top conditions
train_top_conditions = train[train['condition'].isin(top_conditions)]

plt.figure(figsize=(12, 8))
sns.countplot(y='condition', hue='rating', data=train_top_conditions, palette='viridis')
plt.title('Class Distribution Based on Conditions for Top 10 Conditions')
plt.xlabel('Count')
plt.ylabel('Condition')
plt.legend(title='Rating', loc='upper right')
plt.savefig(f'{results_dir}/class_distribution_by_condition.png')
plt.close()

# General Descriptive Statistics
desc_stats = train.describe()
desc_stats.to_csv(f'{results_dir}/descriptive_statistics.csv')

print("Exploratory data analysis completed and results saved.")

# Filtering the top 10 conditions
top_10_conditions = train['condition'].value_counts().head(10).index
filtered_data = train[train['condition'].isin(top_10_conditions)]

# Filtering the top 20 drugs within the top 10 conditions
top_20_drugs = filtered_data['drugName'].value_counts().head(20).index
filtered_data = filtered_data[filtered_data['drugName'].isin(top_20_drugs)]

# Plotting
plt.figure(figsize=(14, 10))
ax = sns.boxplot(data=filtered_data, x='drugName', y='rating', hue='condition')
plt.xticks(rotation=90)
plt.title('Ratings of Top 20 Drugs within Top 10 Conditions')
plt.xlabel('Drug Name')
plt.ylabel('Rating')

# Enhancing the legend - larger font size and placing it inside the plot area
plt.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='large', title_fontsize='x-large')

# Save the plot in the results folder
plt.savefig(f'{results_dir}/top_drugs_conditions_ratings.png', bbox_inches='tight')  # Adjust for the legend
plt.close()

print("Plot saved in the results folder.")
