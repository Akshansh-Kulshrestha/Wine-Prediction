import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
red_wine = pd.read_csv('C:/Users/Akshansh/OneDrive/Desktop/Chinay/Assessment 4/Code/pythonProject/winequality-red.csv')
white_wine = pd.read_csv('C:/Users/Akshansh/OneDrive/Desktop/Chinay/Assessment 4/Code/pythonProject/'
                         'winequality-white.csv')

# Inspect the datasets
print("Red Wine Dataset")
print(red_wine.info())
print(red_wine.describe())

print("\nWhite Wine Dataset")
print(white_wine.info())
print(white_wine.describe())

# Check for missing values
print("\nMissing values in Red Wine Dataset")
print(red_wine.isnull().sum())

print("\nMissing values in White Wine Dataset")
print(white_wine.isnull().sum())

# Visualize the distributions of features
def plot_distributions(df, title):
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    axes = axes.flatten()

    for idx, col in enumerate(df.columns[:-1]):  # Exclude the target variable 'quality'
        sns.histplot(df[col], kde=True, ax=axes[idx])
        axes[idx].set_title(col)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

plot_distributions(red_wine, "Red Wine Features Distribution")
plot_distributions(white_wine, "White Wine Features Distribution")

# Correlation matrix
def plot_correlation_matrix(df, title):
    plt.figure(figsize=(12, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(title)
    plt.show()

plot_correlation_matrix(red_wine, "Red Wine Correlation Matrix")
plot_correlation_matrix(white_wine, "White Wine Correlation Matrix")

# Save cleaned data (if necessary)
red_wine.to_csv('/path/to/cleaned_winequality-red.csv', index=False)
white_wine.to_csv('/path/to/cleaned_winequality-white.csv', index=False)

