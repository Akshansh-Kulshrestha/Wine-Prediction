import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import joblib

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

# Harmonize column names if necessary
red_wine.columns = white_wine.columns

# Combine datasets for unified processing
wine_data = pd.concat([red_wine, white_wine], ignore_index=True)

# Check for duplicates
print(wine_data.duplicated().sum())

# Drop duplicates if any
wine_data = wine_data.drop_duplicates()

# Rescale features

scaler = StandardScaler()
features = wine_data.drop('quality', axis=1)
features_scaled = scaler.fit_transform(features)

# Prepare final dataset
X = features_scaled
y = wine_data['quality']


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Random Forest as a benchmark model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))


# Perform Grid Search for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Evaluate best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best))


# Save the model
joblib.dump(best_model, 'best_wine_quality_model.pkl')

