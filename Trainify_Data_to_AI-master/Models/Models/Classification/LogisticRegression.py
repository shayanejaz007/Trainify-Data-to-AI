

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

# Load your data
csv_path = 'YOUR_DATA_CSV'
target_column = 'TARGET_COLUMN'
df = pd.read_csv(csv_path)

num_columns = df.shape[1]
print("Features = ", num_columns)
num_selected_features = df.shape[1] - 1

print("Original Data:")
print(df.head())

X = df.drop(columns=[target_column])
y = df[target_column]

if y.dtypes == 'object':
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
else:
    y_encoded = y

numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

feature_selector = SelectKBest(score_func=f_classif, k=num_selected_features)

model = LogisticRegression()

model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('selector', feature_selector),
                                  ('model', model)])

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

if y.dtypes == 'object':
    metric = accuracy_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"\nModel: Logistic Regression")
    print(f"F1 Score: {f1}")

    print(f"Performance Metric: {int(metric * 100)}%")


print(f"Model: Logistic Regression")