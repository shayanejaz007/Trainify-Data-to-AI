import pandas as pd
import numpy as np
import warnings

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.model_selection import train_test_split

# Regression Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Classification Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

# Clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer


def supervised_preprocessing(csv_path, target_column, num_selected_features):
    df = pd.read_csv(csv_path)

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

    if y.dtypes == 'object':
        feature_selector = SelectKBest(score_func=f_classif, k=num_selected_features)
        models = [
            ('DecisionTreeClassifier', DecisionTreeClassifier()),
            ('SVC', SVC()),
            ('LogisticRegression', LogisticRegression()),
            ('RandomForestClassifier', RandomForestClassifier()),
            ('KNeighborsClassifier', KNeighborsClassifier()),
            ('MLPClassifier', MLPClassifier(max_iter=5000)),
            ('XGBClassifier', XGBClassifier())
        ]
        print("Classification================")
    else:
        feature_selector = SelectKBest(score_func=f_regression, k=num_selected_features)
        models = [
            ('DecisionTreeRegressor', DecisionTreeRegressor()),
            ('SVR', SVR()),
            ('LinearRegression', LinearRegression()),
            ('ElasticNet', ElasticNet()),
            ('RandomForestRegressor', RandomForestRegressor()),
            ('KNeighborsRegressor', KNeighborsRegressor()),
            ('MLPRegressor', MLPRegressor(max_iter=5000)),
            ('XGBRegressor', XGBRegressor())
        ]
        print("Regression===================")

    best_model_name = None
    best_metric = float('-inf') if y.dtypes == 'object' else float('-inf')

    for model_name, model in models:
        model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('selector', feature_selector),
                                         ('model', model)])

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)

        if y.dtypes == 'object':
            metric = accuracy_score(y_test, y_pred)

            f1 = f1_score(y_test, y_pred, average='weighted')
            print(f"\nModel: {model_name}")
            print(f"F1 Score: {f1}")

            print(f"Performance Metric: {int(metric * 100)}%")

        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"\nModel: {model_name}")
            print(f"r2_Score: {int(r2 * 100)}%")
            # metric = r2 if y.dtypes == 'object' else -mse
            metric = r2

            print(f"Performance Metric: {-mse}")

        # Choose the model with the highest accuracy or the lowest MSE/r2 as the best model
        if (y.dtypes == 'object' and metric > best_metric) or (y.dtypes != 'object' and metric > best_metric):
            best_metric = metric
            best_model_name = model_name

    print(f"\nBest Model: {best_model_name} (based on Performance Metric)")
    return X, y_encoded, best_model_name


def find_optimal_clusters(X, max_clusters=10):
    silhouette_scores = []

    for k in range(2, max_clusters + 1):
        kmeans_model = KMeans(n_clusters=k, random_state=42)
        kmeans_labels = kmeans_model.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, kmeans_labels))

    return silhouette_scores


def generic_clustering(csv_path, column, max_clusters=10):
    df = pd.read_csv(csv_path)

    print("Original Data:")
    print(df.head())

    # Drop the specified column
    df = df.drop(column, axis=1)

    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
        ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        processed_data = preprocessor.fit_transform(df)
        silhouette_scores = find_optimal_clusters(processed_data, max_clusters)

        optimal_num_clusters = np.argmax(silhouette_scores) + 2  # Adding 2 to get the actual number of clusters

        kmeans_model = KMeans(n_clusters=optimal_num_clusters, random_state=42)
        kmeans_labels = kmeans_model.fit_predict(processed_data)

        silhouette_kmeans = silhouette_score(processed_data, kmeans_labels)

        print(f'\nBest Clustering Algorithm: K-Means')
        print(f'Optimal Number of Clusters: {optimal_num_clusters}')
        print(f'Best Silhouette Score: {silhouette_kmeans:.4f}')

        df['Cluster_Labels_KMeans'] = kmeans_labels
        print(df[['Cluster_Labels_KMeans']])


# Example usage:

