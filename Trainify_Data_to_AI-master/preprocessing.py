import pandas as pd
import numpy as np
import warnings
import joblib
import os
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer


def supervised_preprocessing(csv_path, target_column, user_id):
    df = pd.read_csv(csv_path)
    num_selected_features = df.shape[1] - 1

    X = df.drop(columns=[target_column])
    y = df[target_column]
    classes = X.columns.tolist()

    labels_directory = "User_labels"
    os.makedirs(labels_directory, exist_ok=True)
    labels_file_path = os.path.join(labels_directory, f"{user_id}_original_labels.txt")
    if y.dtypes == 'object':
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        with open(labels_file_path, 'w') as file:
            for label in label_encoder.classes_:
                file.write("%s\n" % label)
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
        # print("Classification================")
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
        # print("Regression===================")

    best_model_name = None
    best_metric = float('-inf') if y.dtypes == 'object' else float('-inf')
    accuracy = None
    precision = None
    recall = None
    best_model = None

    for model_name, model in models:
        model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('selector', feature_selector),
                                         ('model', model)])

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)

        if y.dtypes == 'object':
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            metric = f1_score(y_test, y_pred, average='weighted')

        else:

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metric = r2

        if (y.dtypes == 'object' and metric > best_metric) or (y.dtypes != 'object' and metric > best_metric):
            best_metric = metric
            best_model_name = model_name
            best_model = model_pipeline

        if best_model is not None:
            directory_path = "User_models"
            os.makedirs(directory_path, exist_ok=True)
            joblib.dump(best_model, os.path.join(directory_path, f"{user_id}_Best_Model.pkl"))

    return best_model_name, best_metric, accuracy, precision, recall, classes


def find_optimal_clusters(X, max_clusters=10):
    silhouette_scores = []

    for k in range(2, max_clusters + 1):
        kmeans_model = KMeans(n_clusters=k, random_state=42)
        kmeans_labels = kmeans_model.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, kmeans_labels))

    return silhouette_scores


def generic_clustering(csv_path, max_clusters=10):
    df = pd.read_csv(csv_path)

    # print("Original Data:")
    # print(df.head())

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

        # print(f'\nBest Clustering Algorithm: K-Means')

        df['Cluster_Labels_KMeans'] = kmeans_labels
        # print(df[['Cluster_Labels_KMeans']])

        return optimal_num_clusters, round(silhouette_kmeans, 3), kmeans_labels

# Example usage:
