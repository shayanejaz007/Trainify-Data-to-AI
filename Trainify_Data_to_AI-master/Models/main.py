import pandas as pd
from preprocessing import supervised_preprocessing, generic_clustering
import os

Unsupervised = False
Supervised = True

Supervised_models = []
Unsupervised_models = []


def find_model_file(model_name):
    root_dir = 'Models'
    for model_type in os.listdir(root_dir):
        model_type_dir = os.path.join(root_dir, model_type)
        model_file_path = os.path.join(model_type_dir, f"{model_name}.py")
        if os.path.isfile(model_file_path):
            return model_file_path
    return None


if Unsupervised:
    csv_file_path = 'Customer-Data - 2.csv'
    column_to_drop = 'CUST_ID'

    generic_clustering(csv_file_path, column_to_drop)
    # path to download file
    model_name = "KMeans"
    file_path = find_model_file(model_name)
    print(file_path)

elif Supervised:
    csv_file_path = 'breast-cancer-data.csv'
    Your_target_column = 'class'

    df = pd.read_csv(csv_file_path)

    num_columns = df.shape[1]
    print("Features = ", num_columns)
    features = df.shape[1] - 1
    X_processed, y_processed, model_name = supervised_preprocessing(csv_file_path, Your_target_column, features)
    file_path = find_model_file(model_name)
    print(file_path)

else:
    pass
# print(model)
