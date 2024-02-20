from preprocessing import supervised_preprocessing, generic_clustering
import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify, url_for, send_file

app = Flask(__name__)


def find_model_file(model_name):
    root_dir = 'Models'
    for model_type in os.listdir(root_dir):
        model_type_dir = os.path.join(root_dir, model_type)
        model_file_path = os.path.join(model_type_dir, f"{model_name}.py")
        if os.path.isfile(model_file_path):
            return model_file_path
    return None


@app.route('/download_python_file/<model_name>', methods=["GET"])
def download_python_file(model_name):
    try:
        file_path = find_model_file(model_name)

        if file_path is None:
            return "File not found", 404

        return send_file(file_path, as_attachment=True, download_name=f"{model_name}_file.py")
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/download_model_file/<model_name>/<user_id>', methods=["GET"])
def download_model_file(model_name, user_id):
    try:
        file_path = f"User_models/{user_id}_Best_Model.pkl"
        if file_path is None:
            return "File not found", 404
        return send_file(file_path, as_attachment=True, download_name=f"{model_name}_model.pkl")
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/download_labels_file/<model_name>/<user_id>', methods=["GET"])
def download_labels_file(model_name, user_id):
    try:
        file_path = f"User_labels/{user_id}_original_labels.txt"
        if file_path is None:
            return "File not found", 404
        return send_file(file_path, as_attachment=True, download_name=f"{model_name}_labels.txt")
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/predict/<Model_Type>', methods=['POST'])
def predict(Model_Type):
    try:
        model_file = request.files['model']
        labels_file = request.files.get('labels', None)
        loaded_model = joblib.load(model_file)
        data = request.files['file']
        df = pd.read_csv(data)
        df_cleaned = df.dropna(axis=1)
        predictions = loaded_model.predict(df_cleaned)

        if Model_Type == "Classification":
            if labels_file and labels_file.filename:
                labels_content = labels_file.read().decode("utf-8")
                original_labels = [label.strip() for label in labels_content.split("\n")]
                predictions = [original_labels[prediction] for prediction in predictions]
            else:
                return jsonify({"error": "Labels file not found or is empty"}), 400

            response = {
                "Predictions": predictions
            }

            return jsonify(response), 200
        response = {
            "Predictions": predictions.tolist()
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/train', methods=['POST'])
def train():
    try:
        csv_file_path = request.files['file']
        target_column = request.form.get('target_column', None)
        user_id = request.form.get('user_id')

        if target_column is None:
            clusters, score, labels = generic_clustering(csv_file_path)
            model_name = "KMeans"
            labels = labels.tolist()
            response = {
                "Model ": model_name,
                "Model_Type": "Clustering",
                "Labels": labels,
                "Optimal No of Clusters ": int(clusters),
                "Score ": float(score),
                "download_link": url_for('download_python_file', model_name=model_name, _external=True)
            }

            return jsonify(response), 200

        else:
            print("CSV file path:", csv_file_path)  # Debugging output
            model_name, best_metric, accuracy, precision, recall, classes = supervised_preprocessing(csv_file_path,
                                                                                                     target_column,
                                                                                                     user_id)

            if accuracy:
                response = {
                    "Model ": model_name,
                    "Classes": classes,
                    "Model Type": "Classification",
                    "F1_Score": best_metric,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "download_labels": url_for('download_labels_file', model_name=model_name, user_id=user_id,
                                               _external=True),
                    "download_model": url_for('download_model_file', model_name=model_name, user_id=user_id,
                                              _external=True),
                    "download_code": url_for('download_python_file', model_name=model_name, _external=True)
                }
            else:
                response = {
                    "Model ": model_name,
                    "Classes": classes,
                    "Model Type": "Regression",
                    "R2_Score": best_metric,
                    "download_model": url_for('download_model_file', model_name=model_name, user_id=user_id,
                                              _external=True),
                    "download_code": url_for('download_python_file', model_name=model_name, _external=True)
                }
            return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400
        # print("Error", e)


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host="0.0.0.0",port=8080)
