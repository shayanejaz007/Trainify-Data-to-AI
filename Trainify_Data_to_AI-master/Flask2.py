from flask import Flask, send_file, jsonify, url_for
import os
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
    # Find the model file
    file_path = find_model_file(model_name)

    # If file not found, return 404
    if file_path is None:
        return "File not found", 404

    # Send the file for download
    return send_file(file_path, as_attachment=True, download_name=f"{model_name}.py")

@app.route('/other_route', methods=["GET"])
def other_route():
    # Determine the model name
    model_name = "DecisionTreeClassifier"  # Example, replace this with your logic

    # Example JSON response
    data = {
        "Model": model_name,
        "Score": "100",
        "download_link": url_for('download_python_file', model_name=model_name, _external=True)
    }

    # Return the JSON response
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=9090)


