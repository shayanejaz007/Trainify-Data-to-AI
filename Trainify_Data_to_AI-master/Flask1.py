from flask import Flask, request, jsonify
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load data from request
        file = request.files['file']
        target_column = request.form['target_column']

        # Read CSV file
        df = pd.read_csv(file)

        # Extract features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode target if it's categorical
        if y.dtypes == 'object':
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
        else:
            y_encoded = y

        # Preprocessing
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

        # Feature selection
        num_selected_features = X.shape[1] - 1
        feature_selector = SelectKBest(score_func=f_classif, k=num_selected_features)

        # Model
        model = DecisionTreeClassifier()

        # Create pipeline
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('selector', feature_selector),
            ('model', model)
        ])

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Train model
        model_pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = model_pipeline.predict(X_test)

        # Calculate metrics
        f1 = f1_score(y_test, y_pred, average='weighted')
        metric = accuracy_score(y_test, y_pred)

        # Prepare response
        response = {
            "Model": "Decision Tree Classifier",
            "F1 Score": f1,
            "Performance Metric": int(metric * 100)

        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
