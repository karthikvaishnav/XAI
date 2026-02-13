from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import shap 

import traceback

app = FastAPI()

# Global Storage
current_data = None
current_model = None
X_test_global = None
X_global = None # <--- We need this to exist globally

class LoadRequest(BaseModel):
    file_path: str

class TrainRequest(BaseModel):
    file_path: str
    model_type: str
    target_column: str

class ExplainRequest(BaseModel):
    index : int
    model_type : str

@app.get('/')
def read_root():
    return {'status': "ML engine is running"}

@app.post('/load-data')
def load_data(request: LoadRequest):
    global current_data
    try:
        df = pd.read_csv(request.file_path)
        current_data = df
        return {
            "message": "Data Loaded",
            "columns": df.columns.tolist(),
            "preview": df.head(5).to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/train')
def train_model(request: TrainRequest):
    global current_data, current_model, X_test_global, X_global

    # 1. Load Data
    if current_data is None:
        try:
            current_data = pd.read_csv(request.file_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Data not found. Upload first.")

    df = current_data.copy()

    # 2. Preprocessing
    if request.target_column not in df.columns:
         raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' not found.")

    y = df[request.target_column]
    X = df.drop(columns=[request.target_column])
    
    # Encode Categoricals
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Encode Target
    if y.dtype == 'object':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)
    
    # Impute
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # SAVE GLOBAL X FOR EXPLAINABILITY
    X_global = X 

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_global = X_test.reset_index(drop=True)

    # 4. Model Selection
    if request.model_type == 'rf':
        current_model = RandomForestClassifier(n_estimators=100)
    elif request.model_type == 'logistic':
        current_model = LogisticRegression(max_iter=1000)
    elif request.model_type == 'dt':
        current_model = DecisionTreeClassifier()
    else:
        raise HTTPException(status_code=400, detail="Unknown model type")

    # 5. Training
    current_model.fit(X_train, y_train)
    acc = current_model.score(X_test, y_test)

    # 6. Feature Importance
    importance_list = []
    if hasattr(current_model, 'feature_importances_'):
        imps = current_model.feature_importances_
    elif hasattr(current_model, 'coef_'):
        imps = np.abs(current_model.coef_[0])
    else:
        imps = np.zeros(len(X.columns))

    for i, col in enumerate(X.columns):
        importance_list.append({
            'feature': col,
            'importance': float(imps[i])
        })    
    importance_list.sort(key=lambda x: x['importance'], reverse=True)

    # 7. PCA
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(X) 

    scatterData = []
    limit = min(len(pca_results), 1000) 

    for i in range(len(pca_results)):
        scatterData.append({
            'x': float(pca_results[i][0]),
            'y': float(pca_results[i][1]),
            'z': float(pca_results[i][2]),
            'target': int(y[i]),
            'original_index':int(i)
        })

    # 2d pca
    pca1 = PCA(n_components=2)
    pca_results1 = pca1.fit_transform(X)

    scatterData1 = []

    for i in range(len(pca_results1)):
        scatterData1.append({
            'x':float(pca_results1[i][0]),
            'y': float(pca_results1[i][1]),
            'target': int(y[i]),
            'original_index': int(i)
        })

    return {
        'model': request.model_type,
        'accuracy': acc,
        'feature_importance': importance_list[:10],
        'scatter_data': scatterData,
        'scatter_data1': scatterData1
    }   
from fastapi import HTTPException
import numpy as np
import traceback

@app.post('/explain')
def explain_instance(request: ExplainRequest):
    global current_model, X_global

    print(f"[EXPLAIN] Request: index={request.index}, model={request.model_type}")

    if current_model is None:
        raise HTTPException(400, detail="Model not trained yet")
    if X_global is None:
        raise HTTPException(400, detail="No training data available")

    print(f"[EXPLAIN] X_global rows: {len(X_global)}")

    if request.index < 0 or request.index >= len(X_global):
        raise HTTPException(400, detail=f"Index {request.index} out of bounds (0–{len(X_global)-1})")

    print("[EXPLAIN] Index check passed → proceeding to SHAP")

    try:
        row = X_global.iloc[[request.index]]

        if request.model_type in ['rf', 'dt']:
            explainer = shap.TreeExplainer(current_model)
            shap_values = explainer.shap_values(row)

            pred_class = int(current_model.predict(row)[0])
            print(f"[EXPLAIN] Predicted class: {pred_class}")

            # ── Handle SHAP values ────────────────────────────────────────
            if isinstance(shap_values, list):
                # Binary classification: list of two arrays
                class_shap = shap_values[pred_class]
                values = class_shap[0]               # shape (n_features,)
            else:
                # Unusual single-output case
                values = shap_values[0]

            # ── Handle base value correctly ───────────────────────────────
            expected = explainer.expected_value

            if isinstance(expected, (list, np.ndarray)):
                # Binary/multiclass: array with one value per class
                base_value = expected[pred_class]
            else:
                # Rare scalar case
                base_value = expected

        elif request.model_type == 'logistic':
            background = shap.maskers.Independent(X_global, max_samples=100)
            explainer = shap.LinearExplainer(current_model, background)
            shap_obj = explainer(row)
            values = shap_obj.values[0]
            base_value = shap_obj.base_values[0]  # usually scalar

        else:
            raise HTTPException(400, detail="Unsupported model type")

        # Ensure clean types
        values = np.asarray(values).flatten()
        base_value = float(base_value)  # <--- convert here, after selection

        print(f"[EXPLAIN] values shape: {values.shape}, length: {len(values)}")
        print(f"[EXPLAIN] base_value: {base_value} (type: {type(base_value)})")

        explanation = []
        for i, col in enumerate(X_global.columns):
            explanation.append({
                "feature": col,
                "value": float(row.iloc[0, i]),
                "shap_value": float(values[i])
            })

        explanation.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "base_value": base_value,           # already float now
            "explanation": explanation[:10]
        }

    except Exception as e:
        print("[EXPLAIN] ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"SHAP computation failed: {str(e)}")