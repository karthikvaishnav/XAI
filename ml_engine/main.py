from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import traceback
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import shap
app = FastAPI()

# Global Storage
current_data = None
current_model = None
X_test_global = None
X_global = None 
feature_cols_global = []
# --- ADD THIS WITH OTHER GLOBALS ---
model_artifacts = {
    "imputer": None,
    "scaler": None,
    "encoders": {},  # To remember LabelEncodings (e.g. Sex: Male -> 1)
    "poly": None,
    "features": []
}

class LoadRequest(BaseModel):
    file_path: str

class TrainRequest(BaseModel):
    file_path: str
    model_type: str
    target_column: str
    # New fields for Regression
    selected_features: List[str] = [] 
    poly_degree: int = 2

class ExplainRequest(BaseModel):

    index : int

    model_type : str

class SimulationRequest(BaseModel):

    features : dict

    model_type : str

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

# @app.post('/train')
@app.post('/train')
@app.post('/train')
def train_model(request: TrainRequest):
    global current_data, current_model, X_test_global, X_global, feature_cols_global, model_artifacts

    # 1. Load Data
    if current_data is None:
        try:
            current_data = pd.read_csv(request.file_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Data not found. Upload first.")

    df = current_data.copy()
    
    # 2. Validation
    if request.target_column not in df.columns:
         raise HTTPException(status_code=400, detail=f"Target '{request.target_column}' not found.")
    
    # 3. Feature Selection
    if not request.selected_features:
        feature_cols = [c for c in df.columns if c != request.target_column]
    else:
        feature_cols = request.selected_features
    
    # --- 🛡️ USE TEMP ARTIFACTS ---
    temp_artifacts = {
        "features": feature_cols,
        "encoders": {},
        "imputer": None,
        "scaler": None,
        "poly": None
    }
    
    X = df[feature_cols]
    y = df[request.target_column]

    # 4. Preprocessing
    # Handle Categoricals in X
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        temp_artifacts['encoders'][col] = le

    # Encode Target if categorical
    if y.dtype == 'object':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)
    
    # Impute X
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
    temp_artifacts['imputer'] = imputer
    X = X_imputed 
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
    temp_artifacts['scaler'] = scaler

    # 5. Determine Task
    REGRESSION_MODELS = ['linear', 'ridge', 'lasso', 'poly']
    is_regression = request.model_type in REGRESSION_MODELS

    if request.model_type == 'poly':
        poly = PolynomialFeatures(degree=request.poly_degree)
        temp_artifacts['poly'] = poly
    else:
        temp_artifacts['poly'] = None

    response_data = {}

    if is_regression:
        if y.dtype == 'object':
             raise HTTPException(status_code=400, detail="Target is categorical. Select numeric for Regression.")
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Model Init
        if request.model_type == 'linear': model = LinearRegression()
        elif request.model_type == 'ridge': model = Ridge(alpha=1.0)
        elif request.model_type == 'lasso': model = Lasso(alpha=0.1)
        elif request.model_type == 'poly':
            X_train = poly.fit_transform(X_train)
            model = LinearRegression()
        
        model.fit(X_train, y_train)
        
        # Predict
        if request.model_type == 'poly':
            y_pred = model.predict(poly.transform(X_test)) 
        else:
            y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        visuals = {}
        visuals['actual_vs_pred'] = {'y_true': y_test.tolist(), 'y_pred': y_pred.tolist()}
        
        # --- A. 1 Feature (Line) ---
        if len(feature_cols) == 1:
             col_name = feature_cols[0]
             line_x = np.linspace(X[col_name].min(), X[col_name].max(), 100).reshape(-1, 1)
             line_x_scaled = scaler.transform(pd.DataFrame(line_x, columns=feature_cols))
             if request.model_type == 'poly': line_y = model.predict(poly.transform(line_x_scaled))
             else: line_y = model.predict(line_x_scaled)
             
             visuals['regression_line'] = {'x': line_x.flatten().tolist(), 'y': line_y.tolist(), 'feature_name': col_name}
             visuals['scatter_raw'] = {'x': X[col_name].tolist(), 'y': y.tolist()}

        # --- B. 2 Features (Surface) ---
        elif len(feature_cols) == 2:
            x1_col, x2_col = feature_cols[0], feature_cols[1]
            x1_range = np.linspace(X[x1_col].min(), X[x1_col].max(), 20)
            x2_range = np.linspace(X[x2_col].min(), X[x2_col].max(), 20)
            xx1, xx2 = np.meshgrid(x1_range, x2_range)
            grid_flat = np.c_[xx1.ravel(), xx2.ravel()]
            grid_scaled = scaler.transform(pd.DataFrame(grid_flat, columns=feature_cols))
            
            if request.model_type == 'poly': z_pred = model.predict(poly.transform(grid_scaled))
            else: z_pred = model.predict(grid_scaled)
            
            visuals['surface'] = {'x': xx1.tolist(), 'y': xx2.tolist(), 'z': z_pred.reshape(xx1.shape).tolist(), 'features': [x1_col, x2_col]}
            visuals['scatter_3d'] = {'x': X[x1_col].tolist(), 'y': X[x2_col].tolist(), 'z': y.tolist()}

        # --- C. 3+ Features (PCA Cloud) ---
        else:
            # Fallback: Use PCA to project High-Dim Data into 3D so user sees *something*
            pca = PCA(n_components=3)
            pca_res = pca.fit_transform(X_scaled)
            visuals['scatter_3d'] = {
                'x': pca_res[:, 0].tolist(),
                'y': pca_res[:, 1].tolist(),
                'z': pca_res[:, 2].tolist(),
                'target': y.tolist()
            }
            # Note: We cannot draw a regression "surface" easily in PCA space, so we just show the data points

        # --- FIX: Extract Real Coefficients ---
        coefs = {}
        if request.model_type == 'poly':
             coefs = {"info": "Higher Order Polynomial Terms (Hidden)"}
        else:
             # Standard Linear/Ridge/Lasso
             coefs = {col: float(val) for col, val in zip(feature_cols, model.coef_)}
             coefs['intercept'] = float(model.intercept_)

        response_data = {
            "model": request.model_type, "task": "regression",
            "features": {"X": feature_cols, "y": request.target_column},
            "metrics": {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2},
            "coefficients": coefs, 
            "visuals": visuals
        }
        current_model = model

    else:
        # --- CLASSIFICATION LOGIC (Keep existing logic mostly same, just updating variables) ---
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        X_test_global = X_test.reset_index(drop=True)

        if request.model_type == 'rf': model = RandomForestClassifier(n_estimators=100)
        elif request.model_type == 'logistic': model = LogisticRegression(max_iter=1000)
        elif request.model_type == 'dt': model = DecisionTreeClassifier()
        
        model.fit(X_train, y_train)
        current_model = model 

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        # Feature Importance
        importance_list = []
        if hasattr(model, 'feature_importances_'): imps = model.feature_importances_
        elif hasattr(model, 'coef_'): imps = np.abs(model.coef_[0])
        else: imps = np.zeros(len(feature_cols))
        
        for i, col in enumerate(feature_cols):
            importance_list.append({'feature': col, 'importance': float(imps[i])})
        importance_list.sort(key=lambda x: x['importance'], reverse=True)

        # Dynamic PCA (Safe)
        n_features = X_scaled.shape[1]
        scatter_data = []
        scatter_data1 = []

        if n_features >= 3:
            pca = PCA(n_components=3)
            pca_res = pca.fit_transform(X_scaled)
            for i in range(min(len(pca_res), 1000)):
                scatter_data.append({'x': float(pca_res[i][0]), 'y': float(pca_res[i][1]), 'z': float(pca_res[i][2]), 'target': int(y[i]), 'original_index': int(i)})
        else:
            # Fallback for low dimensions
            for i in range(min(len(X_scaled), 1000)):
                scatter_data.append({'x': float(X_scaled.iloc[i, 0]), 'y': float(X_scaled.iloc[i, 1]) if n_features > 1 else 0.0, 'z': 0.0, 'target': int(y[i]), 'original_index': int(i)})

        if n_features >= 2:
            pca2 = PCA(n_components=2)
            pca2_res = pca2.fit_transform(X_scaled)
            for i in range(min(len(pca2_res), 1000)):
                scatter_data1.append({'x': float(pca2_res[i][0]), 'y': float(pca2_res[i][1]), 'target': int(y[i]), 'original_index': int(i)})
        else:
             for i in range(min(len(X_scaled), 1000)):
                scatter_data1.append({'x': float(X_scaled.iloc[i, 0]), 'y': 0.0, 'target': int(y[i]), 'original_index': int(i)})

        response_data = {
            "model": request.model_type, "task": "classification",
            "metrics": {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1},
            "confusion_matrix": {"z": cm.tolist(), "x": [str(c) for c in set(y)], "y": [str(c) for c in set(y)]},
            "feature_importance": importance_list[:10],
            "scatter_data": scatter_data,
            "scatter_data1": scatter_data1
        }

    # ✅ COMMIT ARTIFACTS
    model_artifacts = temp_artifacts
    X_global = X
    feature_cols_global = feature_cols

    return response_data
# from fastapi import HTTPException
# import numpy as np
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
    
@app.post('/simulate')
def simulate_prediction(request: SimulationRequest):
    global current_model,X_global

    if current_model is None:
        raise HTTPException(status_code = 400, detail = "train model first")

    feature_order = X_global.columns.tolist()
    input_data = pd.DataFrame(0,index=[0],columns=feature_order)

    for col,val in request.features.items():
        if col in input_data.columns:
            input_data.at[0,col] = float(val)

    if hasattr(current_model,"predict_proba"):
        probs = current_model.predict_proba(input_data)[0]
        prob_value = probs[1] if len(probs)>1 else probs[0]
    else:
        prob_value = float(current_model.predict(input_data)[0])
    
    return{
        "probability":float(prob_value),
        "prediction": int(prob_value>0.5)
    }

from sklearn.tree import _tree

def tree_to_json(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    def recurse(node):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = f"{feature_name[node]} <= {tree_.threshold[node]:.2f}"
            return {
                "name": name,
                "attributes": {
                    "gini": float(f"{tree_.impurity[node]:.3f}"),
                    "samples": int(tree_.n_node_samples[node])
                },
                "children": [
                    recurse(tree_.children_left[node]),
                    recurse(tree_.children_right[node])
                ]
            }
        else:
            return {
                "name": f"🍃 Leaf",
                "attributes": {
                    "gini": float(f"{tree_.impurity[node]:.3f}"),
                    "samples": int(tree_.n_node_samples[node]),
                    "value": str(tree_.value[node])
                }
            }
    return recurse(0)

@app.post("/decision_tree")
def get_decision_tree(request: TrainRequest):
    global current_data
    
    # 1. Use existing data
    df = current_data
    if df is None:
        try:
            df = pd.read_csv(request.file_path)
        except:
            raise HTTPException(400, "Data not found. Upload first.")

    # 2. Prepare Data
    y = df[request.target_column]
    X = df.drop(columns=[request.target_column])
    
    # Encode Categoricals
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    if y.dtype == 'object':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)
        
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # 3. Train Shallow Tree (Max Depth 3 for visualization)
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)

    # 4. Return JSON
    return {"tree_structure": tree_to_json(clf, X.columns.tolist())}

# --- NEW: MANUAL PREDICTION ENDPOINT ---
class PredictRequest(BaseModel):
    inputs: dict  # Example: {"Age": 25, "Sex": "Male"}
    model_type: str
@app.post('/predict_manual')
def predict_manual(request: PredictRequest):
    global current_model, model_artifacts

    if current_model is None:
        raise HTTPException(400, detail="Train model first")
    
    try:
        # 1. Align Inputs
        features = model_artifacts['features']
        input_data = [] # Use list to preserve order
        
        for f in features:
            val = request.inputs.get(f)
            
            # Handle categorical inputs
            if f in model_artifacts['encoders']:
                le = model_artifacts['encoders'][f]
                try:
                    val = le.transform([str(val)])[0]
                except:
                    val = 0
            
            try:
                input_data.append(float(val))
            except:
                input_data.append(0.0)

        # Create DataFrame with correct column names
        X_new = pd.DataFrame([input_data], columns=features)

        # 2. Apply Saved Preprocessing (Restoring DataFrame structure)
        if model_artifacts['imputer']:
            X_new_arr = model_artifacts['imputer'].transform(X_new)
            X_new = pd.DataFrame(X_new_arr, columns=features)
        
        if model_artifacts['scaler']:
            X_new_arr = model_artifacts['scaler'].transform(X_new)
            X_new = pd.DataFrame(X_new_arr, columns=features)

        # 3. Apply Polynomial (Returns Array, fine for LinearRegression)
        if request.model_type == 'poly' and model_artifacts['poly']:
            X_new = model_artifacts['poly'].transform(X_new)

        # 4. Predict
        prediction = current_model.predict(X_new)[0]

        # 5. Confidence
        confidence = 0.0
        if hasattr(current_model, "predict_proba"):
            try:
                probs = current_model.predict_proba(X_new)[0]
                confidence = float(max(probs))
            except:
                pass
        
        result = float(prediction) if isinstance(prediction, (np.float64, float)) else str(prediction)

        return {
            "prediction": result,
            "confidence": confidence,
            "is_regression": request.model_type in ['linear', 'ridge', 'lasso', 'poly']
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=f"Prediction failed: {str(e)}")