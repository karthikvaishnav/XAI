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
from sklearn.svm import SVC
import lime
import lime.lime_tabular
app = FastAPI()

# Global Storage
current_data = None
current_model = None
X_test_global = None
X_global = None        # Scaled version — for SHAP/LIME (matches what model was trained on)
X_global_raw = None    # Imputed-but-unscaled version — for human-readable slider display
feature_cols_global = []
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
    selected_features: List[str] = []
    poly_degree: int = 2
    test_size: float = 0.2  # configurable train-test split (0.1 – 0.4)

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

        # Build per-column statistics for the dataset preview panel
        stats = {}
        for col in df.columns:
            col_stats = {
                "dtype": str(df[col].dtype),
                "nulls": int(df[col].isnull().sum())
            }
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats["min"] = round(float(df[col].min()), 4)
                col_stats["max"] = round(float(df[col].max()), 4)
                col_stats["mean"] = round(float(df[col].mean()), 4)
            else:
                col_stats["unique"] = int(df[col].nunique())
            stats[col] = col_stats

        return {
            "message": "Data Loaded",
            "columns": df.columns.tolist(),
            "shape": {"rows": len(df), "cols": len(df.columns)},
            "preview": df.head(10).fillna("").to_dict(orient="records"),
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/train')
def train_model(request: TrainRequest):
    global current_data, current_model, X_test_global, X_global, X_global_raw, feature_cols_global, model_artifacts

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
        "label_encoder_y": None,
        "class_labels": None,
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

    # Encode Target if categorical — preserve original labels for display
    class_labels = None
    if y.dtype == 'object':
        le_y = LabelEncoder()
        class_labels = list(le_y.fit(y).classes_)  # e.g. ['Absence', 'Presence']
        y = le_y.transform(y)
        temp_artifacts['label_encoder_y'] = le_y
        temp_artifacts['class_labels'] = class_labels
    else:
        class_labels = sorted(y.unique().tolist())
    
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
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=request.test_size, random_state=42)
        
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
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=request.test_size, random_state=42)
        X_test_global = X_test.reset_index(drop=True)

        if request.model_type == 'rf': model = RandomForestClassifier(n_estimators=100)
        elif request.model_type == 'logistic': model = LogisticRegression(max_iter=1000)
        elif request.model_type == 'dt': model = DecisionTreeClassifier()
        elif request.model_type == 'svm': model = SVC(kernel='linear', probability=True, random_state=42)
        
        model.fit(X_train, y_train)
        current_model = model 

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        # Use macro averaging: treats all classes equally, shows more distinct values
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        # Per-class breakdown for frontend display
        unique_classes = sorted(set(y_test))
        per_class_precision = precision_score(y_test, y_pred, average=None, zero_division=0, labels=unique_classes)
        per_class_recall = recall_score(y_test, y_pred, average=None, zero_division=0, labels=unique_classes)
        per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0, labels=unique_classes)
        
        # Map class index back to original labels for display
        if class_labels:
            display_labels = [str(class_labels[int(c)]) if int(c) < len(class_labels) else str(c) for c in unique_classes]
        else:
            display_labels = [str(c) for c in unique_classes]
        
        per_class_metrics = [
            {
                "class": display_labels[i],
                "precision": float(per_class_precision[i]),
                "recall": float(per_class_recall[i]),
                "f1": float(per_class_f1[i])
            }
            for i in range(len(unique_classes))
        ]
        
        print(f"--- TRAINED CLASSIFIER: {request.model_type} | ACCURACY: {acc:.4f} | MACRO P/R/F1: {precision:.4f}/{recall:.4f}/{f1:.4f} | FEATURES: {len(feature_cols)} ---")
        
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

        # Helper: map encoded int back to original label (e.g. 0 → 'Absence')
        # Defensive: works whether y is an int-encoded numpy array OR a raw string Series
        def get_label(i):
            val = y.iloc[i] if hasattr(y, 'iloc') else y[i]
            try:
                idx = int(val)
                if class_labels and idx < len(class_labels):
                    return str(class_labels[idx])
                return str(idx)
            except (ValueError, TypeError):
                # y was not encoded — return the raw string as-is
                return str(val)

        if n_features >= 3:
            pca = PCA(n_components=3)
            pca_res = pca.fit_transform(X_scaled)
            for i in range(min(len(pca_res), 1000)):
                scatter_data.append({'x': float(pca_res[i][0]), 'y': float(pca_res[i][1]), 'z': float(pca_res[i][2]), 'target': get_label(i), 'original_index': int(i)})
        else:
            for i in range(min(len(X_scaled), 1000)):
                scatter_data.append({'x': float(X_scaled.iloc[i, 0]), 'y': float(X_scaled.iloc[i, 1]) if n_features > 1 else 0.0, 'z': 0.0, 'target': get_label(i), 'original_index': int(i)})

        if n_features >= 2:
            pca2 = PCA(n_components=2)
            pca2_res = pca2.fit_transform(X_scaled)
            for i in range(min(len(pca2_res), 1000)):
                scatter_data1.append({'x': float(pca2_res[i][0]), 'y': float(pca2_res[i][1]), 'target': get_label(i), 'original_index': int(i)})
        else:
            for i in range(min(len(X_scaled), 1000)):
                scatter_data1.append({'x': float(X_scaled.iloc[i, 0]), 'y': 0.0, 'target': get_label(i), 'original_index': int(i)})

        # Use original string labels for confusion matrix axes if available
        cm_labels = class_labels if class_labels else [str(c) for c in sorted(set(y))]
        response_data = {
            "model": request.model_type, "task": "classification",
            "metrics": {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1},
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": {"z": cm.tolist(), "x": cm_labels, "y": cm_labels},
            "feature_importance": importance_list[:10],
            "scatter_data": scatter_data,
            "scatter_data1": scatter_data1
        }

    # ✅ COMMIT ARTIFACTS
    # X_global = scaled data for SHAP/LIME (must match model input)
    # X_global_raw = imputed-but-unscaled data for human-readable slider display
    model_artifacts = temp_artifacts
    X_global = X_scaled
    X_global_raw = X_imputed  # unscaled — used to show real feature values in simulator
    feature_cols_global = feature_cols

    return response_data

@app.post('/explain')
def explain_instance(request: ExplainRequest):
    global current_model, X_global, X_global_raw

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

            pred_label = current_model.predict(row)[0]
            try:
                pred_class = list(current_model.classes_).index(pred_label)
            except (ValueError, AttributeError):
                pred_class = int(pred_label)
            print(f"[EXPLAIN] Predicted class index: {pred_class}")

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
            raw_values = shap_obj.values[0]  # shape: (n_features,) binary OR (n_features, n_classes) multiclass
            if raw_values.ndim == 2:
                # Multiclass: pick the column for the predicted class
                pred_label = current_model.predict(row)[0]
                try:
                    pred_class = list(current_model.classes_).index(pred_label)
                except (ValueError, AttributeError):
                    pred_class = int(pred_label)
                values = raw_values[:, pred_class]
                base_val_raw = shap_obj.base_values[0]
                base_value = base_val_raw[pred_class] if hasattr(base_val_raw, '__len__') else float(base_val_raw)
            else:
                values = raw_values
                base_value = shap_obj.base_values[0]

        elif request.model_type == 'svm':
            background = shap.sample(X_global, 50)
            explainer = shap.KernelExplainer(current_model.predict_proba, background)
            shap_values = explainer.shap_values(row)
            pred_label = current_model.predict(row)[0]
            try:
                pred_class = list(current_model.classes_).index(pred_label)
            except (ValueError, AttributeError):
                pred_class = int(pred_label)
            if isinstance(shap_values, list):
                values = shap_values[pred_class][0]
                base_value = explainer.expected_value[pred_class] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            else:
                values = shap_values[0]
                if isinstance(explainer.expected_value, (list, np.ndarray)):
                    base_value = explainer.expected_value[pred_class] if len(explainer.expected_value) > pred_class else explainer.expected_value[0]
                else:
                    base_value = explainer.expected_value

        else:
            raise HTTPException(400, detail="Unsupported model type")

        # Ensure clean types
        values = np.asarray(values).flatten()
        base_value = float(base_value)  # <--- convert here, after selection

        print(f"[EXPLAIN] base_value: {base_value} (type: {type(base_value)})")

        explanation = []
        for i, col in enumerate(X_global.columns):
            # Use raw (unscaled) value for the slider so the UI shows human-readable numbers
            raw_val = float(X_global_raw.iloc[request.index, i]) if X_global_raw is not None else float(row.iloc[0, i])
            explanation.append({
                "feature": col,
                "value": raw_val,          # human-readable, unscaled
                "shap_value": float(values[i])
            })

        explanation.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        # Generate LIME explanation
        lime_explanation = []
        try:
            if hasattr(current_model, "predict_proba"):
                explainer_lime = lime.lime_tabular.LimeTabularExplainer(
                    training_data=X_global.values,
                    feature_names=X_global.columns.tolist(),
                    mode='classification'
                )
                exp = explainer_lime.explain_instance(
                    data_row=row.values[0],
                    predict_fn=current_model.predict_proba,
                    num_features=10
                )
                lime_explanation = [{"feature": f, "weight": float(w)} for f, w in exp.as_list()]
        except Exception as lime_e:
            print(f"[EXPLAIN] LIME failed: {lime_e}")

        return {
            "base_value": base_value,           # already float now
            "explanation": explanation[:10],
            "lime_explanation": lime_explanation
        }

    except Exception as e:
        print("[EXPLAIN] ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"SHAP computation failed: {str(e)}")
    
@app.post('/simulate')
def simulate_prediction(request: SimulationRequest):
    global current_model, X_global, model_artifacts

    if current_model is None:
        raise HTTPException(status_code=400, detail="train model first")

    feature_order = model_artifacts['features']
    # Build input from the raw (unscaled) values sent by the UI
    input_data = pd.DataFrame(0.0, index=[0], columns=feature_order)

    for col, val in request.features.items():
        if col in input_data.columns:
            input_data.at[0, col] = float(val)

    # Apply imputer first (handles edge cases where value might be NaN)
    if model_artifacts['imputer'] is not None:
        imputed_arr = model_artifacts['imputer'].transform(input_data)
        input_data = pd.DataFrame(imputed_arr, columns=feature_order)

    # Then scale exactly once (same pipeline as training)
    if model_artifacts['scaler'] is not None:
        scaled_arr = model_artifacts['scaler'].transform(input_data)
        input_data = pd.DataFrame(scaled_arr, columns=feature_order)

    if hasattr(current_model, "predict_proba"):
        probs = current_model.predict_proba(input_data)[0]
        # For binary: class 1 probability; for multiclass: max class probability
        prob_value = probs[1] if len(probs) == 2 else float(max(probs))
    else:
        prob_value = float(current_model.predict(input_data)[0])

    return {
        "probability": float(prob_value),
        "prediction": int(prob_value > 0.5)
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

    # 1. Use existing data — always copy to avoid mutating global state
    if current_data is not None:
        df = current_data.copy()
    else:
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

        # 6. Decode prediction label back to original string (e.g. 1 → "Presence")
        is_regression = request.model_type in ['linear', 'ridge', 'lasso', 'poly']
        if not is_regression and model_artifacts.get('label_encoder_y') is not None:
            try:
                prediction = model_artifacts['label_encoder_y'].inverse_transform([int(prediction)])[0]
            except Exception:
                pass

        result = float(prediction) if isinstance(prediction, (np.float64, float)) else str(prediction)

        return {
            "prediction": result,
            "confidence": confidence,
            "is_regression": is_regression
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=f"Prediction failed: {str(e)}")


# ─────────────────────────────────────────────────────────────
# NEW: DECISION BOUNDARY ENDPOINT
# ─────────────────────────────────────────────────────────────
class BoundaryRequest(BaseModel):
    feature1: str
    feature2: str

@app.post("/decision_boundary")
def decision_boundary(request: BoundaryRequest):
    global current_model, X_global, model_artifacts

    if current_model is None:
        raise HTTPException(400, detail="Train a model first")
    if X_global is None:
        raise HTTPException(400, detail="No training data available")

    features = model_artifacts['features']
    if request.feature1 not in features or request.feature2 not in features:
        raise HTTPException(400, detail="Selected features not in training set")

    try:
        f1_idx = features.index(request.feature1)
        f2_idx = features.index(request.feature2)

        # Build 2D grid over the two features (everything else = mean = 0 in scaled space)
        x1_vals = X_global.iloc[:, f1_idx]
        x2_vals = X_global.iloc[:, f2_idx]

        x1_range = np.linspace(x1_vals.min() - 0.5, x1_vals.max() + 0.5, 80)
        x2_range = np.linspace(x2_vals.min() - 0.5, x2_vals.max() + 0.5, 80)
        xx1, xx2 = np.meshgrid(x1_range, x2_range)

        # Build full feature grid (mean=0 for other features since data is scaled)
        n = xx1.shape[0] * xx1.shape[1]
        grid = np.zeros((n, len(features)))
        grid[:, f1_idx] = xx1.ravel()
        grid[:, f2_idx] = xx2.ravel()

        grid_df = pd.DataFrame(grid, columns=features)
        Z = current_model.predict(grid_df)

        # Encode Z to numeric if needed
        try:
            Z_num = Z.astype(float)
        except (ValueError, AttributeError):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            Z_num = le.fit_transform(Z).astype(float)

        # Scatter: real data points projected to 2 chosen features
        scatter = []
        le_y = model_artifacts.get('label_encoder_y')
        class_labels = model_artifacts.get('class_labels')
        for i in range(min(len(X_global), 800)):
            raw_y_val = X_global.index[i]  # not the label
            # We'll get actual label from Z (train predict) — approximate
            scatter.append({
                "x": float(X_global.iloc[i, f1_idx]),
                "y": float(X_global.iloc[i, f2_idx]),
                "original_index": int(i)
            })

        return {
            "x": xx1.tolist(),
            "y": xx2.tolist(),
            "z": Z_num.reshape(xx1.shape).tolist(),
            "feature1": request.feature1,
            "feature2": request.feature2,
            "scatter": scatter
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=f"Boundary failed: {str(e)}")


# ─────────────────────────────────────────────────────────────
# NEW: MULTI-MODEL COMPARISON ENDPOINT
# ─────────────────────────────────────────────────────────────
class CompareRequest(BaseModel):
    file_path: str
    target_column: str
    selected_features: List[str] = []
    test_size: float = 0.2
    model_types: List[str] = ['rf', 'logistic', 'dt', 'svm']

@app.post("/compare")
def compare_models(request: CompareRequest):
    global current_data

    if current_data is None:
        try:
            current_data = pd.read_csv(request.file_path)
        except Exception as e:
            raise HTTPException(400, detail="Data not loaded. Upload first.")

    df = current_data.copy()

    if request.target_column not in df.columns:
        raise HTTPException(400, detail=f"Target '{request.target_column}' not found.")

    feature_cols = request.selected_features if request.selected_features else \
        [c for c in df.columns if c != request.target_column]

    X = df[feature_cols].copy()
    y = df[request.target_column].copy()

    # Encode categoricals
    encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    if y.dtype == 'object':
        le_y = LabelEncoder()
        y = pd.Series(le_y.fit_transform(y))

    imputer = SimpleImputer(strategy="mean")
    X_arr = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_arr), columns=feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=request.test_size, random_state=42)

    results = []
    model_map = {
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
        'dt': DecisionTreeClassifier(random_state=42),
        'svm': SVC(kernel='linear', probability=True, random_state=42)
    }
    model_labels = {
        'rf': 'Random Forest',
        'logistic': 'Logistic Regression',
        'dt': 'Decision Tree',
        'svm': 'SVM'
    }

    for mtype in request.model_types:
        if mtype not in model_map:
            continue
        try:
            m = model_map[mtype]
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            results.append({
                "model": model_labels.get(mtype, mtype),
                "model_type": mtype,
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "precision": round(float(precision_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
                "f1": round(float(f1_score(y_test, y_pred, average='weighted', zero_division=0)), 4)
            })
        except Exception as e:
            results.append({"model": model_labels.get(mtype, mtype), "model_type": mtype, "error": str(e)})

    return {"comparison": results, "test_size": request.test_size, "features": feature_cols}