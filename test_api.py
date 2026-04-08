import requests
import json
import glob
import time

filepaths = glob.glob('C:/Users/ASUS/xai-workbench/server/uploads/*')
found = False
target_file = filepaths[0]
for f in filepaths:
    if f.endswith('.csv'): continue # uploaded files don't have extension usually
    target_file = f
    break

print(f"Testing against file: {target_file}")
for model in ['rf', 'logistic', 'dt', 'svm']:
    try:
        payload = {
            "file_path": target_file,
            "target_column": "Presence",  # Assumption based on heart dataset
            "model_type": model,
            "selected_features": [],
            "poly_degree": 2
        }
        res = requests.post("http://localhost:5000/api/train", json=payload)
        data = res.json()
        print(f"[{model}] ACC:", data.get('metrics', {}).get('accuracy', 'FAIL'))
    except Exception as e:
        print(f"[{model}] ERROR:", e)
