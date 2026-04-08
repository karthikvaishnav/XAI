import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import glob

for f in glob.glob('C:/Users/ASUS/xai-workbench/server/uploads/*'):
    try:
        df = pd.read_csv(f)
        if len(df) == 270:
            print(f"File {f} is 270 rows!")
            X = df.drop(columns=[df.columns[-1]])
            y = df[df.columns[-1]]
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            X_scaled = StandardScaler().fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
            lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
            dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
            svm = SVC(kernel='linear', probability=True, random_state=42).fit(X_train, y_train)

            print('RF:', accuracy_score(y_test, rf.predict(X_test)))
            print('LR:', accuracy_score(y_test, lr.predict(X_test)))
            print('DT:', accuracy_score(y_test, dt.predict(X_test)))
            print('SVM:', accuracy_score(y_test, svm.predict(X_test)))
            break
    except:
        pass
