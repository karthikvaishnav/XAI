import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv('C:/Users/ASUS/xai-workbench/server/uploads/22dcff4e943895929beb3846af185538')
X = df[['age', 'sex', 'cp', 'fbs']]
y = df['target']

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
