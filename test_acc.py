import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import sys

try:
    df = pd.read_csv("c:/Users/ASUS/xai-workbench/uploads/titanic.csv") # we don't know the dataset exactly, but wait, the user showed "Chest pain type", "Sex", "Age". This is heart disease!
except:
    pass

sys.exit(0)
