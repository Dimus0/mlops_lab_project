import pandas as pd
import sys 
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

input_path = sys.argv[1]
output_path = sys.argv[2]

df = pd.read_csv("data/raw/telco_dataset.csv",sep=",")
df.drop(columns=["customerID"], inplace=True)

# Preprocessing steps
df.dropna(inplace=True)

encoder = LabelEncoder()
for column in df.columns:
    if df[column].dtype == "str" or df[column].dtype == "object":
        df[column] = encoder.fit_transform(df[column])

X = df.drop(columns=["Churn"])
y = df["Churn"]

# Треба замінити по іншому, щоб відбувалися тільки на train sample
smote = SMOTE(random_state=42)
X_balance, y_balanced = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_balance, y_balanced, test_size=0.2, random_state=42)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

os.makedirs(output_path, exist_ok=True)
train_df.to_csv(os.path.join(output_path, "train.csv"), index=False)
test_df.to_csv(os.path.join(output_path, "test.csv"), index=False)