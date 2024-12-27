import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np



df = pd.read_csv('heart_failure_clinical_records1.csv')
df = df.dropna()

X = df[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
            'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking']]
y = df['DEATH_EVENT']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

input_data = np.array([[55,0,748,0,45,0,263358.03,1.3,137,1,1]])
input_data = scaler.transform(input_data)

prediction = model.predict(input_data)[0]
percent = model.predict_proba(input_data)[0][1]

print(prediction)
print(percent)