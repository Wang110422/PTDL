from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import  numpy as np
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("back.html")
@app.route("/menu")
def home1():
    return render_template("haha.html")
@app.route("/submit_patient_info", methods=["POST"])
def xuly():
    data = request.json["data"]
    age = data['age']
    gender = data['gender']
    anaemia = data['anaemia']
    cpk = data['cpk']
    diabetes = data['diabetes']
    ejection_fraction = data['ejection_fraction']
    high_blood_pressure = data['high_blood_pressure']
    platelets = data['platelets']
    serum_creatinine = data['serum_creatinine']
    serum_sodium = data['serum_sodium']
    smoking = data['smoking']
    prediction , percent = trave(age, gender, anaemia, cpk, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, smoking)

    if prediction == 1:
        result = "Bệnh nhân có nguy cơ bị suy tim."
    else:
        result = "Bệnh nhân không có nguy cơ bị suy tim."
    result1 = f"Bệnh nhân có nguy cơ suy tim: {percent * 100:.2f}%"
    return jsonify(response=f"{result}\n{result1}")

def trave(age, gender, anaemia, cpk, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, smoking):
    ## Doc data
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

    input_data = np.array([[age, anaemia, cpk, diabetes, ejection_fraction, high_blood_pressure, platelets,serum_creatinine, serum_sodium,gender , smoking]])
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)[0]
    percent = model.predict_proba(input_data)[0][1]
    return prediction,percent
if __name__ == "__main__":
    app.run(debug=True)
