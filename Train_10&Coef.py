import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

def analyze_logistic_regression_coefficients():
    # Đọc dữ liệu từ tệp CSV
    df = pd.read_csv('heart_failure_clinical_records.csv')

    # Kiểm tra và xử lý dữ liệu bị khuyết
    df = df.dropna()

    # Chọn các đặc điểm (features) và nhãn (label)
    X = df[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
            'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking']]
    y = df['DEATH_EVENT']

    columns = X.columns  # Lưu trữ các tên cột

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Chuyển đổi trở lại thành DataFrame sau khi chuẩn hóa
    X_train = pd.DataFrame(X_train, columns=columns)
    X_test = pd.DataFrame(X_test, columns=columns)

    # Tạo mô hình hồi quy logistic
    model = LogisticRegression()

    # Huấn luyện mô hình
    model.fit(X_train, y_train)

    # Lấy hệ số hồi quy
    coefficients = model.coef_[0]
    feature_importance = pd.Series(coefficients, index=columns).sort_values(ascending=False)

    print("Hệ số hồi quy logistic:")
    print(feature_importance)

    # Tính xác suất tỷ lệ (odds ratio)
    odds_ratios = np.exp(coefficients)
    odds_ratios_series = pd.Series(odds_ratios, index=columns).sort_values(ascending=False)

    print("\nTỷ lệ odds:")
    print(odds_ratios_series)

# Gọi hàm để phân tích hệ số hồi quy logistic
analyze_logistic_regression_coefficients()
