import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Đọc dữ liệu từ tệp CSV
df = pd.read_csv('heart_failure_clinical_records.csv')

# Kiểm tra và xử lý dữ liệu bị khuyết
df = df.dropna()

# Chọn các đặc điểm (features) và nhãn (label)
X = df[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
        'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking']]
y = df['DEATH_EVENT']

columns = X.columns

# Khởi tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# Danh sách lưu trữ kết quả MSE và R^2 cho từng lần chia
mse_scores = []
r2_scores = []

for i in range(10):
    # Chia dữ liệu thành tập huấn luyện (70%) và tập kiểm tra (30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i+10)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=columns)
    X_test = pd.DataFrame(X_test, columns=columns)

    # Huấn luyện mô hình
    model.fit(X_train, y_train)

    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)

    # Tính toán MSE và R^2
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mse_scores.append(mse)
    r2_scores.append(r2)

    print(f'Lần chia thứ {i + 1}:')
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    print('---')

# In kết quả đánh giá tổng thể
print(f'Độ chính xác trung bình MSE: {np.mean(mse_scores)}')
print(f'Độ chính xác trung bình R^2: {np.mean(r2_scores)}')

# Tính hệ số hồi quy
coefficients = model.coef_
feature_importance = pd.Series(coefficients, index=columns).sort_values(ascending=False)

print("Hệ số hồi quy cho từng đặc điểm:")
print(feature_importance)
