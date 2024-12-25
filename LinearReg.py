import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Đọc dữ liệu từ tệp CSV
df = pd.read_csv('heart_failure_clinical_records.csv')

# Kiểm tra và xử lý dữ liệu bị khuyết
df = df.dropna()

# Chọn các đặc điểm (features) và nhãn (label)
X = df[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
        'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking']]
y = df['DEATH_EVENT']

columns = X.columns
# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=columns)
X_test = pd.DataFrame(X_test, columns=columns)

# Tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Tính toán MSE và R^2
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

coefficients = model.coef_
feature_importance = pd.Series(coefficients, index=X.columns).sort_values(ascending=False)

print(feature_importance)
# Trực quan hóa kết quả
plt.scatter(y_test, y_pred)
plt.xlabel('Thực tế')
plt.ylabel('Dự đoán')
plt.title('Thực tế vs Dự đoán')
plt.show()
