import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def plot_boxplots():
    # Đọc dữ liệu từ tệp CSV
    df = pd.read_csv('heart_failure_clinical_records.csv')

    # Kiểm tra và xử lý dữ liệu bị khuyết
    df = df.dropna()

    # Chọn các đặc điểm (features)
    features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
                'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking','time']

    # Tạo một khung hình với 11 bảng, mỗi hàng có 4 bảng
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(13, 10))
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    for i, feature in enumerate(features):
        row = i // 4
        col = i % 4
        box = axes[row, col].boxplot(df[feature], patch_artist=True,
                                     boxprops=dict(facecolor='blue', color='blue'),
                                     whiskerprops=dict(color='blue'),
                                     capprops=dict(color='blue'),
                                     medianprops=dict(color='red'),
                                     flierprops=dict(markerfacecolor='blue', marker='o', color='blue'))
        axes[row, col].set_title(f'Biểu đồ hộp cho cột {feature}', fontsize=10)
        axes[row, col].set_xlabel(feature, fontsize=10)
        axes[row, col].set_ylabel('Giá trị', fontsize=8)

    # Ẩn bảng trống (ô thứ 12)

    # Điều chỉnh khoảng cách giữa các bảng
    plt.tight_layout()

    # Hiển thị biểu đồ
    plt.show()

# Gọi hàm để vẽ box plot
plot_boxplots()
