import pandas as pd
import matplotlib.pyplot as plt

def plot_histograms():
    # Đọc dữ liệu từ tệp CSV
    df = pd.read_csv('heart_failure_clinical_records.csv')

    # Kiểm tra và xử lý dữ liệu bị khuyết
    df = df.dropna()

    # Chọn các đặc điểm (features)
    features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
                'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking','time']

    # Tạo một khung hình với 11 bảng, mỗi hàng có 4 bảng
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(13, 10))

    for i, feature in enumerate(features):
        row = i // 4
        col = i % 4
        axes[row, col].hist(df[feature], bins=30, alpha=0.7, edgecolor='black')
        axes[row, col].set_title(f'Biểu đồ histogram cho cột {feature}',fontsize = "10")
        axes[row, col].set_xlabel({feature})
        axes[row, col].set_ylabel('Tần suất')

    # Ẩn bảng trống (ô thứ 12)
    # Điều chỉnh khoảng cách giữa các bảng
    plt.tight_layout()

    # Hiển thị biểu đồ
    plt.show()

# Gọi hàm để vẽ histogram
plot_histograms()
