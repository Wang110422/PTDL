import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_multiple_scatter_with_logistic_regression():
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
        sns.regplot(x=feature, y='DEATH_EVENT', data=df, logistic=True, scatter_kws={'alpha':0.2, 's':10, 'color':'blue'}, line_kws={'color':'red'}, ax=axes[row, col])
        axes[row, col].set_title(f'{feature} vs DEATH_EVENT', fontsize=10)
        axes[row, col].set_xlabel(feature, fontsize=8)
        axes[row, col].set_ylabel('DEATH_EVENT', fontsize=8)

    # Ẩn bảng trống (ô thứ 12)

    # Điều chỉnh khoảng cách giữa các bảng
    plt.tight_layout()

    # Hiển thị biểu đồ
    plt.show()

# Gọi hàm để vẽ scatter plot với đường hồi quy logistic
plot_multiple_scatter_with_logistic_regression()
