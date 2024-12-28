import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score

def k_fold_logistic_regression_stratified_z_score():
    # Đọc dữ liệu từ tệp CSV
    df = pd.read_csv('heart_failure_clinical_records_clean.csv')

    # Chọn các đặc điểm (features) và nhãn (label)
    X = df[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
            'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking']]
    y = df['DEATH_EVENT']

    # Thiết lập StratifiedKFold cross-validation với k = 10
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Lưu trữ các kết quả đánh giá
    accuracies = []
    aucs = []
    f1s = []
    conf_matrices = []
    reports = []

    fold = 1
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Tạo mô hình hồi quy logistic
        model = LogisticRegression()
        # Huấn luyện mô hình
        model.fit(X_train, y_train)

        # Dự đoán trên tập kiểm tra
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Đánh giá mô hình
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Lưu trữ kết quả
        accuracies.append(accuracy)
        aucs.append(auc)
        f1s.append(f1)
        conf_matrices.append(conf_matrix)
        reports.append(report)

        # In kết quả của từng fold
        print(f"\nFold {fold}")
        print(f"Accuracy: {accuracy}")
        print(f"AUC: {auc}")
        print(f"F1 Score: {f1}")
        print("Confusion Matrix:")
        print(conf_matrix)

        fold += 1

    # In kết quả tổng quan
    mean_accuracy = sum(accuracies) / len(accuracies)
    mean_auc = sum(aucs) / len(aucs)
    mean_f1 = sum(f1s) / len(f1s)
    std_accuracy = pd.Series(accuracies).std()
    std_auc = pd.Series(aucs).std()
    std_f1 = pd.Series(f1s).std()

    print(f'\nMean Accuracy: {mean_accuracy}')
    print(f'Mean AUC: {mean_auc}')
    print(f'Mean F1 Score: {mean_f1}')
    print(f'Standard Deviation of Accuracy: {std_accuracy}')
    print(f'Standard Deviation of AUC: {std_auc}')
    print(f'Standard Deviation of F1 Score: {std_f1}')

# Gọi hàm để thực hiện dự đoán và đánh giá
k_fold_logistic_regression_stratified_z_score()
