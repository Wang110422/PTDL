import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("heart_failure_clinical_records.csv")

# BOX PLOT : AGE VÀ DEATH_EVENT

# plt.figure(figsize=(10,6))
# sns.boxplot(x='DEATH_EVENT', y='age', data=df)
# plt.title('Biểu đồ so sánh độ tuổi giữa hai nhóm: tử vong và không tử vong.')
# plt.xlabel('DEATH_EVENT')
# plt.ylabel('Age')
# plt.show()


# HISTOGRAM : AGE VÀ DEATH_EVENT

# Tạo hai nhóm dữ liệu dựa trên DEATH_EVENT
# df_death = df[df['DEATH_EVENT'] == 1]
# df_no_death = df[df['DEATH_EVENT'] == 0]

# Vẽ biểu đồ Histogram
# plt.figure(figsize=(10, 6))
# plt.hist(df_no_death['age'], bins=10, alpha=0.7, label='Không tử vong', color='blue', edgecolor='black')
# plt.hist(df_death['age'], bins=10, alpha=0.7, label='Tử vong', color='red', edgecolor='black')
#
# plt.title('Biểu đồ phân bố tuổi của bệnh nhân theo DEATH_EVENT')
# plt.xlabel('Tuổi')
# plt.ylabel('Tần suất')
# plt.legend(loc='upper right')
# plt.show()


# Phân suất tống máu và DEATH_EVENT
# Box Plot

# plt.figure(figsize=(10,6))
# sns.boxplot(x='DEATH_EVENT', y='ejection_fraction', data=df)
# plt.title('Biểu đồ so sánh phân suất tống máu giữa hai nhóm tử vong và không tử vong. ')
# plt.xlabel('DEATH_EVENT')
# plt.ylabel('Phân suất tống máu(%)')
# plt.show()

#Violin Plot
# plt.figure(figsize=(10, 6))
# sns.violinplot(x='DEATH_EVENT', y='ejection_fraction', data=df)
# plt.title('Biểu đồ phân bố Phân suất tống máu theo DEATH_EVENT')
# plt.xlabel('DEATH_EVENT')
# plt.ylabel('Phân suất tống máu (%)')
# plt.show()


# Thiếu máu và DEATH_EVENT
#Bar Chart
# Tạo bảng đếm số lượng bệnh nhân tử vong và không tử vong theo thiếu máu
death_counts = df.groupby('anaemia')['DEATH_EVENT'].value_counts().unstack().fillna(0)

# Vẽ biểu đồ Bar Chart
# death_counts.plot(kind='bar', stacked=True, color=['blue', 'red'], alpha=0.7, edgecolor='black')
# plt.title('Biểu đồ thể hiện Tỷ lệ tử vong giữa bệnh nhân thiếu máu và không thiếu máu')
# plt.xlabel('Thiếu máu')
# plt.ylabel('Số lượng bệnh nhân')
# plt.legend(['Không tử vong', 'Tử vong'])
# plt.show()



#Phân tích nhiều biến (Multivariate Analysis)
#Pair Plot (Seaborn)

#Heatmap
# correlation_matrix = df.corr()
#
# plt.figure(figsize=(12, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
# plt.title('Ma trận tương quan giữa các biến ')
# plt.show()



#Tuổi và Phân suất tống máu
# Vẽ biểu đồ Scatter Plot
# plt.figure(figsize=(10, 6))
# plt.scatter(df['age'], df['ejection_fraction'], color='blue', alpha=0.7, edgecolors='w', linewidth=0.5)
# plt.title(' Biểu đồ thể hiện Mối quan hệ giữa Tuổi và Phân suất tống máu')
# plt.xlabel('Tuổi (năm)')
# plt.ylabel('Phân suất tống máu (%)')
# plt.grid(True)
# plt.show()



# Vẽ biểu đồ histogram cho biến "age"
plt.hist(df['age'], bins=10, edgecolor='black')

# Thêm tiêu đề và nhãn
plt.title('Biểu đồ Histogram thể hiện tần suất độ tuổi mắc bệnh của bện nhân ')
plt.xlabel('Age (years)')
plt.ylabel('Frequency')

# Hiển thị biểu đồ
plt.show()


# # Vẽ biểu đồ Box & Whisker cho biến "age"
# plt.boxplot(df['age'])
#
# # Thêm tiêu đề và nhãn
# plt.title(' Biểu đồ Box & Whisker Plot cho biến Age')
# plt.ylabel('Age (years)')
#
# # Hiển thị biểu đồ
# plt.show()
