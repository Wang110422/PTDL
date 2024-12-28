import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Doc file csv
df = pd.read_csv("heart_failure_clinical_records.csv")

#Kiem tra dữ liệu có bao nhiêu cột nul
tf = df.isnull()
print(tf.sum())

#Xoa cac cot co du lieu khuyet
df_clearn = df.dropna()

#Chon các cột trừ cột DEATH_EVENT
df_choose = df_clearn.select_dtypes(include="number").drop(columns=["DEATH_EVENT"]);
df_death_event = df_clearn["DEATH_EVENT"]


#Chuẩn hóa dữ liệu
scaler = StandardScaler()
df_norma = scaler.fit_transform(df_choose)
df_norma = pd.DataFrame(df_norma, columns= df_choose.columns)

#Ghép lại cột Death_even vào
df_norma['DEATH_EVENT'] = df_death_event.values

#Xoa cot time vi không cần thiết
df_norma = df_norma.drop(columns=['time'])

#Ghi lại vào file mới
df_norma.to_csv('heart_failure_clinical_records_clean.csv', index=False)

#Hiển thị file sau chẩn hóa
print(df_norma.to_string())

