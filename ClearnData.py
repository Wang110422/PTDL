import pandas as pd
import numpy as np

# Doc file csv
df = pd.read_csv("heart_failure_clinical_records.csv")
# Lay cac cot dữ liệu dạng số
df_number = df.select_dtypes(include="number")
tf = df_number.isnull()
df_clearn = df_number.dropna()
# print(df_clearn.count())
# print(df_clearn.dtypes)



print(df_clearn['time'].dtypes)

