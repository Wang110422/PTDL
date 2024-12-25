
import pandas as pd
import numpy as np

# Doc file csv
df = pd.read_csv("heart_failure_clinical_records.csv")

# Lay cac cot dữ liệu dạng số
df_number = df.select_dtypes(include="number")

#Tom luoc du lieu

# Dem cac cot
df_count = df_number.count()
print(df_count)

#Tinh min
df_min = df_number.min()
print(df_min)

#Tinh max
df_max = df_number.max()
print(df_max)

# Tinh trung binh
df_mean = df_number.mean()
print(df_mean)

#Tinh trung vi
df_median = df_number.median()
print(df_median)

#Tinh mode
df_mode = df_number.mode()
print(df_mode)

#Tinh Q1, Q2 , Q3
df_Q1 = df_number.quantile(0.25)
df_Q2 = df_number.mean()
df_Q3 = df_number.quantile(0.75)
print(df_Q1,df_Q2,df_Q3)

#Tinh IQR
df_iqr = df_Q3-df_Q1
print(df_iqr)

#Tinh Phuong sai
df_var = df_number.var()
print(df_var)

#Tinh Do lech chuan
df_stdev = np.sqrt(df_var)
print(df_stdev)


def display(column_count , column_min , column_max , column_mean, column_median ,column_mode , column_Q1 , column_Q2 , column_Q3 , column_iqr , column_var , column_stdev):
    data = {'Count':[i for i in column_count],
            'Min': [i for i in column_min],
            'Max' : [i for i in column_max],
            'Mean' : [i for i in column_mean],
            'Median' : [i for i in column_median],
            'Mode': [i for i in column_mode.values[0]],
            'Q1' : [i for i in column_Q1],
            'Q2' : [i for i in column_Q2],
            'Q3' : [i for i in column_Q3],
            'IQR': [i for i in column_iqr],
            'Variance' : [i for i in column_var],
            'Stdev ' : [i for i in column_stdev]}
    df_convert = pd.DataFrame(data)
    df_convert.index = df_number.keys()
    data_convert = df_convert.transpose()

    #Them cot moi

    # Luu thanh file moi
    data_convert.to_csv('heart_failure_data_summarization.csv',index = False)

    print(data_convert.to_string())

# # Hien thi cac bang trong file csv
display(df_count,df_min,df_max,df_mean,df_median,df_mode,df_Q1,df_Q2,df_Q3,df_iqr,df_var,df_stdev)

