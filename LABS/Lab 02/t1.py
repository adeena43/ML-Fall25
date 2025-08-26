import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Hello")
df1 = pd.read_csv('Lab2 D1A.csv')
df2 = pd.read_csv('Lab2 D1B.csv')
df3 = pd.read_csv('Lab2 D1C.csv')
print(df1)
newdf = pd.concat([df1, df2], ignore_index = True, axis = 0)
print(newdf)

newdf.drop_duplicates(keep=False, inplace=True)
newdf.isnull().sum()
df_duplicate=newdf.duplicated()
df_duplicate
df1.head()
df3.head()
comboAC = df1.merge(df2, how = 'inner', on = 'county')
print(comboAC)
