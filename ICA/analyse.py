import pandas 
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs
import numpy as np

data=pandas.read_csv('Employee_Data - Sheet1.csv')
d = data.head()
print(d)

desc = data.describe()
print(desc)

features=['Average montly hours','Average daily hours','Stressful days in week','Department']
fig=plt.subplots(figsize=(10,15))
for i, j in enumerate(features):
    plt.subplot(4, 2, i+1)
    plt.subplots_adjust(hspace = 1.0)
    sns.countplot(x=j,data = data)
    plt.xticks(rotation=90)
    plt.title("Employee")
    plt.show()