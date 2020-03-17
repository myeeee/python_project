import pandas as pd
import missingno as msno
import datetime
import matplotlib.pyplot as plt
import numpy as np
import calendar
from datetime import datetime
import seaborn as sn

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data["date"] = train_data.datetime.apply(lambda x : x.split()[0])
train_data["hour"] = train_data.datetime.apply(lambda x : x.split()[1].split(":")[0])
train_data["weekday"] = train_data.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString, '%Y-%m-%d').weekday()])

print(train_data.shape)
print(train_data.dtypes)
print(train_data.head(2))
msno.matrix(train_data,figsize=(12,5))

fig, axes = plt.subplots(nrows=2, ncols=2)

train_data_withoutOutlier = train_data[np.abs(train_data['count'] - train_data['count'].mean()) <= 3 * train_data['count'].std()]

corrMatt = train_data_withoutOutlier[["temp","atemp","casual","registered","humidity","windspeed","count"]].corr()
mask = np.array(corrMatt)
print('before: ', mask)

print(np.tril_indices_from(mask))

mask[np.tril_indices_from(mask)] = False
print('after: ', mask)
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)