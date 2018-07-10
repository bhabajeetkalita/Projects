'''
Author: Bhabajeet Kalita
Date: 28 - 11 - 2017
Description: Use reservation and visitation data to predict the total number of visitors to a restaurant for future dates.
'''

import pandas as pd
#1st dataset:
'''
This file contains reservations made in the air system.
reserve_datetime -> time when the reservation was created.
visit_datetime -> time in the future where the visit will occur.

air_store_id - the restaurant's id in the air system
visit_datetime - the time of the reservation
reserve_datetime - the time the reservation was made
reserve_visitors - the number of visitors for that reservation
'''
df_air_reserve = pd.read_csv("air_reserve.csv")
df_air_reserve.head()

print("Unique: air_store_id: "+str(df_air_reserve['air_store_id'].nunique()))
print("Unique: visit_datetime: "+str(df_air_reserve['visit_datetime'].nunique()))
print("Unique: reserve_datetime: "+str(df_air_reserve['reserve_datetime'].nunique()))
print("Unique: reserve_visitors: "+str(df_air_reserve['reserve_visitors'].nunique()))

df_air_reserve['visit_date'] = pd.to_datetime(df_air_reserve['visit_datetime']).dt.date
df_air_reserve['visit_time'] = pd.to_datetime(df_air_reserve['visit_datetime']).dt.time
df_air_reserve['reserve_date'] = pd.to_datetime(df_air_reserve['reserve_datetime']).dt.date
df_air_reserve['reserve_time'] = pd.to_datetime(df_air_reserve['reserve_datetime']).dt.time
df_air_reserve.head()

#2nd dataset:
'''
This file contains information about select air restaurants. Column names and contents are self-explanatory.

air_store_id
air_genre_name
air_area_name
latitude
longitude
'''
df_air_store_info = pd.read_csv("air_store_info.csv")
df_air_store_info.head()

print("Unique: air_store_id: "+str(df_air_store_info['air_store_id'].nunique()))
print("Unique: air_genre_name: "+str(df_air_store_info['air_genre_name'].nunique()))
print("Unique: air_area_name: "+str(df_air_store_info['air_area_name'].nunique()))
print("Unique: latitude: "+str(df_air_store_info['latitude'].nunique()))
print("Unique: longitude: "+str(df_air_store_info['longitude'].nunique()))

#3rd dataset:

'''
This file contains historical visit data for the air restaurants.

air_store_id
visit_date - the date
visitors - the number of visitors to the restaurant on the date
'''
df_air_visit_data = pd.read_csv("air_visit_data.csv")
df_air_visit_data.head()

print("Unique: air_store_id: "+str(df_air_visit_data['air_store_id'].nunique()))
print("Unique: visit_date: "+str(df_air_visit_data['visit_date'].nunique()))
print("Unique: visitors: "+str(df_air_visit_data['visitors'].nunique()))

#4th dataset:

'''
This file gives basic information about the calendar dates in the dataset.

calendar_date
day_of_week
holiday_flg - is the day a holiday in Japan

'''
df_date_info = pd.read_csv("date_info.csv")
df_date_info.head()

print("Unique: calendar_date: "+str(df_date_info['calendar_date'].nunique()))
print("Unique: day_of_week: "+str(df_date_info['day_of_week'].nunique()))
print("Unique: holiday_flg: "+str(df_date_info['holiday_flg'].nunique()))

#5th dataset:
'''
This file contains reservations made in the hpg system.

hpg_store_id - the restaurant's id in the hpg system
visit_datetime - the time of the reservation
reserve_datetime - the time the reservation was made
reserve_visitors - the number of visitors for that reservation
'''
df_hpg_reserve = pd.read_csv("/Users/Gourhari/Desktop/DS_3/untitled folder/hpg_reserve.csv")
df_hpg_reserve.head()

print("Unique: hpg_store_id: "+str(df_hpg_reserve['hpg_store_id'].nunique()))
print("Unique: visit_datetime: "+str(df_hpg_reserve['visit_datetime'].nunique()))
print("Unique: reserve_datetime: "+str(df_hpg_reserve['reserve_datetime'].nunique()))
print("Unique: reserve_visitors: "+str(df_hpg_reserve['reserve_visitors'].nunique()))

df_hpg_reserve['visit_date'] = pd.to_datetime(df_hpg_reserve['visit_datetime']).dt.date
df_hpg_reserve['visit_time'] = pd.to_datetime(df_hpg_reserve['visit_datetime']).dt.time
df_hpg_reserve['reserve_date'] = pd.to_datetime(df_hpg_reserve['reserve_datetime']).dt.date
df_hpg_reserve['reserve_time'] = pd.to_datetime(df_hpg_reserve['reserve_datetime']).dt.time
df_hpg_reserve.head()

#6th dataset:
'''
This file contains information about select hpg restaurants. Column names and contents are self-explanatory.

hpg_store_id
hpg_genre_name
hpg_area_name
latitude
longitude
Note: latitude and longitude are the latitude and longitude of the area to which the store belongs
'''
df_hpg_store_info = pd.read_csv("hpg_store_info.csv")
df_hpg_store_info.head()

print("Unique: hpg_store_id: "+str(df_hpg_store_info['hpg_store_id'].nunique()))
print("Unique: hpg_genre_name: "+str(df_hpg_store_info['hpg_genre_name'].nunique()))
print("Unique: hpg_area_name: "+str(df_hpg_store_info['hpg_area_name'].nunique()))
print("Unique: latitude: "+str(df_hpg_store_info['latitude'].nunique()))
print("Unique: longitude: "+str(df_hpg_store_info['longitude'].nunique()))

#7th dataset:
'''
This file shows a submission in the correct format, including the days for which you must forecast.

id - the id is formed by concatenating the air_store_id and visit_date with an underscore
visitors- the number of visitors forecasted for the store and date combination

'''
df_sample_submission = pd.read_csv("sample_submission.csv")
df_sample_submission.head()

print("Unique: id: "+str(df_sample_submission['id'].nunique()))
print("Unique: visitors: "+str(df_sample_submission['visitors'].nunique()))

#8th dataset:

'''
store_id_relation.csv

This file allows you to join select restaurants that have both the air and hpg system.

hpg_store_id
air_store_id
'''
df_store_id_relation = pd.read_csv("store_id_relation.csv")
df_store_id_relation.head()

print("Unique: air_store_id: "+str(df_store_id_relation['air_store_id'].nunique()))
print("Unique: hpg_store_id: "+str(df_store_id_relation['hpg_store_id'].nunique()))

from sklearn import preprocessing
l = preprocessing.LabelEncoder()
train['air_store_id'] = l.fit_transform(train['air_store_id'])
train['visit_date'] = l.fit_transform(train['visit_date'])
train['air_genre_name'] = l.fit_transform(train['air_genre_name'])
train['air_area_name'] = l.fit_transform(train['air_area_name'])
train['day_of_week'] = l.fit_transform(train['day_of_week'])
train['holiday_flg'] = l.fit_transform(train['holiday_flg'])
train['visit_time'] = l.fit_transform(train['visit_time'])
train['reserve_date'] = l.fit_transform(train['reserve_date'])
train['reserve_time'] = l.fit_transform(train['reserve_time'])
train['visitors'] = train['visitors'].fillna(-1)
train['reserve_visitors'] = train['reserve_visitors'].fillna(-1)
train.head()

test['air_store_id'] = l.fit_transform(test['air_store_id'])
test['visit_date'] = l.fit_transform(test['visit_date'])
test['air_genre_name'] = l.fit_transform(test['air_genre_name'])
test['air_area_name'] = l.fit_transform(test['air_area_name'])
test['day_of_week'] = l.fit_transform(test['day_of_week'])
test['holiday_flg'] = l.fit_transform(test['holiday_flg'])
test['visit_time'] = l.fit_transform(test['visit_time'])
test['reserve_date'] = l.fit_transform(test['reserve_date'])
test['reserve_time'] = l.fit_transform(test['reserve_time'])
test['reserve_visitors'] = test['reserve_visitors'].fillna(0)
test.head()

x = train.drop(["visitors"],axis=1)
y = train['visitors']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.ensemble import RandomForestRegressor
x_train = train.drop(["visitors"],axis=1)
y_train = train['visitors']
x_test = test.drop(["visitors"],axis=1)

clf = RandomForestRegressor(n_estimators=100)
model = clf.fit(x_train,y_train)
test["visitors"] = model.predict(x_test)
test_final = pd.read_csv("sample_submission.csv")
test_final["visitors"] = test["visitors"]
test_final.to_csv("/Users/Gourhari/Desktop/DS_3/raina.csv",index=False,encoding="utf-8")
