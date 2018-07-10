'''
Author: Bhabajeet Kalita

Date: 21/12/2017

Description: Predicting the fare for bikes.


'''

#Uploading the training & test sets
import pandas as pd
df_1 = pd.read_csv("/Users/Gourhari/Downloads/DS/train_stud.csv")
df_2 = pd.read_csv("/Users/Gourhari/Downloads/DS/test_stud_revised.csv")
df_3 = pd.read_csv("/Users/Gourhari/Downloads/DS/sample_submission_blank.csv")
df_1.head()

df_1.describe

#Visualising the data types -
df_1.dtypes

df_1["fare_amount"].nunique()

#Dropping the first column of index which is unnecessary

df_1.drop(df_1.columns[[0]], axis=1, inplace=True)
df_1.head()

df_2.drop(df_2.columns[[0]], axis=1, inplace=True)
df_2.head()

#Dropping the driver_id column
df_1.drop(df_1.columns[[0]], axis=1, inplace=True)
df_1.head()

df_2.drop(df_2.columns[[0]], axis=1, inplace=True)
df_2.head()

#Filling the null values of the training & testing tests using specific values

df_1["vehicle_id"].fillna(1000,inplace=True)
df_1["vendor_id"].fillna(1,inplace=True)
df_1["rate_code"].fillna(6,inplace=True)
df_1["passenger_count"].fillna(3,inplace=True)
df_1["pickup_date"].fillna(1001325,inplace=True)
df_1["pickup_longitude"].fillna(-73.988052,inplace=True)
df_1["pickup_latitude"].fillna(40.738071,inplace=True)
df_1["dropoff_date"].fillna(1539102,inplace=True)
df_1["dropoff_longitude"].fillna(-74.004219,inplace=True)
df_1["dropoff_latitude"].fillna(40.742226,inplace=True)
df_1["payment_type"].fillna(1,inplace=True)
df_1["mta_tax"].fillna(0.5,inplace=True)
df_1["tip_amount"].fillna(0.00,inplace=True)
df_1["tolls_amount"].fillna(0.00,inplace=True)


df_2["vehicle_id"].fillna(1000,inplace=True)
df_2["vendor_id"].fillna(1,inplace=True)
df_2["rate_code"].fillna(6,inplace=True)
df_2["passenger_count"].fillna(3,inplace=True)
df_2["pickup_date"].fillna(1001325,inplace=True)
df_2["pickup_longitude"].fillna(-73.988052,inplace=True)
df_2["pickup_latitude"].fillna(40.738071,inplace=True)
df_2["dropoff_date"].fillna(1539102,inplace=True)
df_2["dropoff_longitude"].fillna(-74.004219,inplace=True)
df_2["dropoff_latitude"].fillna(40.742226,inplace=True)
df_2["payment_type"].fillna(1,inplace=True)
df_2["mta_tax"].fillna(0.5,inplace=True)
df_2["tip_amount"].fillna(0.00,inplace=True)
df_2["tolls_amount"].fillna(0.00,inplace=True)

from sklearn import preprocessing
l = preprocessing.LabelEncoder()


df_1['payment_type'] = l.fit_transform(df_1['payment_type'])
df_1['vendor_id'] = l.fit_transform(df_1['vendor_id'])
df_1['pickup_date'] = l.fit_transform(df_1['pickup_date'])
df_1['dropoff_date'] = l.fit_transform(df_1['dropoff_date'])



df_2['payment_type'] = l.fit_transform(df_2['payment_type'])
df_2['vendor_id'] = l.fit_transform(df_2['vendor_id'])
df_2['pickup_date'] = l.fit_transform(df_2['pickup_date'])
df_2['dropoff_date'] = l.fit_transform(df_2['dropoff_date'])

x_train = df_1.drop(['fare_amount'], axis=1)
y_train = df_1['fare_amount']

#Using Random Forest Classifier to solve the problem
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=10)
#Fitting the model
model = clf.fit(x_train,y_train)

#Predicting on the Test Set
final_output = model.predict(df_2)
len(final_output)

df_2["fare_amount"] = final_output
df_2.head()

df_2["fare_amount"].to_csv("/Users/Gourhari/Desktop/Bhabajeet.csv",index="id")
