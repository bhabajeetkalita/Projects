'''
Author: Bhabajeet Kalita

Date: 14-12-2017

Description: Predict the Gender of a person visiting the Lesara Website.

1. First, we import both the training & test sets using the Pandas Library.
2. Extract the month, day, hour, minute & second from the click_time column of the training & testing sets.
3. Categorise the host_name & page_path columns of the training & testing sets.
4. Visualize the training set using the Seaborn library.
5. Select the appropriate columns needed for making the predictive model.
6. Encode the training and test sets for faster processing using StandardScaler.
7. Use Random Forest Classifier to predict the Gender.
8. Predict on the Test Set.
9. Export the results to another csv file.
10. Repeat steps 7-9 using Artificial Neural Networks (ANN).

Better models could be made on the basis of the page_path if more information is obtained on it other than the hash values.
Recurrent Neural Networks can be also used like predicting stock prices data.
Cross validation (GridSearchCV) could be used as an extension on the ANN to figure out cross val score and tune better models.
Neural N/w training takes a long time if the systems are not fast enough.
www.lesara.co.uk has very few values in this dataset and so, making 2 separate models for www.lesara.it & www.lesara.co.uk could help
Training Testing on the existing training data sets can be used to figure out the confusion matrix and try different methods.


'''
#Importing the training and test sets
import pandas as pd
df_1 = pd.read_csv("/Users/Gourhari/Downloads/Lesara/train.csv")
df_2 = pd.read_csv("/Users/Gourhari/Downloads/Lesara/test.csv")

df_1.head()

#Extracting the month, day, hour, minute & second from the click_time column of the training & testing sets
df_1["Month"]= pd.to_datetime(df_1["click_time"],unit='ms').dt.month
df_2["Month"]= pd.to_datetime(df_2["click_time"],unit='ms').dt.month

df_1["Day"]= pd.to_datetime(df_1["click_time"],unit='ms').dt.day
df_2["Day"]= pd.to_datetime(df_2["click_time"],unit='ms').dt.day

df_1["Hour"]= pd.to_datetime(df_1["click_time"],unit='ms').dt.hour
df_2["Hour"]= pd.to_datetime(df_2["click_time"],unit='ms').dt.hour

df_1["Minute"]= pd.to_datetime(df_1["click_time"],unit='ms').dt.minute
df_2["Minute"]= pd.to_datetime(df_2["click_time"],unit='ms').dt.minute

df_1["Second"]= pd.to_datetime(df_1["click_time"],unit='ms').dt.second
df_2["Second"]= pd.to_datetime(df_2["click_time"],unit='ms').dt.second
df_1.head(10)

#Categorising the host_name & page_path columns of the training & testing sets
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
df_1["page_path"] = labelencoder_X_1.fit_transform(df_1["page_path"])

labelencoder_X_2 = LabelEncoder()
df_1["host_name"] = labelencoder_X_2.fit_transform(df_1["host_name"])

labelencoder_Y_1 = LabelEncoder()
df_2["page_path"] = labelencoder_Y_1.fit_transform(df_2["page_path"])

labelencoder_Y_2 = LabelEncoder()
df_2["host_name"] = labelencoder_Y_2.fit_transform(df_2["host_name"])
df_1.head(10)

#Checking the data types
print(df_1.dtypes)

#Visualizing the training set using Seaborn library
import seaborn as sns

#We see that www.lesara.co.uk has quite less data

sns.countplot(x='host_name',data=df_1)

#We see that gender value "2" has more than double the value than gender value "1"
sns.countplot(x='gender',data=df_1)

#On days 10-18, both the genders visit more
sns.violinplot(x='gender',y='Day',data=df_1)

#Building the predicitve model
#Setting the dependent variables for the model
#client_id column is dropped because it will have no impact on the model because it is not relevant

x_train = df_1.drop(['client_id','click_time','gender'], axis=1)
x_test = df_2.drop(['client_id','click_time'], axis=1)

#Independent variable gender which is to be predicted in the test set
y_train = df_1["gender"]

x_train.head(10)

#Encoding the training and test sets for faster processing using RFC & Artificial Neural Networks
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#Using Random Forest Classifier to solve the problem
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)

#Fitting the model
model = clf.fit(x_train,y_train)

#Predicting on the Test Set
final_output = model.predict(x_test)

#Exporting the results to a csv file
import pandas
df_result = pandas.DataFrame(final_output)
df_result.columns = ['gender']
df_result.head(10)
df_result.to_json("/Users/Gourhari/Desktop/result.json",orient="records")

#A sample example of prediction
print(model.predict([[0,14763,6,15,9,23,15]]))



'''
#Using Artificial Neural Networks to solve the problem
#I did not run this due to my laptop's low ram, it was taking around many hours so, I stopped the kernel, but it works

#Import the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialise the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
#Output dim value has been taken as half of the sum of the dependent and independent variables
#input_dim=7 because of 7 independent variables
classifier.add(Dense(output_dim=4,init= 'uniform',activation = 'relu',input_dim=7))

#Adding the second hidden layer
classifier.add(Dense(output_dim=4,init= 'uniform',activation = 'relu'))

#Adding the output layer
classifier.add(Dense(output_dim=1,init= 'uniform',activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer='adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

#Fitting the ANN to the Training Set
classifier.fit(x_train,y_train,batch_size=10,nb_epoch=10)

#Predicting on the Test Set
y_pred = classifier.predict(x_test)

#Exporting the results to a csv file
import pandas
df_result = pandas.DataFrame(y_pred)
df_result.columns = ['gender']
df_result.head(10)
df_result.to_csv("/Users/Gourhari/Desktop/result.csv",encoding="utf-8",index=None)

'''
