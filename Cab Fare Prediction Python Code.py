#!/usr/bin/env python
# coding: utf-8

# In[38]:


#Importing the necessary packages
import os
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import GradientBoostingRegressor


# In[3]:


#Set working directory
os.chdir("C:/Users/sai/Documents/Python Scripts")
print(os.getcwd())


# In[4]:


# Loading the the train and test datasets associated with the Cab Fare: 
df_train = pd.read_csv("train_cab.csv")
df_test = pd.read_csv("test.csv")

#Check the first 5 rows of our datasets
print(df_train.head(30))
print("\n")
print(df_test.head(5))


# In[5]:


#Data Types of the columns
print(df_train.dtypes)
print(df_test.dtypes)

#Converting the fare amount column into numeric data form
df_train["fare_amount"] = pd.to_numeric(df_train["fare_amount"],errors='coerce')

print(df_train.dtypes)


# In[6]:


#description of our datatests
print(df_train.describe())
print(df_test.describe())


# In[9]:


########################################## MisssingValue Analysis ###################################################

def missin_val(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() * 100 /df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return(missing_data)

print("The missing value percentage in training data : \n\n",missin_val(df_train))
print("\n")
print("The missing value percentage in test data : \n\n",missin_val(df_test))
print("\n")

df_train["passenger_count"] = df_train["passenger_count"].fillna(df_train["passenger_count"].mean())
df_train["fare_amount"] = df_train["fare_amount"].fillna(df_train["fare_amount"].mean())

print("Is there still any missing value in the training data:\n\n",missin_val(df_train))
print("\n")


# In[10]:


######################### Proper Aligning the Dataset ############################################################

## The fare amount column is having some neagative values, Lets Check it
print(Counter(df_train['fare_amount']<0))
print(Counter(df_train['fare_amount']>1000))
#Also there are values like more 6 persons in a cab, Lets cross check
print(Counter(df_train['passenger_count']>6))
print(Counter(df_train['passenger_count']<0))

#Originally, Latitudes range from -90 to 90.
#Originally, Longitudes range from -180 to 180.
#But our data is purely negative Longitudes and purely positive latitudes
#lets align our data in its respective minimum and maximum Longitudes 
#and latitudes values, also removing fare amount,passenger count those are negative and above optimum level.


df_train = df_train[((df_train['pickup_longitude'] > -79) & (df_train['pickup_longitude'] < -70)) & 
           ((df_train['dropoff_longitude'] > -79) & (df_train['dropoff_longitude'] < -70)) & 
           ((df_train['pickup_latitude'] > 36) & (df_train['pickup_latitude'] < 45)) & 
           ((df_train['dropoff_latitude'] > 36) & (df_train['dropoff_latitude'] < 45)) & 
           ((df_train['passenger_count'] > 0) & (df_train['passenger_count'] < 7))  &
           ((df_train['fare_amount'] > 0)& (df_train['fare_amount'] < 1000))]
    

df_test = df_test[((df_test['pickup_longitude'] > -79) & (df_test['pickup_longitude'] < -70)) & 
           ((df_test['dropoff_longitude'] > -79) & (df_test['dropoff_longitude'] < -70)) & 
           ((df_test['pickup_latitude'] > 36) & (df_test['pickup_latitude'] < 45)) & 
           ((df_test['dropoff_latitude'] > 36) & (df_test['dropoff_latitude'] < 45)) & 
           (df_test['passenger_count'] > 0) ]


# In[11]:


#Split our Datetime into individual columns for ease of data processing and modelling
def align_datetime(df):
    df["pickup_datetime"] = df["pickup_datetime"].map(lambda x: str(x)[:-3])
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], format='%Y-%m-%d %H:%M:%S')
    df['year'] = df.pickup_datetime.dt.year
    df['month'] = df.pickup_datetime.dt.month
    df['day'] = df.pickup_datetime.dt.day
    df['weekday'] = df.pickup_datetime.dt.weekday
    df['hour'] = df.pickup_datetime.dt.hour
    return(df["pickup_datetime"].head())
    
align_datetime(df_train)
align_datetime(df_test)


#Remove the datetime column
df_train.drop('pickup_datetime', axis=1, inplace=True)
df_test.drop('pickup_datetime', axis=1, inplace=True)


#Checking NA in the fresh Dataset
print(df_train.isnull().sum())
df_train=df_train.fillna(df_train.mean())
print(df_train.isnull().sum())

print(df_train.head(5))
print("\n")
print(df_test.head(5))


# In[12]:


#Setting proper data type for each columns
print(df_train.dtypes)
print(df_test.dtypes)

df_train= df_train.astype({"passenger_count":int,"year":int,"month":int ,"day" :int,"weekday":int,"hour":int})
print(df_train.dtypes)


# In[14]:


##################################################  Outlier Analysis ###############################################################################################
df_train.plot(kind='box', subplots=True, layout=(8,3), sharex=False, sharey=False, fontsize=8)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top= 3,wspace=0.2, hspace=0.2)
plt.show()


##Detect and delete outliers from data
def outliers_analysis(df): 
    for i in df.columns:
        print(i)
        q75, q25 = np.percentile(df.loc[:,i], [75 ,25])
        iqr = q75 - q25

        min = q25 - (iqr*1.5)
        max = q75 + (iqr*1.5)
        print(min)
        print(max)
    
        df = df.drop(df[df.loc[:,i] < min].index)
        df = df.drop(df[df.loc[:,i] > max].index)
    return(df)
 
    
def eliminate_rows_with_zero_value(df):
    df= df[df!= 0]
    df=df.fillna(df.mean())
    return(df)

df_train = outliers_analysis(df_train)
df_train = eliminate_rows_with_zero_value(df_train)

df_train.plot(kind='box', subplots=True, layout=(8,3), sharex=False, sharey=False, fontsize=8)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top= 3,wspace=0.2, hspace=0.2)
plt.show()

   



# In[15]:


df_test.plot(kind='box', subplots=True, layout=(8,3), sharex=False, sharey=False, fontsize=8)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top= 3,wspace=0.2, hspace=0.2)
plt.show()

df_test = outliers_analysis(df_test)
df_test = eliminate_rows_with_zero_value(df_test)

df_test.plot(kind='box', subplots=True, layout=(8,3), sharex=False, sharey=False, fontsize=8)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top= 3,wspace=0.2, hspace=0.2)
plt.show()


# In[16]:


######################################## univarate analysis #########################################################################################################

#Histogram Plot of passenger_count Column
plt.figure(figsize=(7,7))
plt.hist(df_train['passenger_count'],bins = 6)
plt.xlabel('No. of Passengers')
plt.ylabel('Frequency')

#Histogram Plot of passenger_count Column
plt.figure(figsize=(7,7))
plt.hist(df_train['fare_amount'],bins=25)
plt.xlabel('Different amount of Fare')
plt.ylabel('Frequency')

#Histogram Plot of day Column
plt.figure(figsize=(7,7))
plt.hist(df_train['day'],bins=10)
plt.xlabel('Different Days of the month')
plt.ylabel('Frequency')

#Histogram Plot of weekday Column
plt.figure(figsize=(7,7))
plt.hist(df_train['weekday'],bins=10)
plt.xlabel('Different Days of the week')
plt.ylabel('Frequency')

#Histogram Plot of hour Column
plt.figure(figsize=(7,7))
plt.hist(df_train['hour'],bins=10)
plt.xlabel('Different hours of the Day')
plt.ylabel('Frequency')

#Histogram Plot of month Column
plt.figure(figsize=(7,7))
plt.hist(df_train['month'],bins=10)
plt.xlabel('Different Months of the year')
plt.ylabel('Frequency')

#Histogram Plot of year Column
plt.figure(figsize=(7,7))
plt.hist(df_train['year'],bins=10)
plt.xlabel('Years')
plt.ylabel('Frequency')

#Histogram Plot of year Column
plt.figure(figsize=(7,7))
plt.hist(df_train['dropoff_latitude'])
plt.xlabel('dropoff_latitude')
plt.ylabel('Frequency')
#Histogram Plot of year Column
plt.figure(figsize=(7,7))
plt.hist(df_train['dropoff_longitude'])
plt.xlabel('dropoff_longitude')
plt.ylabel('Frequency')

#Histogram Plot of year Column
plt.figure(figsize=(7,7))
plt.hist(df_train['dropoff_latitude'])
plt.xlabel('dropoff_latitude')
plt.ylabel('Frequency')
#Histogram Plot of year Column
plt.figure(figsize=(7,7))
plt.hist(df_train['pickup_latitude'])
plt.xlabel('pickup_latitude')
plt.ylabel('Frequency')



# In[17]:



##################################################  Density Plots ##################################################################################################
fig,x = plt.subplots(nrows=5,ncols=2)
fig.set_size_inches(10,12)
sns.kdeplot(df_train['fare_amount'], shade = True,ax=x[0][0])
sns.kdeplot(df_train['passenger_count'], shade = True,ax=x[0][1])
sns.kdeplot(df_train['month'], shade = True,ax=x[1][0])
sns.kdeplot(df_train['day'], shade = True,ax=x[1][1])
sns.kdeplot(df_train['weekday'], shade = True,ax=x[2][0])
sns.kdeplot(df_train['hour'], shade = True,ax=x[2][1])
sns.kdeplot(df_train['dropoff_latitude'], shade = True,ax=x[3][0])
sns.kdeplot(df_train['pickup_longitude'], shade = True,ax=x[3][1])
sns.kdeplot(df_train['pickup_latitude'], shade = True,ax=x[4][0])
sns.kdeplot(df_train['dropoff_longitude'], shade = True,ax=x[4][1])



# In[24]:



################################################## Bivariate Plots #################################################################################################
fig,x = plt.subplots(nrows=6,ncols=2)
fig.set_size_inches(10,12)

sns.scatterplot(x="passenger_count", y="fare_amount", data= df_train, palette="Set2",ax=x[0][0])
sns.scatterplot(x="month", y="fare_amount", data= df_train, palette="Set2",ax=x[0][1])
sns.scatterplot(x="weekday", y="fare_amount", data= df_train, palette="Set2",ax=x[1][0])
sns.scatterplot(x="hour", y="fare_amount", data= df_train, palette="Set2",ax=x[1][1])
sns.scatterplot(x="pickup_longitude", y="fare_amount", data= df_train, palette="Set2",ax=x[2][0])
sns.scatterplot(x="pickup_latitude", y="fare_amount", data= df_train, palette="Set2",ax=x[2][1])
sns.scatterplot(x="pickup_latitude", y="pickup_longitude", data= df_train, palette="Set2",ax=x[3][0])
sns.scatterplot(x="dropoff_latitude", y="dropoff_longitude", data= df_train, palette="Set2",ax=x[3][1])
sns.scatterplot(x="dropoff_longitude", y="fare_amount", data= df_train, palette="Set2",ax=x[4][0])
sns.scatterplot(x="dropoff_latitude", y="fare_amount", data= df_train, palette="Set2",ax=x[4][1])
sns.scatterplot(x="day", y="fare_amount", data= df_train, palette="Set2",ax=x[5][0])


# In[25]:


########################### Feature Selection ############################# 
##Correlation analysis
#Correlation plot
def Correlation(df):
    df_corr = df.loc[:,df.columns]
    sns.set()
    plt.figure(figsize=(9, 9))
    corr = df_corr.corr()
    sns.heatmap(corr, annot= True,fmt = " .3f", linewidths = 0.5,
            square=True)
    
Correlation(df_train)
Correlation(df_test)


# In[27]:


## Splitting DataSets######
train,test = train_test_split(df_train, test_size = 0.2,random_state = 123 )
X_train = train.loc[:,train.columns != 'fare_amount']
Y_train = train['fare_amount']
X_test = test.loc[:,test.columns != 'fare_amount']
Y_test = test['fare_amount']

print(train.head(5))
print(test.head(5))


# In[28]:


############################ Feature Scaling ##############################
# #Normalisation
def Normalisation(df):
    for i in df.columns:
        print(i)
        df[i] = (df[i] - df[i].min())/(df[i].max() - df[i].min())
        
Normalisation(X_train)
Normalisation(X_test)

print(X_train.head(5))
print(X_test.head(5))


# In[29]:


#Calculate MAPE
def MAPE(y_true, y_pred): 
    MAE = np.mean(np.abs((y_true - y_pred)))
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    print("MAPE is: ",mape)
    print("MAE is: ",MAE)              
    return mape

def RMSE(y_test,y_predict):
    mse = np.mean((y_test-y_predict)**2)
    print("Mean Square : ",mse)
    rmse=np.sqrt(mse)
    print("Root Mean Square : ",rmse)
    return rmse


# In[31]:


dt_lnr_model = sm.OLS(train.iloc[:,0],train.iloc[:,1:]).fit()

#Summary of model
print(dt_lnr_model.summary())

#predict the  model

dt_predict_LR = dt_lnr_model.predict(test.iloc[:,1:])



MAPE(test.iloc[:,0], dt_predict_LR)
RMSE(test.iloc[:,0], dt_predict_LR)

#MAPE is:  0.4002251684131252
#MAE is:  2.9476151408555484
#Mean Square :  14.171074990188412
#Root Mean Square :  3.76444882953513


# In[33]:


###### KNN Modelling ########

KNN_model = KNeighborsRegressor(n_neighbors= 40).fit(X_train , Y_train)
KNN_pred= KNN_model.predict(X_test)


df_results = pd.DataFrame({'actual': Y_test, 'pred': KNN_pred})
print(df_results.head())

MAPE(Y_test, KNN_pred)
RMSE(Y_test, KNN_pred)

#MAPE is:  0.3084418890595272
#MAE is:  2.470047013144009
#Mean Square :  10.781339613948111
#Root Mean Square :  3.283495030291368


# In[34]:


#################################################Decision Tree##################################################
dt_model = DecisionTreeRegressor(random_state=123).fit(train.iloc[:,1:], train.iloc[:,0])

print(dt_model)

dt_predictions = dt_model.predict(test.iloc[:,1:])


df_results = pd.DataFrame({'actual': test.iloc[:,0], 'pred': dt_predictions})
df_results.head()


MAPE(test.iloc[:,0], dt_predictions)
RMSE(test.iloc[:,0], dt_predictions)

#MAPE is:  0.2870557144870078
#MAE is:  2.248405252730194
#Mean Square :  10.056863408315259
#Root Mean Square :  3.1712558093467105


# In[46]:


############################################Random Forest ################################################

# Set best parameters given by random search  
rf.set_params(max_features = 'auto',
               max_depth =8, 
               n_estimators = 500,
              bootstrap = 'True'
                )

rf.fit(train.iloc[:,1:], train.iloc[:,0])

# Use the forest's predict method on the test data
dt_rfPredictions = rf.predict(test.iloc[:,1:])

df_results = pd.DataFrame({'actual': test.iloc[:,0], 'pred': dt_rfPredictions})
print(df_results.head())

MAPE(test.iloc[:,0], dt_rfPredictions)
RMSE(test.iloc[:,0], dt_rfPredictions)

#MAPE is:  0.2970395006180068
#MAE is:  2.1666677762006925
#Mean Square :  8.021622657590507
#Root Mean Square :  2.832246927368888


# In[40]:


###### GBR Modelling ##########

        
gbr_model = GradientBoostingRegressor(max_depth= 2,learning_rate = 0.1).fit(train.iloc[:,1:],train.iloc[:,0])
gbr_pred= gbr_model.predict(test.iloc[:,1:])

MAPE(test.iloc[:,0], gbr_pred)
RMSE(test.iloc[:,0], gbr_pred)

#MAPE is:  0.3024254007844994
#MAE is:  2.1996787335105425
#Mean Square :  8.13426907872889
#Root Mean Square :  2.8520640032665625


# In[41]:


df_rmse = pd.DataFrame({"rmse":[3.28,3.76,3.17,2.85,2.83],                   "Model" : ['KNN Regression' ,'Linear Regression','Decision Trees', "GBDT", "Random Forest"]})
print(df_rmse)


# In[56]:


#output for given Test data with best model obtained
dt_rfPredictions_test = rf.predict(df_test)
dt_rfPredictions_test = pd.DataFrame(dt_rfPredictions_test, columns = ["fare_amount"])
dt_rfPredictions_test.to_csv("TestdataPrediction.csv",index=False)


# In[ ]:




