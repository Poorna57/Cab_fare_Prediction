#Clean the environment
rm(list = ls())

#Set working directory
setwd("C:/Users/sai/Documents/New folder R")
getwd()


#import libraries

library(readr) 
library(gridExtra)
library(corrgram)
library(caret)
library(tidyr)
library(rpart)
library(randomForest)
library(dplyr)
library(ggplot2)
library(data.table)
library(gbm)
library(usdm)
library(DMwR)

#Lets name our training dataset as df_train
df_train <- read_csv("train_cab.csv")


#Check the first 10 observations
View(df_train)

#Also  name our testing dataset as df_test
df_test <-read_csv("test.csv")

#Check the first 10 observations
View(df_test)

#Getting the column names of the dataset
colnames(df_train)
colnames(df_test)

#Getting the structure of the dataset
str(df_train)
str(df_test)

#Getting the number of variables and obervation in the datasets
dim(df_train)
dim(df_test)

#datatypes of train and test dataset
map(df_train, class)
map(df_test, class)

#summaries of train and test data's

summary(df_train)
summary(df_test)

#Clearly in the summary it can be seen there are so many anamolies:
# 1. fare amount is negative at few places and at a certain place it is 54343 which is not possible.
# 2. There are missing values in fare_amount and passenger_count
# 3. Number of passengers in few rows is 500 which is not possible for a cab to carry.

##################################Missing Values Analysis###############################################

#checking presence of Missing values in training set

missing_val = data.frame(apply(df_train,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(df_train)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]

View(missing_val)
## < - passenger_count, fare_amount is having missing values

#Plot these Missing values
ggplot(data = missing_val[1:3,], aes(x=reorder(Columns, -Missing_percentage),y = Missing_percentage))+
  geom_bar(stat = "identity",fill = "grey")+xlab("Parameter")+
  ggtitle("Missing data percentage (Train)") + theme_bw()

#checking presence of Missing values in test set

missing_val1 = data.frame(apply(df_test,2,function(x){sum(is.na(x))}))
missing_val1$Columns = row.names(missing_val1)
names(missing_val1)[1] =  "Missing_percentage"
missing_val1$Missing_percentage = (missing_val1$Missing_percentage/nrow(df_test)) * 100
missing_val1 = missing_val1[order(-missing_val1$Missing_percentage),]
row.names(missing_val1) = NULL
missing_val1 = missing_val1[,c(2,1)]
View(missing_val1)
## < - No missing values in test data 

#Mean Method Imputation
df_train$passenger_count[is.na(df_train$passenger_count)] = mean(df_train$passenger_count, na.rm = T)
df_train$fare_amount[is.na(df_train$fare_amount)] = mean(df_train$fare_amount, na.rm = T)

#check the number of Missing values after Imputation
sum(is.na(df_train))

## < - now no missing values are present

#Now lets convert our pickup_datetime to numeric
df_train$pickup_datetime = gsub( " UTC", "", as.character(df_train$pickup_datetime))
df_test$pickup_datetime = gsub( " UTC", "", as.character(df_test$pickup_datetime))

Split_datetime = function(df){
  df = separate(df, "pickup_datetime", c("Date", "Time"), sep = " ")
  df = separate(df, "Date", c("Year", "Month", "Day"), sep = "-")
  df = separate(df, "Time", c("Hour"), sep = ":")
  print(sum(is.na(df)))
  df$Year = as.numeric(df$Year)
  df$Month = as.numeric(df$Month)
  df$Day = as.numeric(df$Day)
  df$Hour = as.numeric(df$Hour)
  df$Year[is.na(df$Year)] = mean(df$Year, na.rm = T)
  df$Month[is.na(df$Month)] = mean(df$Month, na.rm = T)
  df$Day[is.na(df$Day)] = mean(df$Day, na.rm = T)
  df$Hour[is.na(df$Hour)] = mean(df$Hour, na.rm = T)
  print(sum(is.na(df)))
  return(df)}
df_train = Split_datetime(df_train)
df_test = Split_datetime(df_test)
          
################################## Removing Anamolies ###############################################

#Remove unwanted values and bring our dataset in proper shape
df_train = df_train[((df_train['fare_amount'] >= 0) & (df_train['fare_amount'] <=600)) & ((df_train['pickup_longitude']> -79)  &  (df_train['pickup_longitude'] < - 70)) & ((df_train['pickup_latitude']) > 36 & (df_train['pickup_latitude'] < 45)) & ((df_train['dropoff_longitude']) > -79 & (df_train['dropoff_longitude'] < -70)) & ((df_train['dropoff_latitude'] >= 36)  & (df_train['dropoff_latitude'] < 45)) & ((df_train['passenger_count']  >= 1) & (df_train['passenger_count'] <= 7)),]

df_train$pickup_latitude = gsub( "0.00000", "0", as.numeric(df_train$pickup_latitude))
df_train$dropoff_latitude = gsub( "0.00000", "0", as.numeric(df_train$dropoff_latitude))
df_train$pickup_longitude= gsub( "0.00000", "0", as.numeric(df_train$pickup_longitude))
df_train$dropoff_longitude = gsub( "0.00000", "0", as.numeric(df_train$dropoff_longitude))

#Remove rows containing 0 as value
df_train = df_train[apply(df_train, 1, function(row) all(row !=0)),]
df_train <- data.frame(sapply(df_train, function(x) as.numeric(as.character(x))))
sapply(df_train, class)

df_test = df_test[apply(df_test, 1, function(row) all(row !=0)),]
df_test <- data.frame(sapply(df_test, function(x) as.numeric(as.character(x))))
sapply(df_test, class)

############################################ Outlier Analysis #############################################

## BoxPlots - Distribution and Outlier Check

numeric_index = sapply(df_train,is.numeric) 

numeric_data = df_train[,numeric_index]

cnames = colnames(numeric_data)

for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), group = 1), data = subset(df_train))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],X="count")+
           ggtitle(paste("Box plot for",cnames[i])))
}

gridExtra::grid.arrange(gn2,gn3,ncol=2)
gridExtra::grid.arrange(gn6,gn7,ncol=2)
gridExtra::grid.arrange(gn4,gn5,ncol=2)
gridExtra::grid.arrange(gn8,gn9,ncol=2)

for(i in cnames){
  print(i)
  val = df_train[,i][df_train[,i] %in% boxplot.stats(df_train[,i])$out]
  print(length(val))
  df_train[,i][df_train[,i] %in% val] = NA
}
df_train <- data.frame(sapply(df_train, function(x) ifelse(is.na(x), mean( x, na.rm = TRUE),x)))

num = sapply(df_test,is.numeric) 

num_data = df_test[,num]

cnamestest = colnames(num_data)

for(c in cnamestest){
  print(c)
  val = df_test[,c][df_test[,c] %in% boxplot.stats(df_test[,c])$out]
  print(length(val))
  df_test[,c][df_test[,c] %in% val] = NA
}
df_test <- data.frame(sapply(df_test, function(y) ifelse(is.na(y), mean(y, na.rm = TRUE),y)))

df_train$passenger_count =  round(df_train$passenger_count)
df_test$passenger_count =  round(df_test$passenger_count)

ggplot(df_train, aes(x = fare_amount, y = pickup_latitude, group = 1)) +  geom_boxplot()
ggplot(df_train, aes(x = fare_amount, y = pickup_longitude  , group = 1)) +  geom_boxplot()
ggplot(df_train, aes(x = fare_amount, y = dropoff_longitude  , group = 1)) +  geom_boxplot()
ggplot(df_train, aes(x = fare_amount, y = dropoff_latitude  , group = 1)) +  geom_boxplot()
ggplot(df_train, aes(x = fare_amount, y = passenger_count  , group = 1)) +  geom_boxplot()

############################################ Univarate Analysis #############################################

hist(df_train$fare_amount)
hist(df_train$pickup_latitude)
hist(df_train$pickup_longitude)
hist(df_train$dropoff_latitude)
hist(df_train$dropoff_longitude)
hist(df_train$passenger_count)
hist(df_train$Year)
hist(df_train$Month)
hist(df_train$Day)
hist(df_train$Hour)

######################################### Bivariate  Relationship #####################################################
scat1 = ggplot(df_train, aes(x = fare_amount, y = pickup_latitude, group = 1)) +  geom_point()
scat2 = ggplot(df_train, aes(x = fare_amount, y = Year, group = 1)) +  geom_point()
scat3 = ggplot(df_train, aes(x = fare_amount, y = pickup_longitude  , group = 1)) +  geom_point()
scat4 = ggplot(df_train, aes(x = fare_amount, y = dropoff_longitude  , group = 1)) +  geom_point()
scat5 = ggplot(df_train, aes(x = fare_amount, y = dropoff_latitude  , group = 1)) +  geom_point()
scat6 = ggplot(df_train, aes(x = fare_amount, y = passenger_count  , group = 1)) +  geom_point()
scat7 = ggplot(df_train, aes(x = fare_amount, y = Day  , group = 1)) +  geom_point()
scat8 = ggplot(df_train, aes(x = fare_amount, y = Hour  , group = 1)) +  geom_point()
scat9 = ggplot(df_train, aes(x = fare_amount, y = Month  , group = 1)) +  geom_point()

gridExtra::grid.arrange(scat1,scat2,scat3,scat4,scat5,scat6,scat7,scat8,scat9,ncol=3)
################################## Feature Selection ################################################
## Correlation Plot 
corrgram(df_train[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

vifcor(df_train)

##### Splitting the data into train and test
set.seed(12)
n = nrow(df_train)
trainIndex = sample(1:n, size = round(0.8*n), replace=FALSE)
train = df_train[trainIndex ,]
test = df_train[-trainIndex ,]
X_train = subset(train,select = -c(fare_amount))
y_train = subset(train,select = c(fare_amount))

X_test = subset(test,select = -c(fare_amount))
y_test = subset(test,select = c(fare_amount))

############################################ Scaling the data #######################################
#Standardisation

for(i in colnames(X_train))
{
  print(i)
  X_train[,i] = (X_train[,i] - min(X_train[,i]))/(max(X_train[,i])-min(X_train[,i]))
}
for(i in colnames(X_test))
{
  print(i)
  X_test[,i] = (X_test[,i] - min(X_test[,i]))/(max(X_test[,i])-min(X_test[,i]))
}

######################################## Machine learning model##########################################

#### KNN #######################################
#Develop Model on training data

fit_knn = knnreg(fare_amount~ ., data =c(X_train,y_train))

#Lets predict for testing data
pred_knn_test = predict(fit_knn,X_test)

# Results
regr.eval(trues = y_test$fare_amount, preds = pred_knn_test, stats = c("mae","mse","rmse","mape"))
df_KN = data.frame("actual"=y_test$fare_amount, "pred"=pred_knn_test)
head(df_KN)

#mae        mse       rmse       mape 
#2.5410089 11.9405885  3.4555157  0.3069297



################# Multiple Linear Regression ############################

set.seed(100)
#Develop Model on training data
fit_LR = lm(fare_amount~ ., data = train)
summary(fit_LR)

#Lets predict for testing data
pred_LR_test = predict(fit_LR,test[,-1])

# Results
regr.eval(trues = y_test$fare_amount, preds = pred_LR_test, stats = c("mae","mse","rmse","mape"))
df = data.frame("actual"=y_test$fare_amount, "pred"=pred_LR_test)
head(df)

#mae        mse       rmse       mape 
#2.9642533 14.8734695  3.8566137  0.3968254

###### Decision Tree#########################

#Develop Model on training data
fit_DT = rpart(fare_amount ~., data = train, method = 'anova')
pred_DT_test = predict(fit_DT,test[,-1])

# Results

regr.eval(trues = y_test$fare_amount, preds = pred_DT_test, stats = c("mae","mse","rmse","mape"))

df_dt = data.frame("actual"=y_test$fare_amount, "pred"=pred_DT_test)
head(df_dt)

#mae        mse       rmse       mape 
#2.7778769 13.2811050  3.6443250  0.3665036 

################### Random Forest#################################
#Develop Model on training data
fit_RF = randomForest(fare_amount ~., data = train,ntree = 500 ,nodesize =8,importance=TRUE)
pred_RF_test = predict(fit_RF,test[,-1])

# Results
regr.eval(trues = y_test$fare_amount, preds = pred_RF_test, stats = c("mae","mse","rmse","mape"))
df_RF = data.frame("actual"=y_test$fare_amount, "pred"=pred_RF_test)
head(df_RF)

#mae       mse      rmse      mape 
#1.9770044 7.6294763 2.7621507 0.2494035

#################### GBDT###########################

#Develop Model on training data
fit_GBDT = gbm(fare_amount ~., data = train, n.trees = 500, interaction.depth = 2)

#Lets predict for testing data
pred_GBDT_test = predict(fit_GBDT,test[,-1], n.trees = 500)

# Results
regr.eval(trues = y_test$fare_amount, preds = pred_GBDT_test, stats = c("mae","mse","rmse","mape"))
df_GB = data.frame("actual"=y_test$fare_amount, "pred"=pred_GBDT_test)
head(df_GB)

#mae       mse      rmse      mape 
#2.0899910 8.4587235 2.9083885 0.2658953

df_rmse = data.frame("rmse" = c("3.4","3.8","3.64","2.76","2.90" ),"Model" = c("KNN","Linear Regression","Decision tree","Random forest","GBDT"))
print(df_rmse)

#Prediction for given test data with best fit model random forest
pred_RF_test = predict(fit_RF,df_test)
pred_RF_test_R_DF = data.frame("fare_amount" = pred_RF_test )
write.csv(pred_RF_test_R_DF,"Test_Predictions_R.csv",row.names = FALSE)