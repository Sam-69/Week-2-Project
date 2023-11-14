Week 2 Project  
Question 2). Let’s say we want to build a model to predict booking prices on Airbnb. Between linear regression and random forest regression, which model would perform better and why?

Airbnb price prediction

import pyforest

data=pd.read_csv("train.csv")

data.head()

Data analysis and exploration

data.shape

data.amenities[0]

​
data = {'sets_column': [{'Wireless Internet', 'Air conditioning', 'Kitchen', 'Heating', 'Family/kid friendly', 'Essentials', 'Hair dryer', 'Iron', 'translation missing: en.hosting_amenity_50'}]}

​
df = pd.DataFrame(data)

​
df['list_column'] = df['sets_column'].apply(lambda x: [f'"{item}"' for item in x])
​
print(df)
​
data.amenities=data.amenities.apply(lambda x:  [f'"{item}"' for item in x])
​
data.amenities

data.info()

data.drop(data[["amenities"]],inplace=True ,axis=1)

data.head()

data["description"][0]

data.thumbnail_url[2]

data.info()

data.isna().sum()

clean=['first_review','host_response_rate','last_review','review_scores_rating']

data.drop(data[clean],inplace=True,axis=1)
​
data.head()

data.info()

categorical=[]

numerical=[]

for i in data.columns:
    if data[i].dtype=='object' or data[i].dtype=='bool':
        categorical.append(i)
    else:
        numerical.append(i)
    
categorical,numerical

data['host_has_profile_pic'].value_counts()

data_num=data[numerical]

data_num.head()

data_cat=data[categorical]

data_cat.head()

from sklearn.impute import KNNImputer

imputer=KNNImputer(n_neighbors=2)

filled=pd.DataFrame(imputer.fit_transform(data_num),columns=data_num.columns)

filled.isna().sum()

data_cat.isna().sum()

data_cat.dropna(inplace=True,axis=1)

data_cat.isna().sum()
​
merged_df = pd.concat([data_cat, filled], axis=1)

# Merge based on a common column (e.g., 'key_column')
​
​
merged_df.isna().sum()

merged_df.head()

merged_df.describe()

Data visualization

import warnings 

sns.countplot(merged_df["bed_type"])
​
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import LabelEncoder

lb=LabelEncoder()

columns_to_encode = ["property_type", "room_type", "bed_type", "cancellation_policy", "cleaning_fee", "instant_bookable"]

​
# Loop through each column and apply LabelEncoder
for column in columns_to_encode:
    merged_df[column] = lb.fit_transform(merged_df[column])
​
​
​
​
​
​
​
​
​
merged_df.drop(["city","description","name","id","beds","property_type","bedrooms","bed_type"],axis=1,inplace=True)

from sklearn.preprocessing import StandardScaler

scalling=StandardScaler()

scalling.fit(merged_df)

merged_df= pd.DataFrame(scalling.transform(merged_df), columns=merged_df.columns)
​
merged_df.head()

x=merged_df.drop(["log_price"], axis=1)

y=merged_df["log_price"]

print( x.shape)

print(y.shape)

merged_df.shape

corr=merged_df.corr()

plt.figure(figsize=(14,9))

sns.heatmap(corr,annot=True)

#spliting the train and test data

x_train, x_test,y_train,y_test=train_test_split(x,y, test_size=0.2,random_state=2)

print(y_train.shape, y_test.shape)

using linear regression

model = LinearRegression()

model.fit(x_train, y_train)
​
​
pred=model.predict(x_test)

error_score=metrics.r2_score(y_test,pred)

print(error_score)

using random forest regression

models=RandomForestRegressor()

models.fit(x_train, y_train)

pred=models.predict(x_test)

error_score=metrics.r2_score(y_test,pred)

print(error_score)
RandomForest Regression worked better than Linear Regression
                                                                                                                                                                                                                    
