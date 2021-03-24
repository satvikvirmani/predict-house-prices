import numpy
import pandas
from sklearn import linear_model,model_selection,metrics
import matplotlib.pyplot as plt
import seaborn
from scipy import stats

#Import the data
data = pandas.read_csv('data.csv')
df = data.copy()
print((df.dtypes).to_markdown())

#Relation of variables with price
print((df.corr()).to_markdown())
print((df.isnull().sum()).to_markdown())
print((df[df==0].count()).to_markdown())

#Plot sqft_living for 0 price
plt.figure(figsize=(9,6))
ax = seaborn.distplot(df[df["price"]==0].sqft_living)
ax.set_title('Sqft_living for 0 price', fontsize=14)
plt.show()

#Features of 0 price houses
print((df[df["price"]==0].describe()).to_markdown())

#Mean Price of houses with features similar to 0 price house
df_other = df[(df["bedrooms"]<4) & (df["bedrooms"]>2) & (df["bathrooms"]<3) & (df["bedrooms"]>2) & (df["sqft_living"]>2500) & (df["sqft_living"]<3000)]
print(df_other["price"].mean())

#Replacing 0 price with mean price of similar features houses
df["price"].replace(to_replace=0, value=678000, inplace=True)

#Plot bedrooms vs price
plt.figure(figsize=(9,6))
ax = seaborn.barplot(x=df['bedrooms'], y=df['price'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Bedrooms VS Price', fontsize=14)
plt.show()

#Plot bedrooms vs sqft_living
plt.figure(figsize=(9,6))
ax = seaborn.barplot(x=df['bedrooms'], y=df['sqft_living'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Bedrooms VS Sqft_living', fontsize=14)
plt.show()

#Replacing 0 bedrooms with 8 as they have similar sqft_living
df["bedrooms"].replace(to_replace=0,value=8,inplace=True)

#Plot bathrooms vs price
plt.figure(figsize=(9,6))
ax = seaborn.barplot(x=df['bathrooms'], y=df['price'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Bathrooms VS Price', fontsize=14)
plt.show()

#Plot bathrooms vs sqft_living
plt.figure(figsize=(9,6))
ax = seaborn.barplot(x=df['bathrooms'], y=df['sqft_living'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Bathrooms VS Sqft_living', fontsize=14)
plt.show()

#Replacing 0 bathrooms with 4 as they have similar sqft_living
df["bathrooms"].replace(to_replace=0,value=4,inplace=True)

#Displot of Price
plt.figure(figsize=(9,6))
ax = seaborn.distplot(df['price'], kde=True)
ax.set_title('Distplot of Price', fontsize=14)
plt.show()

#Removing Outliers as Price distplot is negatively skewed
df['price'] = df['price'].replace([data['price'][numpy.abs(stats.zscore(data['price'])) > 3]],numpy.median(df['price']))

#Displot of Price
plt.figure(figsize=(9,6))
ax = seaborn.distplot(df['price'], kde=True)
ax.set_title('Distplot of Price', fontsize=14)
plt.show()

#Scatterplot of sqft_living
plt.figure(figsize=(9,6))
ax = seaborn.scatterplot(data=df, x="sqft_living", y="price")
ax.set_title('Sqft_living VS Price', fontsize=14)
plt.show()

#Removing Outliers of sqft_living
df['sqft_living'] = numpy.where((df.sqft_living >6000 ), 6000, df.sqft_living)

#Scatterplot of sqft_living
plt.figure(figsize=(9,6))
ax = seaborn.scatterplot(data=df, x="sqft_living", y="price")
ax.set_title('Sqft_living VS Price', fontsize=14)
plt.show()

#Scatterplot of sqft_lot
plt.figure(figsize=(9,6))
ax = seaborn.scatterplot(data=df, x="sqft_lot", y="price")
ax.set_title('Sqft_lot VS Price', fontsize=14)
plt.show()

#Removing Outliers of sqft_lot
df['sqft_lot'] = numpy.where((df.sqft_lot >250000 ), 250000, df.sqft_lot)

#Scatterplot of sqft_lot
plt.figure(figsize=(9,6))
ax = seaborn.scatterplot(data=df, x="sqft_lot", y="price")
ax.set_title('Sqft_lot VS Price', fontsize=14)
plt.show()

#Scatterplot of sqft_above
plt.figure(figsize=(9,6))
ax = seaborn.scatterplot(data=df, x="sqft_above", y="price")
ax.set_title('Sqft_above VS Price', fontsize=14)
plt.show()

#Removing Outliers of sqft_above
df['sqft_above'] = numpy.where((df.sqft_above >5000 ), 5000, df.sqft_above)

#Scatterplot of sqft_above
plt.figure(figsize=(9,6))
ax = seaborn.scatterplot(data=df, x="sqft_above", y="price")
ax.set_title('Sqft_above VS Price', fontsize=14)
plt.show()

#Scatterplot of sqft_basement
plt.figure(figsize=(9,6))
ax = seaborn.scatterplot(data=df, x="sqft_basement", y="price")
ax.set_title('Sqft_basement VS Price', fontsize=14)
plt.show()

#Removing Outliers of sqft_basement
df['sqft_basement'] = numpy.where((df.sqft_basement >2000 ), 2000, df.sqft_basement)

#Scatterplot of sqft_basement
plt.figure(figsize=(9,6))
ax = seaborn.scatterplot(data=df, x="sqft_basement", y="price")
ax.set_title('Sqft_basement VS Price', fontsize=14)
plt.show()

#Handling discrete values of bedrooms
print(df['bedrooms'].nunique())
bedrooms = df.groupby(['bedrooms']).price.agg([len, min, max])
print(bedrooms.to_markdown())

#To prevent disturbance in data 
df['bedrooms'] = numpy.where(df.bedrooms > 6, 6 ,df.bedrooms)

#Handling discrete values of bathrooms
print(df['bathrooms'].nunique())
bathrooms = df.groupby(['bathrooms']).price.agg([len, min, max])
print(bathrooms.to_markdown())

#To prevent disturbance in data 
df['bathrooms'] = numpy.where(df.bathrooms == 0.75, 1 ,df.bathrooms)
df['bathrooms'] = numpy.where(df.bathrooms == 1.25, 1 ,df.bathrooms)
df['bathrooms'] = numpy.where(df.bathrooms > 4.5, 4.5 ,df.bathrooms)

#Handling discrete values of floors
print(df['floors'].nunique())
floors = df.groupby(['floors']).price.agg([len, min, max])
print(floors.to_markdown())

#To prevent disturbance in data 
df['floors'] = numpy.where(df.floors ==3.5, 3 ,df.floors)

#Handling discrete values of waterfront
print(df['waterfront'].nunique())
waterfront = df.groupby(['waterfront']).price.agg([len, min, max])
print(waterfront.to_markdown())

#To prevent disturbance in data
#No disturbance

#Handling discrete values of condition
print(df['condition'].nunique())
condition = df.groupby(['condition']).price.agg([len, min, max])
print(condition.to_markdown())

#To prevent disturbance in data 
df['condition'] = numpy.where(df.condition ==1, 2 ,df.condition)

#Plot of Heatmap
plt.figure(figsize=(9,6))
ax = seaborn.heatmap(df.corr(),annot = True)
ax.set_title('CORRELATION MATRIX', fontsize=14)
plt.show()

#Make dummies of statezip to use it as a variable
df = pandas.get_dummies(df, columns=['statezip'], prefix = ['statezip'])

#Create Model
X = df.drop(columns=["price","date","street","city","country"])
y = df[["price"]]

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.2,random_state=50)
model = linear_model.LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print("explained_variance_score : ",metrics.explained_variance_score(y_test,y_pred))
print("max_error : ",metrics.max_error(y_test,y_pred))
print("mean_absolute_error : ",metrics.mean_absolute_error(y_test,y_pred))
print("mean_squared_error : ",metrics.mean_squared_error(y_test,y_pred))
print("mean_squared_log_error : ",metrics.mean_squared_log_error(y_test,y_pred))
print("mean_absolute_percentage_error : ",metrics.mean_absolute_percentage_error(y_test,y_pred))
print("median_absolute_error : ",metrics.median_absolute_error(y_test,y_pred))
print("r2_score : ",metrics.r2_score(y_test,y_pred))