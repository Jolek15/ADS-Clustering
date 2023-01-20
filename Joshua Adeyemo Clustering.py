#!/usr/bin/env python
# coding: utf-8

#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit
from scipy.stats import norm

### CLUSTERING
"""
This code uses KMeans clustering to cluster countries based on their GDP growth from 1960 to 2020. 
It imports the necessary libraries, defines a function to call a file, reads the GDP file, 
renames the columns, creates a new dataframe with the relevant columns, and creates a scatter plot of the data. 
The data is then scaled and the KMeans clustering model is used to create 3 clusters. 
The cluster centers are plotted, it finds the optimal number of clusters using the elbow method, the sum of squared error is plotted, 
and new dataframes are created for each cluster.

"""

#read the data 
df = pd.read_excel('https://api.worldbank.org/v2/en/indicator/NY.GDP.PCAP.CD?downloadformat=excel', skiprows=3)

def call_file(indicator_used, sheet_name):
    
    """    
    Read in an excel file from a provided URL and sheet name.        
    
    Parameters:    
    indicator_used (str): The URL of the excel file.    
    sheet_name (str): The name of the sheet.        
    
    Returns:    
    df (dataframe): The dataframe read in from the excel file.    
    
    
    """
    
    df = pd.read_excel(indicator_used, sheet_name=sheet_name, skiprows=3)
    return df

#read in the GDP and CO2 emission files

#define indicators
GDP = ('https://api.worldbank.org/v2/en/indicator/NY.GDP.PCAP.CD?downloadformat=excel')
CO2_emission = ('https://api.worldbank.org/v2/en/indicator/EN.ATM.CO2E.KT?downloadformat=excel')

sheet_name = ('Data')

#call files
gdp_file = call_file(GDP, sheet_name)
co2_file = call_file(CO2_emission, sheet_name)
print(gdp_file)

#cleaning the data
#extract GDP growth from 1990 to 2020 
df1 = gdp_file[gdp_file.columns[:65]]
#years we'll focus on
year1 = '1960'
year2 = '2020'

#extract the required data for the clustering
df_clust = df.loc[df.index, ['Country Name', year1, year2]]

#drop any row with missing values as its going to affect our clustering
df_clust = df_clust.dropna()

print(df_clust)

#create a scatter plot of the data
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,10))
plt.scatter(df_clust['1960'], df_clust['2020'])
plt.xlabel('1960')
plt.ylabel('2020')
plt.title('Scatter plot of 1960 and 2020 GDP')
plt.legend()
plt.savefig('scatter1.png')
plt.show()

#find the optimal number of clusters using the elbow method
k_rng = range(1, 11)
sse = []
for i in k_rng:
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_clust[['1960', '2020']])
    sse.append(kmeans.inertia_)

print(sse)

#plot the elbow graph
plt.figure(figsize=(10,10))
plt.plot(k_rng, sse)
plt.xlabel('No of Clusters')
plt.ylabel('Sum of squared error')
plt.title('The Elbow Method')
plt.legend()
plt.savefig('clusters.png')
plt.show()
    
X = df_clust[["1960", "2020"]].copy()
print(X)

#set the number of clusters
km = KMeans(n_clusters=3, random_state=0)
print(km)

#fit the data and predict clusters
y_predicted = km.fit_predict(X)
print(y_predicted)

#add the cluster column to the dataframe
X['cluster'] = y_predicted

X.head()

#create separate dataframes for each cluster
df1 = X[X.cluster==0]
df2 = X[X.cluster==1]
df3 = X[X.cluster==2]

#create a scatter plot for each cluster
plt.figure(figsize=(10,10))
plt.scatter(df1['1960'], df1['2020'], color='green', label = 'cluster 0')
plt.scatter(df2['1960'], df2['2020'], color='black', label = 'cluster 1')
plt.scatter(df3['1960'], df3['2020'], color='red', label = 'cluster 2')

plt.xlabel('1960')
plt.ylabel('2020')
plt.title('Scatter plot showing cluster membership')
plt.legend()
plt.savefig('scatter2.png')
plt.show()

#list of columns to be scaled
cols_to_scale = ['1960', '2020']

#create scaler instance
scaler = MinMaxScaler()

#scale the columns in the dataset
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

#assign scaled data to new_X
new_X = X[cols_to_scale]

print(new_X)

#create kmeans with 3 clusters
km = KMeans(n_clusters=3, random_state=0)

#fit the clustered data
y_predicted = km.fit_predict(new_X)
print(y_predicted)

#add the cluster column to the original data
X['cluster'] = y_predicted

X.head(10)

#the scaled cluster centers
km.cluster_centers_

#dataframes for the clusters
df1 = X[X.cluster==0]
df2 = X[X.cluster==1]
df3 = X[X.cluster==2]

#visualising the clusters and centroids
plt.figure(figsize=(10,10))
plt.scatter(df1['1960'], df1['2020'], color='green', label = 'cluster 0')
plt.scatter(df2['1960'], df2['2020'], color='black', label = 'cluster 1')
plt.scatter(df3['1960'], df3['2020'], color='red', label = 'cluster 2')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='orange',marker='X',label='centroid')
plt.title('Scatter plot showing cluster membership and cluster centres')
plt.xlabel('1960')
plt.ylabel('2020')
plt.legend()
plt.savefig('scatter3.png')
plt.show()

#assign the labels of the clusters to the cluster column of the original dataframe
df_clust['cluster'] = km.labels_

df_clust.head(20)

#create new dataframes for each cluster
first_cluster = df_clust.loc[df_clust['cluster'] == 0]

second_cluster = df_clust.loc[df_clust['cluster'] == 1]

third_cluster = df_clust.loc[df_clust['cluster'] == 2]

print(first_cluster)
print(second_cluster)
print(third_cluster)

#plot a correlation heat map to show the correlation coefficient between the years
import seaborn as sns
plt.figure(figsize=(8,5))
sns.heatmap(new_X.corr(),annot=True)
plt.title('Correlation heatmap of GDP from 1960 to 2020')
plt.savefig('corr.png')
plt.legend()
plt.show()

#plot the bar chart
plt.figure(figsize=(10,10))
plt.bar([1,2,3,4,5,6], [first_cluster['1960'].mean(), first_cluster['2020'].mean(), second_cluster['1960'].mean(), second_cluster['2020'].mean(), third_cluster['1960'].mean(), third_cluster['2020'].mean()], tick_label = ['1960 cluster0', '2020 cluster0', '1960 cluster1',  '2020 cluster1', '1960 cluster2', '2020 cluster2'])
plt.xlabel('Clusters')
plt.ylabel('Mean GDP')
plt.xticks(rotation=45)
plt.title('Comparison of developments in clusters using their mean')
plt.savefig('clusters_bar.png')
plt.legend()
plt.show()

#using united states as a case study we extract USA data
usa_df = gdp_file[gdp_file['Country Name'] == 'United States']
print(usa_df)

#cleaning the data and extracing what we need
m = usa_df[['1960','1980','2000','2010','2021']].copy()
#Create DataFrame
df_new = m.T
df_new.index = df_new.index.set_names('Year')
df_new.columns = ['GDP']
df_new.reset_index()
print(df_new)

#line plot showing U.S.A GDP growth over the years
plt.figure(figsize=(10,10))
plt.plot(df_new.index, df_new['GDP'])
plt.xlabel('Year')
plt.ylabel('GDP')
plt.title('USA GDP growth analysis from 1960 to 2021')
plt.savefig('chart.png')
plt.legend()
plt.show()


### FITTING

'''
This code imports CO2 emissions data from the world bank api and uses it to create a logistic curve fit and generate predictions 
for the next 20 and 30 years

'''

#calling co2 data
print(co2_file)
#drop unnecessary columns
data = co2_file.drop(co2_file.loc[:, 'Country Code':'1989'].columns,axis = 1)
data = data.drop(data.loc[:, '2020':'2021'].columns,axis = 1)
#drop NaN Values
data = data.dropna()
print(data)

#selecting data for United States 
country1_df = data[data['Country Name'] == 'United States']
print(country1_df)

#transpose the data
df_usa = country1_df.T
headers = df_usa.iloc[0]
#Create DataFrame
df_usa = pd.DataFrame(df_usa.values[1:], columns=headers)
#Insert year column
df_usa.insert(0, "Year", ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019'], True)

print(df_usa)

#Convert to numeric
df_fitting = df_usa[['Year', 'United States']].apply(pd.to_numeric, 
                                               errors='coerce')
#Select only numerical values
data1 = df_fitting.dropna().values

#Fit curve for data
x_years = data1[:,0] #Selecting years from 1990 - 2019
y_data = data1[:,1] #Selecting CO2 emissions


# Define logistic function
def logistic_func(x, k, x0, L):
    """
    This function calculates the logistic function of a given set of parameters. 

    Parameters: 
    x (float): The independent variable. 
    k (float): The growth rate parameter. 
    x0 (float): The x-value of the sigmoid's midpoint. 
    L (float): The curve's maximum value. 

    Returns: 
    float: The calculated logistic function. 

    """
    return L / (1 + np.exp(-k*(x-x0)))

#Fit curve
popt, pcov = curve_fit(logistic_func, x_years, y_data, p0=[2, 2000, 5775810])
print(popt)
print(pcov)

# Calculate the predicted values for 2020
x20 = 2000 + 20
y20 = logistic_func(x20, *popt)

x = np.linspace(1990, 2020, 30)
y = logistic_func(x, *popt)

#a plot to show the predicted co2 emission for 2020
plt.figure(figsize=(10,10))
plt.plot(x, y, 'r-', label='Fitting curve')

plt.scatter(x_years, y_data)
plt.scatter(x20, y20, label='20 years prediction', color='green')

plt.xlabel('Year', fontsize=20)
plt.ylabel('CO2 emission (kt)', fontsize=20)
plt.title('CO2 Emissions in U.S.A. by 2020', fontsize=20)
plt.legend(fontsize=15)
plt.savefig('pred.png')
plt.show()

#a forecast of the co2 emission rate by 2030
year = np.arange(1990, 2031)
print(year)
forecast = logistic_func(year, *popt)
print(forecast)

# Estimate lower and upper limits of the confidence range for 30 years forecast

def err_ranges(x, y, popt):
    """
    This function calculates the upper and lower limits of the 95% confidence level of a forecast given the x and y data points,
    and the optimal parameters obtained from the fit. It first calculates the residuals, 
    then creates a z-score for the 95% confidence level, and finally calculates the upper and lower limits of the forecast. 
    """
    # define 95% confidence level
    confidence = 0.95
    residuals = y - y_data

    # calculate z-score
    z = norm.ppf((1 + confidence) / 2)

    # calculate upper and lower limits
    lower_limit = forecast - z * np.std(residuals)
    upper_limit = forecast + z * np.std(residuals)

    return upper_limit, lower_limit

upper_limit, lower_limit = err_ranges(x, y, popt)


# print upper and lower limits
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)


#prediction of CO2 emission by 2030
x30 = 2000 + 30
y30 = logistic_func(x30, *popt)
print("The CO2 emission of the United States by 2030 is predicted to be", y30, "kt")

#a plot showing the best fitting function and the confidence range for 30 years prediction
plt.figure(figsize=(10,10))
plt.plot(x_years, y_data, 'o')
plt.plot(year, forecast, 'r-', label='Fitting curve', alpha=0.3)
plt.plot(year, upper_limit, 'g--', label='Upper limit')
plt.plot(year, lower_limit, 'b--', label='Lower limit')

plt.scatter(x_years, y_data)
plt.scatter(x20, y20, label='20 years prediction', color='purple', alpha=0.7)
plt.scatter(x30, y30, label='30 years prediction', color='black', alpha=0.7)

plt.fill_between(year, lower_limit, upper_limit, color="yellow", alpha=0.1)
plt.xlabel('Year', fontsize=20)
plt.ylabel('CO2 emission (kt)', fontsize=20)
plt.title('U.S.A CO2 Emissions prediction for 30 years', fontsize=20)
plt.legend(fontsize=15)
plt.savefig('scatter.png')
plt.show()

