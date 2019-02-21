# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 12:16:47 2019

@author: yang
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly as py
import plotly.graph_objs as go

from sklearn.cluster import KMeans

df=pd.read_csv('Mall_Customers.csv')
df.info()
df.describe()
df.isnull().sum()

print(len(df['CustomerID'].unique()))

# EDA
fig,ax=plt.subplots()
ax=sns.countplot(x='Gender',data=df)
ax.set_title('Gender distribution')
plt.show()
## Individual customers, where the majority pertains to women.

fig,ax=plt.subplots()
sns.distplot(df[df['Gender']=='Female']['Age'],color='pink')
sns.distplot(df[df['Gender']=='Male']['Age'],color='blue')
plt.show()

## The Age variable would be a good indicator of the targeted Age groups.
## It is quite interesting that there is a difference between the two genders.
## It appears that in both groups (i.e. Males & Females) there is a strong activity at the ages 25-35, while the data shows another frequent group from the female part at the age of around 45 years old. 
## In contrast, the group of men curve declines as the age reaches the maximum age of 70.

fig,ax=plt.subplots()
sns.distplot(df[df['Gender']=='Female']['Annual Income (k$)'],color='pink')
sns.distplot(df[df['Gender']=='Male']['Annual Income (k$)'],color='blue')
plt.xlabel('Income')
plt.show()


fig,ax=plt.subplots()
sns.distplot(df[df['Gender']=='Female']['Spending Score (1-100)'],color='pink')
sns.distplot(df[df['Gender']=='Male']['Spending Score (1-100)'],color='blue')
plt.xlabel('Spending Score')
plt.show()


corr = df[['Age','Annual Income (k$)','Spending Score (1-100)']].corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

## The income variable and the spending score variables are the ones that most interests us since we are going to keep this variables to perform our clustering.
## From heatmap, we can see all there's no relationship between variables.

'''
Approach - Machine Learning
Unsupervised Learning is a class of Machine Learning techniques to find the patterns in data. The data given to unsupervised algorithm are not labelled, which means only the input variables(X) are given with no corresponding output variables. In unsupervised learning, the algorithms are left to themselves to discover interesting structures in the data.
There are some analytics techniques that can help you with segmenting your customers. These are useful especially when you have a large number of customers and it’s hard to discover patterns in your customer data just by looking at transactions. The two most common ones are:

Clustering
Clustering is an exploration technique for datasets where relationships between different observations may be too hard to spot with the eye.

'''

# Kmeans cluster
## Determine the # of clusters K
## The main input for k-means clustering is the number of clusters. 
## This is derived using the concept of minimizing within cluster sum of square (WCSS).
## For our dataset, we will arrive at the optimum number of clusters using the elbow method:

X=df.iloc[:,[3,4]].values
wcss=[]

for i in range(1,11):
    km=KMeans(n_clusters=i,init='k-means++',max_iter=300, n_init=10,random_state=0)
    km.fit(X)
    wcss.append(km.inertia_)

plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()

## based on the elbow plot, we could choose 5 clusters.

km2=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
km2.fit(X)
y_means=km2.fit_predict(X)

plt.scatter(X[y_means==0,0],X[y_means==0,1],s=50,c='purple',label='Cluster1')
plt.scatter(X[y_means==1,0],X[y_means==1,1],s=50,c='blue',label='Cluster2')
plt.scatter(X[y_means==2,0],X[y_means==2,1],s=50,c='green',label='Cluster3')
plt.scatter(X[y_means==3,0],X[y_means==3,1],s=50,c='cyan',label='Cluster4')
plt.scatter(X[y_means==4,0],X[y_means==4,0],s=50,c='red',label='Cluster5')
plt.scatter(km2.cluster_centers_[:,0],km2.cluster_centers_[:,1],s=300,c='yellow',label='centroids')
plt.legend()
plt.show()

--------------------------------------------------------------
# cluster with boundary
labels1=km2.labels_
cent1=km2.cluster_centers_
h=0.02
xmin,xmax=X[:,0].min()-1,X[:,0].max()+1
ymin,ymax=X[:,1].min()-1,X[:,1].max()+1
xx,yy=np.meshgrid(np.arange(xmin,xmax,h),np.arange(ymin,ymax,h))
z=km2.predict(np.c_[xx.ravel(),yy.ravel()])

plt.figure(1,figsize=(15,7))
plt.clf()
z=z.reshape(xx.shape)
plt.imshow(z, interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = df , c = labels1 , 
            s = 200 )
plt.ylabel('Spending Score (1-100)') , plt.xlabel('Annual Income (k$)')
plt.show()
------------------------------------------------------------------
## model interpretation
## cluster 1-> earning high but spending less
## cluster 2-> average in terms of earning and spending
## cluster 3-> earning high and spending more
## cluster 4-> earning less but spending more
## cluster 5-> earning less, spending less

'''
Marketing strategies for the customer segments
Based on the 5 clusters, we could formulate marketing strategies relevant to each cluster:

1. A typical strategy would focus certain promotional efforts for the high value customers of Cluster 3.
2. Cluster 4 is a unique customer segment, where in spite of their relatively lower annual income, these customers tend to spend more on the site, indicating their loyalty. There could be some discounted pricing based promotional campaigns for this group so as to retain them.
3. For Cluster 5 where both the income and annual spend are low, further analysis could be needed to find the reasons for the lower spend and price-sensitive strategies could be introduced to increase the spend from this segment.
4. Customers in clusters 1 and 2 are not spending enough on the site in spite of a good annual income — further analysis of these segments could lead to insights on the satisfaction / dissatisfaction of these customers or lesser visibility of the e-commerce site to these customers. Strategies could be evolved accordingly.
'''

## cluster based on age, spending, income
X2=df.iloc[:,[2,3,4]].values
wcss=[]

for i in range(1,11):
    km3=KMeans(n_clusters=i,init='k-means++',max_iter=300, n_init=10,random_state=0)
    km3.fit(X2)
    wcss.append(km3.inertia_)

plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()

km3=KMeans(n_clusters=6,init='k-means++',max_iter=300,n_init=10,random_state=0)
km3.fit(X2)
labels2=km3.labels_
cen2=km3.cluster_centers_

## use 3-d plot segmentation
df['label2'] =  labels2
trace1 = go.Scatter3d(
    x= df['Age'],
    y= df['Spending Score (1-100)'],
    z= df['Annual Income (k$)'],
    mode='markers',
     marker=dict(
        color = df['label2'], 
        size= 20,
        line=dict(
            color= df['label2'],
            width= 12
        ),
        opacity=0.8
     )
)
data = [trace1]
layout = go.Layout(

    title= 'Clusters',
    scene = dict(
            xaxis = dict(title  = 'Age'),
            yaxis = dict(title  = 'Spending Score'),
            zaxis = dict(title  = 'Annual Income')
        )
)
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)
plt.draw()