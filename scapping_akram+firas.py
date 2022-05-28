#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import codecs


# In[2]:


from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import urllib
import urllib.request as uReq
site = "http://www.imdb.com/chart/tvmeter"
uClient = uReq.urlopen(site)
page_html = uClient.read()
uClient.close()

import bs4                           
from bs4 import BeautifulSoup 

import pickle                                 # important to save data 

soup = BeautifulSoup(page_html,"html.parser")

containers = soup.findAll("tbody",{"class":"lister-list"})
print(len(containers))    #1

print(type(containers[0]))

rows = containers[0].findAll("tr")

dataSet = {
	'Name_Of_Show':[],
	'Year_Of_Release':[],
	'Ranking_Change_Jump':[],
	'Users_count':[],
	'Rating':[]

}

# for saving the data
file = open("TV_Shows.csv","w+")

# Columns of the dataset
file.write('Name_Of_Show'+','+'Year_Of_Release'+','+'Ranking_Change_Jump'+','+'Users_count'+','+'Rating'+'\n')


# Iterating over column and saving results to the dataset
for i in range(len(rows)):
	
    
    # Add Names of the show to the dataset Dictionary
	dataSet['Name_Of_Show'].append(rows[i].find("td",{"class":"titleColumn"}).find("a").text.replace(',', ' '))


    # Add Release Year of the shows to the dataset Dictionary
	dataSet['Year_Of_Release'].append(int(rows[i].find("td",{"class":"titleColumn"}).find("span").text.split('(')[1].split(')')[0]))
	
     # Adding jump in change of popularity
	if (rows[i].find("td",{"class":"titleColumn"}).find("div").text.encode('cp1252').decode('utf-8').split('(')[1].split(')')[0] == 'no change'):
		dataSet['Ranking_Change_Jump'].append('0')
	else:
		dataSet['Ranking_Change_Jump'].append(rows[i].find("td",{"class":"titleColumn"}).find("div").text.split('(')[1].split(')')[0].split('\n')[2].replace(',',''))
	

    # Adds totalno of users_count 
	try :
		dataSet['Users_count'].append(str(int(rows[i].find("td",{"class":["ratingColumn","imdbRating"]}).find("strong")['title'].split()[3].replace(',',''))))
	except TypeError:
		dataSet['Users_count'].append("")
	
    
    
    #Adds Rating of the TV Shows
	dataSet['Rating'].append(rows[i].find("td",{"class":["ratingColumn","imdbRating"]}).text.split('\n')[1])


    # Write  All Scrapped details to the dataset
	file.write(str(dataSet['Name_Of_Show'][i])+","+str(dataSet['Year_Of_Release'][i])+","+str(dataSet['Ranking_Change_Jump'][i])+","+str(dataSet['Users_count'][i])+","+str(dataSet['Rating'][i])+"\n")

file.close()	
#print(dataSet)


# In[3]:


pd.read_csv('TV_Shows.csv', index_col=0  , encoding='ISO-8859-1')


# In[39]:


import pandas as pd
# Charger des données
df_products=pd.read_csv(r"C:\Users\ak.benhassine\TV_Shows.csv", index_col=0, encoding='ISO-8859-1')
# Remplacer date de recrutement par annee de recrutement
from datetime import datetime

def transform_annee_date(date):
    #date = datetime.strptime(date, '%d-%m-%y')
    annee_now=datetime.now().year
    #print(annee_now)
    #annee="01-01-"+str(date)
    #return annee_now-date
    return date

df_products['Users_count'] = df_products['Users_count'].replace(np.nan, 0)
df_products['Rating'] = df_products['Rating'].replace(np.nan, 0)
df_products['Year_Of_Release']=df_products.Year_Of_Release.apply(transform_annee_date)

def label_popularity (row):
    if (row['Rating'] < 7.3):
        return  "not popular"
    elif (7.3 <row['Rating']<7.5):
         return  "popular"
    else:
         return  "very popular"

df_products["popularity"] =df_products.apply (lambda row: label_popularity(row), axis=1)
df_products.head()


# In[47]:


from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
def calculer_score(client):
    return client['Users_count']/(client['Ranking_Change_Jump']+1)

class TransformationTVShows :
    def __init__(self, calcul_score=False):
        self.calcul_score=calcul_score
    def fit(self, df_products, y=None):
        X_=df_products.copy()
        self.ohe=OneHotEncoder()
        self.ohe.fit(X_.loc[:,['popularity']])
        return self
    def transform(self, X, y=None):
        X_=X.copy()
        
        # transformer niveau d'etudes selon OHE
        popularity_encoded=self.ohe.transform(X_.loc[:,['popularity']])
        df_popularity_encoded=pd.DataFrame(popularity_encoded.toarray(), 
                                                 columns='niveau_'+self.ohe.categories_[0],
                                                 index=X_.index)
        X_=pd.concat([X_,df_popularity_encoded], axis=1)
        X_.drop('popularity', axis=1, errors='ignore', inplace=True)

        if self.calcul_score==True:
            X_['score']=X_.apply(calculer_score, axis=1)
        return X_


# In[50]:


trsf=TransformationTVShows(calcul_score=True)
trsf.fit(df_products)
X_trsf=trsf.transform(df_products)
X_trsf


# In[51]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(X_trsf)
df_products_ss=ss.transform(X_trsf)


# In[17]:


df_products_ss


# In[52]:


from sklearn.decomposition import PCA
import numpy as np
df_products.replace([np.inf, -np.inf], np.nan, inplace=True)
pca=PCA(n_components=2)
pca.fit(df_products_ss)
df_products_pca=pca.transform(df_products_ss)


# In[54]:


from sklearn.cluster import KMeans
km=KMeans(n_clusters=2)
km.fit(df_products_pca)


# In[55]:


labels=km.predict(df_products_pca)
labels


# In[57]:


# NOus allons utiliser la techniques de classification SVM
# SVM : Support Vector Machine
# L'idée
from sklearn.svm import SVC
svm=SVC(kernel='linear')
svm.fit(df_products_pca,labels)


# In[62]:


import matplotlib.pyplot as plt
plt.plot(df_products_pca[labels==0,0],df_products_pca[labels==0,1],'bo', label='Cluster 0')
plt.plot(df_products_pca[labels==1,0],df_products_pca[labels==1,1],'yo', label='Cluster 1')

plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')

for nom,x,y in zip(df_products.index,df_products_pca[:,0],df_products_pca[:,1]):
        plt.annotate(nom,
            xy=(x, y), 
            xycoords='data',
            xytext=(x+2, y+2), 
            textcoords='offset points')
plt.legend()
plt.title("Clustering des Funs")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




