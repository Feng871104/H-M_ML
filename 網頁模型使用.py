#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#可能要安裝的
#!pip install tensorflow
#!pip install xgboost
#pip install pandas

# In[5]:


#載入套件
import pandas as pd
import pickle

#載入資料
df=pd.read_csv('C:/Users/tcfst207/Downloads/0416_機器學習.csv')
#忘記有沒有打亂 打亂一下
from sklearn.utils import shuffle
df=shuffle(df)

df=df.drop('department_name',axis=1) #沒刪乾淨 之後確定
df=df.drop('FN',axis=1)
df=df.drop('club_member_status',axis=1)
df=df.drop('Active',axis=1)
df=df.drop('fashion_news_frequency',axis=1)

#建立一行資料  網頁輸入資料連接  F01、02就是特徵1 對應下方
f01=pd.Series('1')  
f02=pd.Series('Underwear') 
f03=pd.Series('Solid')
f04=pd.Series('Yellow') 
f05=pd.Series('Womens Lingerie') 
f06=pd.Series('Trousers') 
f07=pd.Series('2020') 
f08=pd.Series('Spring') 
f09=pd.Series('99元以下') 
f10=pd.Series('熱門') 


TestX=pd.DataFrame({'sales_channel_id':f01,
                   'product_group_name':f02,
                   'graphical_appearance_name':f03,
                   'colour_group_name':f04,
                   'section_name':f05,
                   'garment_group_name':f06,
                   'Year':f07,
                   'season':f08,
                   'pricelabel':f09,
                   'salelabel':f10,
                   })

#加入進大表最後一筆
df=df.append(TestX,ignore_index=True)

#改類型再做LABEL
df['sales_channel_id']=df['sales_channel_id'].astype('int32')
df['product_group_name']=df['product_group_name'].astype('str')
df['graphical_appearance_name']=df['graphical_appearance_name'].astype('str')
df['colour_group_name']=df['colour_group_name'].astype('str')
df['section_name']=df['section_name'].astype('str')
df['garment_group_name']=df['garment_group_name'].astype('str')

df['Year']=df['Year'].astype('str')
df['season']=df['season'].astype('str')
df['pricelabel']=df['pricelabel'].astype('str')
df['salelabel']=df['salelabel'].astype('str')

from sklearn.preprocessing import LabelEncoder #需要特徵都轉數值
labelencoder = LabelEncoder()

df['product_group_name']= labelencoder.fit_transform(df['product_group_name'])
df['graphical_appearance_name']= labelencoder.fit_transform(df['graphical_appearance_name'])
df['colour_group_name']= labelencoder.fit_transform(df['colour_group_name'])
df['section_name']= labelencoder.fit_transform(df['section_name'])
df['garment_group_name']= labelencoder.fit_transform(df['garment_group_name'])
#以下實驗
df['Year']= labelencoder.fit_transform(df['Year'])
df['pricelabel']= labelencoder.fit_transform(df['pricelabel'])
df['season']= labelencoder.fit_transform(df['season'])
df['salelabel']= labelencoder.fit_transform(df['salelabel'])

 #得到要得X_TEST
TestX=df.tail(1)

#要移除article_id才是特徵
TestX=TestX.drop('article_id',axis=1)

# # 使用保存模型預測
model = pickle.load(open('C:/Users/tcfst207/Downloads/MODEL_x2article.pkl','rb'))

prediction = model.predict(TestX)
print(prediction)


