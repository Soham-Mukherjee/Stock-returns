#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import math
from scipy.stats import f_oneway
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats
import datetime
import warnings
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from statsmodels.graphics.regressionplots import influence_plot
from sklearn.metrics import r2_score,mean_squared_error
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
import statsmodels.stats.outliers_influence
from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier
import pydotplus as pdot
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle


# In[2]:


from nsepy import get_history
from datetime import datetime
import matplotlib.pyplot as plt
from nsepy import get_index_pe_history
from nsetools import Nse
nse=Nse()
print(nse)
from nsepy.symbols import get_symbol_list


# In[3]:


start_cp=datetime(2018,1,1)
end_cp=datetime(2018,12,31)


# In[13]:


stock_cp=get_history(symbol='HDFC',start=start_cp,end=end_cp)


# In[7]:


codes=nse.get_stock_codes().keys()


# In[8]:


codes=codes-set(['SYMBOL'])


# In[15]:


for i in codes:
    df_cp=get_history(symbol=i,start=start_cp,end=end_cp)
    stock_cp=stock_cp.append(df_cp,sort=False)


# In[22]:


cost_price=pd.DataFrame({'Symbol':'HDFC','Highest_price_of_the_year':stock_cp[stock_cp['Symbol']=='HDFC'].High.describe()['max']},index=[0])


# In[23]:


for i in codes:
    cost_df=pd.DataFrame({'Symbol':i,'Highest_price_of_the_year':stock_cp[stock_cp['Symbol']==i].High.describe()['max']},index=[1])
    cost_price=cost_price.append(cost_df,sort=False)


# In[25]:


stock_cp[stock_cp['Symbol']=='SBIN'].Symbol.value_counts()


# In[27]:


traded_codes=list(traded_stocks[traded_stocks['Days_traded']>245].Symbol)


# In[35]:


traded_stocks=pd.DataFrame(stock_cp.Symbol.value_counts()).reset_index()


# In[39]:


traded_codes=list(traded_stocks[traded_stocks['Symbol']>245]['index'])


# In[41]:


len(traded_codes)


# In[43]:


start_sp=datetime(2019,1,1)
end_sp=datetime(2019,12,31)


# In[46]:


stock_sp=get_history(symbol='HDFC',start=start_sp,end=end_sp)


# In[47]:


for i in traded_codes:
    df_sp=get_history(symbol=i,start=start_sp,end=end_sp)
    stock_sp=stock_sp.append(df_sp,sort=False)


# In[48]:


sell_price=pd.DataFrame({'Symbol':'HDFC','Lowest_25_pct_price_of_the_year':stock_sp[stock_sp['Symbol']=='HDFC'].Low.describe()['25%']},index=[0])


# In[49]:


for i in traded_codes:
    sell_df=pd.DataFrame({'Symbol':i,'Lowest_25_pct_price_of_the_year':stock_sp[stock_sp['Symbol']==i].Low.describe()['25%']},index=[1])
    sell_price=sell_price.append(sell_df,sort=False)


# In[50]:


return_2019_20=cost_price.merge(sell_price,on='Symbol',how='outer')


# In[51]:


return_2019_20=return_2019_20.dropna()


# In[52]:


return_2019_20['Return']=((return_2019_20['Lowest_25_pct_price_of_the_year']/return_2019_20['Highest_price_of_the_year'])-1)*100


# In[58]:


return_stat=pd.DataFrame(return_2019_20['Return'].describe()).reset_index()


# In[66]:


plot=pd.DataFrame({'Mean':return_2019_20['Return'].describe()['mean'],'Lowest':return_2019_20['Return'].describe()['min'],'25 PERCENTILE':return_2019_20['Return'].describe()['25%'],'50 PERCENTILE':return_2019_20['Return'].describe()['50%'],'75 PERCENTILE':return_2019_20['Return'].describe()['75%'],'Highest Return':return_2019_20['Return'].describe()['max']},index=[0])


# In[67]:


plot


# In[75]:


return_stat_plot=return_stat.drop(0,axis=0)


# In[83]:


return_stat_plot=return_stat_plot.drop(2,axis=0)


# In[84]:


sns.barplot(x='index',y='Return',data=return_stat_plot)


# In[85]:


return_2019_20[return_2019_20['Return']>0]


# In[86]:


high_codes=list(return_2019_20[return_2019_20['Return']>0].Symbol)


# In[88]:


start_high=datetime(2020,1,1)
end_high=datetime(2020,5,30)


# In[89]:


cost_price_high=pd.DataFrame({'Symbol':'HDFC','Highest_75_pct_price_of_the_year':stock_sp[stock_sp['Symbol']=='HDFC'].High.describe()['75%']},index=[0])


# In[90]:


for  i in high_codes:
    df_high=pd.DataFrame({'Symbol':i,'Highest_75_pct_price_of_the_year':stock_sp[stock_sp['Symbol']==i].High.describe()['75%']},index=[1])
    cost_price_high=cost_price_high.append(df_high,sort=False)


# In[92]:


cost_price_high=cost_price_high.drop(0,axis=0)


# In[93]:


cost_price_high


# In[96]:


stock_high=get_history(symbol='HDFC',start=start_high,end=end_high)


# In[97]:


for i in high_codes:
    stock_df_high=get_history(symbol=i,start=start_high,end=end_high)
    stock_high=stock_high.append(stock_df_high,sort=False)


# In[101]:


test_returns=pd.DataFrame({'Symbol':'HDFC','Lowest_price':stock_high[stock_high['Symbol']=='HDFC'].Low.describe()['min'],'Lowest_25_percentile_price':stock_high[stock_high['Symbol']=='HDFC'].Low.describe()['25%']},index=[0])


# In[104]:


for i in high_codes:
    test_returns_temp=pd.DataFrame({'Symbol':i,'Lowest_price':stock_high[stock_high['Symbol']==i].Low.describe()['min'],'Lowest_25_percentile_price':stock_high[stock_high['Symbol']==i].Low.describe()['25%']},index=[1])
    test_returns=test_returns.append(test_returns_temp,sort=False)


# In[106]:


test_returns=test_returns.drop(0,axis=0)


# In[109]:


test_return_high=test_returns.merge(cost_price_high,on='Symbol',how='outer')


# In[111]:


test_return_high['test_returns_lowest']=((test_return_high['Lowest_price']/test_return_high['Highest_75_pct_price_of_the_year'])-1)*100


# In[119]:


test_return_high['test_return_at_25_percentile']=((test_return_high['Lowest_25_percentile_price']/test_return_high['Highest_75_pct_price_of_the_year'])-1)*100


# In[121]:


test_return_high=test_return_high.drop('return_at_25_percentile',axis=1)


# In[122]:


test_return_high


# In[117]:


test_codes_high=list(test_return_high[test_return_high['return_at_25_percentile']>0].Symbol)


# In[118]:


test_codes_high


# In[123]:


returns=return_2019_20.merge(test_return_high,on='Symbol',how='outer')


# In[125]:


returns=returns.dropna()


# In[126]:


returns


# In[129]:


returns.columns


# In[130]:


cnames=['Symbol','Return','test_returns_lowest','test_return_at_25_percentile']


# In[131]:


return_plot=returns[cnames]


# In[132]:


return_plot


# In[142]:


sns.barplot(y='Symbol',x='Return',data=return_plot)


# In[143]:


sns.barplot(y='Symbol',x='test_returns_lowest',data=return_plot)


# In[144]:


sns.barplot(y='Symbol',x='test_return_at_25_percentile',data=return_plot)


# In[ ]:




