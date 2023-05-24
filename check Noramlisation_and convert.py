# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:11:43 2023

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel(r"C:\Users\user\Downloads\archive\superstore.xls")
df    
     
# 1. (Visual Method) Create a histogram.
# If the histogram is roughly “bell-shaped”, 
# then the data is assumed to be normally distributed.

import math
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt
    
plt.hist(df.Profit, edgecolor='black', bins=10)
plt.hist(df.Sales, edgecolor='black', bins=10)
    
#******************************************************
# 2. (Visual Method) Create a Q-Q plot.
import math
import numpy as np
from scipy.stats import lognorm
import statsmodels.api as sm
import matplotlib.pyplot as plt

#create Q-Q plot with 45-degree line added to plot
fig = sm.qqplot(df.Sales, line='45')
fig = sm.qqplot(df.Profit, line='45')


#******************************************************
# 3. (Formal Statistical Test) Perform a Shapiro-Wilk Test.
# If the p-value of the test is greater than α = .05, 
# then the data is assumed to be normally distributed.

import math
import numpy as np
from scipy.stats import shapiro 
from scipy.stats import lognorm



#perform Shapiro-Wilk test for normality
shapiro(df.Profit)
shapiro(df.Sales)


#******************************************************

# 4. (Formal Statistical Test) Perform a Kolmogorov-Smirnov Test.
# If the p-value of the test is greater than α = .05, 
# then the data is assumed to be normally distributed.


import math
import numpy as np
from scipy.stats import kstest
from scipy.stats import lognorm
   

#perform Kolmogorov-Smirnov test for normality
kstest(df.Profit, 'norm')
kstest(df.Sales, 'norm')
 
#****************** Convert into normal distribution***************

#1. Log Transformation in Python  
import numpy as np
import matplotlib.pyplot as plt

#make this example reproducible
  
#create beta distributed random variable with 200 values
data=df.Sales
#create log-transformed data
data_log = np.log(df.Sales)

#define grid of plots
fig, axs = plt.subplots(nrows=1, ncols=2)    

#create histograms
axs[0].hist(data, edgecolor='black')
axs[1].hist(data_log, edgecolor='black')

#add title to each histogram
axs[0].set_title('Original Data')
axs[1].set_title('Log-Transformed Data')



#2 suare root Transformation 

#works on percentage 
data_log = np.sqrt(df.Sales)
plt.hist(data_log, edgecolor='black')

data_log = np.sqrt(df.Profit)
plt.hist(data_log, edgecolor='black')

data_log = np.sqrt(df.Discount)
plt.hist(data_log, edgecolor='black')

data_log = np.sqrt(df.Quantity)
plt.hist(data_log, edgecolor='black')
   
