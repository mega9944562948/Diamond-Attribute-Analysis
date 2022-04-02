#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries to use in EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import scipy.stats as stats
import statsmodels.api as sm
#libraries for transformation
from scipy.special import inv_boxcox
#libraries for modeling and validation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor


# In[2]:


#set pandas display options for better visualization of the data when called upon
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# In[3]:


#set the random state seed fro future use
seed = 88


# # Data Aquisition: 

# In[4]:


# Read in the data data set from file 
df1 = pd.read_csv('/Users/benjaminmcdaniel/Desktop/D214/d214task2/diamond_master.csv', sep=',')
df1.head()


# # Data Exploration:

# ## Inspect the data set for composition, shape, and completeness. 

# In[5]:


#Inspect the dateframe for shape, data types, null values
df1.info()


# In[6]:


# Inspect for duplicated rows in the dataframe
print(df1.duplicated().unique())


# In[7]:


#Inspect for missing values in the data set after exporting from postgresql
df1.isna().sum()


# ## Explore each feature in the data set individually

# ### Define a function to return exploratory information about continuous variables

# In[8]:


#describe the distribution of a continuous variable center, spread, skew, modality, outliers, normality. 
def univariate_continuous(cont_col):
    """takes a pandas series or column from a data frame that contains a continuous variable and returns:
    Mean, Median, Mode, Variance, Standard Deviation, Minimum value, Maximum value, Range, Interquartile Range,
    Skewness, Kurtosis, Outlier Boundries, Anderson Darling normality test results, Boxplot, Histogram. 
    Unless stated otherwise values are rounded to 3 decimal places"""
    
    #function for Anderson Darling normality test results
    def anderson_darling(cont_col=cont_col):
        """takes a continuous pandas series and returns the statistic, critical value, significance level,
        and indication of normality"""
        normality = stats.anderson(cont_col)
        display('Anderson Darling Test for Normality Results Below: ')
        display('Statistic: {}'.format(round(normality.statistic, 3)))
        for i in range(len(normality.critical_values)):
            significance, critical_value = normality.significance_level[i],normality.critical_values[i]
            normality.critical_values[i]
            if normality.statistic < critical_value:
                display('Indicates Gaussian Distribution: Critical Value {0} at Significance Level {1}'.format(
                critical_value, significance))
            else:
                display('Indicates Not Gaussian Distribution: Critical Value {0} at Significance Level {1}'.format(
                critical_value, significance))
        display('End of Anderson Darling Test for Normality Results')
    #define variables for use in output      
    iqr = round(stats.iqr(cont_col),3)
    sigma = round(np.std(cont_col),3)
    var = round(statistics.variance(cont_col, np.mean(cont_col)),3)
    range_cont_col = cont_col.max() - cont_col.min()
    mean = round(np.mean(cont_col),3)
    minimum = cont_col.min()
    maximum = cont_col.max()
    median = cont_col.median()
    skewness = round(cont_col.skew(),3)
    kurtosis = round(stats.kurtosis(cont_col),3)
    first_quartile = np.quantile(cont_col, 0.25)
    second_quartile = np.quantile(cont_col, 0.5)
    third_quartile = np.quantile(cont_col, 0.75)
    
    #output strings for statistics
    display('The Mean for {0} is: {1}'.format(cont_col.name.title(), mean))
    display('The Median for {0} is: {1}'.format(cont_col.name.title(), median))
    display('The Mode of {0} is: {1}'.format(cont_col.name.title(), stats.mode(cont_col)))
    display('The Variance for {0} is: {1}'.format(cont_col.name.title(), var))
    display('The Standard Deviation for {0} is: {1}'.format(cont_col.name.title(), sigma))
    display('The Minimum for {0} is: {1}'.format(cont_col.name.title(), minimum))
    display('The Maximum for {0} is: {1}'.format(cont_col.name.title(), maximum))
    display('The Range for {0} is: {1}'.format(cont_col.name.title(), range_cont_col))
    display('The First Quartile for {0} is: {1}'.format(cont_col.name.title(), first_quartile))
    display('The Second Quartile for {0} is: {1}'.format(cont_col.name.title(), second_quartile))
    display('The Third Quartile for {0} is: {1}'.format(cont_col.name.title(), third_quartile))
    display('The IQR for {0} is: {1}'.format(cont_col.name.title(), iqr))
    
    display('The Skewness of {0} is: {1}'.format(cont_col.name.title(), skewness))
    display('The Kurtosis of {0} is: {1}'.format(cont_col.name.title(), kurtosis))
    
    display('Outliers have values above {}, and below {}'.format(mean + 3*sigma, mean - 3*sigma))
    
    #run Anderson Darling test
    anderson_darling(cont_col)
    
    #Visualize the column of values as a boxplot
    plt.style.use('fivethirtyeight')
    plt.boxplot(cont_col)
    plt.xlabel('{}'.format(cont_col.name.title()))
    plt.ylabel('Count')
    plt.title('{}'.format(cont_col.name.title()))
    plt.show()
    plt.clf()
    
    #Visualize the column of values as a histogram
    plt.style.use('fivethirtyeight')
    sns.displot(cont_col,color='blue', label="Univariate Distribution")
    plt.xlabel('Values')
    plt.ylabel('Count')
    plt.title('{}'.format(cont_col.name.title()))
    plt.legend()
    plt.show()
    plt.clf()
    
    #Visualize the distribution as a comparison to a normal distribution with a QQ-plot
    fig, ax = plt.subplots(figsize = (4,4))
    stats.probplot(cont_col, plot=ax)


# In[9]:


univariate_continuous(df1['price'])


# In[10]:


univariate_continuous(df1['znorm_price'])


# In[11]:


univariate_continuous(df1['carat'])


# In[12]:


univariate_continuous(df1['price_per_ct'])


# ### Remove outliers found in the data set in relation to the response variable

# In[13]:


#define a function to remove rows of a data set that contain outliers of the response variable
def clean_response_outliers(df, res_col):
    """Takes a dataframe and designated response variable column and returns a printed count of outlier rows, 
    and dataframe reduced by the number of outliers found"""
    drop_list = []
    sigma = np.std(res_col)
    mean = np.mean(res_col)
    high_outlier = mean + (3*sigma)
    low_outlier = mean - (3*sigma)
    for x in res_col:
        if x > high_outlier:
            drop_list.append(x)
        elif x < low_outlier:
            drop_list.append(x)
    print('{} observations will be dropped from this dataframe'.format(len(drop_list)))
    
    df_dropped = df[~df['{}'.format(res_col.name)].isin(drop_list)]
    return df_dropped
    


# In[14]:


df1 = clean_response_outliers(df1, res_col = df1['znorm_price'])


# In[15]:


univariate_continuous(df1['price'])


# ### Define a function to return exploratory information about categorical variables

# In[16]:


#describe the distribution of a categorical variable 
def univariate_categorical(cat_col):
    """takes a categorical pandas series or categorical column from a dataframe and returns the unique 
    label values included, frequency of each labels occurance, a table of proportions, the mode of the
    data set, and a barchart of the distribution of the labels"""
    
    #ignore depreciation warnings 
    import warnings
    warnings.filterwarnings('ignore')
    
    # Assign variable values for output
    label_values = cat_col.unique() #unique values assigned to the observations
    label_freq = cat_col.value_counts() #frequency of label occurance
    prop_table = cat_col.value_counts()/len(cat_col) #table of proportions for categorical variable
    mode = label_freq.index[0]
    
    display('The values of this variable include:')
    display(list(label_values))
    display('The Frequency of occurance for each label is: ')
    display(label_freq)
    display('The table of proportions for this variable is below:')
    display(prop_table)
    display('The most frequent value of this variable is: {}'.format(mode))
    

    #construct a bar chart to represent the data in the variable
    sns.countplot(cat_col)
    plt.xticks(rotation = 30)
    plt.title('Frequency Distribution of Labels')
    plt.show()
    plt.clf()


# In[17]:


univariate_categorical(df1['shape'])


# In[18]:


univariate_categorical(df1['cut'])


# In[19]:


univariate_categorical(df1['color'])


# In[20]:


univariate_categorical(df1['clarity'])


# In[21]:


univariate_categorical(df1['report'])


# In[22]:


univariate_categorical(df1['dia_type'])


# ## Explore bivariate relationships between variables in the data set

# ### Relationship between two categorical variables

# In[23]:


#define function to explore bivariate relationships between categorical variables
def bivar_categorical(cat_col1, cat_col2):
    """takes two categorical pandas series or columns from a dataframe. Returns a contingency table, 
    table of proportions, marginal proportions of each categorical column, an expected values table,
    observed values table, Chi-square statistic and associated p-value, a stated determination of 
    the Chi-square results in regards to the null-hypothesis. A stacked bar chart of the combined variables
    distribution"""
    #ignore depreciation warnings 
    import warnings
    warnings.filterwarnings('ignore')
    
    #summarize the two variables at the same time using a contingency table
    freq_table = pd.crosstab(cat_col1, cat_col2)
    print('Contingency Table')
    print(freq_table)
    print('')
    #develop a table of proportions to better describe the variables as percentages
    prop_table = freq_table/len(cat_col1)
    print('Table of Proportions')
    print(prop_table)
    
    
    print('')
    #determine marginal proportions by column and by row
    marginal_prop_cat2= prop_table.sum(axis=0)
    print('The marginal proportion of {} by type is: '.format(cat_col2.name.title()), marginal_prop_cat2)
    
    print('')
    marginal_prop_cat1=prop_table.sum(axis=1)
    print('The marginal proportion of {} by type is: '.format(cat_col1.name.title()), marginal_prop_cat1)
    
    print('')
    #calculate expected contingency
    from scipy.stats import chi2_contingency
    chi2, pval, dof, expected = chi2_contingency(freq_table)
    print('Expected Values')
    print(np.round(expected))
    print('')
    print('Observed values')
    print(freq_table)
    
    print('')
    
    #perform Chi-Square test for association 
    print('H0: The two variables are independent.')
    print('H1: The two variables are not independent.')    
    print('')
    chi2, pval, dof, expected = chi2_contingency(freq_table)
    print("The Chi-Square statistic is: {}".format(chi2))
    print('')
    print("The P_value is: {}".format(pval))
    print('')
    if pval > 0.05:
        print('Determination:')
        print('Fail to reject the null hypothesis (H0) that the two variables are independent')
    else:
        print('Determination:')
        print('Reject the null hypothesis (H0) that the two variables are independent')
    
    #stacked bar chart of the two variables
    plt.style.use('fivethirtyeight')
    print('')
    pd.crosstab(cat_col1, cat_col2).plot(kind='bar', stacked=True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    plt.clf();
    


# In[24]:


bivar_categorical(df1['shape'], df1['cut'])


# In[25]:


bivar_categorical(df1['shape'], df1['color'])


# In[26]:


bivar_categorical(df1['shape'], df1['clarity'])


# In[27]:


bivar_categorical(df1['shape'], df1['report'])


# In[28]:


bivar_categorical(df1['shape'], df1['dia_type'])


# In[29]:


bivar_categorical(df1['cut'], df1['color'])


# In[30]:


bivar_categorical(df1['cut'], df1['clarity'])


# In[31]:


bivar_categorical(df1['cut'], df1['report'])


# In[32]:


bivar_categorical(df1['cut'], df1['dia_type'])


# In[33]:


bivar_categorical(df1['color'], df1['clarity'])


# In[34]:


bivar_categorical(df1['color'], df1['report'])


# In[35]:


bivar_categorical(df1['color'], df1['dia_type'])


# In[36]:


bivar_categorical(df1['clarity'], df1['report'])


# In[37]:


bivar_categorical(df1['clarity'], df1['dia_type'])


# In[38]:


bivar_categorical(df1['report'], df1['dia_type'])


# In[ ]:





# In[ ]:





# ### Relationship between two continuous variables

# In[39]:


#define a function to explore the relationsip between two continuous variables
def bivar_continuous(cont_col1,cont_col2):
    """takes two continuous pandas series or columns from a dataframe and returns pearson-r coefficient, 
    and associated p-value, spearman rank order coefficient and associated p-value, simple linear regression
    summary parameters and equation, scatterplot of the distribution of the two continuous columns"""
    #determine correlation between two continuous variables
    pearson, p = stats.pearsonr(cont_col1,cont_col2) #better for normal distributions
    print('Pearson correlation coefficient: {}'.format(pearson))
   
    print('Preason correlation p-value: {}'.format(p))
    print('')
    
    spearman, sp = stats.spearmanr(cont_col1, cont_col2) #better for non-normal distributions 
    print('The spearman rank order correlation coefficient is: {}'.format(spearman))
   
    print('The spearman rank order correlation p-value is: {}'.format(sp))
    print('')
    
    print('Covariance Matrix for {0} and {1}'.format(cont_col1.name.title(),cont_col2.name.title()))
    #determine the covariance between cont_col1 and cont_col2
    covar_matrix = np.cov(cont_col1, cont_col2)
    print(covar_matrix)
    print('')
    
    #simple linear regression for bivariate analysis:
    y = cont_col1
    x = cont_col2
    x = sm.add_constant(x)
    model = sm.OLS(y,x)
    results = model.fit()
    print('Model summary of simple linear regression:')
    print('')
    print(results.summary())
    print('')
    
    print("Parameters:\n", results.params)
    print('')
    
    print('The fitted regression equation is:')
    print('Price = {} + {} * {} value'.format(results.params[0], results.params[1], cont_col2.name.title()))
    print('')
    
    #generate a scatter plot of the two continuous variables
    plt.style.use('fivethirtyeight')
    #generate a scatterplot to visualize the relationship between these two variables
    plt.scatter(x=cont_col2, y=cont_col1)
    plt.xlabel('{}'.format(cont_col2.name.title()))
    plt.ylabel('{}'.format(cont_col1.name.title()))
    
    plt.show()
    plt.clf()
    
    

    
    
    


# In[40]:


bivar_continuous(df1['price'],df1['carat'])


# In[ ]:





# In[41]:


# define a function to explore the relationship between a continuous variable and a categorical variable
def bivar_cont_cat(cont_col, cat_col, df):
    """Takes a pandas series object of continuous values and a pandas series object of categorical values
    and returns a boxplot of continuous values by categorical value, """
    plt.style.use('fivethirtyeight')
    df.boxplot(by='{}'.format(cat_col.name), column='{}'.format(cont_col.name))
    plt.xticks(rotation=35)
    plt.title('')
    plt.ylabel('{}'.format(cont_col.name.title()))
    plt.xlabel('{}'.format(cat_col.name.title()))
    
    
    
               
    


# In[42]:


bivar_cont_cat(df1['price'], df1['shape'], df1)


# In[43]:


bivar_cont_cat(df1['price'], df1['cut'], df1)


# In[44]:


bivar_cont_cat(df1['price'], df1['color'], df1)


# In[45]:


bivar_cont_cat(df1['price'], df1['clarity'], df1)


# In[46]:


bivar_cont_cat(df1['price'], df1['report'], df1)


# In[47]:


bivar_cont_cat(df1['price'], df1['dia_type'], df1)


# ### Select variables for regression analysis

# In[48]:


def correlation_plt(df,figsize):
    """Takes a dataframe and figsize tuple as input and returns a correlation martix and correlation heatmap
    of correlation coefficients pandas and seaborn required for proper function """
    corr_mat = df.corr()
    display(corr_mat)
    plt.figure(figsize = figsize)
    sns.heatmap(corr_mat, xticklabels = corr_mat.columns, yticklabels = corr_mat.columns,
            vmin = -1, center = 0, vmax = 1, cmap='PuOr', annot=True, fmt='.1g', square = True)


# # Data Transformation

# In[49]:


#reduce the data set to informative predictors of price that are not highly correlated with eachother
df1 = df1[['price','carat', 'shape', 'cut', 'clarity','color','report','dia_type']]
df1.head()


# ### Encode categorical variables dropping one column to reduce collinearity

# In[50]:


# Encode categorical variables dropping the first column
df1 = pd.get_dummies(df1, columns = ['shape', 'cut','clarity','color','report','dia_type'], drop_first=True)
df1.head()


# In[51]:


#inspect the correlation matrix and plot again
correlation_plt(df1,(50,25))


# In[52]:


#the report columns add multicollinearity dropping to reduce their effect on the outcome
df1=df1.drop(columns=['report_GIA','report_IGI','report_HRD'])


# In[53]:


#re-inspect correlation 
correlation_plt(df1,(50,25))


# In[54]:


#log transform the response variable to fit a more normal distribution
#define a function that will perform power transformations using box-cox
def transform_col(df, cont_col,lambda_val):
    """ Requires scipy stats as stats. Takes a dataframe a continuous column, and lambda value as inputs, performs a box-cox
    power transformation based on the lambda value provided and returns a new column of transformed data.
    lambda values: -1 = reciprocal, -0.5 = reciprocal square root, 0 = log transform, 0.5 = square root,
    1 = no transform"""
    if lambda_val == -1:
        x=df['{}_reciprocal'.format(cont_col.name)] = stats.boxcox(cont_col,lambda_val)
        print('Reciprocal Transformation Complete')
    elif lambda_val == -0.5:
        x=df['{}_recip_sqrt'.format(cont_col.name)] = stats.boxcox(cont_col,lambda_val)
        print('Reciprocal Square Root Transformation Complete')
    elif lambda_val == 0:
        x=df['{}_nat_log'.format(cont_col.name)] = stats.boxcox(cont_col,lambda_val)
        print('Natural Log Transformation Complete')
    elif lambda_val == 0.5:
        x=df['{}_sqrt'.format(cont_col.name)] = stats.boxcox(cont_col,lambda_val)
        print('Square Root Transformation Complete')
    elif lambda_val == 1:
        x=df['{}_no_transformed'.format(cont_col.name)] = stats.boxcox(cont_col,lambda_val)
        print('Non Transformation Complete')
  

    #Visualize the column of values as a histogram
    plt.style.use('fivethirtyeight')
    sns.displot(x,color='blue', label="Univariate Distribution")
    plt.xlabel('Values')
    plt.ylabel('Count')
    plt.title('Distribution of Transformed {}'.format(cont_col.name.title()))
    plt.legend()
    plt.show()
    plt.clf()
    return x
    


# In[55]:


#test
transform_col(df1, df1['price'],0)
df1.head()


# In[56]:


univariate_continuous(df1['price_nat_log'])


# In[57]:


correlation_plt(df1,(50,25))


# ## Begin T-Test for difference of means transformed price column

# ### Natural vs synthetic populations
# ### H0 There is no statistically significant difference of means between natural and synthetic populations
# ### H1 There is a statistically significant difference in the means natural and synthetic populations
# #### p-val less than 0.05 = statistically significant results reject the null hypothesis

# In[58]:


#create a subsamle of diamonds lab created between 0.5 and 1ct
lab_over_half_g_ideal_df = df1[(df1['dia_type_natural'] == 0)
            & (df1['shape_Round']==1)
            & (df1['carat'] <= 1)
            & (df1['carat'] > 0.5)
            & (df1['color_G']==1)
            & (df1['cut_Ideal']==1)
            & (df1['clarity_VS1']==1)]
print('lab diamond data shape is: {}'.format(lab_over_half_g_ideal_df.shape))


#create a subsample of diamonds natural between 0.5 and 1ct
nat_over_half_g_ideal_df = df1[(df1['dia_type_natural'] == 1) 
            & (df1['carat'] <= 1)
            & (df1['carat'] > 0.5)
            & (df1['shape_Round']==1)
            & (df1['color_G']== 1) 
            & (df1['cut_Ideal'] == 1)
            & (df1['clarity_VS1'] == 1)]
print('natural diamond data shape is: {}'.format(nat_over_half_g_ideal_df.shape))


# In[59]:


#sample from the data for t-test
lab_dia = lab_over_half_g_ideal_df.sample(n=30, random_state=seed)
print(lab_dia.shape)

nat_dia = nat_over_half_g_ideal_df.sample(n=30, random_state=seed)
print(nat_dia.shape)


# In[60]:


#check the homogeneity of the variance of the two populations
levene = stats.levene(lab_dia['price_nat_log'], nat_dia['price_nat_log'])
# if levene p < 0.05: we reject its null hypothesis of equal population variances.
print('The Levene statistic is {0[0]}, the p_value is: {0[1]}'.format(levene))
# Test for normality with Shapiro-Wilke p-val greater than 0.05, the data is normal below 0.05 indicates non-normal
lab_shapiro = stats.shapiro(lab_dia['price_nat_log'])
print('The Shapiro statistic for lab diamonds is: {0[0]}, the p_value is: {0[1]}'.format(lab_shapiro))
nat_shapiro = stats.shapiro(nat_dia['price_nat_log'])
print('The Shapiro statistic for natural diamonds is: {0[0]}, the p_value is: {0[1]}'.format(nat_shapiro))


# In[61]:


#t-test for difference of means between natural and synthetic diamonds of similar carat size
#p-val less than 0.05 indicates statistically significant difference in means
t_result = stats.ttest_ind(nat_dia['price_nat_log'],
                          lab_dia['price_nat_log'],
                          equal_var=True)
print(t_result)


# In[62]:


#subsample lab created diamonds from df1
lab_less_half_g_ideal_df = df1[(df1['dia_type_natural'] == 0) 
            & (df1['carat'] <= 0.5)
            & (df1['shape_Round']==1)
            & (df1['color_G']==1) 
            & (df1['cut_Ideal'] == 1)
            & (df1['clarity_VS1'] == 1)]
print('lab diamond data shape is: {}'.format(lab_less_half_g_ideal_df.shape))

nat_less_half_g_ideal_df = df1[(df1['dia_type_natural'] == 1) 
            & (df1['carat'] <= 0.5)
            & (df1['shape_Round']==1)
            & (df1['color_G']== 1) 
            & (df1['cut_Ideal'] == 1)
            & (df1['clarity_VS1'] == 1)]
print('natural diamond data shape is: {}'.format(nat_less_half_g_ideal_df.shape))


# In[63]:


#sample from the data for t-test
lab_dia2 = lab_less_half_g_ideal_df.sample(n=22, random_state=seed)
print(lab_dia2.shape)

nat_dia2 = nat_less_half_g_ideal_df.sample(n=22, random_state=seed)
print(nat_dia2.shape)


# In[64]:


#check the homogeneity of the variance of the two populations
levene = stats.levene(lab_dia2['price_nat_log'], nat_dia2['price_nat_log'])
# if levene p < 0.05: we reject its null hypothesis of equal population variances.
print('The Levene statistic is {0[0]}, the p_value is: {0[1]}'.format(levene))


#Test for normality with Shapiro-Wilke p-val greater than 0.05, the data is normal below 0.05 indicates non-normal
lab_shapiro = stats.shapiro(lab_dia2['price_nat_log'])
print('The Shapiro statistic for lab diamonds is: {0[0]}, the p_value is: {0[1]}'.format(lab_shapiro))
nat_shapiro = stats.shapiro(nat_dia2['price_nat_log'])
print('The Shapiro statistic for natural diamonds is: {0[0]}, the p_value is: {0[1]}'.format(nat_shapiro))


# In[65]:


#Run non-parametric test for difference as natural diamond distribution is non normal 
W_result = stats.wilcoxon(nat_less_half_g_ideal_df['price_nat_log'][0:15],
                          lab_less_half_g_ideal_df['price_nat_log'][0:15],
                          )
#p-value less than 0.05 indicates that there is a statistically significant difference between the two group means 
print('Wilcoxon Signed-Rank test statistic is: {0[0]}, p-value is: {0[1]}'.format(W_result))


# In[66]:


#t-test for difference of means between natural and synthetic diamonds of similar carat size
t_result = stats.ttest_ind(nat_less_half_g_ideal_df['price_nat_log'],
                          lab_less_half_g_ideal_df['price_nat_log'],
                          equal_var=True)
print(t_result)


# # END T-TEST

# In[67]:


#define a function to reverse power transformations made on any predictions made by regressive analysis 
# on the transformed data
def inverse_transform_bxcx(predictions, lambda_val):
    """Takes an array of predictions of a transformed column and returns the predictions in their
    origional state, must have forknowlege of transformation method used in the firstplace: 
    lambda values: -1 = reciprocal, -0.5 = reciprocal square root, 0 = log transform, 0.5 = square root,
    1 = no transform"""
    if lambda_val == -1:
        x= inv_boxcox(predictions,lambda_val)
        print('inverse reciprocal transformation performed')
    elif lambda_val == -0.5:
        x= inv_boxcox(predictions,lambda_val)
        print('inverse reciprocal square root transformation performed')
    elif lambda_val == 0:
        x=inv_boxcox(predictions,lambda_val)
        print('inverse natural log transformation performed')
    elif lambda_val == 0.5:
        x=inv_boxcox(predictions,lambda_val)
        print('inverse square root transformation performed')
    elif lambda_val == 1:
        x=inv_boxcox(predictions,lambda_val)
        print('inverse no transformation performed')
    #Visualize the column of values as a histogram
    plt.style.use('fivethirtyeight')
    sns.displot(x,color='blue', label="Univariate Distribution")
    plt.xlabel('Values')
    plt.ylabel('Count')
    plt.title('Distribution of Inverse Transformed Predictions')
    plt.legend()
    plt.show()
    plt.clf()
    return x
  
    


# In[68]:


df1.head()


# In[ ]:





# In[ ]:





# In[69]:


X = df1.drop(columns=['price','price_nat_log'])
y = df1['price_nat_log']
feature_names = X.columns


# In[70]:


#train test validation split function that was written for D213 task 2
def train_val_test_split(X,y,train_size,random_state):
    """ Takes the feature array, target array, training set size, random_state. 
    OUTPUT FORMAT:X_train, X_val, X_test, y_train, y_val, y_test, shape 
    This function requires sklearn.model_selection.train_test_split,
    and conducts two train_test_splits that are shuffled. 
    The output is a training set of selected size as X_train and y_train, validation and test sets
    that are made up of a 50/50 split of the remaining data as X_val, y_val, X_test, y_test respectively. 
    The final output is the shape of the resulting split sets as shape. """
    
    #split one
    X_train, X_valtest, y_train, y_valtest = train_test_split(X,y,train_size=train_size,random_state=random_state,
                                                              shuffle=True)
    
    #split two
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size = 0.5,random_state=random_state,
                                                    shuffle=True)
    
    #print out the shape of the  resulting splits for comparison
    shape = print('The shape of X_train: {}'.format(X_train.shape)),
    print('The shape of y_train: {}'.format(y_train.shape)),
    print('The shape of X_val: {}'.format(X_val.shape)),
    print('The shape of y_val: {}'.format(y_val.shape)),
    print('The shape of X_test: {}'.format(X_test.shape)),
    print('The shape of y_test: {}'.format(y_test.shape))
    
    return X_train, X_val, X_test, y_train, y_val, y_test, shape


# In[ ]:





# In[71]:


X_train, X_val, X_test, y_train, y_val, y_test, shape = train_val_test_split(X,y,train_size=0.8,random_state=seed)


# # Data Regression Analysis

# In[ ]:



    


# In[72]:


#define steps for a linear regression pipeline that uses an SGDregressor
steps = [('scaler', StandardScaler()),('SGDreg',SGDRegressor())]
pipeline = Pipeline(steps)


# In[73]:


#assemble a parameter grid for gridsearch cv object
parameter_grid = { 'SGDreg__alpha': np.arange(0.0001,0.0011,0.0001),
                  'SGDreg__penalty': ['l1','l2'],                  
}


# In[74]:


#instantiate gridsearch cv object
gscv = GridSearchCV(pipeline, parameter_grid, n_jobs=2)


# In[75]:


#Fit the model on the training data
gscv.fit(X_train, y_train)
print("Best parameter CV score: {}".format(gscv.best_score_))
print('Best Parameters Selected: {}'.format(gscv.best_params_))


# In[76]:


cvdf = pd.DataFrame(gscv.cv_results_)
display(cvdf)


# In[77]:


display('Best Esitmator: {}'.format(gscv.best_estimator_))
display('Best Score: {}'.format(gscv.best_score_))
display('Index: {}'.format(gscv.best_index_))


# In[78]:


y_pred = gscv.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rsqr = gscv.score(X_train,y_train)
rsqr_val = gscv.score(X_val,y_val)
print('The R-Square of the regressor on the training data is: {}'.format(rsqr))
print('The R-Square of the regressor on the validation data is: {}'.format(rsqr_val))
print('The Mean Squared Error of predictions made on the validation set is: {}'.format(mse))


# In[79]:


#Check the cross validation score on the test set
cross_v = cross_val_score(gscv,X_test, y_test, cv = 10)
print('Best Score on the test set is:{}'.format(max(cross_v)))


# In[80]:


#visualize coefficients and the intercept
print('Coefficients: ')
print(gscv.best_estimator_.steps[1][1].coef_)
print('Intercept: ')
print(gscv.best_estimator_.steps[1][1].intercept_)


# In[81]:


#Assign coefficints and intercept to variables 
coefficients = gscv.best_estimator_.steps[1][1].coef_
intercept = gscv.best_estimator_.steps[1][1].intercept_


# In[82]:


feature_coeff = pd.DataFrame(
{'Features': feature_names,
'coefficients': coefficients})


# In[83]:


#visualize the predictions versus the observations 
plt.figure(figsize=(50,25))
plt.style.use('fivethirtyeight')
x_ax = range(len(y_val))
plt.plot(gscv.predict(X_val), color = 'green', alpha = 0.2, label='Prediction Line')
plt.scatter(x_ax, y_val, label="Actual")
plt.scatter(x_ax, y_pred, label="Predicted", alpha = 0.5)
plt.title('Observed VS. Predicted')
plt.legend(prop={'size':30})
plt.show() 
plt.clf()


# In[84]:


#inspect the features and coefficients
feature_coeff = feature_coeff.sort_values(by=['coefficients'], ascending=False)
display(feature_coeff)


# In[85]:


#plot feature coefficients to visualize feature importance
plt.style.use('fivethirtyeight')
plt.figure(figsize=(50,25))
plt.bar(feature_coeff['Features'], feature_coeff['coefficients'])
plt.tick_params(axis='x', labelsize=40)
plt.tick_params(axis='y', labelsize=40)
plt.ylabel('Coefficient Value', fontsize=50)
plt.xlabel('Diamond Attributes', fontsize=50)
plt.title('Diamond Inventory Feature Importance', fontsize=80)
plt.xticks(rotation=90)
plt.show()
plt.clf()


# ## Fit a second model with top two most important features

# In[86]:


X2 = df1[['carat', 'dia_type_natural']]
y2 = df1['price_nat_log']
feature_names = X2.columns


# In[87]:


X_train2, X_val2, X_test2, y_train2, y_val2, y_test2, shape = train_val_test_split(X2,y2,train_size=0.8,random_state=seed)


# In[ ]:





# In[88]:


#define steps for a linear regression pipeline that uses an SGDregressor
steps2 = [('scaler', StandardScaler()),('SGDreg',SGDRegressor())]
pipeline2 = Pipeline(steps2)
#assemble a parameter grid for gridsearch cv object
parameter_grid2 = { 'SGDreg__alpha': np.arange(0.0001,0.0011,0.0001),
                  'SGDreg__penalty': ['l1','l2'],                  
}

#instantiate gridsearch cv object
gscv2 = GridSearchCV(pipeline2, parameter_grid2, n_jobs=2)
#Fit the model on the training data
gscv2.fit(X_train2, y_train2)
print("Best parameter CV score: {}".format(gscv2.best_score_))
print('Best Parameters Selected: {}'.format(gscv2.best_params_))

display('Best Esitmator: {}'.format(gscv2.best_estimator_))
display('Best Score: {}'.format(gscv2.best_score_))

display('Index: {}'.format(gscv2.best_index_))

y_pred2 = gscv2.predict(X_val2)
mse2 = mean_squared_error(y_val2, y_pred2)
rsqr2 = gscv2.score(X_train2,y_train2)
rsqr_val2 = gscv2.score(X_val2,y_val2)
print('The R-Square of the regressor on the training data is: {}'.format(rsqr2))
print('The R-Square of the regressor on the validation data is: {}'.format(rsqr_val2))
print('The Mean Squared Error of predictions made on the validation set is: {}'.format(mse2))

#Check the cross validation score on the test set
cross_v2 = cross_val_score(gscv2,X_test2, y_test2, cv = 10)
print('Best Score on the test set is:{}'.format(max(cross_v2)))

print('Coefficients: ')
print(gscv2.best_estimator_.steps[1][1].coef_)
print('Intercept: ')
print(gscv2.best_estimator_.steps[1][1].intercept_)

coefficients2 = gscv2.best_estimator_.steps[1][1].coef_
intercept2 = gscv2.best_estimator_.steps[1][1].intercept_

feature_coeff2 = pd.DataFrame(
{'Features': feature_names,
'coefficients': coefficients2})

#visualize the predictions versus the observations 
plt.figure(figsize=(50,25))
plt.style.use('fivethirtyeight')
x_ax2 = range(len(y_val2))
plt.plot(gscv2.predict(X_val2), color = 'green', alpha = 0.2, label='Prediction Line')
plt.scatter(x_ax2, y_val2, label="Actual")
plt.scatter(x_ax2, y_pred2, label="Predicted", alpha = 0.5)
plt.title('Observed VS. Predicted')
plt.legend(prop={'size':30})
plt.show() 
plt.clf()


# In[89]:


#inspect the residuals and fitted values of each model on the data set
fitted_values1 = gscv.predict(X_test)
#print(fitted_values1.head())
fitted_values2 = gscv2.predict(X_test2)
#print(fitted_values2.head())
residuals1 = abs(y_test - fitted_values1)
residuals2 = abs(y_test2 - fitted_values2)
print(residuals1.head())
print(residuals2.head())


# In[90]:


#inspect the distribution of the residuals1
plt.style.use('fivethirtyeight')
sns.kdeplot(residuals1, shade=True)
plt.title('Residuals GSVC SGDRegression model 1')
plt.show()
plt.clf()

#inspect the distribution of the residuals2
plt.style.use('fivethirtyeight')
sns.kdeplot(residuals2, shade=True)
plt.title('Residuals GSCV SGDRegression model 2')
plt.show()
plt.clf()


# In[91]:


#inspect gscv SGDRegressor model 1 residuals for homoscedasticity
plt.scatter(fitted_values1, residuals1)
plt.title('SGDRegressor model 1 residual values vs fitted values')
plt.show()
plt.clf()

#inspect gscv SGDRegressor model 2 residuals for homoscedasticity

plt.scatter(fitted_values2, residuals2)
plt.title('SGDRegressor model 2 residual values vs fitted values')
plt.show()
plt.clf()


# In[92]:


#visualize the residual plot with overlad smoothed version as a line 
fig, ax = plt.subplots(figsize=(5,5))
sns.regplot(fitted_values1, residuals1,
           scatter_kws = {'alpha':0.25}, line_kws = {'color':'C1'},
           lowess=True, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Absolute Value of Residual')
ax.set_title('SGDRegressor model 1 residuals & smoothed residuals')
plt.show()
plt.clf()


# In[93]:


#visualize the residual plot with overlad smoothed version as a line 
fig, ax = plt.subplots(figsize=(5,5))
sns.regplot(fitted_values2, residuals2,
           scatter_kws = {'alpha':0.25}, line_kws = {'color':'C1'},
           lowess=True, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Absolute Value of Residual')
ax.set_title('SGDRegressor model 2 residuals & smoothed residuals')
plt.show()
plt.clf()


# # Results & Recommendations

# ## Feature Importance:
# Carat weight is the most informative feature of diamond price included in this data set. 
# The second most informative feature is the determination of a diamond being of natural origin.
# Recommendation: Conduct a similar independent analysis with the populations natural and synthetic split using more data than what was available this time.

# ## Diffenrence in means:
# Two t-tests were run to determine if there is a statistically significant difference in the means between the synthetic and natural diamond categories all other factors being equal. 
# 
# The first t-test determined that there is a statistically significant difference in the means between the two groups. 
# 
# The second t-test also indicated that there was a statistically significant difference int he means between the two groups an d may have been influenced by a non normal distribution in the natural diamond sample. A Wilcoxon Signed Rank non-parametric test was run to confirm the findings of the t-test that there is a statistically significant difference in the means of the two groups for price_nat_log.
# 

# ## Predictive power of the Stochastic Gradient Descent Linear Regression:
# Two models were fit and evaluated on the data-set using gridsearch cross vaidation.  The first model that included the most features has the lowest mean squared error and the higher accuracy score of 0.85.
# 
# Heteroskedastic error indicates that prediction error is different for different ranges of the predictor variables included in the data set. This error is pronounced in observations of higher values of the response variable.
# 
# The recomendation moving forward is to separate the populations of natural and synthetic diamonds perform cluster analysis to determine how many groups are included in the data and make predictions with an ensemble regressor that has been trained on a larger data set. 
# 
# A random forrest regressor is recomended.  

# ## Limitations of Analysis:
# The response variable is of a non normal distribution and needed to be transformed before drawing meaningful conclusions from information provided. 
# 
# The data set is non temporal and does not indicate changes in values over time, analysis applied is only informative of the response variable at the time it was made available.
# 
# The data set is limited to diamonds that had been selected for sale by one vendor and may not represent a more generalized population of diamonds in the retail market. 
# 
# The label "Super Ideal" is a branded label used by this retailer and is not a standard industry label.
# Diamond attribute labels assigned by different gemological labratories may differ.

# In[ ]:




