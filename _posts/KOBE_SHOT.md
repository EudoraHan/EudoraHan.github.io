
<a href="https://colab.research.google.com/github/EudoraHan/KobeShot-LightGBM/blob/master/KOBE_SHOT.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Kobe Shot Evaluation and Prediction


### Yun Han  7/18/2019

### In this project, we use random forest, LGBM and GBDT model to evaluate and predict Kobe shot dataset.


```
from google.colab import files
uploaded = files.upload()
```



     <input type="file" id="files-a4660a74-bcb3-4549-9f3c-3c6920678482" name="files[]" multiple disabled />
     <output id="result-a4660a74-bcb3-4549-9f3c-3c6920678482">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script src="/nbextensions/google.colab/files.js"></script> 


    Saving kobe_shot.csv to kobe_shot.csv



```
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import io
```

## Part 1: Data Exploration


```
# 1.1 Read data:
data = pd.read_csv('kobe_shot.csv')
print(data.shape) # (30697, 25)
data.head()
```

    (30697, 25)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>action_type</th>
      <th>combined_shot_type</th>
      <th>game_event_id</th>
      <th>game_id</th>
      <th>lat</th>
      <th>loc_x</th>
      <th>loc_y</th>
      <th>lon</th>
      <th>minutes_remaining</th>
      <th>period</th>
      <th>playoffs</th>
      <th>season</th>
      <th>seconds_remaining</th>
      <th>shot_distance</th>
      <th>shot_made_flag</th>
      <th>shot_type</th>
      <th>shot_zone_area</th>
      <th>shot_zone_basic</th>
      <th>shot_zone_range</th>
      <th>team_id</th>
      <th>team_name</th>
      <th>game_date</th>
      <th>matchup</th>
      <th>opponent</th>
      <th>shot_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jump Shot</td>
      <td>Jump Shot</td>
      <td>10</td>
      <td>20000012</td>
      <td>33.9723</td>
      <td>167</td>
      <td>72</td>
      <td>-118.1028</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>2000-01</td>
      <td>27</td>
      <td>18</td>
      <td>NaN</td>
      <td>2PT Field Goal</td>
      <td>Right Side(R)</td>
      <td>Mid-Range</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>Los Angeles Lakers</td>
      <td>31/10/00</td>
      <td>LAL @ POR</td>
      <td>POR</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jump Shot</td>
      <td>Jump Shot</td>
      <td>12</td>
      <td>20000012</td>
      <td>34.0443</td>
      <td>-157</td>
      <td>0</td>
      <td>-118.4268</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>2000-01</td>
      <td>22</td>
      <td>15</td>
      <td>0.0</td>
      <td>2PT Field Goal</td>
      <td>Left Side(L)</td>
      <td>Mid-Range</td>
      <td>8-16 ft.</td>
      <td>1610612747</td>
      <td>Los Angeles Lakers</td>
      <td>31/10/00</td>
      <td>LAL @ POR</td>
      <td>POR</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jump Shot</td>
      <td>Jump Shot</td>
      <td>35</td>
      <td>20000012</td>
      <td>33.9093</td>
      <td>-101</td>
      <td>135</td>
      <td>-118.3708</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>2000-01</td>
      <td>45</td>
      <td>16</td>
      <td>1.0</td>
      <td>2PT Field Goal</td>
      <td>Left Side Center(LC)</td>
      <td>Mid-Range</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>Los Angeles Lakers</td>
      <td>31/10/00</td>
      <td>LAL @ POR</td>
      <td>POR</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jump Shot</td>
      <td>Jump Shot</td>
      <td>43</td>
      <td>20000012</td>
      <td>33.8693</td>
      <td>138</td>
      <td>175</td>
      <td>-118.1318</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>2000-01</td>
      <td>52</td>
      <td>22</td>
      <td>0.0</td>
      <td>2PT Field Goal</td>
      <td>Right Side Center(RC)</td>
      <td>Mid-Range</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>Los Angeles Lakers</td>
      <td>31/10/00</td>
      <td>LAL @ POR</td>
      <td>POR</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Driving Dunk Shot</td>
      <td>Dunk</td>
      <td>155</td>
      <td>20000012</td>
      <td>34.0443</td>
      <td>0</td>
      <td>0</td>
      <td>-118.2698</td>
      <td>6</td>
      <td>2</td>
      <td>0</td>
      <td>2000-01</td>
      <td>19</td>
      <td>0</td>
      <td>1.0</td>
      <td>2PT Field Goal</td>
      <td>Center(C)</td>
      <td>Restricted Area</td>
      <td>Less Than 8 ft.</td>
      <td>1610612747</td>
      <td>Los Angeles Lakers</td>
      <td>31/10/00</td>
      <td>LAL @ POR</td>
      <td>POR</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```
# 1.2: Variable Selection
# 1. Remove the row without target variable: shot_made_flag
data = data[pd.notnull(data['shot_made_flag'])]
data.shape # (25697, 25)
```




    (25697, 25)




```
# 2. Compare the variable lat,lon & loc_x,loc_y. 
### Histogram
sns.distplot(data['loc_y'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fad94aea240>




![png](KOBE_SHOT_files/KOBE_SHOT_8_1.png)



```
sns.distplot(data['lat'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fad94a65da0>




![png](KOBE_SHOT_files/KOBE_SHOT_9_1.png)



```
### Distribution Plot
plt.figure(figsize = (10,10))
 
plt.subplot(1,2,1)
plt.scatter(data.loc_x,data.loc_y,color ='r',alpha = 0.05)
plt.title('loc_x and loc_y')
 
plt.subplot(1,2,2)
plt.scatter(data.lon,data.lat,color ='b',alpha = 0.05)
plt.title('lat and lon')
```




    Text(0.5, 1.0, 'lat and lon')




![png](KOBE_SHOT_files/KOBE_SHOT_10_1.png)



```
### Correlations Matrix and heatmap
corr = data[["lat", "lon", "loc_x", "loc_y"]].corr()
sns.heatmap(corr)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fad94988dd8>




![png](KOBE_SHOT_files/KOBE_SHOT_11_1.png)


#### Both the distribution plot and the correlation matrix showed that the lat,lon & loc_x,loc_y represent the same position. We can delete one of them.




```
# 3. Time remain: 
"""
The variable 'minutes_remaining' and 'seconds_remaining' contain the same information，
we can combine the two variable together.

"""
data['remain_time'] = data['minutes_remaining'] * 60 + data['seconds_remaining']

```


```
# 4. shot_distance and shot_zone_range
corr = data[["shot_distance", "shot_zone_range"]].corr()
corr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>shot_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>shot_distance</th>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



#### The correlation between shot distance and shot zone range are 1, we can delete one of them.

## Part 2:  Feature Preprocessing


```
# Delete the duplicated variable:
data = data.drop(['lat','lon','minutes_remaining', 'seconds_remaining','matchup',
                  'shot_id', 'team_id','team_name', 'shot_zone_range','game_date'], axis = 1)

```

### Input our auto data preprocessing function.


```
"""
Created on Wed Jul 17 13:42:10 2019
@author: Yun Han


Automatic Datapreprocessing Function
1. Data format 
2. Missing Value
3. Outlier Dectect


"""

### Data Preprocessing

### Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Part 1. Data formatting

"""

## Label Encoding for String
from sklearn.preprocessing import LabelEncoder

def labelencode(data):
    labelencoder_X = LabelEncoder()
    data = labelencoder_X.fit_transform(data)
    integer_mapping = {l: i for i, l in enumerate(labelencoder_X.classes_)}
    
    return data, integer_mapping

"""
Part 2. Use different method to deal with missing value

"""
# =============================================================================
# ### Detect missing value
# df.info() # the overview information for the dataframe
# df.describe() # basic stats
# df.isnull().sum() # the number of rows with NaN for each column
# df.notnull().sum() # the number of rows without NaN for each column
# 
# =============================================================================


def missing_value(data, method):
    if method == 'delete':
        return data.dropna(inplace=True)
    
    elif method == '0 impute':
        return data.fillna(0, inplace=True) 
    
    elif method == 'mean':
        return data.fillna(data.mean(), inplace=True)
    
    elif method == 'median':
        return data.fillna(data.median(), inplace=True)
    
    elif method == 'ffill':
        return data.fillna(method='ffill', inplace = True)
    
    elif method == 'bfill':
        return data.fillna(method='bfill', inplace = True)
    
    elif method == 'interpolation':
        return data.interpolate()

# =============================================================================
# ### KNN for imputation
#         
# from sklearn.neighbors import KNeighborsClassifier
# # construct X matrix
# X = df.iloc[:, :-1].values
# column_new = ['RevolvingUtilizationOfUnsecuredLines', 'age', 
#               'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 
#               'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 
#               'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 
#               'NumberOfDependents']
# X = pd.DataFrame(data=X, columns = column_new)
# 
# # select the rows with missing values as testing data
# idx_with_nan = X.age.isnull()
# X_with_nan = X[idx_with_nan]
# 
# # select the rows without missing values as training data
# X_no_nan = X[-idx_with_nan]
# 
# # drop name column, set age as target variable and train the model, 
# clf = KNeighborsClassifier(3, weights='distance')
# clf.fit(X_no_nan[['RevolvingUtilizationOfUnsecuredLines',
#                   'NumberOfTime30-59DaysPastDueNotWorse', 
#                   'NumberOfOpenCreditLinesAndLoans', 
#                   'NumberOfTimes90DaysLate', 
#                   'NumberRealEstateLoansOrLines', 
#                   'NumberOfTime60-89DaysPastDueNotWorse']], X_no_nan['age'])
# 
# # impute the NA value
# x_imputed = clf.predict(X_with_nan[['RevolvingUtilizationOfUnsecuredLines',
#                   'NumberOfTime30-59DaysPastDueNotWorse', 
#                   'NumberOfOpenCreditLinesAndLoans', 
#                   'NumberOfTimes90DaysLate', 
#                   'NumberRealEstateLoansOrLines', 
#                   'NumberOfTime60-89DaysPastDueNotWorse']])
# X_with_imputed = X.copy()
# X_with_imputed.loc[idx_with_nan,'age'] = x_imputed.reshape(-1)
# 
# =============================================================================

"""
Part 3. Anomaly detection/ Outliers detection
https://www.zhihu.com/question/38066650

"""

# =============================================================================
# ### Handle Outliers
# import seaborn as sns
# 
# # Simulate data
# sns.set_style('whitegrid')
# sns.distplot(df.DebtRatio)
# sns.distplot(df.MonthlyIncome)
# sns.boxplot(df.age,orient='v')
# 
# 
# =============================================================================

##### auto function for outlier

from scipy.stats import skew
    
# 1. Numeric Outlier: define a function remove outlier using IQR: one dimension
def iqr_outlier(data):
    lq,uq=np.percentile(data,[25,75])
    lower_l=lq - 1.5*(uq-lq)
    upper_l=uq + 1.5*(uq-lq)
    
    # calculate the ratio of outliers
    ratio = (len(data[(data > upper_l)])+len(data[(data < lower_l)]))/len(data)
    # if ratio is large, we might replace the outlier with boundary value.
    if ratio > 0.1:
        
        return data
    
    elif ratio > 0.05:
        data[data < lower_l] = lower_l
        data[data > upper_l] = upper_l
        print ("%d upper is:", upper_l)
        print (data,"lower is:", lower_l)
        return data
        
    else:
        return data[(data >=lower_l)&(data<=upper_l)]
    
    
# 2. Z-score：one dimension or low dimension
def z_score_outlier(data):
    
    threshold=3
    mean_y = np.mean(data)
    stdev_y = np.std(data)
    z_scores = [(y - mean_y) / stdev_y for y in data]
    
    return data[np.abs(z_scores) < threshold]

"""
Auto function for outlier： 
combine the first two function

"""

def outlier(data):
    skewness = skew(data)    
    if skewness > 1:
        remove_outlier = iqr_outlier(data)
        
    else:
        remove_outlier = z_score_outlier(data)
    
    return remove_outlier

# =============================================================================
# ### Isolation Forest: one dimension or high dimension） 
#     
# # https://zhuanlan.zhihu.com/p/27777266
#     
# from sklearn.ensemble import IsolationForest
# import pandas as pd
# 
# 
# clf = IsolationForest(max_samples=100, random_state=42)
# clf.fit(train)
# y_pred = clf.predict(train)
# y_pred = [1 if x == -1 else 0 for x in y_pred]
# y_pred.count(0) # 94714
# y_pred.count(1) # 10524  
# 
# 
# =============================================================================


"""
Part 4. Auto Datapreprocessing Function 
1. a single variable
2. Whole dataset

"""
# For a single variable

def preprocessing(i, data, type, method):    
    
    
    if type == 'numeric':    
        if data[i].dtype == 'O':
            data[i] = pd.to_numeric(data[i], errors='coerce')
            
        missing_value(data[i], method)
        clean_data = outlier(data[i])        
        return clean_data
    
    elif type == 'categorical':
        missing_value(data[i], method)
        pre_index = data[i].index
        
        if data[i].dtype == 'O':
            data, dictionary = labelencode(data[i])        
            data = pd.Series(data, name = i)
            data.index = pre_index
            clean_data = outlier(data)    
        else:
            clean_data = outlier(data[i])  
        return clean_data


# For a whole dataset

def clean_all(df, categorical, method_cate, method_numeric):    
    for i in df.columns:
        if i not in categorical: 
            clean = preprocessing(i, df, 'numeric', method_numeric)
            if len(clean) < len(df):
                df = pd.merge(clean, df, left_index=True,right_index=True, how='left',suffixes=('_x', '_delete')) # left_on, right_on
            else:
                df = pd.merge(clean, df, left_index=True,right_index=True, how='right',suffixes=('_x', '_delete')) # left_on, right_on
      
        else:
            clean = preprocessing(i, df, 'categorical', method_cate)
            if len(clean) < len(df):
                df = pd.merge(clean, df, left_index=True,right_index=True, how='left',suffixes=('_x', '_delete')) # left_on, right_on
            else:
                df = pd.merge(clean, df, left_index=True,right_index=True, how='right',suffixes=('_x', '_delete')) # left_on, right_on
    
    for name in df.columns:
        if "_delete"  in name:
            df = df.drop([name], axis=1)
    
    return df


```

### Use auto-data preprocessing function 


```
# Use auto-data preprocessing fucntion
cat = ['action_type', 'combined_shot_type', 'period', 'playoffs','season', 'shot_made_flag',
       'shot_type', 'shot_zone_area', 'shot_zone_basic', 'opponent']        
after_Clean = clean_all(data, cat, 'delete', 'median')        

```


```
corr = after_Clean[['loc_x_x', 'loc_y_x','action_type_x', 'combined_shot_type_x', 'period_x', 'playoffs_x',
             'season_x', 'shot_made_flag_x','shot_type_x', 'shot_zone_area_x', 'shot_zone_basic_x',
             'opponent_x','game_event_id_x','game_id_x','shot_distance_x', 'remain_time_x']].corr()
sns.heatmap(corr)

```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fad952011d0>




![png](KOBE_SHOT_files/KOBE_SHOT_22_1.png)


#### From the correlation matrix, we find that the correlation between shot_distance and loc_y is 0.79, game_event_id and period is 0.96, game_id and playoffs is 0.92, hence, we can delete one of each pair.



```
# Delete the high correlation variable: shot_distance, game_event_id,game_id
shot = after_Clean.drop(['shot_distance_x','game_event_id_x','game_id_x'], axis = 1)

```

## Part 3: Model Training --- Random Forest


```
X = shot.drop(['shot_made_flag_x'], axis = 1)
y = shot.iloc[:, 5].values
```


```
# One HotEncoder: Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [1,2,3,5,7,10,11])
X = onehotencoder.fit_transform(X).toarray()

```

    /usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_encoders.py:415: FutureWarning: The handling of integer data will change in version 0.22. Currently, the categories are determined based on the range [0, max(values)], while in the future they will be determined based on the unique values.
    If you want the future behaviour and silence this warning, you can specify "categories='auto'".
    In case you used a LabelEncoder before this OneHotEncoder to convert the categories to integers, then you can now use the OneHotEncoder directly.
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_encoders.py:451: DeprecationWarning: The 'categorical_features' keyword is deprecated in version 0.20 and will be removed in 0.22. You can use the ColumnTransformer instead.
      "use the ColumnTransformer instead.", DeprecationWarning)


### Part 3.1: Split dataset


```
# 3.1: Split dataset
# Splite data into training and testing
from sklearn import model_selection
# Reserve 20% for testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2,random_state = 0)

```

### Part 3.2: Model Training and Selection


```
# 3.2: Model Training and Selection
from sklearn.ensemble import RandomForestClassifier
# Random Forest
classifier_RF = RandomForestClassifier()

```


```
# Train the model
classifier_RF.fit(X_train, y_train)
# Prediction of test data
classifier_RF.predict(X_test)

```

    /usr/local/lib/python3.6/dist-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)





    array([0., 0., 0., ..., 0., 1., 1.])




```
# Accuracy of test data
classifier_RF.score(X_test, y_test) 

```




    0.6301567656765676




```
# Use 5-fold Cross Validation to get the accuracy # 0.6192
cv_score = model_selection.cross_val_score(classifier_RF, X_train, y_train, cv=5)
print('Model accuracy of Random Forest is:',cv_score.mean())

```

    Model accuracy of Random Forest is: 0.6267477710944356


###  Part 3.3: Use Grid Search to Find Optimal Hyperparameters


```
# 3.3: Use Grid Search to Find Optimal Hyperparameters
from sklearn.model_selection import GridSearchCV

# helper function for printing out grid search results 
def print_grid_search_metrics(gs):
    print ("Best score: %0.3f" % gs.best_score_)
    print ("Best parameters set:")
    best_parameters = gs.best_params_
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


```


```
# Possible hyperparamter options for Random Forest
# Choose the number of trees
parameters = {
    'n_estimators' : [80,90],
    'max_features' : ['auto','sqrt','log2'],
    'min_samples_leaf' : [1,10,50,100]
    
}

# Use time function to measure time elapsed
import time
start = time.time()

Grid_RF = GridSearchCV(RandomForestClassifier(),parameters, cv=5)
Grid_RF.fit(X_train, y_train)

end = time.time()
print(end - start)
```

    187.79808497428894



```
# best number of tress
print_grid_search_metrics(Grid_RF)

```

    Best score: 0.668
    Best parameters set:
    	max_features: 'auto'
    	min_samples_leaf: 10
    	n_estimators: 80



```
# best random forest
best_RF_model = Grid_RF.best_estimator_

```

### Part 3.4: Use Random Search to Find Optimal Hyperparameters



```
# 3.4: Use Random Search to Find Optimal Hyperparameters
from sklearn.model_selection import RandomizedSearchCV

# helper function for printing out grid search results 
def print_random_search_metrics(rs):
    print ("Best score: %0.3f" % rs.best_score_)
    print ("Best parameters set:")
    best_parameters = rs.best_params_
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))



```


```

# Possible hyperparamter options for Random Forest
# Choose the number of trees
parameters = {
    'n_estimators' : [80,90],
    'max_features' : ['auto','sqrt','log2'],
    'min_samples_leaf' : [1,10,50,100]
}

# Use time function to measure time elapsed
import time
start = time.time()

Random_RF = RandomizedSearchCV(RandomForestClassifier(),parameters, cv=5)
Random_RF.fit(X_train, y_train)

end = time.time()
print(end - start)
```

    88.73095989227295



```
# best number of tress
print_grid_search_metrics(Random_RF)
```

    Best score: 0.668
    Best parameters set:
    	max_features: 'sqrt'
    	min_samples_leaf: 10
    	n_estimators: 90



```

```

### Part 3.5: Use Bayesian Optimization to Find Optimal Hyperparameters


```
!pip install bayesian-optimization
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
```

    Collecting bayesian-optimization
      Downloading https://files.pythonhosted.org/packages/72/0c/173ac467d0a53e33e41b521e4ceba74a8ac7c7873d7b857a8fbdca88302d/bayesian-optimization-1.0.1.tar.gz
    Requirement already satisfied: numpy>=1.9.0 in /usr/local/lib/python3.6/dist-packages (from bayesian-optimization) (1.16.4)
    Requirement already satisfied: scipy>=0.14.0 in /usr/local/lib/python3.6/dist-packages (from bayesian-optimization) (1.3.0)
    Requirement already satisfied: scikit-learn>=0.18.0 in /usr/local/lib/python3.6/dist-packages (from bayesian-optimization) (0.21.2)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.6/dist-packages (from scikit-learn>=0.18.0->bayesian-optimization) (0.13.2)
    Building wheels for collected packages: bayesian-optimization
      Building wheel for bayesian-optimization (setup.py) ... [?25l[?25hdone
      Stored in directory: /root/.cache/pip/wheels/1d/0d/3b/6b9d4477a34b3905f246ff4e7acf6aafd4cc9b77d473629b77
    Successfully built bayesian-optimization
    Installing collected packages: bayesian-optimization
    Successfully installed bayesian-optimization-1.0.1



```
rf = RandomForestClassifier()
def rf_cv(n_estimators, min_samples_leaf, max_features):
    val = cross_val_score(
        RandomForestClassifier(n_estimators=int(n_estimators),
            min_samples_leaf=int(min_samples_leaf),
            max_features=min(max_features, 0.999), # float
            random_state=2
        ),
        X, y, scoring= 'accuracy', cv=5
    ).mean()
    return val
```


```
rf_bo = BayesianOptimization(
        rf_cv,
        {'n_estimators': (10, 250),
         'min_samples_leaf': (2, 25),
         'max_features': (0.1, 0.999)
        }
)
```


```
# Use time function to measure time elapsed
import time
start = time.time()

num_iter = 10
init_points = 5
rf_bo.maximize(init_points=init_points,n_iter=num_iter)

end = time.time()
print(end - start)
```

    |   iter    |  target   | max_fe... | min_sa... | n_esti... |
    -------------------------------------------------------------
    | [0m 1       [0m | [0m 0.6362  [0m | [0m 0.3618  [0m | [0m 3.493   [0m | [0m 12.25   [0m |
    | [95m 2       [0m | [95m 0.6646  [0m | [95m 0.1639  [0m | [95m 14.34   [0m | [95m 204.0   [0m |
    | [0m 3       [0m | [0m 0.6588  [0m | [0m 0.8935  [0m | [0m 23.97   [0m | [0m 148.7   [0m |
    | [0m 4       [0m | [0m 0.6475  [0m | [0m 0.8533  [0m | [0m 2.733   [0m | [0m 203.6   [0m |
    | [0m 5       [0m | [0m 0.6587  [0m | [0m 0.8141  [0m | [0m 22.84   [0m | [0m 240.9   [0m |
    | [0m 6       [0m | [0m 0.6551  [0m | [0m 0.999   [0m | [0m 25.0    [0m | [0m 10.0    [0m |
    | [0m 7       [0m | [0m 0.6591  [0m | [0m 0.9986  [0m | [0m 24.95   [0m | [0m 201.6   [0m |
    | [95m 8       [0m | [95m 0.667   [0m | [95m 0.1127  [0m | [95m 24.84   [0m | [95m 68.14   [0m |
    | [0m 9       [0m | [0m 0.6582  [0m | [0m 0.1071  [0m | [0m 2.027   [0m | [0m 108.4   [0m |
    | [0m 10      [0m | [0m 0.6656  [0m | [0m 0.1     [0m | [0m 24.63   [0m | [0m 101.6   [0m |
    | [0m 11      [0m | [0m 0.6632  [0m | [0m 0.107   [0m | [0m 5.121   [0m | [0m 250.0   [0m |
    | [0m 12      [0m | [0m 0.6659  [0m | [0m 0.1     [0m | [0m 25.0    [0m | [0m 35.28   [0m |
    | [0m 13      [0m | [0m 0.6664  [0m | [0m 0.1137  [0m | [0m 24.62   [0m | [0m 179.8   [0m |
    | [0m 14      [0m | [0m 0.6664  [0m | [0m 0.1197  [0m | [0m 24.87   [0m | [0m 216.6   [0m |
    | [0m 15      [0m | [0m 0.6664  [0m | [0m 0.1219  [0m | [0m 24.28   [0m | [0m 249.7   [0m |
    =============================================================
    831.7602977752686



```
rf_bo.max
```




    {'params': {'max_depth': 5.08900552171501,
      'max_features': 0.18226085009058457,
      'min_samples_split': 24.775708453366086,
      'n_estimators': 174.04350508403547},
     'target': 0.6683580716980221}



## Part 4: Model Training --- LGBM


```
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import KFold, train_test_split
from lightgbm import LGBMClassifier

```

### Part 4.1: Model Training and Selection


```
lgbm = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=50,
    colsample_bytree=.8,
    subsample=.9,
    max_depth=7,
    reg_alpha=.1,
    reg_lambda=.1,
    min_split_gain=.01,
    min_child_weight=2,
    silent=-1,
    verbose=-1,
)
```

#### Fit the model


```
lgbm.fit(X_train, y_train, 
    eval_set= [(X_train, y_train), (X_test, y_test)], 
    eval_metric='auc', verbose=100, early_stopping_rounds=30  #30
)

```

    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [53]	training's binary_logloss: 0.601406	training's auc: 0.737307	valid_1's binary_logloss: 0.619088	valid_1's auc: 0.688686





    LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.8,
                   importance_type='split', learning_rate=0.05, max_depth=7,
                   min_child_samples=20, min_child_weight=2, min_split_gain=0.01,
                   n_estimators=400, n_jobs=-1, num_leaves=50, objective=None,
                   random_state=None, reg_alpha=0.1, reg_lambda=0.1, silent=-1,
                   subsample=0.9, subsample_for_bin=200000, subsample_freq=0,
                   verbose=-1)




```
# Use 5-fold Cross Validation to get the accuracy # 0.6192
cv_score = model_selection.cross_val_score(lgbm, X_train, y_train, cv=5)
print('Model accuracy of LGBM is:',cv_score.mean())

```

    Model accuracy of LGBM is: 0.656866550527722


### Part 4.2: Use Grid Search to Find Optimal Hyperparameters


```
# Possible hyperparamter options for LigntGBM
# Choose the number of trees
parameters = {
    'n_estimators' : [100, 200, 300, 400],
    'learning_rate' : [0.03, 0.05, 0.08, 0.1, 0.2],
    'num_leaves' : [30, 50, 80],
    'max_depth': [5, 10, 20, 30]
}

# Use time function to measure time elapsed
import time
start = time.time()

Grid_LGBM = GridSearchCV(LGBMClassifier(),parameters, cv=5)
Grid_LGBM.fit(X_train, y_train)


end = time.time()
print(end - start)
```

    940.9568140506744



```
# best number of tress
print_grid_search_metrics(Grid_LGBM)
```

    Best score: 0.668
    Best parameters set:
    	learning_rate: 0.03
    	max_depth: 5
    	n_estimators: 300
    	num_leaves: 30


### Part 4.3: Use Random Search to Find Optimal Hyperparameters


```
# 4.3: Use Random Search to Find Optimal Hyperparameters
from sklearn.model_selection import RandomizedSearchCV

# helper function for printing out grid search results 
def print_random_search_metrics(rs):
    print ("Best score: %0.3f" % rs.best_score_)
    print ("Best parameters set:")
    best_parameters = rs.best_params_
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))



```


```
# Possible hyperparamter options for LightGBM
# Choose the number of trees
parameters = {
    'n_estimators' : [60, 80, 100, 200, 300],
    'learning_rate' : [0.03, 0.05, 0.08, 0.1, 0.2],
    'num_leaves' : [20, 30, 50],
    'max_depth': [10, 20, 30]
}

# Use time function to measure time elapsed
import time
start = time.time()

Random_LGBM = RandomizedSearchCV(LGBMClassifier(),parameters, cv=5)
Random_LGBM.fit(X_train, y_train)

end = time.time()
print(end - start)
```

    18.503878831863403



```
# best number of tress
print_grid_search_metrics(Random_LGBM)
```

    Best score: 0.667
    Best parameters set:
    	learning_rate: 0.05
    	max_depth: 30
    	n_estimators: 60
    	num_leaves: 50


### Part 4.4: Use Bayesian Optimazition to Find Optimal Hyperparameters


```
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
```


```
LGBM = LGBMClassifier()
def LGBM_cv(n_estimators, learning_rate, num_leaves, max_depth,lambda_l1):
    val = cross_val_score(
        LGBMClassifier(n_estimators = int(n_estimators),
            learning_rate = learning_rate,
            num_leaves = int(num_leaves),
            max_depth = int(max_depth),
            lambda_l1 = lambda_l1,
            random_state = 2
        ),
        X, y, scoring= 'accuracy', cv=5
    ).mean()
    return val
```


```
LGBM_bo = BayesianOptimization(
        LGBM_cv,
        {'n_estimators': (10, 250),
         'learning_rate' : (0.01, 2.0),
         'num_leaves': (5, 130),
         'max_depth': (4, 20),
         'lambda_l1': (0, 6)}
)
```


```
# Use time function to measure time elapsed
import time
start = time.time()

num_iter = 10
init_points = 5
LGBM_bo.maximize(init_points=init_points,n_iter=num_iter)

end = time.time()
print(end - start)
```

    |   iter    |  target   | lambda_l1 | learni... | max_depth | n_esti... | num_le... |
    -------------------------------------------------------------------------------------
    | [0m 1       [0m | [0m 0.6495  [0m | [0m 1.961   [0m | [0m 0.1331  [0m | [0m 7.443   [0m | [0m 188.6   [0m | [0m 31.95   [0m |
    | [0m 2       [0m | [0m 0.5554  [0m | [0m 1.387   [0m | [0m 1.985   [0m | [0m 15.65   [0m | [0m 27.77   [0m | [0m 55.66   [0m |
    | [0m 3       [0m | [0m 0.5164  [0m | [0m 1.74    [0m | [0m 1.991   [0m | [0m 6.956   [0m | [0m 70.04   [0m | [0m 82.4    [0m |
    | [0m 4       [0m | [0m 0.5969  [0m | [0m 0.2214  [0m | [0m 1.075   [0m | [0m 5.229   [0m | [0m 109.6   [0m | [0m 123.1   [0m |
    | [0m 5       [0m | [0m 0.6251  [0m | [0m 3.696   [0m | [0m 0.8189  [0m | [0m 13.79   [0m | [0m 57.76   [0m | [0m 54.53   [0m |
    | [95m 6       [0m | [95m 0.662   [0m | [95m 6.0     [0m | [95m 0.01    [0m | [95m 20.0    [0m | [95m 250.0   [0m | [95m 130.0   [0m |
    | [0m 7       [0m | [0m 0.5742  [0m | [0m 6.0     [0m | [0m 0.01    [0m | [0m 4.0     [0m | [0m 10.0    [0m | [0m 5.0     [0m |
    | [95m 8       [0m | [95m 0.6666  [0m | [95m 6.0     [0m | [95m 0.01    [0m | [95m 20.0    [0m | [95m 250.0   [0m | [95m 5.0     [0m |
    | [0m 9       [0m | [0m 0.4921  [0m | [0m 0.0     [0m | [0m 2.0     [0m | [0m 20.0    [0m | [0m 114.6   [0m | [0m 5.0     [0m |
    | [0m 10      [0m | [0m 0.6625  [0m | [0m 6.0     [0m | [0m 0.01    [0m | [0m 20.0    [0m | [0m 178.8   [0m | [0m 130.0   [0m |
    | [0m 11      [0m | [0m 0.6638  [0m | [0m 0.0     [0m | [0m 0.01    [0m | [0m 4.0     [0m | [0m 250.0   [0m | [0m 68.13   [0m |
    | [0m 12      [0m | [0m 0.6627  [0m | [0m 6.0     [0m | [0m 0.01    [0m | [0m 20.0    [0m | [0m 147.6   [0m | [0m 77.09   [0m |
    | [95m 13      [0m | [95m 0.667   [0m | [95m 6.0     [0m | [95m 0.01    [0m | [95m 4.0     [0m | [95m 216.9   [0m | [95m 130.0   [0m |
    | [0m 14      [0m | [0m 0.663   [0m | [0m 6.0     [0m | [0m 0.01    [0m | [0m 20.0    [0m | [0m 219.2   [0m | [0m 63.12   [0m |
    | [0m 15      [0m | [0m 0.5144  [0m | [0m 6.0     [0m | [0m 2.0     [0m | [0m 4.0     [0m | [0m 250.0   [0m | [0m 5.0     [0m |
    =====================================================================================
    133.16940212249756



```
LGBM_bo.max
```




    {'params': {'lambda_l1': 6.0,
      'learning_rate': 0.01,
      'max_depth': 4.0,
      'n_estimators': 216.90519644528626,
      'num_leaves': 130.0},
     'target': 0.6669551420508177}



## Part 5: Model Training --- GBDT


```
from sklearn.ensemble import GradientBoostingClassifier
```

### Part 5.1: Model Training and Selection




```
gbr = GradientBoostingClassifier(n_estimators=300, max_depth=2, min_samples_split=2, learning_rate=0.1)
gbr.fit(X_train, y_train)
```




    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                               learning_rate=0.1, loss='deviance', max_depth=2,
                               max_features=None, max_leaf_nodes=None,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=300,
                               n_iter_no_change=None, presort='auto',
                               random_state=None, subsample=1.0, tol=0.0001,
                               validation_fraction=0.1, verbose=0,
                               warm_start=False)




```
# Prediction of test data
gbr.predict(X_test)
```




    array([0., 0., 0., ..., 0., 0., 1.])




```
# Accuracy of test data
gbr.score(X_test, y_test) 
```




    0.6745049504950495




```

```

### Part 5.2: Use Grid Search to Find Optimal Hyperparameters



```
# Possible hyperparamter options for GBDT
# Choose the number of trees
parameters = {
    'n_estimators' : [100],
    'learning_rate' : [0.1],
    'min_samples_split' :[2],
    'max_depth': [2]

}

# Use time function to measure time elapsed
import time
start = time.time()

Grid_GBDT = GridSearchCV(GradientBoostingClassifier(),parameters, cv=3)
Grid_GBDT.fit(X_train, y_train)


end = time.time()
print(end - start)
```

    11.35716199874878



```
# best number of tress
print_grid_search_metrics(Grid_GBDT)
```

    Best score: 0.668
    Best parameters set:
    	learning_rate: 0.1
    	max_depth: 2
    	min_samples_split: 2
    	n_estimators: 100


### Part 5.3: Use Random Search to Find Optimal Hyperparameters


```
# 5.3: Use Random Search to Find Optimal Hyperparameters
from sklearn.model_selection import RandomizedSearchCV

# helper function for printing out grid search results 
def print_random_search_metrics(rs):
    print ("Best score: %0.3f" % rs.best_score_)
    print ("Best parameters set:")
    best_parameters = rs.best_params_
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))



```


```
# Possible hyperparamter options for LightGBM
# Choose the number of trees
parameters = {
    'n_estimators' : [60, 80, 100, 200, 300],
    'learning_rate' : [0.03, 0.05, 0.08, 0.1, 0.2],
    'min_samples_split' :[2, 25],
    'max_depth': [2, 5, 8]
        
}

# Use time function to measure time elapsed
import time
start = time.time()

Random_GBDT = RandomizedSearchCV(GradientBoostingClassifier(),parameters, cv=5)
Random_GBDT.fit(X_train, y_train)

end = time.time()
print(end - start)
```

    472.7206542491913



```
# best number of tress
print_grid_search_metrics(Random_GBDT)
```

    Best score: 0.667
    Best parameters set:
    	learning_rate: 0.08
    	max_depth: 2
    	min_samples_split: 25
    	n_estimators: 60


### Part 5.4: Use Bayesian Optimazition to Find Optimal Hyperparameters


```
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
```


```
GBDT = GradientBoostingClassifier()
def GBDT_cv(n_estimators, learning_rate, min_samples_split, max_depth):
    val = cross_val_score(
        GradientBoostingClassifier(n_estimators = int(n_estimators),
            learning_rate = learning_rate,
            min_samples_split=int(min_samples_split),
            max_depth = int(max_depth),
            random_state = 2
        ),
        X, y, scoring= 'accuracy', cv=5
    ).mean()
    return val
```


```
GBDT_bo = BayesianOptimization(
        GBDT_cv,
        {'n_estimators': (100, 300),
         'learning_rate' : (0.01, 0.2),
         'min_samples_split' :(2, 25),
         'max_depth': (2, 10)
        }
)
```


```
# Use time function to measure time elapsed
import time
start = time.time()

num_iter = 10
init_points = 5
GBDT_bo.maximize(init_points=init_points,n_iter=num_iter)

end = time.time()
print(end - start)
```

    |   iter    |  target   | learni... | max_depth | min_sa... | n_esti... |
    -------------------------------------------------------------------------
    | [0m 20      [0m | [0m 0.6601  [0m | [0m 0.1183  [0m | [0m 3.98    [0m | [0m 24.39   [0m | [0m 190.4   [0m |
    | [0m 21      [0m | [0m 0.6601  [0m | [0m 0.0304  [0m | [0m 5.925   [0m | [0m 20.28   [0m | [0m 222.8   [0m |
    | [0m 22      [0m | [0m 0.6614  [0m | [0m 0.06898 [0m | [0m 4.586   [0m | [0m 18.85   [0m | [0m 146.2   [0m |
    | [0m 23      [0m | [0m 0.6526  [0m | [0m 0.08733 [0m | [0m 5.392   [0m | [0m 21.93   [0m | [0m 211.0   [0m |
    | [0m 24      [0m | [0m 0.6645  [0m | [0m 0.04976 [0m | [0m 2.35    [0m | [0m 24.66   [0m | [0m 166.9   [0m |
    | [0m 25      [0m | [0m 0.6535  [0m | [0m 0.02413 [0m | [0m 9.515   [0m | [0m 2.007   [0m | [0m 139.0   [0m |
    | [0m 26      [0m | [0m 0.664   [0m | [0m 0.09761 [0m | [0m 2.075   [0m | [0m 24.96   [0m | [0m 266.0   [0m |
    | [0m 27      [0m | [0m 0.6654  [0m | [0m 0.02553 [0m | [0m 2.094   [0m | [0m 12.34   [0m | [0m 132.1   [0m |
    | [0m 28      [0m | [0m 0.6557  [0m | [0m 0.01    [0m | [0m 9.445   [0m | [0m 25.0    [0m | [0m 283.3   [0m |
    | [0m 29      [0m | [0m 0.6644  [0m | [0m 0.0751  [0m | [0m 2.013   [0m | [0m 2.319   [0m | [0m 153.6   [0m |
    | [0m 30      [0m | [0m 0.6455  [0m | [0m 0.05067 [0m | [0m 9.923   [0m | [0m 24.97   [0m | [0m 238.6   [0m |
    | [0m 31      [0m | [0m 0.6168  [0m | [0m 0.1351  [0m | [0m 9.448   [0m | [0m 2.356   [0m | [0m 299.6   [0m |
    | [0m 32      [0m | [0m 0.6643  [0m | [0m 0.04323 [0m | [0m 2.032   [0m | [0m 14.63   [0m | [0m 294.7   [0m |
    | [0m 33      [0m | [0m 0.6635  [0m | [0m 0.1374  [0m | [0m 2.115   [0m | [0m 24.95   [0m | [0m 135.2   [0m |
    | [95m 34      [0m | [95m 0.6684  [0m | [95m 0.01    [0m | [95m 2.0     [0m | [95m 14.22   [0m | [95m 106.0   [0m |
    =========================================================================
    2270.5618958473206



```
GBDT_bo.max
```




    {'params': {'learning_rate': 0.01,
      'max_depth': 2.000000029868887,
      'min_samples_split': 14.215169189609608,
      'n_estimators': 106.04429937904912},
     'target': 0.6684405969713871}


