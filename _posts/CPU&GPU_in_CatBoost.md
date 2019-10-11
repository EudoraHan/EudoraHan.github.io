
# CatBoost Property Test

### Yun Han 7/30/2019

# 1. Epilson


```
!pip install catboost
```


```
from catboost.datasets import epsilon

train, test = epsilon()

X_train, y_train = train.iloc[:,1:], train[0]
X_test, y_test = test.iloc[:,1:], test[0]

```


```
X_train.shape
```




    (400000, 2000)




```
X_test.shape
```




    (100000, 2000)




```
!pip install -U -q PyDrive
```

    [K     |████████████████████████████████| 993kB 9.6MB/s 
    [?25h  Building wheel for PyDrive (setup.py) ... [?25l[?25hdone



```
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
# This only needs to be done once in a notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
# Create & upload a text file.
#你想要导出的文件的名字
uploaded = drive.CreateFile({'title': 'OK.csv'})
#改为之前生成文件的名字
uploaded.SetContentFile('over.csv')
uploaded.Upload()
print('Uploaded file with ID {}'.format(uploaded.get('id')))
```

    Uploaded file with ID 1DPAcA9VzfP6HuhhSt0ifk9vXSsdCgobH



```

```


```
X_train.shape

```




    (400000, 2000)



### Training on CPU


```
from catboost import CatBoostClassifier
import timeit

def train_on_cpu():  
  model = CatBoostClassifier(
      iterations=1000,
      learning_rate=0.03,
      boosting_type = 'Plain'
  )
  
  model.fit(
      X_train, y_train,
      eval_set=(X_test, y_test),
      verbose=10
  );   
      
cpu_time = timeit.timeit('train_on_cpu()', 
                         setup="from __main__ import train_on_cpu", 
                         number=1)

print('Time to fit model on CPU: {} sec'.format(int(cpu_time)))
```

### Training on GPU


```
from catboost import CatBoostClassifier
import timeit

def train_on_gpu():  
  model = CatBoostClassifier(
      iterations=1000,
      learning_rate=0.03,
      boosting_type = 'Plain',
      task_type='GPU'
  )
  
  model.fit(
      X_train, y_train,
      eval_set=(X_test, y_test),
      verbose=10
  );     
      
gpu_time = timeit.timeit('train_on_gpu()', 
                         setup="from __main__ import train_on_gpu", 
                         number=1)

print('Time to fit model on GPU: {} sec'.format(int(gpu_time)))
print('GPU speedup over CPU: ' + '%.2f' % (cpu_time/gpu_time) + 'x')
```


```
from catboost import CatBoostClassifier
classifier_cat = CatBoostClassifier(iterations = 100, task_type = 'GPU')
```


```
# Train the model
classifier_cat.fit(X_train, y_train)
```


```
# Prediction of test data
classifier_cat.predict(X_test)
```


```
# Accuracy of test data
classifier_cat.score(X_test, y_test)
```


```
import lightgbm as lgb
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'learning_rate': 0.03
}

```


```
# Use time function to measure time elapsed
import time
start = time.time()


gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_train)

end = time.time()
print(end - start)
```

# 2. Credit card lost


```
from google.colab import files
uploaded = files.upload()
```



     <input type="file" id="files-e6110ee2-0f38-4084-a083-baf51740d777" name="files[]" multiple disabled />
     <output id="result-e6110ee2-0f38-4084-a083-baf51740d777">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script src="/nbextensions/google.colab/files.js"></script> 


    Saving creditcard lost.csv to creditcard lost.csv



```
import io
import pandas as pd

df = pd.read_csv(io.BytesIO(uploaded['creditcard lost.csv']))
#df.head()
```


```
df.head()
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
      <th>y</th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
      <th>x4</th>
      <th>x5</th>
      <th>x6</th>
      <th>x7</th>
      <th>x8</th>
      <th>x9</th>
      <th>x10</th>
      <th>x11</th>
      <th>x12</th>
      <th>x13</th>
      <th>x14</th>
      <th>x15</th>
      <th>x16</th>
      <th>x17</th>
      <th>x18</th>
      <th>x19</th>
      <th>x20</th>
      <th>x21</th>
      <th>x22</th>
      <th>x23</th>
      <th>x24</th>
      <th>x25</th>
      <th>x26</th>
      <th>x27</th>
      <th>x28</th>
      <th>x29</th>
      <th>x30</th>
      <th>x31</th>
      <th>x32</th>
      <th>x33</th>
      <th>x34</th>
      <th>x35</th>
      <th>x36</th>
      <th>x37</th>
      <th>x38</th>
      <th>x39</th>
      <th>...</th>
      <th>x81</th>
      <th>x82</th>
      <th>x83</th>
      <th>x84</th>
      <th>x85</th>
      <th>x86</th>
      <th>x87</th>
      <th>x88</th>
      <th>x89</th>
      <th>x90</th>
      <th>x91</th>
      <th>x92</th>
      <th>x93</th>
      <th>x94</th>
      <th>x95</th>
      <th>x96</th>
      <th>x97</th>
      <th>x98</th>
      <th>x99</th>
      <th>x100</th>
      <th>x101</th>
      <th>x102</th>
      <th>x103</th>
      <th>x104</th>
      <th>x105</th>
      <th>x106</th>
      <th>x107</th>
      <th>x108</th>
      <th>x109</th>
      <th>x110</th>
      <th>x111</th>
      <th>x112</th>
      <th>x113</th>
      <th>x114</th>
      <th>x115</th>
      <th>x116</th>
      <th>x117</th>
      <th>x118</th>
      <th>x119</th>
      <th>x120</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5.04</td>
      <td>5.77</td>
      <td>6.04</td>
      <td>3.91</td>
      <td>0.66</td>
      <td>1.04</td>
      <td>1.83</td>
      <td>5.41</td>
      <td>5.45</td>
      <td>6.91</td>
      <td>6.70</td>
      <td>1.17</td>
      <td>3.08</td>
      <td>3.44</td>
      <td>1.83</td>
      <td>5.32</td>
      <td>5.16</td>
      <td>4.62</td>
      <td>4.58</td>
      <td>3.98</td>
      <td>6.57</td>
      <td>5.18</td>
      <td>1.79</td>
      <td>2.99</td>
      <td>7.95</td>
      <td>21.02</td>
      <td>1.70</td>
      <td>0.00</td>
      <td>4.55</td>
      <td>6.90</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.25</td>
      <td>3.16</td>
      <td>5.85</td>
      <td>6.26</td>
      <td>5.06</td>
      <td>0.00</td>
      <td>...</td>
      <td>4.94</td>
      <td>5.75</td>
      <td>6.23</td>
      <td>5.80</td>
      <td>1.45</td>
      <td>2.63</td>
      <td>2.19</td>
      <td>5.18</td>
      <td>8.17</td>
      <td>3.39</td>
      <td>1.14</td>
      <td>7.95</td>
      <td>7.95</td>
      <td>8.52</td>
      <td>3.41</td>
      <td>3.98</td>
      <td>0.0</td>
      <td>3.45</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>3.45</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>20.69</td>
      <td>7.94</td>
      <td>7.70</td>
      <td>2.97</td>
      <td>6.88</td>
      <td>4.54</td>
      <td>0.68</td>
      <td>7.31</td>
      <td>8.04</td>
      <td>10.73</td>
      <td>5.58</td>
      <td>0.00</td>
      <td>7.52</td>
      <td>5.37</td>
      <td>6.18</td>
      <td>4.24</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5.98</td>
      <td>2.90</td>
      <td>1.94</td>
      <td>1.42</td>
      <td>0.10</td>
      <td>3.29</td>
      <td>0.81</td>
      <td>9.09</td>
      <td>8.04</td>
      <td>4.65</td>
      <td>2.65</td>
      <td>0.46</td>
      <td>2.25</td>
      <td>6.63</td>
      <td>0.00</td>
      <td>6.19</td>
      <td>5.06</td>
      <td>0.56</td>
      <td>4.80</td>
      <td>6.55</td>
      <td>3.49</td>
      <td>0.44</td>
      <td>0.00</td>
      <td>3.93</td>
      <td>10.62</td>
      <td>4.28</td>
      <td>3.83</td>
      <td>3.98</td>
      <td>7.52</td>
      <td>24.07</td>
      <td>9.26</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.47</td>
      <td>0.04</td>
      <td>10.31</td>
      <td>9.35</td>
      <td>4.45</td>
      <td>0.04</td>
      <td>...</td>
      <td>5.17</td>
      <td>7.54</td>
      <td>13.27</td>
      <td>3.82</td>
      <td>0.00</td>
      <td>0.34</td>
      <td>0.00</td>
      <td>10.48</td>
      <td>6.55</td>
      <td>0.44</td>
      <td>0.00</td>
      <td>12.83</td>
      <td>12.83</td>
      <td>5.31</td>
      <td>2.51</td>
      <td>1.33</td>
      <td>0.0</td>
      <td>4.63</td>
      <td>1.85</td>
      <td>0.0</td>
      <td>2.78</td>
      <td>2.78</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.93</td>
      <td>12.21</td>
      <td>6.96</td>
      <td>0.10</td>
      <td>4.20</td>
      <td>0.88</td>
      <td>0.19</td>
      <td>10.44</td>
      <td>7.17</td>
      <td>2.96</td>
      <td>6.75</td>
      <td>0.17</td>
      <td>5.40</td>
      <td>8.32</td>
      <td>5.40</td>
      <td>1.57</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>7.04</td>
      <td>8.92</td>
      <td>3.24</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>4.50</td>
      <td>0.00</td>
      <td>8.41</td>
      <td>6.97</td>
      <td>12.65</td>
      <td>9.24</td>
      <td>0.00</td>
      <td>0.54</td>
      <td>10.11</td>
      <td>0.06</td>
      <td>10.43</td>
      <td>12.13</td>
      <td>1.42</td>
      <td>13.91</td>
      <td>10.66</td>
      <td>3.73</td>
      <td>3.05</td>
      <td>0.00</td>
      <td>2.61</td>
      <td>2.86</td>
      <td>8.57</td>
      <td>10.00</td>
      <td>0.00</td>
      <td>8.57</td>
      <td>11.11</td>
      <td>3.70</td>
      <td>0.00</td>
      <td>3.70</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>9.51</td>
      <td>5.19</td>
      <td>6.20</td>
      <td>0.00</td>
      <td>...</td>
      <td>7.11</td>
      <td>8.28</td>
      <td>8.66</td>
      <td>5.33</td>
      <td>0.00</td>
      <td>0.13</td>
      <td>0.00</td>
      <td>6.01</td>
      <td>6.16</td>
      <td>0.08</td>
      <td>0.00</td>
      <td>10.00</td>
      <td>7.14</td>
      <td>11.43</td>
      <td>20.00</td>
      <td>2.86</td>
      <td>0.0</td>
      <td>7.41</td>
      <td>14.81</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>11.11</td>
      <td>8.47</td>
      <td>9.45</td>
      <td>0.00</td>
      <td>4.60</td>
      <td>0.43</td>
      <td>0.00</td>
      <td>3.77</td>
      <td>2.70</td>
      <td>6.84</td>
      <td>7.28</td>
      <td>0.00</td>
      <td>5.60</td>
      <td>7.52</td>
      <td>5.93</td>
      <td>3.89</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>13.30</td>
      <td>7.16</td>
      <td>5.69</td>
      <td>1.98</td>
      <td>0.71</td>
      <td>2.20</td>
      <td>0.22</td>
      <td>13.89</td>
      <td>1.75</td>
      <td>0.00</td>
      <td>3.27</td>
      <td>0.00</td>
      <td>10.26</td>
      <td>9.46</td>
      <td>4.42</td>
      <td>4.14</td>
      <td>4.08</td>
      <td>4.32</td>
      <td>4.22</td>
      <td>4.11</td>
      <td>3.83</td>
      <td>4.08</td>
      <td>4.36</td>
      <td>3.99</td>
      <td>7.96</td>
      <td>6.19</td>
      <td>5.31</td>
      <td>4.42</td>
      <td>11.50</td>
      <td>32.14</td>
      <td>3.57</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.58</td>
      <td>0.13</td>
      <td>5.83</td>
      <td>1.45</td>
      <td>11.60</td>
      <td>0.00</td>
      <td>...</td>
      <td>4.11</td>
      <td>4.15</td>
      <td>4.13</td>
      <td>4.20</td>
      <td>4.37</td>
      <td>4.30</td>
      <td>4.54</td>
      <td>4.15</td>
      <td>4.11</td>
      <td>4.28</td>
      <td>0.00</td>
      <td>9.73</td>
      <td>4.42</td>
      <td>3.54</td>
      <td>7.96</td>
      <td>4.42</td>
      <td>0.0</td>
      <td>7.14</td>
      <td>3.57</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>3.57</td>
      <td>2.95</td>
      <td>7.28</td>
      <td>0.00</td>
      <td>9.54</td>
      <td>5.71</td>
      <td>0.48</td>
      <td>4.77</td>
      <td>1.13</td>
      <td>2.67</td>
      <td>8.83</td>
      <td>0.00</td>
      <td>4.09</td>
      <td>4.14</td>
      <td>4.16</td>
      <td>4.23</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5.24</td>
      <td>5.09</td>
      <td>5.31</td>
      <td>1.05</td>
      <td>0.00</td>
      <td>4.24</td>
      <td>0.34</td>
      <td>5.25</td>
      <td>5.59</td>
      <td>9.58</td>
      <td>3.32</td>
      <td>0.00</td>
      <td>5.85</td>
      <td>10.89</td>
      <td>2.16</td>
      <td>6.13</td>
      <td>5.88</td>
      <td>4.22</td>
      <td>5.15</td>
      <td>5.54</td>
      <td>5.85</td>
      <td>4.68</td>
      <td>1.15</td>
      <td>1.54</td>
      <td>9.42</td>
      <td>7.48</td>
      <td>2.77</td>
      <td>3.88</td>
      <td>6.65</td>
      <td>13.64</td>
      <td>3.03</td>
      <td>4.55</td>
      <td>4.55</td>
      <td>0.24</td>
      <td>0.08</td>
      <td>8.67</td>
      <td>8.11</td>
      <td>4.08</td>
      <td>0.00</td>
      <td>...</td>
      <td>5.93</td>
      <td>6.01</td>
      <td>6.05</td>
      <td>6.07</td>
      <td>1.64</td>
      <td>3.17</td>
      <td>2.25</td>
      <td>6.44</td>
      <td>6.60</td>
      <td>2.91</td>
      <td>0.28</td>
      <td>9.14</td>
      <td>9.42</td>
      <td>4.99</td>
      <td>3.60</td>
      <td>2.49</td>
      <td>0.0</td>
      <td>15.15</td>
      <td>1.52</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>4.55</td>
      <td>9.09</td>
      <td>7.24</td>
      <td>7.63</td>
      <td>0.00</td>
      <td>5.11</td>
      <td>3.71</td>
      <td>0.00</td>
      <td>6.01</td>
      <td>4.16</td>
      <td>4.49</td>
      <td>6.14</td>
      <td>0.00</td>
      <td>6.18</td>
      <td>6.10</td>
      <td>6.06</td>
      <td>5.86</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 121 columns</p>
</div>




```
X
```




    array([[5.04, 5.77, 6.04, ..., 5.37, 6.18, 4.24],
           [5.98, 2.9 , 1.94, ..., 8.32, 5.4 , 1.57],
           [7.04, 8.92, 3.24, ..., 7.52, 5.93, 3.89],
           ...,
           [2.74, 5.42, 8.79, ..., 6.75, 4.98, 5.21],
           [7.51, 6.76, 4.73, ..., 4.94, 5.  , 4.93],
           [3.81, 6.79, 7.66, ..., 5.17, 5.06, 4.72]])




```
# Get X and y
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
```


```
# Split data into training and testing
from sklearn import model_selection

# Reserve 20% for testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

print('Training data has %d observation with %d features' % X_train.shape)
print('Test data has %d observation with %d features' % X_test.shape)
```

    Training data has 12000 observation with 120 features
    Test data has 3000 observation with 120 features



```
!pip install catboost
from catboost import CatBoostClassifier
classifier_cat = CatBoostClassifier(iterations = 100, task_type = 'GPU')
```


```
X_train.shape

```




    (12000, 120)




```
import timeit
def train_on_cpu():  
  model = CatBoostClassifier(
      iterations=1000,
      eval_metric = 'AUC',
      boosting_type = 'Plain',
      learning_rate=0.03
  )
  
  model.fit(
      X_train, y_train,
      eval_set=(X_test, y_test),
      verbose=100
  );   
      
cpu_time = timeit.timeit('train_on_cpu()', 
                         setup="from __main__ import train_on_cpu", 
                         number=1)

print('Time to fit model on CPU: {} sec'.format(int(cpu_time)))
```

    0:	learn: 0.6676061	test: 0.6675290	best: 0.6675290 (0)	total: 38ms	remaining: 37.9s
    100:	learn: 0.3157643	test: 0.3226858	best: 0.3226858 (100)	total: 4.22s	remaining: 37.6s
    200:	learn: 0.3003360	test: 0.3215573	best: 0.3214868 (186)	total: 8.91s	remaining: 35.4s
    300:	learn: 0.2866442	test: 0.3216649	best: 0.3214678 (221)	total: 13.5s	remaining: 31.5s
    400:	learn: 0.2729592	test: 0.3216150	best: 0.3214438 (359)	total: 18.2s	remaining: 27.1s
    500:	learn: 0.2592651	test: 0.3218496	best: 0.3214438 (359)	total: 22.8s	remaining: 22.7s
    600:	learn: 0.2460502	test: 0.3213318	best: 0.3212648 (585)	total: 27.4s	remaining: 18.2s
    700:	learn: 0.2329541	test: 0.3218720	best: 0.3212648 (585)	total: 32.1s	remaining: 13.7s
    800:	learn: 0.2210274	test: 0.3218479	best: 0.3212648 (585)	total: 36.8s	remaining: 9.15s
    900:	learn: 0.2093181	test: 0.3221360	best: 0.3212648 (585)	total: 41.6s	remaining: 4.57s
    999:	learn: 0.1988029	test: 0.3234133	best: 0.3212648 (585)	total: 46.3s	remaining: 0us
    
    bestTest = 0.3212648222
    bestIteration = 585
    
    Shrink model to first 586 iterations.
    Time to fit model on CPU: 46 sec



```
def train_on_gpu():  
  model = CatBoostClassifier(
      iterations=1000,
      learning_rate=0.03,
      eval_metric = 'AUC',
      boosting_type = 'Plain',
      task_type='GPU'
  )
  
  model.fit(
      X_train, y_train,
      eval_set=(X_test, y_test),
      verbose=100
  );     
      
gpu_time = timeit.timeit('train_on_gpu()', 
                         setup="from __main__ import train_on_gpu", 
                         number=1)

print('Time to fit model on GPU: {} sec'.format(int(gpu_time)))
print('GPU speedup over CPU: ' + '%.2f' % (cpu_time/gpu_time) + 'x')
```

    0:	learn: 0.5710810	test: 0.5505786	best: 0.5505786 (0)	total: 22.9ms	remaining: 22.9s
    100:	learn: 0.8201020	test: 0.5631405	best: 0.5827100 (49)	total: 2.35s	remaining: 20.9s
    200:	learn: 0.8866887	test: 0.5592176	best: 0.5827100 (49)	total: 4.79s	remaining: 19.1s
    300:	learn: 0.9317816	test: 0.5518306	best: 0.5827100 (49)	total: 7.2s	remaining: 16.7s
    400:	learn: 0.9584097	test: 0.5486118	best: 0.5827100 (49)	total: 9.55s	remaining: 14.3s
    500:	learn: 0.9745375	test: 0.5483546	best: 0.5827100 (49)	total: 12s	remaining: 12s
    600:	learn: 0.9842969	test: 0.5495813	best: 0.5827100 (49)	total: 14.5s	remaining: 9.62s
    700:	learn: 0.9908361	test: 0.5486887	best: 0.5827100 (49)	total: 16.8s	remaining: 7.17s
    800:	learn: 0.9940755	test: 0.5510603	best: 0.5827100 (49)	total: 19.2s	remaining: 4.76s
    900:	learn: 0.9961302	test: 0.5465447	best: 0.5827100 (49)	total: 21.5s	remaining: 2.36s
    999:	learn: 0.9976045	test: 0.5430479	best: 0.5827100 (49)	total: 23.9s	remaining: 0us
    bestTest = 0.5827099681
    bestIteration = 49
    Shrink model to first 50 iterations.
    Time to fit model on GPU: 26 sec
    GPU speedup over CPU: 1.79x



```

```


```
import lightgbm as lgb
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'learning_rate': 0.03
}

```


```
# Use time function to measure time elapsed
import time
start = time.time()


gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_train)

end = time.time()
print(end - start)
```

# 3. Customer Satisfaction


```
from google.colab import files
uploaded = files.upload()
```



     <input type="file" id="files-e5655230-bf1b-40c8-8ab5-975d6b884627" name="files[]" multiple disabled />
     <output id="result-e5655230-bf1b-40c8-8ab5-975d6b884627">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script src="/nbextensions/google.colab/files.js"></script> 


    Saving customer_satisfaction.csv to customer_satisfaction.csv



```
import io
import pandas as pd

data = pd.read_csv(io.BytesIO(uploaded['customer_satisfaction.csv']))
data.head()
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
      <th>ID</th>
      <th>var3</th>
      <th>var15</th>
      <th>imp_ent_var16_ult1</th>
      <th>imp_op_var39_comer_ult1</th>
      <th>imp_op_var39_comer_ult3</th>
      <th>imp_op_var40_comer_ult1</th>
      <th>imp_op_var40_comer_ult3</th>
      <th>imp_op_var40_efect_ult1</th>
      <th>imp_op_var40_efect_ult3</th>
      <th>imp_op_var40_ult1</th>
      <th>imp_op_var41_comer_ult1</th>
      <th>imp_op_var41_comer_ult3</th>
      <th>imp_op_var41_efect_ult1</th>
      <th>imp_op_var41_efect_ult3</th>
      <th>imp_op_var41_ult1</th>
      <th>imp_op_var39_efect_ult1</th>
      <th>imp_op_var39_efect_ult3</th>
      <th>imp_op_var39_ult1</th>
      <th>imp_sal_var16_ult1</th>
      <th>ind_var1_0</th>
      <th>ind_var1</th>
      <th>ind_var2_0</th>
      <th>ind_var2</th>
      <th>ind_var5_0</th>
      <th>ind_var5</th>
      <th>ind_var6_0</th>
      <th>ind_var6</th>
      <th>ind_var8_0</th>
      <th>ind_var8</th>
      <th>ind_var12_0</th>
      <th>ind_var12</th>
      <th>ind_var13_0</th>
      <th>ind_var13_corto_0</th>
      <th>ind_var13_corto</th>
      <th>ind_var13_largo_0</th>
      <th>ind_var13_largo</th>
      <th>ind_var13_medio_0</th>
      <th>ind_var13_medio</th>
      <th>ind_var13</th>
      <th>...</th>
      <th>saldo_medio_var5_ult1</th>
      <th>saldo_medio_var5_ult3</th>
      <th>saldo_medio_var8_hace2</th>
      <th>saldo_medio_var8_hace3</th>
      <th>saldo_medio_var8_ult1</th>
      <th>saldo_medio_var8_ult3</th>
      <th>saldo_medio_var12_hace2</th>
      <th>saldo_medio_var12_hace3</th>
      <th>saldo_medio_var12_ult1</th>
      <th>saldo_medio_var12_ult3</th>
      <th>saldo_medio_var13_corto_hace2</th>
      <th>saldo_medio_var13_corto_hace3</th>
      <th>saldo_medio_var13_corto_ult1</th>
      <th>saldo_medio_var13_corto_ult3</th>
      <th>saldo_medio_var13_largo_hace2</th>
      <th>saldo_medio_var13_largo_hace3</th>
      <th>saldo_medio_var13_largo_ult1</th>
      <th>saldo_medio_var13_largo_ult3</th>
      <th>saldo_medio_var13_medio_hace2</th>
      <th>saldo_medio_var13_medio_hace3</th>
      <th>saldo_medio_var13_medio_ult1</th>
      <th>saldo_medio_var13_medio_ult3</th>
      <th>saldo_medio_var17_hace2</th>
      <th>saldo_medio_var17_hace3</th>
      <th>saldo_medio_var17_ult1</th>
      <th>saldo_medio_var17_ult3</th>
      <th>saldo_medio_var29_hace2</th>
      <th>saldo_medio_var29_hace3</th>
      <th>saldo_medio_var29_ult1</th>
      <th>saldo_medio_var29_ult3</th>
      <th>saldo_medio_var33_hace2</th>
      <th>saldo_medio_var33_hace3</th>
      <th>saldo_medio_var33_ult1</th>
      <th>saldo_medio_var33_ult3</th>
      <th>saldo_medio_var44_hace2</th>
      <th>saldo_medio_var44_hace3</th>
      <th>saldo_medio_var44_ult1</th>
      <th>saldo_medio_var44_ult3</th>
      <th>var38</th>
      <th>TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>23</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>39205.170000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>2</td>
      <td>34</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>300.0</td>
      <td>122.22</td>
      <td>300.0</td>
      <td>240.75</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>49278.030000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>2</td>
      <td>23</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3.00</td>
      <td>2.07</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>67333.770000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>2</td>
      <td>37</td>
      <td>0.0</td>
      <td>195.0</td>
      <td>195.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>195.0</td>
      <td>195.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>195.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>195.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>91.56</td>
      <td>138.84</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>64007.970000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>2</td>
      <td>39</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>40501.08</td>
      <td>13501.47</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>85501.89</td>
      <td>85501.89</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117310.979016</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 371 columns</p>
</div>




```
# Get X and y
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
```


```
# Split data into training and testing
from sklearn import model_selection

# Reserve 20% for testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

print('Training data has %d observation with %d features' % X_train.shape)
print('Test data has %d observation with %d features' % X_test.shape)
```

    Training data has 60816 observation with 370 features
    Test data has 15204 observation with 370 features


## 3.1 CatBoost

### Train in CPU


```
from catboost import CatBoostClassifier
classifier_cat = CatBoostClassifier(iterations = 100, task_type = 'GPU')
```


```
import timeit
def train_on_cpu():  
  model = CatBoostClassifier(
      iterations=1000,
      learning_rate=0.03,
      boosting_type = 'Plain'
  )
  
  model.fit(
      X_train, y_train,
      eval_set=(X_test, y_test),
      verbose=100
  );   
      
cpu_time = timeit.timeit('train_on_cpu()', 
                         setup="from __main__ import train_on_cpu", 
                         number=1)

print('Time to fit model on CPU: {} sec'.format(int(cpu_time)))
```

    0:	learn: 0.6499884	test: 0.6502590	best: 0.6502590 (0)	total: 97.6ms	remaining: 1m 37s
    100:	learn: 0.1356915	test: 0.1439829	best: 0.1439829 (100)	total: 11.9s	remaining: 1m 46s
    200:	learn: 0.1312296	test: 0.1412094	best: 0.1412094 (200)	total: 23.3s	remaining: 1m 32s
    300:	learn: 0.1295643	test: 0.1405838	best: 0.1405772 (299)	total: 34.2s	remaining: 1m 19s
    400:	learn: 0.1280686	test: 0.1401968	best: 0.1401918 (399)	total: 45.2s	remaining: 1m 7s
    500:	learn: 0.1265318	test: 0.1399162	best: 0.1399123 (495)	total: 56.4s	remaining: 56.2s
    600:	learn: 0.1251861	test: 0.1397071	best: 0.1397071 (600)	total: 1m 7s	remaining: 44.8s
    700:	learn: 0.1239926	test: 0.1396878	best: 0.1396694 (695)	total: 1m 18s	remaining: 33.5s
    800:	learn: 0.1225489	test: 0.1395715	best: 0.1395640 (797)	total: 1m 30s	remaining: 22.4s
    900:	learn: 0.1212882	test: 0.1395614	best: 0.1395459 (852)	total: 1m 41s	remaining: 11.2s
    999:	learn: 0.1201106	test: 0.1395218	best: 0.1395129 (989)	total: 1m 52s	remaining: 0us
    
    bestTest = 0.1395128933
    bestIteration = 989
    
    Shrink model to first 990 iterations.
    Time to fit model on CPU: 117 sec


### Train in GPU


```
def train_on_gpu():  
  model = CatBoostClassifier(
      iterations=1000,
      learning_rate=0.03,
      task_type='GPU'
  )
  
  model.fit(
      X_train, y_train,
      eval_set=(X_test, y_test),
      verbose=100
  );     
      
gpu_time = timeit.timeit('train_on_gpu()', 
                         setup="from __main__ import train_on_gpu", 
                         number=1)

print('Time to fit model on GPU: {} sec'.format(int(gpu_time)))
print('GPU speedup over CPU: ' + '%.2f' % (cpu_time/gpu_time) + 'x')
```

    0:	learn: 0.6420095	test: 0.6421342	best: 0.6421342 (0)	total: 12.8ms	remaining: 12.8s
    100:	learn: 0.1350072	test: 0.1434604	best: 0.1434604 (100)	total: 1.01s	remaining: 9.04s
    200:	learn: 0.1307862	test: 0.1409726	best: 0.1409726 (200)	total: 1.88s	remaining: 7.46s
    300:	learn: 0.1288554	test: 0.1404405	best: 0.1404258 (294)	total: 2.75s	remaining: 6.4s
    400:	learn: 0.1271075	test: 0.1399712	best: 0.1399712 (400)	total: 3.63s	remaining: 5.43s
    500:	learn: 0.1255655	test: 0.1397720	best: 0.1397680 (486)	total: 4.51s	remaining: 4.5s
    600:	learn: 0.1243287	test: 0.1396310	best: 0.1396310 (600)	total: 5.57s	remaining: 3.7s
    700:	learn: 0.1229532	test: 0.1396375	best: 0.1395841 (667)	total: 6.65s	remaining: 2.84s
    800:	learn: 0.1216558	test: 0.1396176	best: 0.1395823 (772)	total: 7.75s	remaining: 1.93s
    900:	learn: 0.1204271	test: 0.1396710	best: 0.1395823 (772)	total: 8.86s	remaining: 973ms
    999:	learn: 0.1191901	test: 0.1397230	best: 0.1395823 (772)	total: 10s	remaining: 0us
    bestTest = 0.1395823121
    bestIteration = 772
    Shrink model to first 773 iterations.
    Time to fit model on GPU: 15 sec
    GPU speedup over CPU: 7.48x



```

```

## 3.2 LGBM

### Train in LGBM


```
import lightgbm as lgb
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'learning_rate': 0.03
}

```


```
# Use time function to measure time elapsed
import time
start = time.time()


gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_train)

end = time.time()
print(end - start)
```
