
# coding: utf-8

# # Introduction to the Data Set

# In[1]:


import pandas as pd

cars = pd.read_csv("imports-85.data")
cars.head()


# It looks like there are problems with columns names. We need to fix this.

# In[2]:


cols = ['symboling','normalized_losses','make','fuel_type','aspiration','num_doors','body_style','drive_wheels','engine_location','wheel_base','length','width','height','curb_weight','engine_type','num_cylinders','engine_size','fuel_system','bore','stroke','compression_ratio','horsepower','peak_rpm','city_mpg','highway_mpg','price']
cars = pd.read_csv('imports-85.data', names=cols)
cars.head()


# In[3]:


cars.dtypes


# symboling, wheel_base, length, width, height, curb_weights, engine_size, compression_ratio, city_mpg, highway_mpg can be used as numeric features, price is the target column

# In[4]:


# Lets select only the numeric columns
numeric_cols = ['normalized_losses', 'wheel_base', 'length', 'width', 'height', 'curb_weight', 'bore', 'stroke', 'compression_ratio', 'horsepower', 'peak_rpm', 'city_mpg', 'highway_mpg', 'price']
cars = cars[numeric_cols]
cars.head()


# # Data Cleaning

# In[5]:


cars.dtypes


# In[6]:


import numpy as np
cars = cars.replace("?", np.nan)
cars.info()


# In[7]:


cars = cars.astype('float')


# In[8]:


# Lets see how many missing values each column has
cars.isnull().sum()


# In[9]:


cars = cars.dropna(subset=['price'])
cars.isnull().sum()


# In[10]:


cars = cars.fillna(cars.mean())
cars.info()


# In[11]:


normalized_cars = (cars - cars.mean()) / cars.std()
normalized_cars['price'] = cars['price']
normalized_cars.head()


# # Univariate Model

# In[12]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def knn_train_test(train_col, target_col, df):
    knn = KNeighborsRegressor()
    
    # Randomizing and reordering the rows
    np.random.seed(1)
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)
    
    # Trainig a Holdout Validation Technique
    last_train_row = int(len(rand_df) / 2)
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    # Predicting the price
    knn.fit(train_df[[train_col]], train_df[target_col])
    predicted_labels = knn.predict(test_df[[train_col]])
    
    # Calculate the error metric: RMSE
    mse = mean_squared_error(test_df[target_col], predicted_labels)
    rmse = np.sqrt(mse)
    return rmse


# In[13]:


# Which columns performed the best?
rmse_results = {}
train_cols = cars.columns.drop('price')
train_cols


# In[14]:


for col in train_cols:
    rmse_val = knn_train_test(col, 'price', cars)
    rmse_results[col] = rmse_val
    
rmse_results


# In[15]:


rmse_results_series = pd.Series(rmse_results)
rmse_results_series.sort_values()


# In[16]:


def knn_train_test(train_col, target_col, df):
    knn = KNeighborsRegressor()
    
    # Randomizing and reordering the rows
    np.random.seed(1)
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)
    
    # Trainig a Holdout Validation Technique
    last_train_row = int(len(rand_df) / 2)
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    k_values = [1,3,5,7,9]
    k_rmses = {}
    
    for k in k_values:
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[[train_col]], train_df[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_df[[train_col]])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)
        
        k_rmses[k] = rmse
    return k_rmses                                        


# In[17]:


k_rmse_results = {}
train_cols = cars.columns.drop('price')
for col in train_cols:
    rmse_val = knn_train_test(col, 'price', cars)
    k_rmse_results[col] = rmse_val

k_rmse_results


# In[18]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

for k,v in k_rmse_results.items():
    x = list(v.keys())
    y = list(v.values())
    
    plt.plot(x,y)
    plt.xlabel('k value')
    plt.ylabel('RMSE')


# In[19]:


# Compute average RMSE across different `k` values for each feature.
feature_avg_rmse = {}
for k,v in k_rmse_results.items():
    avg_rmse = np.mean(list(v.values()))
    feature_avg_rmse[k] = avg_rmse
series_avg_rmse = pd.Series(feature_avg_rmse)
series_avg_rmse.sort_values()


# # Multivariate Model

# In[20]:


def knn_train_test(train_cols, target_col, df):
    # Randomizing the reordering the rows
    np.random.seed(1)
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)
    
    # Holdout Validation Technique
    last_train_row = int(len(rand_df) / 2)
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    k_values = [5]
    k_rmses = {}
    
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[train_cols], train_df[target_col])
        predicted_labels = knn.predict(test_df[train_cols])
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)
        
        k_rmses[k] = rmse
    return k_rmses

k_rmse_results = {}


two_best_features = ['horsepower', 'width']
rmse_val = knn_train_test(two_best_features, 'price', cars)
k_rmse_results['two_best_features'] = rmse_val


# In[21]:


three_best_features = ['horsepower', 'width', 'curb_weight']
rmse_val = knn_train_test(two_best_features, 'price', cars)
k_rmse_results['three_best_features'] = rmse_val


# In[22]:


four_best_features = ['horsepower', 'width', 'curb_weight', 'city_mpg']
rmse_val = knn_train_test(four_best_features, 'price', cars)
k_rmse_results["four_best_features"] = rmse_val


# In[23]:


five_best_features = ['horsepower', 'width', 'curb_weight' , 'city_mpg' , 'highway_mpg']
rmse_val = knn_train_test(five_best_features, 'price', cars)
k_rmse_results["five_best_features"] = rmse_val


# In[24]:


k_rmse_results


# # Hyperparameter Tuning

# In[25]:


def knn_train_test(train_cols, target_col, df):
    # Randomizing the reordering the rows
    np.random.seed(1)
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)
    
    # Holdout Validation Technique
    last_train_row = int(len(rand_df) / 2)
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    k_values = [i for i in range(1, 25)]
    k_rmses = {}
    
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[train_cols], train_df[target_col])
        predicted_labels = knn.predict(test_df[train_cols])
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)
        
        k_rmses[k] = rmse
    return k_rmses

k_rmse_results = {}


three_best_features = ['horsepower', 'width', 'curb_weight']
rmse_val = knn_train_test(three_best_features, 'price', cars)
k_rmse_results["three best features"] = rmse_val

four_best_features = ['horsepower', 'width', 'curb_weight', 'city_mpg']
rmse_val = knn_train_test(four_best_features, 'price', cars)
k_rmse_results["four best features"] = rmse_val

five_best_features = ['horsepower', 'width', 'curb_weight' , 'city_mpg' , 'highway_mpg']
rmse_val = knn_train_test(five_best_features, 'price', cars)
k_rmse_results["five best features"] = rmse_val

k_rmse_results


# For the five_best_features, when k=1, the rmse has the lowest value(4509). For the four_best_features, when k=3, the rmse has the lowest value(3980). For the three_best_features, when k=1, the rmse has the lowest value(3981).

# In[26]:


for k,v in k_rmse_results.items():
    x = list(v.keys())
    y = list(v.values())
    
    plt.plot(x,y)
    plt.xlabel('k value')
    plt.ylabel('RMSE')

