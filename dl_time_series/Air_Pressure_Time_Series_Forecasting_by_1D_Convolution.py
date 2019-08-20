#!/usr/bin/env python
# coding: utf-8

# In this notebook, we will use a multi-layer perceptron to develop time series forecasting models.
# The dataset used for the examples of this notebook is on air pollution measured by concentration of
# particulate matter (PM) of diameter less than or equal to 2.5 micrometers. There are other variables
# such as air pressure, air temparature, dewpoint and so on.
# Two time series models are developed - one on air pressure and the other on pm2.5.
# The dataset has been downloaded from UCI Machine Learning Repository.
# https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data

# In[4]:


from __future__ import print_function
import os
import sys
import pandas as pd
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns
import datetime
import keras
import tensorflow as tf
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)


# In[5]:


#set current working directory
#os.chdir('D:/Practical Time Series')


# In[6]:


#Read the dataset into a pandas.DataFrame
df = pd.read_csv('../datasets/PRSA_data_2010.1.1-2014.12.31.csv')


# In[7]:


print('Shape of the dataframe:', df.shape)


# In[8]:


#Let's see the first five rows of the DataFrame
df.head()


# To make sure that the rows are in the right order of date and time of observations,
# a new column datetime is created from the date and time related columns of the DataFrame.
# The new column consists of Python's datetime.datetime objects. The DataFrame is sorted in ascending order
# over this column.

# In[9]:


df['datetime'] = df[['year', 'month', 'day', 'hour']].apply(lambda row: datetime.datetime(year=row['year'], month=row['month'], day=row['day'],
                                                                                          hour=row['hour']), axis=1)
df.sort_values('datetime', ascending=True, inplace=True)


# In[ ]:


#Let us draw a box plot to visualize the central tendency and dispersion of PRES
plt.figure(figsize=(5.5, 5.5))
g = sns.boxplot(df['PRES'])
g.set_title('Box plot of PRES')


# In[ ]:


plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df['PRES'])
g.set_title('Time series of PRES')
g.set_xlabel('Index')
g.set_ylabel('PRES readings')


# Gradient descent algorithms perform better (for example converge faster) if the variables are wihtin range [-1, 1]. Many sources relax the boundary to even [-3, 3]. The PRES variable is mixmax scaled to bound the tranformed variable within [0,1].

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_PRES'] = scaler.fit_transform(np.array(df['PRES']).reshape(-1, 1))


# Before training the model, the dataset is split in two parts - train set and validation set.
# The neural network is trained on the train set. This means computation of the loss function, back propagation
# and weights updated by a gradient descent algorithm is done on the train set. The validation set is
# used to evaluate the model and to determine the number of epochs in model training. Increasing the number of 
# epochs will further decrease the loss function on the train set but might not neccesarily have the same effect
# for the validation set due to overfitting on the train set.Hence, the number of epochs is controlled by keeping
# a tap on the loss function computed for the validation set. We use Keras with Tensorflow backend to define and train
# the model. All the steps involved in model training and validation is done by calling appropriate functions
# of the Keras API.

# In[11]:


"""
Let's start by splitting the dataset into train and validation. The dataset's time period if from
Jan 1st, 2010 to Dec 31st, 2014. The first fours years - 2010 to 2013 is used as train and
2014 is kept for validation.
"""
split_date = datetime.datetime(year=2014, month=1, day=1, hour=0)
df_train = df.loc[df['datetime']<split_date]
df_val = df.loc[df['datetime']>=split_date]
print('Shape of train:', df_train.shape)
print('Shape of test:', df_val.shape)


# In[12]:


#First five rows of train
df_train.head()


# In[13]:


#First five rows of validation
df_val.head()


# In[14]:


#Reset the indices of the validation set
df_val.reset_index(drop=True, inplace=True)


# In[15]:


"""
The train and validation time series of scaled PRES is also plotted.
"""

plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_train['scaled_PRES'], color='b')
g.set_title('Time series of scaled PRES in train set')
g.set_xlabel('Index')
g.set_ylabel('Scaled PRES readings')

plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_val['scaled_PRES'], color='r')
g.set_title('Time series of scaled PRES in validation set')
g.set_xlabel('Index')
g.set_ylabel('Scaled PRES readings')


# Now we need to generate regressors (X) and target variable (y) for train and validation. 2-D array of regressor and 1-D array of target is created from the original 1-D array of columm standardized_PRES in the DataFrames. For the time series forecasting model, Past seven days of observations are used to predict for the next day. This is equivalent to a AR(7) model. We define a function which takes the original time series and the number of timesteps in regressors as input to generate the arrays of X and y.

# In[16]:


def makeXy(ts, nb_timesteps):
    """
    Input: 
           ts: original time series
           nb_timesteps: number of time steps in the regressors
    Output: 
           X: 2-D array of regressors
           y: 1-D array of target 
    """
    X = []
    y = []
    for i in range(nb_timesteps, ts.shape[0]):
        X.append(list(ts.loc[i-nb_timesteps:i-1]))
        y.append(ts.loc[i])
    X, y = np.array(X), np.array(y)
    return X, y


# In[17]:


X_train, y_train = makeXy(df_train['scaled_PRES'], 7)
print('Shape of train arrays:', X_train.shape, y_train.shape)


# In[18]:


X_val, y_val = makeXy(df_val['scaled_PRES'], 7)
print('Shape of validation arrays:', X_val.shape, y_val.shape)


# The input to convolution layers must be of shape (number of samples, number of timesteps, number of features per timestep). In this case we are modeling only PRES hence number of features per timestep is one. Number of timesteps is seven and number of samples is same as the number of samples in X_train and X_val, which are reshaped to 3D arrays.

# In[19]:


#X_train and X_val are reshaped to 3D arrays
X_train, X_val = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)),                 X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
print('Shape of arrays after reshaping:', X_train.shape, X_val.shape)


# Now we define the MLP using the Keras Functional API. In this approach a layer can be declared as the input of the following layer at the time of defining the next layer. 

# In[20]:


from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import ZeroPadding1D
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import AveragePooling1D
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


# In[21]:


#Define input layer which has shape (None, 7) and of type float32. None indicates the number of instances
input_layer = Input(shape=(7,1), dtype='float32')


# ZeroPadding1D layer is added next to add zeros at the beginning and end of each series. Zeropadding ensure that the downstream convolution layer does not reduce the dimension of the output sequences. Pooling layer, added after the convolution layer is used to downsampling the input.

# In[22]:


#Add zero padding
zeropadding_layer = ZeroPadding1D(padding=1)(input_layer)


# The first argument of Conv1D is the number of filters, which determine the number of features in the output. Second argument indicates length of the 1D convolution window. The third argument is strides and represent the number of places to shift the convolution window. Lastly, setting use_bias as True, add a bias value during computation of an output feature. Here, the 1D convolution can be thought of as generating local AR models over rolling window of three time units.

# In[23]:


#Add 1D convolution layer
conv1D_layer = Conv1D(64, 3, strides=1, use_bias=True)(zeropadding_layer)


# AveragePooling1D is added next to downsample the input by taking average over pool size of three with stride of one timesteps. The average pooling in this case can be thought of as taking moving averages over a rolling window of three time units. We have used average pooling instead of max pooling to generate the moving averages.

# In[24]:


#Add AveragePooling1D layer
avgpooling_layer = AveragePooling1D(pool_size=3, strides=1)(conv1D_layer)


# The preceeding pooling layer returns 3D output. Hence before passing to the output layer, a Flatten layer is added. The Flatten layer reshapes the input to (number of samples, number of timesteps*number of features per timestep), which is then fed to the output layer

# In[25]:


#Add Flatten layer
flatten_layer = Flatten()(avgpooling_layer)


# In[26]:


dropout_layer = Dropout(0.2)(flatten_layer)


# In[27]:


#Finally the output layer gives prediction for the next day's air pressure.
output_layer = Dense(1, activation='linear')(dropout_layer)


# The input, dense and output layers will now be packed inside a Model, which is wrapper class for training and making
# predictions. Mean squared error (MSE) is used as the loss function.
# 
# The network's weights are optimized by the Adam algorithm. Adam stands for adaptive moment estimation
# and has been a popular choice for training deep neural networks. Unlike, stochastic gradient descent, adam uses
# different learning rates for each weight and separately updates the same as the training progresses. The learning rate of a weight is updated based on exponentially weighted moving averages of the weight's gradients and the squared gradients.

# In[28]:


ts_model = Model(inputs=input_layer, outputs=output_layer)
ts_model.compile(loss='mean_absolute_error', optimizer='adam')#SGD(lr=0.001, decay=1e-5))
ts_model.summary()


# The model is trained by calling the fit function on the model object and passing the X_train and y_train. The training 
# is done for a predefined number of epochs. Additionally, batch_size defines the number of samples of train set to be
# used for a instance of back propagation.The validation dataset is also passed to evaluate the model after every epoch
# completes. A ModelCheckpoint object tracks the loss function on the validation set and saves the model for the epoch,
# at which the loss function has been minimum.

# In[ ]:


save_weights_at = os.path.join('keras_models', 'PRSA_data_Air_Pressure_1DConv_weights.{epoch:02d}-{val_loss:.4f}.hdf5')
save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min',
                            period=1)
ts_model.fit(x=X_train, y=y_train, batch_size=16, epochs=20,
             verbose=1, callbacks=[save_best], validation_data=(X_val, y_val),
             shuffle=True)


# Prediction are made for the PRES from the best saved model. The model's predictions, which are on the standardized  PRES, are inverse transformed to get predictions of original PRES.

# In[30]:


best_model = load_model(os.path.join('keras_models', 'PRSA_data_Air_Pressure_1DConv_weights.16-0.0097.hdf5'))
preds = best_model.predict(X_val)
pred_PRES = np.squeeze(scaler.inverse_transform(preds))


# In[31]:


from sklearn.metrics import r2_score


# In[32]:


r2 = r2_score(df_val['PRES'].loc[7:], pred_PRES)
print('R-squared for the validation set:', round(r2, 4))


# In[34]:


#Let's plot the first 50 actual and predicted values of PRES.
plt.figure(figsize=(5.5, 5.5))
plt.plot(range(50), df_val['PRES'].loc[7:56], linestyle='-', marker='*', color='r')
plt.plot(range(50), pred_PRES[:50], linestyle='-', marker='.', color='b')
plt.legend(['Actual','Predicted'], loc=2)
plt.title('Actual vs Predicted PRES')
plt.ylabel('PRES')
plt.xlabel('Index')

