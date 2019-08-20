{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this notebook, we will use Gated Recurrent Unit RNN to develop a time series forecasting model.\n",
    "The dataset used for the examples of this notebook is on air pollution measured by concentration of\n",
    "particulate matter (PM) of diameter less than or equal to 2.5 micrometers. There are other variables\n",
    "such as air pressure, air temparature, dewpoint and so on.\n",
    "Two time series models are developed - one on air pressure and the other on pm2.5.\n",
    "The dataset has been downloaded from UCI Machine Learning Repository.\n",
    "https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from __future__ import print_function\n",
    "import os\n",
    "import sys\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "%matplotlib inline\n",
    "from matplotlib import pyplot as plt\n",
    "import seaborn as sns\n",
    "import datetime"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "#set current working directory\n",
    "#os.chdir('D:/Practical Time Series')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Read the dataset into a pandas.DataFrame\n",
    "df = pd.read_csv('../datasets/PRSA_data_2010.1.1-2014.12.31.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape of the dataframe: (43824, 13)\n"
     ]
    }
   ],
   "source": [
    "print('Shape of the dataframe:', df.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>No</th>\n",
       "      <th>year</th>\n",
       "      <th>month</th>\n",
       "      <th>day</th>\n",
       "      <th>hour</th>\n",
       "      <th>pm2.5</th>\n",
       "      <th>DEWP</th>\n",
       "      <th>TEMP</th>\n",
       "      <th>PRES</th>\n",
       "      <th>cbwd</th>\n",
       "      <th>Iws</th>\n",
       "      <th>Is</th>\n",
       "      <th>Ir</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>2010</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>-21</td>\n",
       "      <td>-11.0</td>\n",
       "      <td>1021.0</td>\n",
       "      <td>NW</td>\n",
       "      <td>1.79</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2010</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>-21</td>\n",
       "      <td>-12.0</td>\n",
       "      <td>1020.0</td>\n",
       "      <td>NW</td>\n",
       "      <td>4.92</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>2010</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>NaN</td>\n",
       "      <td>-21</td>\n",
       "      <td>-11.0</td>\n",
       "      <td>1019.0</td>\n",
       "      <td>NW</td>\n",
       "      <td>6.71</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>2010</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>NaN</td>\n",
       "      <td>-21</td>\n",
       "      <td>-14.0</td>\n",
       "      <td>1019.0</td>\n",
       "      <td>NW</td>\n",
       "      <td>9.84</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>2010</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>NaN</td>\n",
       "      <td>-20</td>\n",
       "      <td>-12.0</td>\n",
       "      <td>1018.0</td>\n",
       "      <td>NW</td>\n",
       "      <td>12.97</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   No  year  month  day  hour  pm2.5  DEWP  TEMP    PRES cbwd    Iws  Is  Ir\n",
       "0   1  2010      1    1     0    NaN   -21 -11.0  1021.0   NW   1.79   0   0\n",
       "1   2  2010      1    1     1    NaN   -21 -12.0  1020.0   NW   4.92   0   0\n",
       "2   3  2010      1    1     2    NaN   -21 -11.0  1019.0   NW   6.71   0   0\n",
       "3   4  2010      1    1     3    NaN   -21 -14.0  1019.0   NW   9.84   0   0\n",
       "4   5  2010      1    1     4    NaN   -20 -12.0  1018.0   NW  12.97   0   0"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Let's see the first five rows of the DataFrame\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To make sure that the rows are in the right order of date and time of observations,\n",
    "a new column datetime is created from the date and time related columns of the DataFrame.\n",
    "The new column consists of Python's datetime.datetime objects. The DataFrame is sorted in ascending order\n",
    "over this column."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['datetime'] = df[['year', 'month', 'day', 'hour']].apply(lambda row: datetime.datetime(year=row['year'], month=row['month'], day=row['day'],\n",
    "                                                                                          hour=row['hour']), axis=1)\n",
    "df.sort_values('datetime', ascending=True, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Box plot of Air Pressure')"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 396x396 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Let us draw a box plot to visualize the central tendency and dispersion of PRES\n",
    "plt.figure(figsize=(5.5, 5.5))\n",
    "g = sns.boxplot(df['PRES'])\n",
    "g.set_title('Box plot of Air Pressure')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/manmeet/anaconda3/envs/py35/lib/python3.5/site-packages/seaborn/timeseries.py:183: UserWarning: The `tsplot` function is deprecated and will be removed in a future release. Please update your code to use the new `lineplot` function.\n",
      "  warnings.warn(msg, UserWarning)\n",
      "/home/manmeet/.local/lib/python3.5/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.\n",
      "  return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Air Pressure readings in hPa')"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 396x396 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(5.5, 5.5))\n",
    "g = sns.tsplot(df['PRES'])\n",
    "g.set_title('Time series of Air Pressure')\n",
    "g.set_xlabel('Index')\n",
    "g.set_ylabel('Air Pressure readings in hPa')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Gradient descent algorithms perform better (for example converge faster) if the variables are wihtin range [-1, 1]. The PRES variable is minmax scaled as shown in the following cell in order to bound the scaled features within [0, 1]."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import MinMaxScaler\n",
    "scaler = MinMaxScaler(feature_range=(0, 1))\n",
    "df['scaled_PRES'] = scaler.fit_transform(np.array(df['PRES']).reshape(-1, 1))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "Before training the model, the dataset is split in two parts - train set and validation set.\n",
    "The neural network is trained on the train set. This means computation of the loss function, back propagation\n",
    "and weights updated by a gradient descent algorithm is done on the train set. The validation set is\n",
    "used to evaluate the model and to determine the number of epochs in model training. Increasing the number of \n",
    "epochs will further decrease the loss function on the train set but might not neccesarily have the same effect\n",
    "for the validation set due to overfitting on the train set.Hence, the number of epochs is controlled by keeping\n",
    "a tap on the loss function computed for the validation set. We use Keras with Tensorflow backend to define and train\n",
    "the model. All the steps involved in model training and validation is done by calling appropriate functions\n",
    "of the Keras API."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape of train: (35064, 15)\n",
      "Shape of test: (8760, 15)\n"
     ]
    }
   ],
   "source": [
    "\"\"\"\n",
    "Let's start by splitting the dataset into train and validation. The dataset's time period if from\n",
    "Jan 1st, 2010 to Dec 31st, 2014. The first fours years - 2010 to 2013 is used as train and\n",
    "2014 is kept for validation.\n",
    "\"\"\"\n",
    "split_date = datetime.datetime(year=2014, month=1, day=1, hour=0)\n",
    "df_train = df.loc[df['datetime']<split_date]\n",
    "df_val = df.loc[df['datetime']>=split_date]\n",
    "print('Shape of train:', df_train.shape)\n",
    "print('Shape of test:', df_val.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>No</th>\n",
       "      <th>year</th>\n",
       "      <th>month</th>\n",
       "      <th>day</th>\n",
       "      <th>hour</th>\n",
       "      <th>pm2.5</th>\n",
       "      <th>DEWP</th>\n",
       "      <th>TEMP</th>\n",
       "      <th>PRES</th>\n",
       "      <th>cbwd</th>\n",
       "      <th>Iws</th>\n",
       "      <th>Is</th>\n",
       "      <th>Ir</th>\n",
       "      <th>datetime</th>\n",
       "      <th>scaled_PRES</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>2010</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>-21</td>\n",
       "      <td>-11.0</td>\n",
       "      <td>1021.0</td>\n",
       "      <td>NW</td>\n",
       "      <td>1.79</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2010-01-01 00:00:00</td>\n",
       "      <td>0.545455</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2010</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>-21</td>\n",
       "      <td>-12.0</td>\n",
       "      <td>1020.0</td>\n",
       "      <td>NW</td>\n",
       "      <td>4.92</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2010-01-01 01:00:00</td>\n",
       "      <td>0.527273</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>2010</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>NaN</td>\n",
       "      <td>-21</td>\n",
       "      <td>-11.0</td>\n",
       "      <td>1019.0</td>\n",
       "      <td>NW</td>\n",
       "      <td>6.71</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2010-01-01 02:00:00</td>\n",
       "      <td>0.509091</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>2010</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>NaN</td>\n",
       "      <td>-21</td>\n",
       "      <td>-14.0</td>\n",
       "      <td>1019.0</td>\n",
       "      <td>NW</td>\n",
       "      <td>9.84</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2010-01-01 03:00:00</td>\n",
       "      <td>0.509091</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>2010</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>NaN</td>\n",
       "      <td>-20</td>\n",
       "      <td>-12.0</td>\n",
       "      <td>1018.0</td>\n",
       "      <td>NW</td>\n",
       "      <td>12.97</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2010-01-01 04:00:00</td>\n",
       "      <td>0.490909</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   No  year  month  day  hour  pm2.5  DEWP  TEMP    PRES cbwd    Iws  Is  Ir  \\\n",
       "0   1  2010      1    1     0    NaN   -21 -11.0  1021.0   NW   1.79   0   0   \n",
       "1   2  2010      1    1     1    NaN   -21 -12.0  1020.0   NW   4.92   0   0   \n",
       "2   3  2010      1    1     2    NaN   -21 -11.0  1019.0   NW   6.71   0   0   \n",
       "3   4  2010      1    1     3    NaN   -21 -14.0  1019.0   NW   9.84   0   0   \n",
       "4   5  2010      1    1     4    NaN   -20 -12.0  1018.0   NW  12.97   0   0   \n",
       "\n",
       "             datetime  scaled_PRES  \n",
       "0 2010-01-01 00:00:00     0.545455  \n",
       "1 2010-01-01 01:00:00     0.527273  \n",
       "2 2010-01-01 02:00:00     0.509091  \n",
       "3 2010-01-01 03:00:00     0.509091  \n",
       "4 2010-01-01 04:00:00     0.490909  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#First five rows of train\n",
    "df_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>No</th>\n",
       "      <th>year</th>\n",
       "      <th>month</th>\n",
       "      <th>day</th>\n",
       "      <th>hour</th>\n",
       "      <th>pm2.5</th>\n",
       "      <th>DEWP</th>\n",
       "      <th>TEMP</th>\n",
       "      <th>PRES</th>\n",
       "      <th>cbwd</th>\n",
       "      <th>Iws</th>\n",
       "      <th>Is</th>\n",
       "      <th>Ir</th>\n",
       "      <th>datetime</th>\n",
       "      <th>scaled_PRES</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>35064</th>\n",
       "      <td>35065</td>\n",
       "      <td>2014</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>24.0</td>\n",
       "      <td>-20</td>\n",
       "      <td>7.0</td>\n",
       "      <td>1014.0</td>\n",
       "      <td>NW</td>\n",
       "      <td>143.48</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2014-01-01 00:00:00</td>\n",
       "      <td>0.418182</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>35065</th>\n",
       "      <td>35066</td>\n",
       "      <td>2014</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>53.0</td>\n",
       "      <td>-20</td>\n",
       "      <td>7.0</td>\n",
       "      <td>1013.0</td>\n",
       "      <td>NW</td>\n",
       "      <td>147.50</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2014-01-01 01:00:00</td>\n",
       "      <td>0.400000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>35066</th>\n",
       "      <td>35067</td>\n",
       "      <td>2014</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>65.0</td>\n",
       "      <td>-20</td>\n",
       "      <td>6.0</td>\n",
       "      <td>1013.0</td>\n",
       "      <td>NW</td>\n",
       "      <td>151.52</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2014-01-01 02:00:00</td>\n",
       "      <td>0.400000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>35067</th>\n",
       "      <td>35068</td>\n",
       "      <td>2014</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>70.0</td>\n",
       "      <td>-20</td>\n",
       "      <td>6.0</td>\n",
       "      <td>1013.0</td>\n",
       "      <td>NW</td>\n",
       "      <td>153.31</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2014-01-01 03:00:00</td>\n",
       "      <td>0.400000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>35068</th>\n",
       "      <td>35069</td>\n",
       "      <td>2014</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>79.0</td>\n",
       "      <td>-18</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1012.0</td>\n",
       "      <td>cv</td>\n",
       "      <td>0.89</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2014-01-01 04:00:00</td>\n",
       "      <td>0.381818</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          No  year  month  day  hour  pm2.5  DEWP  TEMP    PRES cbwd     Iws  \\\n",
       "35064  35065  2014      1    1     0   24.0   -20   7.0  1014.0   NW  143.48   \n",
       "35065  35066  2014      1    1     1   53.0   -20   7.0  1013.0   NW  147.50   \n",
       "35066  35067  2014      1    1     2   65.0   -20   6.0  1013.0   NW  151.52   \n",
       "35067  35068  2014      1    1     3   70.0   -20   6.0  1013.0   NW  153.31   \n",
       "35068  35069  2014      1    1     4   79.0   -18   3.0  1012.0   cv    0.89   \n",
       "\n",
       "       Is  Ir            datetime  scaled_PRES  \n",
       "35064   0   0 2014-01-01 00:00:00     0.418182  \n",
       "35065   0   0 2014-01-01 01:00:00     0.400000  \n",
       "35066   0   0 2014-01-01 02:00:00     0.400000  \n",
       "35067   0   0 2014-01-01 03:00:00     0.400000  \n",
       "35068   0   0 2014-01-01 04:00:00     0.381818  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#First five rows of validation\n",
    "df_val.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Reset the indices of the validation set\n",
    "df_val.reset_index(drop=True, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/manmeet/anaconda3/envs/py35/lib/python3.5/site-packages/seaborn/timeseries.py:183: UserWarning: The `tsplot` function is deprecated and will be removed in a future release. Please update your code to use the new `lineplot` function.\n",
      "  warnings.warn(msg, UserWarning)\n",
      "/home/manmeet/.local/lib/python3.5/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.\n",
      "  return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval\n",
      "/home/manmeet/anaconda3/envs/py35/lib/python3.5/site-packages/seaborn/timeseries.py:183: UserWarning: The `tsplot` function is deprecated and will be removed in a future release. Please update your code to use the new `lineplot` function.\n",
      "  warnings.warn(msg, UserWarning)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Scaled Air Pressure readings')"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 396x396 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 396x396 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "\"\"\"\n",
    "The train and validation time series of standardized PRES is also plotted.\n",
    "\"\"\"\n",
    "\n",
    "plt.figure(figsize=(5.5, 5.5))\n",
    "g = sns.tsplot(df_train['scaled_PRES'], color='b')\n",
    "g.set_title('Time series of scaled Air Pressure in train set')\n",
    "g.set_xlabel('Index')\n",
    "g.set_ylabel('Scaled Air Pressure readings')\n",
    "\n",
    "plt.figure(figsize=(5.5, 5.5))\n",
    "g = sns.tsplot(df_val['scaled_PRES'], color='r')\n",
    "g.set_title('Time series of scaled Air Pressure in validation set')\n",
    "g.set_xlabel('Index')\n",
    "g.set_ylabel('Scaled Air Pressure readings')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now we need to generate regressors (X) and target variable (y) for train and validation. 2-D array of regressor and 1-D array of target is created from the original 1-D array of columm scaled_PRES in the DataFrames. For the time series forecasting model, Past seven days of observations are used to predict for the next day. This is equivalent to a AR(7) model. We define a function which takes the original time series and the number of timesteps in regressors as input to generate the arrays of X and y."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "def makeXy(ts, nb_timesteps):\n",
    "    \"\"\"\n",
    "    Input: \n",
    "           ts: original time series\n",
    "           nb_timesteps: number of time steps in the regressors\n",
    "    Output: \n",
    "           X: 2-D array of regressors\n",
    "           y: 1-D array of target \n",
    "    \"\"\"\n",
    "    X = []\n",
    "    y = []\n",
    "    for i in range(nb_timesteps, ts.shape[0]):\n",
    "        X.append(list(ts.loc[i-nb_timesteps:i-1]))\n",
    "        y.append(ts.loc[i])\n",
    "    X, y = np.array(X), np.array(y)\n",
    "    return X, y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape of train arrays: (35057, 7) (35057,)\n"
     ]
    }
   ],
   "source": [
    "X_train, y_train = makeXy(df_train['scaled_PRES'], 7)\n",
    "print('Shape of train arrays:', X_train.shape, y_train.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape of validation arrays: (8753, 7) (8753,)\n"
     ]
    }
   ],
   "source": [
    "X_val, y_val = makeXy(df_val['scaled_PRES'], 7)\n",
    "print('Shape of validation arrays:', X_val.shape, y_val.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The input to RNN layers must be of shape (number of samples, number of timesteps, number of features per timestep). In this case we are modeling only pm2.5 hence number of features per timestep is one. Number of timesteps is seven and number of samples is same as the number of samples in X_train and X_val, which are reshaped to 3D arrays."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape of 3D arrays: (35057, 7, 1) (8753, 7, 1)\n"
     ]
    }
   ],
   "source": [
    "X_train, X_val = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), X_val.reshape((X_val.shape[0], X_val.shape[1], 1))\n",
    "print('Shape of 3D arrays:', X_train.shape, X_val.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now we define the MLP using the Keras Functional API. In this approach a layer can be declared as the input of the following layer at the time of defining the next layer. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "from keras.layers import Dense, Input, Dropout\n",
    "from keras.layers.recurrent import GRU\n",
    "from keras.optimizers import SGD\n",
    "from keras.models import Model\n",
    "from keras.models import load_model\n",
    "from keras.callbacks import ModelCheckpoint"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Define input layer which has shape (None, 7) and of type float32. None indicates the number of instances\n",
    "input_layer = Input(shape=(7,1), dtype='float32')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "#LSTM layer is defined for seven timesteps\n",
    "gru_layer = GRU(64, input_shape=(7,1), return_sequences=False)(input_layer)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "dropout_layer = Dropout(0.2)(gru_layer)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Finally the output layer gives prediction for the next day's air pressure.\n",
    "output_layer = Dense(1, activation='linear')(dropout_layer)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The input, dense and output layers will now be packed inside a Model, which is wrapper class for training and making\n",
    "predictions. Mean square error (mse) is used as the loss function.\n",
    "\n",
    "The network's weights are optimized by the Adam algorithm. Adam stands for adaptive moment estimation\n",
    "and has been a popular choice for training deep neural networks. Unlike, stochastic gradient descent, adam uses\n",
    "different learning rates for each weight and separately updates the same as the training progresses. The learning rate of a weight is updated based on exponentially weighted moving averages of the weight's gradients and the squared gradients."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "input_1 (InputLayer)         (None, 7, 1)              0         \n",
      "_________________________________________________________________\n",
      "gru_1 (GRU)                  (None, 64)                12672     \n",
      "_________________________________________________________________\n",
      "dropout_1 (Dropout)          (None, 64)                0         \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 1)                 65        \n",
      "=================================================================\n",
      "Total params: 12,737\n",
      "Trainable params: 12,737\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "ts_model = Model(inputs=input_layer, outputs=output_layer)\n",
    "ts_model.compile(loss='mse', optimizer='adam')\n",
    "ts_model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 35057 samples, validate on 8753 samples\n",
      "Epoch 1/20\n",
      "35057/35057 [==============================] - 37s 1ms/step - loss: 6.5575e-04 - val_loss: 1.6298e-04\n",
      "Epoch 2/20\n",
      "35057/35057 [==============================] - 38s 1ms/step - loss: 4.0079e-04 - val_loss: 1.5829e-04\n",
      "Epoch 3/20\n",
      "35057/35057 [==============================] - 38s 1ms/step - loss: 3.3174e-04 - val_loss: 1.5003e-04\n",
      "Epoch 4/20\n",
      "35057/35057 [==============================] - 38s 1ms/step - loss: 3.1412e-04 - val_loss: 1.5610e-04\n",
      "Epoch 5/20\n",
      "35057/35057 [==============================] - 38s 1ms/step - loss: 3.1635e-04 - val_loss: 1.3811e-04\n",
      "Epoch 6/20\n",
      "35057/35057 [==============================] - 38s 1ms/step - loss: 3.1433e-04 - val_loss: 1.3788e-04\n",
      "Epoch 7/20\n",
      "35057/35057 [==============================] - 37s 1ms/step - loss: 3.0616e-04 - val_loss: 1.5243e-04\n",
      "Epoch 8/20\n",
      "35057/35057 [==============================] - 39s 1ms/step - loss: 3.0851e-04 - val_loss: 1.8102e-04\n",
      "Epoch 9/20\n",
      "35057/35057 [==============================] - 38s 1ms/step - loss: 3.0800e-04 - val_loss: 1.4096e-04\n",
      "Epoch 10/20\n",
      "35057/35057 [==============================] - 37s 1ms/step - loss: 3.0419e-04 - val_loss: 1.6410e-04\n",
      "Epoch 11/20\n",
      "35057/35057 [==============================] - 35s 1ms/step - loss: 3.0955e-04 - val_loss: 1.3326e-04\n",
      "Epoch 12/20\n",
      "35057/35057 [==============================] - 38s 1ms/step - loss: 3.0101e-04 - val_loss: 1.6389e-04\n",
      "Epoch 13/20\n",
      "35057/35057 [==============================] - 36s 1ms/step - loss: 3.0216e-04 - val_loss: 1.7493e-04\n",
      "Epoch 14/20\n",
      "35057/35057 [==============================] - 35s 995us/step - loss: 3.0241e-04 - val_loss: 1.6030e-04\n",
      "Epoch 15/20\n",
      "35057/35057 [==============================] - 36s 1ms/step - loss: 3.0299e-04 - val_loss: 1.3621e-04\n",
      "Epoch 16/20\n",
      "35057/35057 [==============================] - 36s 1ms/step - loss: 2.9966e-04 - val_loss: 1.4058e-04\n",
      "Epoch 17/20\n",
      "35057/35057 [==============================] - 35s 994us/step - loss: 3.0566e-04 - val_loss: 1.4133e-04\n",
      "Epoch 18/20\n",
      "35057/35057 [==============================] - 35s 998us/step - loss: 3.0022e-04 - val_loss: 1.4219e-04\n",
      "Epoch 19/20\n",
      "35057/35057 [==============================] - 34s 971us/step - loss: 3.0324e-04 - val_loss: 1.5915e-04\n",
      "Epoch 20/20\n",
      "35057/35057 [==============================] - 36s 1ms/step - loss: 2.9921e-04 - val_loss: 1.3672e-04\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7fcdd317ce10>"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\"\"\"\n",
    "The model is trained by calling the fit function on the model object and passing the X_train and y_train. The training \n",
    "is done for a predefined number of epochs. Additionally, batch_size defines the number of samples of train set to be\n",
    "used for a instance of back propagation.The validation dataset is also passed to evaluate the model after every epoch\n",
    "completes. A ModelCheckpoint object tracks the loss function on the validation set and saves the model for the epoch,\n",
    "at which the loss function has been minimum.\n",
    "\"\"\"\n",
    "save_weights_at = os.path.join('keras_models', 'PRSA_data_Air_Pressure_GRU_weights.{epoch:02d}-{val_loss:.4f}.hdf5')\n",
    "save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,\n",
    "                            save_best_only=True, save_weights_only=False, mode='min',\n",
    "                            period=1)\n",
    "ts_model.fit(x=X_train, y=y_train, batch_size=16, epochs=20,\n",
    "             verbose=1, callbacks=[save_best], validation_data=(X_val, y_val),\n",
    "             shuffle=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Prediction are made for the air pressure from the best saved model. The model's predictions, which are on the minmax scaled  air-pressure, are inverse transformed to get predictions on original air pressure. The goodness-of-fit, R-squared is also calculated for the predictions on the original variable."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_model = load_model(os.path.join('keras_models', 'PRSA_data_Air_Pressure_GRU_weights.11-0.0001.hdf5'))\n",
    "preds = best_model.predict(X_val)\n",
    "pred_PRES = scaler.inverse_transform(preds)\n",
    "pred_PRES = np.squeeze(pred_PRES)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R-squared on validation set of the original air pressure: 0.9958023602647159\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import r2_score\n",
    "r2 = r2_score(df_val['PRES'].loc[7:], pred_PRES)\n",
    "print('R-squared on validation set of the original air pressure:', r2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 0, 'Index')"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 396x396 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Let's plot the first 50 actual and predicted values of air pressure.\n",
    "plt.figure(figsize=(5.5, 5.5))\n",
    "plt.plot(range(50), df_val['PRES'].loc[7:56], linestyle='-', marker='*', color='r')\n",
    "plt.plot(range(50), pred_PRES[:50], linestyle='-', marker='.', color='b')\n",
    "plt.legend(['Actual','Predicted'], loc=2)\n",
    "plt.title('Actual vs Predicted Air Pressure')\n",
    "plt.ylabel('Air Pressure')\n",
    "plt.xlabel('Index')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
