{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this notebook, we will use a multi-layer perceptron to develop time series forecasting models.\n",
    "The dataset used for the examples of this notebook is on air pollution measured by concentration of\n",
    "particulate matter (PM) of diameter less than or equal to 2.5 micrometers. There are other variables\n",
    "such as air pressure, air temperature, dew point and so on.\n",
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
    "g.set_title('Box plot of Air Pressure')\n",
    "#plt.savefig('plots/ch5/B07887_05_01.png', format='png', dpi=300)"
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
    "g.set_ylabel('Air Pressure readings in hPa')\n",
    "#plt.savefig('plots/ch5/B07887_05_02.png', format='png', dpi=300)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Gradient descent algorithms perform better (for example converge faster) if the variables are wihtin range [-1, 1]. Many sources relax the boundary to even [-3, 3]. The PRES variable is mixmax scaled to bound the tranformed variable within [0,1]."
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
    "epochs will further decrease the loss function on the train set but might not necessarily have the same effect\n",
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
    "Let's start by splitting the dataset into train and validation. The dataset's time period is from\n",
    "Jan 1st, 2010 to Dec 31st, 2014. The first four years - 2010 to 2013 is used as train and\n",
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
    "The train and validation time series of standardized PRES are also plotted.\n",
    "\"\"\"\n",
    "\n",
    "plt.figure(figsize=(5.5, 5.5))\n",
    "g = sns.tsplot(df_train['scaled_PRES'], color='b')\n",
    "g.set_title('Time series of scaled Air Pressure in train set')\n",
    "g.set_xlabel('Index')\n",
    "g.set_ylabel('Scaled Air Pressure readings')\n",
    "#plt.savefig('plots/ch5/B07887_05_03.png', format='png', dpi=300)\n",
    "\n",
    "plt.figure(figsize=(5.5, 5.5))\n",
    "g = sns.tsplot(df_val['scaled_PRES'], color='r')\n",
    "g.set_title('Time series of scaled Air Pressure in validation set')\n",
    "g.set_xlabel('Index')\n",
    "g.set_ylabel('Scaled Air Pressure readings')\n",
    "#plt.savefig('plots/ch5/B07887_05_04.png', format='png', dpi=300)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now we need to generate regressors (X) and target variable (y) for train and validation. 2-D array of regressors and 1-D array of target is created from the original 1-D array of column standardized_PRES in the DataFrames. For the time series forecasting model, Past seven days of observations are used to predict for the next day. This is equivalent to a AR(7) model. We define a function which takes the original time series and the number of timesteps in regressors as input to generate the arrays of X and y."
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
    "Now we define the MLP using the Keras Functional API. In this approach a layer can be declared as the input of the following layer at the time of defining the next layer. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
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
    "from keras.optimizers import SGD\n",
    "from keras.models import Model\n",
    "from keras.models import load_model\n",
    "from keras.callbacks import ModelCheckpoint"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Define input layer which has shape (None, 7) and of type float32. None indicates the number of instances\n",
    "input_layer = Input(shape=(7,), dtype='float32')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Dense layers are defined with linear activation\n",
    "dense1 = Dense(32, activation='linear')(input_layer)\n",
    "dense2 = Dense(16, activation='linear')(dense1)\n",
    "dense3 = Dense(16, activation='linear')(dense2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Multiple hidden layers and large number of neurons in each hidden layer gives neural networks the ability to model complex non-linearity of the underlying relations between regressors and target. However, deep neural networks can also overfit train data and give poor results on validation or test set. Dropout has been effectively used to regularize deep neural networks. In this example, a Dropout layer is added before the output layer. Dropout randomly sets p fraction of input neurons to zero before passing to the next layer. Randomly dropping inputs essentially acts as a bootstrap aggregating or bagging type of model ensembling. Random forest uses bagging by building trees on random subsets of input features. We use p=0.2 to dropout 20% of randomly selected input features."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "dropout_layer = Dropout(0.2)(dense3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Finally, the output layer gives prediction for the next day's air pressure.\n",
    "output_layer = Dense(1, activation='linear')(dropout_layer)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The input, dense and output layers will now be packed inside a Model, which is wrapper class for training and making\n",
    "predictions. Mean square error (MSE) is used as the loss function.\n",
    "\n",
    "The network's weights are optimized by the Adam algorithm. Adam stands for adaptive moment estimation\n",
    "and has been a popular choice for training deep neural networks. Unlike, stochastic gradient descent, Adam uses\n",
    "different learning rates for each weight and separately updates the same as the training progresses. The learning rate of a weight is updated based on exponentially weighted moving averages of the weight's gradients and the squared gradients."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "input_1 (InputLayer)         (None, 7)                 0         \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 32)                256       \n",
      "_________________________________________________________________\n",
      "dense_2 (Dense)              (None, 16)                528       \n",
      "_________________________________________________________________\n",
      "dense_3 (Dense)              (None, 16)                272       \n",
      "_________________________________________________________________\n",
      "dropout_1 (Dropout)          (None, 16)                0         \n",
      "_________________________________________________________________\n",
      "dense_4 (Dense)              (None, 1)                 17        \n",
      "=================================================================\n",
      "Total params: 1,073\n",
      "Trainable params: 1,073\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "ts_model = Model(inputs=input_layer, outputs=output_layer)\n",
    "ts_model.compile(loss='mean_squared_error', optimizer='adam')\n",
    "ts_model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The model is trained by calling the fit function on the model object and passing the X_train and y_train. The training is done for a predefined number of epochs. Additionally, batch_size defines the number of samples of train set to be used for a instance of back propagation. The validation dataset is also passed to evaluate the model after every epoch completes. A ModelCheckpoint object tracks the loss function on the validation set and saves the model for the epoch, at which the loss function has been minimum."
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
      "Train on 35057 samples, validate on 8753 samples\n",
      "Epoch 1/20\n",
      "35057/35057 [==============================] - 17s 492us/step - loss: 0.0033 - val_loss: 2.8455e-04\n",
      "Epoch 2/20\n",
      "35057/35057 [==============================] - 10s 280us/step - loss: 9.9257e-04 - val_loss: 4.8282e-04\n",
      "Epoch 3/20\n",
      "35057/35057 [==============================] - 10s 280us/step - loss: 8.3610e-04 - val_loss: 1.6989e-04\n",
      "Epoch 4/20\n",
      "35057/35057 [==============================] - 11s 305us/step - loss: 8.1489e-04 - val_loss: 1.8826e-04\n",
      "Epoch 5/20\n",
      "35057/35057 [==============================] - 10s 286us/step - loss: 7.7990e-04 - val_loss: 1.4604e-04\n",
      "Epoch 6/20\n",
      "35057/35057 [==============================] - 10s 282us/step - loss: 7.8052e-04 - val_loss: 1.6379e-04\n",
      "Epoch 7/20\n",
      "35057/35057 [==============================] - 10s 285us/step - loss: 7.5776e-04 - val_loss: 1.3950e-04\n",
      "Epoch 8/20\n",
      "35057/35057 [==============================] - 10s 289us/step - loss: 7.6072e-04 - val_loss: 1.3750e-04\n",
      "Epoch 9/20\n",
      "35057/35057 [==============================] - 11s 301us/step - loss: 7.5212e-04 - val_loss: 1.8095e-04\n",
      "Epoch 10/20\n",
      "35057/35057 [==============================] - 11s 308us/step - loss: 7.5986e-04 - val_loss: 2.7602e-04\n",
      "Epoch 11/20\n",
      "35057/35057 [==============================] - 10s 296us/step - loss: 7.5360e-04 - val_loss: 2.2445e-04\n",
      "Epoch 12/20\n",
      "35057/35057 [==============================] - 10s 291us/step - loss: 7.5283e-04 - val_loss: 1.7190e-04\n",
      "Epoch 13/20\n",
      "35057/35057 [==============================] - 10s 293us/step - loss: 7.6461e-04 - val_loss: 1.4717e-04\n",
      "Epoch 14/20\n",
      "35057/35057 [==============================] - 10s 295us/step - loss: 7.4857e-04 - val_loss: 1.3729e-04\n",
      "Epoch 15/20\n",
      "35057/35057 [==============================] - 10s 292us/step - loss: 7.4545e-04 - val_loss: 1.4115e-04\n",
      "Epoch 16/20\n",
      "35057/35057 [==============================] - 10s 297us/step - loss: 7.3857e-04 - val_loss: 3.7646e-04\n",
      "Epoch 17/20\n",
      "35057/35057 [==============================] - 11s 308us/step - loss: 7.3057e-04 - val_loss: 2.0651e-04\n",
      "Epoch 18/20\n",
      "35057/35057 [==============================] - 10s 285us/step - loss: 7.5421e-04 - val_loss: 2.1660e-04\n",
      "Epoch 19/20\n",
      "35057/35057 [==============================] - 10s 282us/step - loss: 7.4279e-04 - val_loss: 1.9086e-04\n",
      "Epoch 20/20\n",
      "35057/35057 [==============================] - 10s 295us/step - loss: 7.3811e-04 - val_loss: 2.6516e-04\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7fa1bb0e76a0>"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "save_weights_at = os.path.join('keras_models', 'PRSA_data_Air_Pressure_MLP_weights.{epoch:02d}-{val_loss:.4f}.hdf5')\n",
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
    "Prediction are made for the air pressure from the best saved model. The model's predictions, which are on the scaled  air-pressure, are inverse transformed to get predictions on original air pressure. The goodness-of-fit or R squared is also calculated."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_model = load_model(os.path.join('keras_models', 'PRSA_data_Air_Pressure_MLP_weights.14-0.0001.hdf5'))\n",
    "preds = best_model.predict(X_val)\n",
    "pred_PRES = scaler.inverse_transform(preds)\n",
    "pred_PRES = np.squeeze(pred_PRES)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import r2_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R-squared for the validation set: 0.9957\n"
     ]
    }
   ],
   "source": [
    "r2 = r2_score(df_val['PRES'].loc[7:], pred_PRES)\n",
    "print('R-squared for the validation set:', round(r2,4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 0, 'Index')"
      ]
     },
     "execution_count": 28,
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
    "plt.xlabel('Index')\n",
    "#plt.savefig('plots/ch5/B07887_05_05.png', format='png', dpi=300)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "# confirm TensorFlow sees the GPU\n",
    "from tensorflow.python.client import device_lib\n",
    "assert 'GPU' in str(device_lib.list_local_devices())\n",
    "\n",
    "# confirm Keras sees the GPU\n",
    "from keras import backend\n",
    "assert len(backend.tensorflow_backend._get_available_gpus()) > 0\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
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
