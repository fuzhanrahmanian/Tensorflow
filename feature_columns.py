import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
tf.version.VERSION

URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe= pd.read_csv(URL)
dataframe.head()
dataframe.info()

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# creating input pipline
# feature column can map from the columns in pandas dataframe to features used to train a model
