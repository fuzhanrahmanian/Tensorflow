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

#create a tf.data dataset from a panda dataFrame

def df_to_dataset(dataframe, shuffle=True, batch_size=5):
    dataframe= dataframe.copy()
    labels= dataframe.pop('target')
    ds= tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds= ds.shuffle(buffer_size=len(dataframe))
        ds= ds.batch(batch_size)
    return ds

train_ds = df_to_dataset(train)
val_ds = df_to_dataset(val, shuffle=False)
test_ds = df_to_dataset(test, shuffle=False)

for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of ages:', feature_batch['age'])
    print('A varch of targets:', label_batch)

example_batch= next(iter(train_ds))[0]

# create a feature column and to transform a batch of data
def demo(feature_column):
    feature_layer= layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch))


### define a numeric column
age= feature_column.numeric_column("age")
tf.feature_column.numeric_column
print(age)

demo(age)

##define a bucketize columns : split the numerical values inro differnet categories based on numerical ranges
## this is describe by one hot values

age_buckets= tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
demo(age_buckets)
#categorical columns
# dataset cannot be feed as a string into a model therefore we need to convert them to a numeric values with the usage of one hot vector

thal= tf.feature_column.categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'])
thal_one_hot= tf.feature_column.indicator_column(thal)
demo(thal_one_hot)
#embedding: instead of having a large vector of 0 and 1 we assign a value to it
thal_embedding= tf.feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)

#this feature column calculates a hash value of the input and will select one of the hash_bucket_size buckets to encode a string
thal_hashed= tf.feature_column.categorical_column_with_hash_bucket('thal', hash_bucket_size=1000)
demo(tf.feature_column.indicator_column(thal_hashed))

#feature cross represent nonlinear relationship
# cross feature column will combine features into a single feature. Model will learn seperate weights for each combination of features
# cross feature is backed by a hashed_-column and we can choose how large the table is
import sys
sys.maxsize
crossed_feature = tf.feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
demo(tf.feature_column.indicator_column(crossed_feature))

# a summerize of all of the feature columns

feature_columns= []
#numeric column
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(feature_column.numeric_column(header))
#bucketizr column
age_buckets= feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

#indicator cols
thal= feature_column.categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'])
thal_one_hot= feature_column.indicator_column(thal)


# embedding cols
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed cols
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

#converting input feature columns to a keras model by DenseFeatures
feature_layer= tf.keras.layers.DenseFeatures(feature_columns)

#create, compile and train the model
model= tf.keras.Sequential([feature_layer,
                            layers.Dense(128, activation='relu'),
                            layers.Dense(128, activation='relu'),
                            layers.Dense(1)])
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
history= model.fit(train_ds, validation_data=val_ds, epochs=5)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

def plot_curves(history, metrics):
    nrows = 1
    ncols = 2
    fig = plt.figure(figsize=(10,5))

    for idx, key in enumerate(metrics):
        ax = fig.add_subplot(nrows, ncols, idx+1)
        plt.plot(history.history[key])
        plt.plot(history.history['val_{}'.format(key)])
        plt.title('model {}'.format(key))
        plt.ylabel(key)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

plot_curves(history, ['loss', 'accuracy'])