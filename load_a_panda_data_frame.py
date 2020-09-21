import pandas as pd
import tensorflow as tf

csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/applied-dl/heart.csv')

df = pd.read_csv(csv_file)
df.head()
df.dtypes

# change the object in the dataframe to a discrete numerical value.
df['thal']= pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes

#load data using tf.data.Dataset
# for reading the values from a pandas dataframe, we should use tf.data.Dataset.from_tensor_slices
target = df.pop('target')

dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

for feat, targ in dataset.take(5):
    print('Features {}, Target: {}'.format(feat, targ))
tf.constant(df['thal'])

#shuffling and batching
train_dataset= dataset.shuffle(len(df)).batch(1)
# create and train a model

def get_compile_model():
    model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='relu'),
                                 tf.keras.layers.Dense(10, activation='relu'),
                                 tf.keras.layers.Dense(1)])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

model = get_compile_model()
model.fit(train_dataset, epochs=18)

