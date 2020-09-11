# Learning objects
# Load a CSV file into a tf.data.Dataset.
# Load Numpy data

import functools
import numpy as np
import tensorflow as tf

tf.version.VERSION

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

np.set_printoptions(precision=3, suppress=True)

LABEL_COLUMN = 'survived'
LABELS= [0,1]
def get_dataset(file_path, **kwargs):
    dataset= tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size= 4,
        label_name= LABEL_COLUMN,
        na_value= '?',
        num_epochs= 2,
        ignore_errors= True,
        **kwargs)

raw_train_data= get_dataset(train_file_path)
raw_test_data= get_dataset(test_file_path)

def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}, {}".format(key, value.numpy()))

show_batch(raw_train_data)
CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']
temp_dataset = get_dataset(train_file_path, column_names=CSV_COLUMNS)
show_batch(temp_dataset)

#preprocesing for continous data
SELECT_COLUMNS= ['survived', 'age', 'n_siblings_spouse', 'parch', 'fare']
DEFAULTS= [0, 0.0, 0.0, 0.0, 0.0]
temp_dataset = get_dataset(train_file_path, select_columns= SELECT_COLUMNS, column_defaults= DEFAULTS)

# data normalization
import pandas as pd
desc= pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()

MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])

def normalize_numeric_data(data, mean, std):
  # Center the data
    return (data - mean) / std

# See what you just created.
normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)
#print(normalizer)
numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]
numeric_column

# load numpy data
DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
with np.load(path) as data:
  train_examples = data['x_train']
  train_labels = data['y_train']
  test_examples = data['x_test']
  test_labels = data['y_test']

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

