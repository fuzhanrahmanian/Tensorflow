## load data with tensorflow
# tf.data.TextLineDataset will load examples into text files
import tensorflow as tf
import tensorflow_datasets as tfds
import os

DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

for name in FILE_NAMES:
    text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL+name)

parent_dir = os.path.dirname(text_dir)

# load text into dataset

def labeler(example, index):
    return example, tf.cast(index, tf.int64)

labeled_data_sets = []

for i, file_name in enumerate(FILE_NAMES):
    lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
    labeled_data_sets.append(lines_dataset.map(lambda ex: labeler(ex, i)))

#combine the labels into one dataset

all_labeled_data = labeled_data_sets[0]

for labeled_data_set in labeled_data_sets[1:]:
    all_labeled_data.concatenate(labeled_data_set)

all_labeled_data = all_labeled_data.shuffle(buffer_size=50000, reshuffle_each_iteration=False)

for ex in all_labeled_data.take(4):
    print(ex)

#encodes text lines as numbers
#tfds.deprecated.text.TokenTextEncoder> this will create an encoder bz takes in a string of text and return a lost of unique integers
# adding value to a set bz "update"

tokenizer = tfds.deprecated.text.Tokenizer()
# collect the tokens into a python set to remove duplicates
vocabulary_set = set()
for text_tensor, _ in all_labeled_data:
    some_tokens = tokenizer.tokenize(text_tensor.numpy())
    vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)

#tfds.deprecated.text.TokenTextEncoder will takes in a string of text and returns a list of integers

encoder = tfds.deprecated.text.TokenTextEncoder(vocabulary_set)
# looking at a single line
example_text = next(iter(all_labeled_data))[0].numpy()

encoded_exmple = encoder.encode(example_text)

# wrapping a funciton with tf.py_function
# passing the wrap function with map method

def encode(text_tensor, label):
    encode_text = encoder.encode(text_tensor.numpy())
    return  encode_text, label

def encode_map_function(text, label):
    encode_text, label = tf.py_function(encode,
                                        inp=[text, label],
                                        Tout=(tf.int64, tf.int64))
    # manually set the shape of the components
    encode_text.set_shape([None])
    label.set_shape([])
    return  encode_text, label

all_encoded_data = all_labeled_data.map(encode_map_function)

# creating a small dataset with tf.data.Dataset.take
# creating a large training set with tf.data.Dataset.skip

# the size of the batches should be the same and if not then use tf.data.Dataset.padded_batch
BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 5000

train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE)

test_data = all_encoded_data.take(TAKE_SIZE).shuffle(BUFFER_SIZE)
test_data = test_data.padded_batch(BUFFER_SIZE)

sample_text, sample_label = next(iter(test_data))

vocab_size += 1

# build a model
model = tf.keras.Sequential()
# in this model, the first layer converts integer representations to dense vector embeddings
model.add(tf.keras.layers.Embedding(vocab_size, 64))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

for units in [64, 64]:
    model.add(tf.keras.layers.Dense(units, activation='relu'))

model.add(tf.keras.layers.Dense(3))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train the model
model.fit(train_data, epochs=3, validation_data=test_data)

eval_loss, eval_acc = model.evaluate(test_data)