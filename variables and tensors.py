import tensorflow as tf
import numpy as np
print("Tensorflow version", tf.version.VERSION)

rank_0_tensor = tf.constant(4)
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
rank_2_tensor = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float16)
rank_3_tensor = tf.constant([[[0, 1, 2, 3, 4],
                              [5, 6, 7, 8, 9]],
                             [[10, 11, 12, 13, 14],
                              [15, 16, 17, 18, 19]],
                             [[20, 21, 22, 23, 24],
                              [25, 26, 27, 28, 29]],])
np.array(rank_3_tensor)
#or
rank_3_tensor.numpy()

a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]]) # Could have also said `tf.ones([2,2])`

print(tf.add(a, b), "\n")
print(tf.multiply(a, b), "\n")
print(tf.matmul(a, b), "\n")

c = tf.constant([[4.0, 5.0], [10.0, 1.0]])
# Find the largest value
print(tf.reduce_max(c))

# Find the index of the largest value
print(tf.argmax(c))
# Compute the softmax
print(tf.nn.softmax(c))

rank_4_tensor = tf.zeros([3, 2, 4, 5]) #(3 is batch, 2 is width, 4 is height, 5 is features
#element type
rank_4_tensor.dtype
# number of dimensions
rank_4_tensor.ndim
# tensor shape
rank_4_tensor.shape
# elements along axis 0 of tensor
rank_4_tensor.shape[0]
#elements along last axis
rank_4_tensor.shape[-1]
#total number of elements
tf.size(rank_4_tensor).numpy()



rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
rank_1_tensor.numpy()
print("First:", rank_1_tensor[0].numpy())
print("Second:", rank_1_tensor[1].numpy())
print("Last:", rank_1_tensor[-1].numpy())

#indexing
#everything
rank_1_tensor[:].numpy()
# before 4
rank_1_tensor[:4].numpy()
#from 4 to end
rank_1_tensor[4:].numpy()
#from 2 before 7
rank_1_tensor[2:7].numpy()
#every add item
rank_1_tensor[::2].numpy()
#reversed
rank_1_tensor[::-1].numpy()

print(rank_2_tensor.numpy())
#get a single value
rank_2_tensor[1, 1].numpy()

#index using any combination intergers and slices
#second row
rank_2_tensor[1,:].numpy()
#senods column
rank_2_tensor[:, 1].numpy()
#last row
rank_2_tensor[-1,:].numpy()
#first item in last row
rank_2_tensor[0, -1].numpy()
#skip the first row####
rank_2_tensor[1:,:].numpy()

print(rank_3_tensor[:, :, 4])

#manipulating shape
var_x = tf.Variable(tf.constant([[1], [2], [3]]))
var_x.shape
#invert the object to a python list
var_x.shape.as_list()
reshaped = tf.reshape(var_x, [1, 3])
reshaped.shape


print(rank_3_tensor)

# -1 passed in a shape argument says "whatever fits"
tf.reshape(rank_3_tensor, [-1])
print(tf.reshape(rank_3_tensor, [3*2, 5]), "\n")
print(tf.reshape(rank_3_tensor, [3, -1]))


the_f64_tensor = tf.constant([1.1, 2.2, 4.4], dtype= tf.float64)
the_f16_tensor = tf.cast(the_f64_tensor, dtype= tf.float16)
# Now, let's cast to an uint8 and lose the decimal precision
the_u8_tensor = tf.cast(the_f16_tensor, dtype= tf.uint8)
print(the_u8_tensor)

#broadcasting and stretching
x = tf.constant([1, 2, 3])

y = tf.constant(2)
z = tf.constant([2, 2, 2])
# All of these are the same computation
print(tf.multiply(x, 2))
print(x * y)
print(x * z)

x = tf.reshape(x,[3,1])
y = tf.range(1, 5)
print(x, "\n")
print(y, "\n")
print(tf.multiply(x, y))

x_stretch = tf.constant([[1, 1, 1, 1],
                         [2, 2, 2, 2],
                         [3, 3, 3, 3]])

y_stretch = tf.constant([[1, 2, 3, 4],
                         [1, 2, 3, 4],
                         [1, 2, 3, 4]])

print(x_stretch * y_stretch)  # Again, operator overloading

tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3])

ragged_list = [
    [0, 1, 2, 3],
    [4, 5],
    [6, 7, 8],
    [9]]

try:
  tensor = tf.constant(ragged_list)
except Exception as e: print(e)

ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor.shape)

#string tensor
scalar_string_tensor = tf.constant("Gray wolf")
print(scalar_string_tensor)

tensor_of_strings = tf.constant(["Gray wolf",
                                 "Quick brown fox",
                                 "Lazy dog"])
tf.strings.split(scalar_string_tensor, sep=" ")
tf.strings.split(tensor_of_strings)

text = tf.constant("1 10 100")
tf.strings.to_number(tf.strings.split(text, sep=" "))

byte_strings = tf.strings.bytes_split(tf.constant("Duck"))
byte_ints = tf.io.decode_raw(tf.constant("Duck"), tf.uint8)
print("Byte strings:", byte_strings)
print("Bytes:", byte_ints)

# Or split it up as unicode and then decode it
unicode_bytes = tf.constant("アヒル 🦆")
unicode_char_bytes = tf.strings.unicode_split(unicode_bytes, "UTF-8")
unicode_values = tf.strings.unicode_decode(unicode_bytes, "UTF-8")

print("\nUnicode bytes:", unicode_bytes)
print("\nUnicode chars:", unicode_char_bytes)
print("\nUnicode values:", unicode_values)

#sparse tensor
sparse_tensor= tf.sparse.SparseTensor(indices= [[0,0], [1, 2]],
                                      values= [0, 1],
                                      dense_shape= [3, 4])
tf.sparse.to_dense(sparse_tensor)

#variable : define an initial value

my_tensor = tf.constant([[1.1, 2.2], [3.3, 4.4]])
my_variable = tf.Variable(my_tensor)
# Variables can be all kinds of types, just like tensors
bool_variable = tf.Variable([False, False, False, True])
complex_variable = tf.Variable([5 + 4j, 6 + 1j])
print(complex_variable.numpy())

#viewed as a tensor
tf.convert_to_tensor(my_variable)
# index of highest value
tf.argmax(my_variable)
#copying and reshaping
tf.reshape(my_variable, [1,4])
a = tf.Variable([2.0, 3.0])
# This will keep the same dtype, float32
a.assign([1, 2])
# Not allowed as it resizes the variable:
try:
  a.assign([1.0, 2.0, 3.0])
except Exception as e: print(e)

a = tf.Variable([2.0, 3.0])
b = tf.Variable(a)
a.assign([4, 3])
a.numpy()
a.assign_add([1, 1]).numpy()
a.assign_sub([1, 1]).numpy()
step_counter = tf.Variable(1, trainable=False)

with tf.device('CPU:0'):

  # Create some tensors
  a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  c = tf.matmul(a, b)

print(c)

with tf.device('CPU:0'):
    a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.Variable([[1.0, 2.0, 3.0]])

with tf.device('GPU:0'):
    k = a * b
print(k)