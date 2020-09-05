import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

x = tf.constant([2, 3, 4])
x = tf.Variable(2.0, dtype=tf.float32, name='my_variable')
x.numpy()
x.assign(45.9).numpy()
x.assign_add(10).numpy()
x.assign_sub(2).numpy()

#linear regression
X = tf.constant(range(10), dtype=tf.float32)
Y = 2 * X + 10

X_test = tf.constant(range(10, 20), dtype=tf.float32)
Y_test = 2 * X_test + 10

print("X_test:{}".format(X_test))
print("Y_test:{}".format(Y_test))

y_mean = Y.numpy().mean()

def predict_mean(X):
    y_hat = [y_mean] * len(X)
    return y_hat

y_hat = predict_mean(X_test)

#calculate loss
errors= (y_hat - Y)**2
loss = tf.reduce_mean(errors)
print(loss)

def loss_mse(X, Y, w0, w1):
    Y_hat= w0 * X + w1
    errors = (Y_hat - Y)**2
    return tf.reduce_mean(errors)

def compute_gradients(X, Y, w0, w1):
    with tf.GradientTape() as tape:
        loss = loss_mse(X, Y, w0, w1)
    return tape.gradient(loss, [w0, w1])


w0 = tf.Variable(0.0)
w1 = tf.Variable(0.0)

dw0, dw1 = compute_gradients(X, Y, w0, w1)

print("dw0:", dw0.numpy())
print("dw1", dw1.numpy())

STEPS = 700
LEARNING_RATE = .02
MSG = "STEP {step} - loss: {loss}, w0: {w0}, w1: {w1}\n"

w0 = tf.Variable(0.0)
w1 = tf.Variable(0.0)

for step in range(0, STEPS + 1):
    dw0, dw1 = compute_gradients(X, Y, w0, w1)
    print("dw0: {}, dw1: {}".format(dw0.numpy(), dw1.numpy()))
    w0.assign_sub(dw0 *LEARNING_RATE)
    w1.assign_sub(dw1 *LEARNING_RATE)
    if step % 50 == 0:
        loss = loss_mse(X, Y, w0, w1)
        print(MSG.format(step=step, loss=loss, w0=w0.numpy(), w1=w1.numpy()))

loss = loss_mse(X_test, Y_test, w0, w1)
loss.numpy()


#### modelling a non-linear function y = x.e**(-x**2)

X = tf.constant(np.linspace(0, 2, 1000), dtype=tf.float32)
Y = X * tf.exp(-X**2)
plt.plot(X, Y)
plt.show()


def make_features(X):
    f1 = tf.ones_like(X)  # Bias.
    f2 = X
    f3 = tf.square(X)
    f4 = tf.sqrt(X)
    f5 = tf.exp(X)
    print(f1, f2, f3, f4, f5)
    return tf.stack([f1, f2, f3, f4, f5], axis=1)

def predict(X, W):
    return tf.squeeze(X @ W, -1)

def loss_mse(X, Y, W):
    Y_hat = predict(X, W)
    errors = (Y_hat - Y)**2
    return tf.reduce_mean(errors)

def compute_gradients(X, Y, W):
    with tf.GradientTape() as tape:
        loss = loss_mse(Xf, Y, W)
    return tape.gradient(loss, W)

STEPS = 2000
LEARNING_RATE = .02

Xf = make_features(X)
n_weights = Xf.shape[1]

W = tf.Variable(np.zeros((n_weights, 1)), dtype=tf.float32)

# For plotting
steps, losses = [], []
plt.figure()

for step in range(1, STEPS + 1):

    dW = compute_gradients(X, Y, W)
    W.assign_sub(dW * LEARNING_RATE)

    if step % 50 == 0:
        loss = loss_mse(Xf, Y, W)
        print(loss)
        steps.append(step)
        losses.append(loss)
        plt.clf()
        plt.plot(steps, losses)


print("STEP: {} MSE: {}".format(STEPS, loss_mse(Xf, Y, W)))

plt.figure()
plt.plot(X, Y, label='actual')
plt.plot(X, predict(Xf, W), label='predicted')
plt.legend()

