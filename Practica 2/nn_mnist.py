import gzip
import cPickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
validate_x, validate_y = valid_set
test_x, test_y = test_set


# ---------------- Visualizing some element of the MNIST dataset --------------
"""
import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()  # Let's see a sample
print train_y[57]
"""

# TODO: the neural net!!

# 70% de las muestras para entrenamiento
x_train = train_x
y_train = one_hot(train_y, 10)

# 15% de las muestras para validacion
x_validate = validate_x
y_validate = one_hot(validate_y, 10)

# 15% de las muestras para test
x_test = test_x
y_test = one_hot(test_y, 10)

x = tf.placeholder("float", [None, 28*28])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

# Variables capa

W1 = tf.Variable(np.float32(np.random.rand(28*28, 10)) * 0.1)        # 28*28  entradas 10 neuronas
b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)           # 10 bias por neurona

y = tf.nn.softmax(tf.matmul(x, W1) + b1)        # Salidas capa

loss = tf.reduce_sum(tf.square(y_ - y))

# Objeto optimizador + minimizar "loss"

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20     # Lotes de muestras
epoch_error = sess.run(loss, feed_dict={x: x_validate, y_: y_validate})
before_error = 10000
epoch = 0
while(before_error - epoch_error > 0.75):
    for jj in xrange(len(x_train) / batch_size):
        batch_xs = x_train[jj * batch_size: jj * batch_size + batch_size]        # Muestras de la epoca actual
        batch_ys = y_train[jj * batch_size: jj * batch_size + batch_size]        # Etiquetas correspondientes a batch_xs
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})                   # Entrenamiento

    # Valida y muestra resultados
    before_error = epoch_error
    epoch_error = sess.run(loss, feed_dict={x: x_validate, y_: y_validate})
    print "Epoch #:", epoch, "Error: ", epoch_error
    print "----------------------------------------------------------------------------------"
    epoch += 1

# Test
print "Resultados del test:"
print "----------------------------------------------------------------------------------"
result = sess.run(y, feed_dict={x: x_test})

success = 0
fail = 0
for b, r in zip(y_test, result):
    if (np.argmax(b) == np.argmax(r)):
        success += 1
    else:
        fail += 1
    #print b, "-->", r
total = success + fail
print "Numero de aciertos: " , success
print "Numero de fallos: " , fail
print "Porcentaje de aciertos: " , (float(success) / float(total)) * 100 , "%"
print "----------------------------------------------------------------------------------"
