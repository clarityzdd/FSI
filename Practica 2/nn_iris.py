import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]

# Cambia de int a one_hot (vectores binarios)

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


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

# 70% de las muestras para entrenamiento
x_train = data[0:106, 0:4].astype('f4')
y_train = one_hot(data[0:106, 4].astype(int), 3)

# 15% de las muestras para validacion
x_validate = data[106:128, 0:4].astype('f4')
y_validate = one_hot(data[106:128, 4].astype(int), 3)

# 15% de las muestras para test
x_test = data[128:150, 0:4].astype('f4')
y_test = one_hot(data[128:150, 4].astype(int), 3)

print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

# Variables capa 1

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)        # 4 entradas 5 neuronas
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)           # 5 bias por neurona

# Variables capa 2

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)        # 5 entradas 3 neuronas
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)           # 3 salidas

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)        # Salidas capa 1 y entradas capa 2
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)        # Salidas capa 2

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

for epoch in xrange(100):
    for jj in xrange(len(x_train) / batch_size):
        batch_xs = x_train[jj * batch_size: jj * batch_size + batch_size]        # Muestras de la epoca actual
        batch_ys = y_train[jj * batch_size: jj * batch_size + batch_size]        # Etiquetas correspondientes a batch_xs
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})                   # Entrenamiento

    # Valida y muestra resultados
    print "Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: x_validate, y_: y_validate})
    print "----------------------------------------------------------------------------------"

# Test
print "Resultados del test:"
print "----------------------------------------------------------------------------------"
result = sess.run(y, feed_dict={x: x_test})
for b, r in zip(y_test, result):
    print b, "-->", r
print "----------------------------------------------------------------------------------"
