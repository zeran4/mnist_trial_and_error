# Lab 11 MNIST and Deep learning CNN
import tensorflow as tf
import random
# import matplotlib.pyplot as plt
import os
import numpy as np
import sklearn.preprocessing as prep

from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(777)  # reproducibility
tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
test_size = len(mnist.test.labels)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

print("Number of mnist.train.images:", len(mnist.train.images))
print("Number of mnist.test.images:", len(mnist.test.images))

LOG_DIR = './MNIST_cnn_log'    # modulename_log 형태로
LOG_PERIOD = 10    # LOG를 남기는 반복 시점 (대개 epoch와 연계)

NewRecord_Accuracy = None
NewRecord_Epoch = None
NewRecord_LOG_LOWLIMIT = 0.995

# hyper parameters
training_epochs = 0
batch_size = 100

def standard_scale(X_train, X_test):    # sklearn-preprocessing-Standardization.ipynb 참고
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def min_max_scale(X_train, X_test):    # sklearn-preprocessing-Standardization.ipynb 참고
    preprocessor = prep.MinMaxScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def get_random_block_from_data(data, data_label, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)], data_label[start_index:(start_index + batch_size)]

X_train = None
X_test = None

#X_train, X_test = min_max_scale(mnist.train.images, mnist.test.images)    # 시각 정보가 잘 보존됨
#X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)    # 0를 center로 하다보니 시각 정보가 왜곡됨
X_train = mnist.train.images
X_test = mnist.test.images

class Model:

    def __init__(self, sess, name, filter=[32,64,128], kernels=[[3,3],[3,3],[3,3]], dropout_rate=[0.3,0.3,0.3,0.5], learning_rate=0.0001):
        self.sess = sess
        self.name = name
        self.learning_rate = learning_rate
        self._build_net(filter, kernels, dropout_rate)

    def _build_net(self, filters, kernels, dropout_rate):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.training = tf.placeholder(tf.bool)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])

            # img 28x28x1 (black/white), Input Layer
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            #bn0 = tf.layers.batch_normalization(inputs=X_img, training=self.training)
            
            # Convolutional Layer #1 and Pooling Layer #1
            conv1 = tf.layers.conv2d(inputs=X_img, filters=filters[0], kernel_size=kernels[0],
                                     padding="SAME", activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            #bn1 = tf.layers.batch_normalization(inputs=conv1, training=self.training)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1,
                                         rate=dropout_rate[0], training=self.training)
            print(dropout1)
            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=filters[1], kernel_size=kernels[1],
                                     padding="SAME", activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            #bn2 = tf.layers.batch_normalization(inputs=conv2, training=self.training)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2,
                                         rate=dropout_rate[1], training=self.training)
            print(dropout2)
            # Convolutional Layer #3 and Pooling Layer #3
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=filters[2], kernel_size=kernels[2],
                                     padding="SAME", activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            #bn3 = tf.layers.batch_normalization(inputs=conv3, training=self.training)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3,
                                         rate=dropout_rate[2], training=self.training)
            print(dropout3)
            # Convolutional Layer #4 and Pooling Layer #4
            #conv4 = tf.layers.conv2d(inputs=dropout3, filters=128, kernel_size=[3, 3],
            #                         padding="SAME", activation=tf.nn.relu,
            #                         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            #bn4 = tf.layers.batch_normalization(inputs=conv3, training=self.training)
            #pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2],
            #                                padding="SAME", strides=2)
            #dropout4 = tf.layers.dropout(inputs=pool4,
            #                             rate=0.7, training=self.training)
            #print(dropout4)
            # Dense Layer with Relu
            flat = tf.reshape(dropout3, [-1, filters[2] * 4 * 4])
            dense5 = tf.layers.dense(inputs=flat,
                                     units=625, activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
            dropout5 = tf.layers.dropout(inputs=dense5,
                                         rate=dropout_rate[3], training=self.training)
            print(dropout5)
            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.logits = tf.layers.dense(inputs=dropout5, units=10,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

last_epoch = tf.Variable(0, name='last_epoch')    

# for ensemble predict for test images
_predictions = np.zeros(test_size * 10, dtype=np.float32).reshape(test_size, 10)
predictions = tf.placeholder(tf.float32, [test_size, 10])
ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))

# initialize
sess = tf.Session()

models = []
num_models = 5
models.append(Model(sess, "model" + str('1'), filter=[64,128,256], kernels=[[9,9],[7,7],[5,5]], dropout_rate=[0.5,0.6,0.7,0.5], learning_rate=0.00005))
models.append(Model(sess, "model" + str('2'), filter=[64,128,256], kernels=[[9,9],[7,7],[5,5]], dropout_rate=[0.5,0.6,0.7,0.5], learning_rate=0.00007))
models.append(Model(sess, "model" + str('3'), filter=[64,128,256], kernels=[[9,9],[7,7],[5,5]], dropout_rate=[0.5,0.6,0.7,0.5], learning_rate=0.00009))
models.append(Model(sess, "model" + str('4'), filter=[64,128,256], kernels=[[9,9],[7,7],[5,5]], dropout_rate=[0.5,0.6,0.7,0.5], learning_rate=0.00011))
models.append(Model(sess, "model" + str('5'), filter=[64,128,256], kernels=[[9,9],[7,7],[5,5]], dropout_rate=[0.5,0.6,0.7,0.5], learning_rate=0.00013))

sess.run(tf.global_variables_initializer())

global_step = 0

# Saver and Restore
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(LOG_DIR)

if checkpoint and checkpoint.model_checkpoint_path:
    try:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    except:
        print("Error on loading old network weights")
else:
    print("Could not find old network weights")

start_from = sess.run(last_epoch)

print('Start learning from:', start_from)

total_batch = int(mnist.train.num_examples / batch_size)

# train my model
for epoch in range(start_from, training_epochs+1):
    print('Start Epoch:', epoch+1)
    
    avg_cost_list = np.zeros(len(models))

    for i in range(total_batch):
        batch_xs, batch_ys = get_random_block_from_data(X_train, mnist.train.labels, batch_size)

        # train each model
        for m_idx, m in enumerate(models):
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)

    # Test model and check accuracy
    _accuracy = 0
    _predictions[:] = 0    # initialize 0
    for m_idx, m in enumerate(models):
        _accuracy = m.get_accuracy(X_test, mnist.test.labels)
        print(m_idx, 'Accuracy:', _accuracy)
        p = m.predict(X_test)
        _predictions += p

    if num_models > 1:    # avoid duplicate run if num_models is 1
        _accuracy = sess.run(ensemble_accuracy, feed_dict={predictions:_predictions})
    if NewRecord_Accuracy == None or NewRecord_Accuracy < _accuracy:
        NewRecord_Accuracy = _accuracy
        NewRecord_Epoch = epoch+1
        print('Ensemble accuracy(New Record!!!!!!!!!!!!!!!!!!!!!!):', _accuracy)

        if NewRecord_LOG_LOWLIMIT <= NewRecord_Accuracy:
            print("Saving network...")
            sess.run(last_epoch.assign(epoch + 1))
            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)
            saver.save(sess, os.path.join(LOG_DIR, "model" + str(int(NewRecord_Accuracy*100000)) + ".ckpt"), global_step=epoch)
            print("Saved")
    else:
        print('Ensemble accuracy:', _accuracy)

print('Finished!')

# code 정리 필요 : Epoch 단계에서 중복된 code를 사용함
# Test model and check accuracy
predictions = np.zeros(test_size * 10).reshape(test_size, 10)
for m_idx, m in enumerate(models):
    print(m_idx, 'Accuracy:', m.get_accuracy(
        X_test, mnist.test.labels))
    p = m.predict(X_test)
    predictions += p

ensemble_correct_prediction = tf.equal(
    tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(
    tf.cast(ensemble_correct_prediction, tf.float32))
_ensemble_accuracy = sess.run(ensemble_accuracy)
print('Ensemble accuracy:', _ensemble_accuracy)
