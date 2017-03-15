import constants as C
from NeuralNetwork import NeuralNetwork as NN 
from preprocessing.preprocess import process as pre_processor
import pickle
from load_labels import get_sign_titles
import numpy as np
import tensorflow as tf
import time
import tensorflow.contrib.slim as slim
import PIL.Image as Image
import os

TRAINING_MODE = False
# TODO: Fill this in based on where you saved the training and testing data

training_file = './train.p'
validation_file = './valid.p'
testing_file = './test.p'
model_save_file = C.MODEL_PATH

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

n_train = len(train['features'])
n_test = len(test['features'])
n_valid = len(valid['features'])
image_shape = np.array(train['features'][0]).shape
n_classes = len(np.unique(train['labels']))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

with open('train_aug.p', mode='rb') as f:
    train = pickle.load(f)
X_train, y_train = train['features'], train['labels']

X_train, y_train = pre_processor(X_train, y_train)
X_valid, y_valid = pre_processor(X_valid, y_valid)
X_test, y_test = pre_processor(X_test, y_test)


x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int32, [None, C.N_CLASSES])
keep_prob = tf.placeholder(tf.float32)

def neural_network(x):
     with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm):
         logits = (NN(x)
                   .convolution_layer(output_depth=16, filter_width=3, stride=1, padding='SAME', scope='conv_1', initial_layer=True)
                   .max_pool_layer(filter_width=3, stride=1, padding='SAME', scope='conv1_maxpool')
                   .convolution_layer(output_depth=64, filter_width=5, stride=3, padding='VALID', scope='conv_2')
                   .max_pool_layer(filter_width=3, stride=1, padding='VALID', scope='conv2_maxpool')
                   .convolution_layer(output_depth=128, filter_width=3, stride=1, padding='SAME', scope='conv3')
                   .convolution_layer(output_depth=64, filter_width=3, stride=1, padding='SAME', scope='conv4')
                   .max_pool_layer(filter_width=3, stride=1, padding='VALID', scope='conv4_maxpool')
                   .flatten()
                   .fully_connected_layer(output_depth=1024, scope='fc1')
                   .dropout_layer()
                   .fully_connected_layer(output_depth=1024, scope='fc2')
                   .dropout_layer()
                   .fully_connected_layer(output_depth=C.N_CLASSES, scope='fc3'))

         return logits.current_layer

logits = neural_network(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = C.ALPHA)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, C.EVAL_BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+C.EVAL_BATCH_SIZE], y_data[offset:offset+C.EVAL_BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: C.KEEP_PROB})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def test_on_images(print_top_k=False):
    paths = ['american_signs/' + file for file in os.listdir('american_signs')]
    test_images = []

    for img in paths:
        image = Image.open(img)
        image = image.convert('RGB')
        image = image.resize((32, 32), Image.ANTIALIAS)
        image = np.array(list(image.getdata()))
        image = np.reshape(image, (32, 32, 3))
        test_images.append(image)

    test_images = np.array(test_images, dtype='uint8')
    sign_titles = get_sign_titles()
    test_images, _ = pre_processor(test_images, np.array([0 for _ in range(test_images.shape[0])]))

    with tf.Session() as sess:
        logits = neural_network(test_images)
        predictions = tf.argmax(logits, 1)
        saver = tf.train.Saver()
        saver.restore(sess, model_save_file)
        lgts, actual_predictions = sess.run([logits, predictions], feed_dict={x: test_images, keep_prob: 1.})

    if print_top_k:
        def display_top_k(path, values, indexes):
            print('Top 5 predictions for ', path, ' (prediction: probability)')
            top_k_predictions = [sign_titles[i] for i in indexes]
            for i in range(5):
                print('%s: %.2f%%' % (top_k_predictions[i].replace('\n', ''), values[i] * 100))

        with tf.Session() as sess:
            logits = tf.placeholder('float', [None, C.N_CLASSES])
            k_logits, k_preds = tf.nn.top_k(tf.nn.softmax(logits), k=5)
            top_k_logits, top_k_preds = sess.run([k_logits, k_preds], feed_dict={logits: lgts})

            for i, test_img in enumerate(test_images):
                display_top_k(paths[i], top_k_logits[i], top_k_preds[i])
    else:
        prediction_titles = [sign_titles[pred] for pred in actual_predictions]

        for i in range(test_images.shape[0]):
            print(paths[i], " --- ", prediction_titles[i])

def train_NN(sess, resume=C.RESUME):
    if resume:
        saver = tf.train.Saver()
        saver.restore(sess, model_save_file)
    else:
        sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    print("Training...")
    print()
    total_time = time.time()
    for i in range(C.EPOCHS):
        epoch_time = time.time()
        for offset in range(0, num_examples, C.BATCH_SIZE):
            end = offset + C.BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: C.KEEP_PROB})
        training_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Training Accuracy = {:.3f}".format(training_accuracy),
              "Validation Accuracy = {:.3f}".format(validation_accuracy), "Epoch Time: ", time.time() - epoch_time)
        print()
    print("Time for Training: ", time.time() - total_time)
    saver = tf.train.Saver()
    saver.save(sess, model_save_file)

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


with tf.Session() as sess:
    if not TRAINING_MODE:
        test_on_images()
    else:
        train_NN(sess)

