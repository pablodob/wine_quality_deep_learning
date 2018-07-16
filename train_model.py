import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

SCRIPT_DIR = Path.cwd()
sys.path.append(str(SCRIPT_DIR.parent))

from data_utils.data_iterator import DataIterator

training_data = np.load(open("training_data/winered_training_data.npy", 'rb'))
training_labels = np.load(open("training_data/winered_training_labels.npy", 'rb'))
validation_data = np.load(open("training_data/winered_validation_data.npy", 'rb'))
validation_labels = np.load(open("training_data/winered_validation_labels.npy", 'rb'))
test_data = np.load(open("training_data/winered_test_data.npy", 'rb'))
test_labels = np.load(open("training_data/winered_test_labels.npy", 'rb'))

# Hyperparameters
epochs = 50
learning_rate = 0.001
batch_size = 50

# Parameters
layers_cant = 3
layer_units = 50
layer_units_regresive = True
hidden_activation = tf.nn.relu

x = tf.placeholder(dtype=tf.float32, shape=[None, training_data.shape[1]], name="inputs")
y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="labels")

fully_connected = [x]
for i in range(layers_cant):
    hidden_name = "hidden_"+str(i)
    if i == layers_cant - 1:
        hidden_name = "output"
        layer_units = 1
    elif i == 0:
        hidden_name = "input"

    fully_connected.append(tf.layers.dense(fully_connected[i], units=layer_units, activation=hidden_activation, name=hidden_name))


    if layer_units_regresive:
        layer_units = layer_units//2

init = tf.global_variables_initializer()

with tf.name_scope("Cost_and_Optimizer"):
    cost = tf.reduce_mean(tf.square(fully_connected[-1] - y), name="cost")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cost)

print(cost)

tf.summary.scalar("Cost", cost)
for i in range(layers_cant):
    layer_name = "hidden_"+str(i)
    if i == layers_cant - 1:
        layer_name = "output"
    elif i == 0:
        layer_name = "input"
    tf.summary.histogram(layer_name, fully_connected[0])
    tf.summary.histogram("output_layer", fully_connected[-1])

    with tf.variable_scope(layer_name, reuse=True):
        weights = tf.get_variable('kernel')

    tf.summary.histogram("weights_" + layer_name, weights)

training_iterator = DataIterator(training_data, training_labels, batch_size=batch_size)


with tf.Session() as sess:
    sess.run(init)

    merged_summary = tf.summary.merge_all()

    # Handle old tensorboard file with same hyperparameters
    tensorboard_job_name = "lr_{}-e_{}-b_{}".format(learning_rate, epochs, batch_size)
    tensorboard_log_dir = Path(Path.cwd(), "tensoboard_logs", tensorboard_job_name)
    writer = tf.summary.FileWriter("./tensoboard_logs/{}".format(tensorboard_job_name))

    if len(list(tensorboard_log_dir.iterdir())) > 0:
        for file in list(tensorboard_log_dir.iterdir()):
            file.unlink()

    writer.add_graph(tf.get_default_graph())

    training_steps = 0

    for e in range(epochs):

        for i, training_data_batch, training_labels_batch in training_iterator:

            if i % 5 == 0:
                output, batch_cost = sess.run([fully_connected[-1], cost], feed_dict={x: training_data_batch,
                                                                               y: training_labels_batch})

                predictions = np.rint(output)
                accuracy = np.sum(training_labels_batch == predictions) / training_labels_batch.shape[0]

                validation_output, validation_cost = sess.run([fully_connected[-1], cost], feed_dict={x: validation_data,
                                                                                               y: validation_labels})
                validation_predictions = np.rint(validation_output)
                validation_accuracy = np.sum(validation_labels == validation_predictions) / validation_labels.shape[0]

                print("Epoch N°: {} - Batch N°: {}\n"
                      "Training cost = {} - Training accuracy = {}%\n"
                      "Validation cost = {} - Validation accuracy = {}%".format(e, i, batch_cost, accuracy * 100,
                                                                                validation_cost,
                                                                                validation_accuracy * 100))

            s = sess.run(merged_summary, feed_dict={x: training_data_batch, y: training_labels_batch})
            writer.add_summary(s, training_steps)
            training_steps += 1

            sess.run(train, feed_dict={x: training_data_batch, y: training_labels_batch})

    output_test, cost_test = sess.run([fully_connected[-1], cost], feed_dict={x : test_data, y : test_labels})

    test_predictions = np.rint(output_test)
    test_accuracy = np.sum(test_labels == test_predictions) / test_labels.shape[0]

    print("Test cost = {} - Test accuracy = {}%".format(cost_test, test_accuracy * 100))
