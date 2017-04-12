# Lab 6 Softmax Classifier
import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility

# Predicting animal type based on various features

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

learning_rate = 0.001
training_epochs = 20
batch_size = 170

def read_my_file_format(filename_queue):
  reader = tf.SomeReader()
  key, record_string = reader.read(filename_queue)
  example, label = tf.some_decoder(record_string)
  processed_example = tf.some_processing(example)
  return processed_example, label

def input_pipeline(filenames, batch_size, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
  example, label = read_my_file_format(filename_queue)
  # min_after_dequeue defines how big a buffer we will randomly sample
  #   from -- bigger means better shuffling but slower start up and more
  #   memory used.
  # capacity must be larger than min_after_dequeue and the amount larger
  #   determines the maximum we will prefetch.  Recommendation:
  #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  example_batch, label_batch = tf.train.shuffle_batch(
      [example, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return example_batch, label_batch

train_x_batch, train_y_batch = input_pipeline('data-04-zoo.csv',batch_size=batch_size,num_epochs=training_epochs)


#print(x_data.shape, y_data.shape)

nb_classes = 4  # 0 ~ 3

X = tf.placeholder(tf.float32, [None, 5])
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 3
Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape", Y_one_hot)

keep_prob = tf.placeholder(tf.float32)
with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal([5, 20]), name='weight1')
    b1 = tf.Variable(tf.random_normal([20]), name='bias1')
    layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)

    w1_hist = tf.summary.histogram("weights1", W1)
    b1_hist = tf.summary.histogram("bias1", b1)
    layer1_hist = tf.summary.histogram("layer1", layer1)

with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_normal([20, 20]), name='weight2')
    b2 = tf.Variable(tf.random_normal([20]), name='bias2')
    layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
    layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

    w2_hist = tf.summary.histogram("weights2", W2)
    b2_hist = tf.summary.histogram("bias2", b2)
    layer2_hist = tf.summary.histogram("layer2", layer2)

with tf.name_scope("layer3") as scope:
    W3 = tf.Variable(tf.random_normal([20, 5]), name='weight3')
    b3 = tf.Variable(tf.random_normal([5]), name='bias3')
    layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)
    layer3 = tf.nn.dropout(layer3, keep_prob=keep_prob)

    w3_hist = tf.summary.histogram("weights1", W3)
    b3_hist = tf.summary.histogram("bias1", b3)
    layer3_hist = tf.summary.histogram("layer3", layer3)

with tf.name_scope("layer4") as scope:
    W4 = tf.Variable(tf.random_normal([5, nb_classes]), name='weight4')
    b4 = tf.Variable(tf.random_normal([nb_classes]), name='bias4')
    layer4 = tf.matmul(layer3, W4) + b4

    w4_hist = tf.summary.histogram("weights4", W4)
    b4_hist = tf.summary.histogram("bias4", b4)
    layer4_hist = tf.summary.histogram("layer4", layer4)

saver = tf.train.Saver()

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(layer4)

with tf.name_scope("cost") as scope:
    # Cross entropy cost/loss
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer4,
                                                                  labels=Y_one_hot))
    cost_summ = tf.summary.scalar("cost", cost)

with tf.name_scope("train") as scope:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_summ = tf.summary.scalar("accuracy", accuracy)

# Launch graph
with tf.Session() as sess:
    # tensorboard --logdir=./logs/xor_logs
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/ubitus")
    writer.add_graph(sess.graph)  # Show the graph

    sess.run(tf.global_variables_initializer())

    #saver.restore(sess, "/tmp/model3.ckpt")

    for step in range(200000):
        feed_dict = {X: train_x_batch, Y: train_y_batch, keep_prob: 0.7}
        summary, _ = sess.run([merged_summary, optimizer], feed_dict=feed_dict)
        writer.add_summary(summary, global_step=step)

        if step % 200 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict=
                feed_dict)
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, loss, acc))
            save_path = saver.save(sess, "/tmp/model3.ckpt")
'''
    pred = sess.run(prediction, feed_dict={X: [[7,4,0.1303,0.00,0.01],[17,4,0.1304,0.00,0.05]]})

    for p in zip(pred, y_data.flatten()):

        print("Prediction: {} ".format(p))
'''

'''
    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})

    #y_data: (N,1) = flatten => (N, ) matches pred.shape
    i=0

    for p, y in zip(pred, y_data.flatten()):
        i = i +1
        print(i,end='')
        print(" row [{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

'''

'''
Step:     0 Loss: 5.106 Acc: 37.62%
Step:   100 Loss: 0.800 Acc: 79.21%
Step:   200 Loss: 0.486 Acc: 88.12%
Step:   300 Loss: 0.349 Acc: 90.10%
Step:   400 Loss: 0.272 Acc: 94.06%
Step:   500 Loss: 0.222 Acc: 95.05%
Step:   600 Loss: 0.187 Acc: 97.03%
Step:   700 Loss: 0.161 Acc: 97.03%
Step:   800 Loss: 0.140 Acc: 97.03%
Step:   900 Loss: 0.124 Acc: 97.03%
Step:  1000 Loss: 0.111 Acc: 97.03%
Step:  1100 Loss: 0.101 Acc: 99.01%
Step:  1200 Loss: 0.092 Acc: 100.00%
Step:  1300 Loss: 0.084 Acc: 100.00%
...
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 3 True Y: 3
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 3 True Y: 3
[True] Prediction: 3 True Y: 3
[True] Prediction: 0 True Y: 0
'''
