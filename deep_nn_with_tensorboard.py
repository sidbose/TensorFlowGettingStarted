import tensorflow as tf

# load mnist data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# tuning parameters
batch_size = 100
learning_rate = 0.01
training_epochs = 10
logs_path = "tensorboard_logs/"

# input images
with tf.name_scope('input'):
    # None -> batch size can be any size, 784 -> flattened mnist image
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
    # target 10 output classes
    y = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

with tf.name_scope('hidden_layer'):
    # model parameters will change during training so we use tf.Variable
    with tf.name_scope("weights"):
        w = tf.Variable(tf.zeros([784, 100]), name='w')
    # bias
    with tf.name_scope("biases"):
        b = tf.Variable(tf.zeros([100]), name='b')
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.add(tf.matmul(x, w), b)
        layer_output = tf.nn.relu(wx_plus_b, name='relu')

with tf.name_scope('output_layer'):
    w_out = tf.Variable(tf.zeros([100, 10]))
    b_out = tf.Variable(tf.zeros([10]))
    output = tf.add(tf.matmul(layer_output, w_out), b_out)

# implement model
with tf.name_scope('softmax_and_cross_entropy'):
    # this is our cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))

# specify optimizer
with tf.name_scope('train'):
    # optimizer is an "operation" which we can execute in a session
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope('Accuracy'):
    # Accuracy
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# create a summary for our cost and accuracy
tf.summary.scalar('cost', cost)
tf.summary.scalar('accuracy', accuracy)

# merge all summaries into a single "operation" which we can execute in a session
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    # variables need to be initialized before we can use them
    sess.run(tf.global_variables_initializer())

    # create log writer object
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # perform training cycles
    for epoch in range(training_epochs):

        # number of batches in one epoch
        batch_count = int(55000 / batch_size)

        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # perform the operations we defined earlier on batch
            _, summary = sess.run([train_op, summary_op], feed_dict={x: batch_x, y: batch_y})

            # write log
            writer.add_summary(summary, epoch * batch_count + i)

        if epoch % 5 == 0:
            print
            "Epoch: ", epoch
    print("Accuracy: ", accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    print("done")



