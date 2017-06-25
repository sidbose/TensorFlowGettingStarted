import tensorflow as tf

# load mnist data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# tuning parameters
batch_size = 100
learning_rate = 0.001
training_epochs = 10
logs_path = "tensorboard_logs/"

'''
This method defines layers to our network. 
Args:
    data: input data
    input_dim: number of input nodes to the layer
    output_dim: number of nodes in the layer
    layer_name: layer name to be used in tensorboard graph
    activation_function: activation function if any to be used ex. tf.nn.relu 
'''
def add_layer(data, input_dim, output_dim, layer_name, activation_function=None):
    with tf.name_scope(layer_name):
        # model parameters will change during training so we use tf.Variable
        with tf.name_scope("weights"):
            w = tf.Variable(tf.random_normal([input_dim, output_dim]), name='w')
        # bias
        with tf.name_scope("biases"):
            b = tf.Variable(tf.random_normal([output_dim]), name='b')
        with tf.name_scope('wx_plus_b'):
            wx_plus_b = tf.add(tf.matmul(data, w), b)
            if activation_function is None:
                layer_output = tf.nn.relu(wx_plus_b)
            else:
                layer_output = activation_function(wx_plus_b, name='activation_function')
                tf.summary.histogram("activations", layer_output)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        return layer_output

# input images placeholders
with tf.name_scope('input'):
    # None -> batch size can be any size, 784 -> flattened mnist image
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
    # target 10 output classes
    y = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

# Adding one hidden layer with relu activation
hidden_layer = add_layer(x, 784, 100, 'hidden_layer_1', tf.nn.relu)
# Adding output layer with sigmoid activation
output = add_layer(hidden_layer, 100, 10, 'output_layer', tf.nn.sigmoid)

# implement model
with tf.name_scope('softmax_and_cross_entropy'):
    # this is our cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    tf.summary.scalar('cost', cost)

# specify optimizer
with tf.name_scope('train'):
    # optimizer is an "operation" which we can execute in a session
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope('Accuracy'):
    # Accuracy
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
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
        epoch_loss = 0
        # number of batches in one epoch
        batch_count = int(mnist.train.images.shape[0] / batch_size)

        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # perform the operations we defined earlier on batch
            c, summary = sess.run([cost, summary_op], feed_dict={x: batch_x, y: batch_y})
            epoch_loss += c
            # write log
            writer.add_summary(summary, epoch * batch_count + i)
        print('Epoch:', epoch, ' completed out of ', training_epochs, ' loss:', epoch_loss)

    correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))






