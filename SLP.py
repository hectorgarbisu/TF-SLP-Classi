import tensorflow as tf

class SLP:
    def __init__(self, input_size, hidden_size, output_size, alpha=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W_hidden = tf.Variable(tf.truncated_normal([input_size, hidden_size]))
        self.b_hidden = tf.Variable(tf.truncated_normal([hidden_size]))
        self.W_output = tf.Variable(tf.truncated_normal([hidden_size, output_size]))
        self.b_output = tf.Variable(tf.truncated_normal([output_size]))
        self.x = tf.placeholder("float", [None, self.input_size])  # "None" as dimension for versatility between batches and non-batches
        self.y_ = tf.placeholder("float", [None, self.output_size])
        y_hidden = tf.sigmoid(tf.matmul(self.x, self.W_hidden) + self.b_hidden)
        # y = tf.tanh(tf.matmul(y_hidden, self.W_output) + self.b_output)
        # self.error_measure = tf.reduce_sum(tf.square(self.y_ - y))
        self.y = tf.nn.softmax(tf.matmul(y_hidden, self.W_output) + self.b_output)
        self.error_measure = tf.reduce_mean(tf.reduce_sum(-self.y_*tf.log(self.y)))
        self.train = tf.train.GradientDescentOptimizer(alpha).minimize(self.error_measure)
        self.init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(self.init)

    def feed_batch(self, batch, expected_outputs):
        self.sess.run(self.train, feed_dict={self.x: batch, self.y_: expected_outputs})

    def error(self,batch,expected_outputs):
        return self.sess.run(self.error_measure, feed_dict={self.x: batch, self.y_: expected_outputs})

    def categorize(self, data):
        return self.sess.run(self.y, feed_dict={self.x: data})