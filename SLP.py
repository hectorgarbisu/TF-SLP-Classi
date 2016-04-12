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
        self.init = tf.initialize_all_variables()
        x = tf.placeholder("float", [None, self.input_size])  # "None" as dimension for versatility between batches and non-batches
        y_ = tf.placeholder("float", [None, self.output_size])
        y_hidden = tf.tanh(tf.matmul(x, self.W_hidden) + self.b_hidden)
        y = tf.tanh(tf.matmul(y_hidden, self.W_output) + self.b_output) # If 2 layers
        error_measure = tf.reduce_sum(tf.square(y_ - y))
        self.train = tf.train.GradientDescentOptimizer(alpha).minimize(error_measure)
        self.sess = tf.Session()

    def feed_batch(self, batch, labels):
        print "hi"
