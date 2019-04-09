from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# load mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# hyperparameters
lr = 0.001      #learning rate
training_iters = 100000
batch_size = 128

n_inputs = 28   # MNIST data input (img shape: 28*28)
n_steps = 28    # time steps
n_hidden_units = 128   # neurons in hidden layer
n_classes = 10      # MNIST classes (0-9 digits)

# define placeholder for input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# define w and b
weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),       # 每个cell输入的全连接层参数(28,128)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))      # 定义用于输出的全连接层参数(128,10)
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),         # (128, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))              # (10, )
}

def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    ##########################################

    # basic LSTM Cell.
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)  # 初始的bias=1,不希望遗忘任何信息
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state,  time_major=False)  # time_major的意思是：是否steps为第一个参数，这里不是，则false

    # hidden layer for output as the final results
    #############################################
    # final_state为[2,80,128]     则final_state[1]为[80,128]
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # # or
    # unpack to list [(batch, outputs)..] * steps
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # shape = (128, 10)

    return results


pred = RNN(x, weights, biases)   # shape:(128, 10)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  # tf.argmax()返回最大数值的下标 通常和tf.equal()一起使用，计算模型准确度
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# init session
sess = tf.Session()
# init all variables
sess.run(tf.global_variables_initializer())
# start training


with tf.Session() as sess:
    # 初始化
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    # 持续迭代
    while step * batch_size < training_iters:
        # 随机抽出这一次迭代训练时用的数据
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 对数据进行处理，使得其符合输入
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        #迭代
        sess.run([train_op], feed_dict={x: batch_xs,y: batch_ys,})
        # 在特定的迭代回合进行数据的输出
        if step % 20 == 0:
            #输出准确度
            print(sess.run(accuracy, feed_dict={x: batch_xs,y: batch_ys,}))
        step += 1
        # test_data = mnist.test.images
        # test_label = mnist.test.labels
        # print("Testing Accuracy: ", sess.run(accuracy, feed_dict={x: test_data, y: test_label,batch_size:test_data.shape[0]}))