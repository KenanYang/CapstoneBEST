# This file is based on the idea of Kevin Xu's tensorflow tutorial
# Retrieved from https://github.com/kevin28520/My-TensorFlow-tutorials
# %%

import tensorflow as tf


# %%
def inference(images, batch_size, n_classes,keep_prob1,keep_prob2):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    # conv1, shape = [kernel size, kernel size, channels, kernel numbers]

    with tf.variable_scope('conv1') as scope:
        weights1 = tf.get_variable('weights',
                                  shape=[20, 20, 3, 24],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases1 = tf.get_variable('biases',
                                 shape=[24],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv1 = tf.nn.conv2d(images, weights1, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation1 = tf.nn.bias_add(conv1, biases1)
        conv11 = tf.nn.relu(pre_activation1, name=scope.name)

    # pool1 and norm1
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        weights2 = tf.get_variable('weights',
                                  shape=[20, 20, 24, 144],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases2 = tf.get_variable('biases',
                                 shape=[144],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv2 = tf.nn.conv2d(norm1, weights2, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation2 = tf.nn.bias_add(conv2, biases2)
        conv22 = tf.nn.relu(pre_activation2, name='conv2')

    # pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        pool2 = tf.nn.max_pool(conv22, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling2')
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm2')


    # conv3
    with tf.variable_scope('conv3') as scope:
        weights3 = tf.get_variable('weights',
                                  shape=[20, 20, 144, 432],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases3 = tf.get_variable('biases',
                                 shape=[432],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv3 = tf.nn.conv2d(norm2, weights3, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation3 = tf.nn.bias_add(conv3, biases3)
        conv33 = tf.nn.relu(pre_activation3, name='conv3')

    # pool3 and norm3
    with tf.variable_scope('pooling3_lrn') as scope:
        pool3 = tf.nn.max_pool(conv33, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling3')
        norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm3')

    # conv4
    with tf.variable_scope('conv4') as scope:
         weights4 = tf.get_variable('weights',
                                   shape=[20, 20, 432, 864],
                                   dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
         biases4 = tf.get_variable('biases',
                                  shape=[864],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
         conv4 = tf.nn.conv2d(norm3, weights4, strides=[1, 1, 1, 1], padding='SAME')
         pre_activation4 = tf.nn.bias_add(conv4, biases4)
         conv44 = tf.nn.relu(pre_activation4, name='conv4')

    # pool4 and norm4
    with tf.variable_scope('pooling4_lrn') as scope:
        pool4 = tf.nn.max_pool(conv44, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling4')
        norm4 = tf.nn.lrn(pool4, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                         beta=0.75, name='norm4')




    # local3   fully connected
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(norm4, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weightsf1 = tf.get_variable('weightsf1',
                                  shape=[dim, 1024],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biasesf1 = tf.get_variable('biasesf1',
                                 shape=[1024],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weightsf1) + biasesf1, name=scope.name)
    # Drop out 1

        local3_drop = tf.nn.dropout(local3, keep_prob1)


        # local4
    with tf.variable_scope('local4') as scope:
        weightsf2 = tf.get_variable('weightsf2',
                                  shape=[1024, 1024],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biasesf2 = tf.get_variable('biasesf2',
                                 shape=[1024],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3_drop, weightsf2) + biasesf2, name='local4')

    # Drop out 2

    local4_drop = tf.nn.dropout(local4, keep_prob2)

# fully connected 3/local5
    with tf.variable_scope('local5') as scope:
        weightsf3 = tf.get_variable('weightsf3',
                                  shape=[1024, 1024],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biasesf3 = tf.get_variable('biasesf3',
                                 shape=[1024],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local5 = tf.nn.relu(tf.matmul(local4_drop, weightsf3) + biasesf3, name='local5')

    # Drop out 3

    local5_drop = tf.nn.dropout(local5, keep_prob2)

    # fully connected 4/local6
    with tf.variable_scope('local6') as scope:
        weightsf4 = tf.get_variable('weightsf4',
                                    shape=[1024, 1024],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biasesf4 = tf.get_variable('biasesf4',
                                   shape=[1024],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))
        local6 = tf.nn.relu(tf.matmul(local5_drop, weightsf4) + biasesf4, name='local6')

    # Drop out 4

    local6_drop = tf.nn.dropout(local6, keep_prob2)

    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights_sl = tf.get_variable('softmax_linear',
                                  shape=[1024, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases_sl = tf.get_variable('biases_sl',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        label_conv = tf.matmul(local6_drop, weights_sl)+biases_sl

    return label_conv


# %%
def losses(logits, labels):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]

    Returns:
        loss tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


# %%
def trainning(loss, learning_rate):
    '''Training ops, the Op returned by this function is what must be passed to
        'sess.run()' call to cause the model to train.

    Args:
        loss: loss tensor, from losses()

    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


# %%
def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).
    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    with tf.variable_scope('accuracy') as scope:
        correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels, 1))
        correct = tf.cast(correct_prediction, tf.float16)
        accuracy = tf.reduce_mean(correct)

        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy

# %%
