# This file is based on the idea of Kevin Xu's tensorflow tutorial
# Retrieved from https://github.com/kevin28520/My-TensorFlow-tutorials

# %%

import os
import numpy as np
import tensorflow as tf

import import_data
import model

# %%

N_CLASSES = 6
IMG_W = 150  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 150
BATCH_SIZE = 20
CAPACITY = 256
MAX_STEP = 5000  # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001  # with current parameters, it is suggested to use learning rate<0.0001
keep_prob1 = 0.5
keep_prob2 = 0.5


# %%
def run_training():
    # you need to change the directories to yours.
    train_dir = '/home/awsgui/Desktop/TrainingSet/'
    logs_train_dir = '/home/awsgui/PycharmProjects/pyproject/logs/'

    train, train_label = import_data.get_files(train_dir)

    train_batch, train_label_batch = import_data.get_batch(train,
                                               train_label,
                                               IMG_W,
                                               IMG_H,
                                               BATCH_SIZE,
                                               CAPACITY)
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES, keep_prob1, keep_prob2)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.trainning(train_loss, learning_rate)
    train__acc = model.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])

            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 5000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

