# This file is based on the idea of Kevin Xu's tensorflow tutorial
# Retrieved from https://github.com/kevin28520/My-TensorFlow-tutorials

# %%

import tensorflow as tf
import numpy as np
import os

# %%

# you need to change this to your data directory
#train_dir = '/home/awsgui/Desktop/TrainingSet/'


def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    raymond = []
    label_raymond = []
    willsonhall = []
    label_willsonhall = []
    biochemist = []
    label_biochemist = []
    EEE = []
    label_EEE = []
    OldE = []
    label_OldE = []
    OldM = []
    label_OldM = []
    for file in os.listdir(file_dir):
        name = file.split(sep='_')
        if name[0] == 'raymond':
            raymond.append(file_dir + file)
            label_raymond.append(0)
        elif name[0] == 'willsonhall':
            willsonhall.append(file_dir + file)
            label_willsonhall.append(1)
        elif name[0] == 'biochemist':
            biochemist.append(file_dir + file)
            label_biochemist.append(2)
        elif name[0] == 'EEE':
            EEE.append(file_dir + file)
            label_EEE.append(3)
        elif name[0] == 'OldE':
            OldE.append(file_dir + file)
            label_OldE.append(4)
        else:
            OldM.append(file_dir + file)
            label_OldM.append(5)


    print('There are %d raymond\nThere are %d willsonhall\nThere are %d biochemist\n \
          There are %d EEE\nThere are %d OldE\nThere are %d OldM'\
          % (len(raymond), len(willsonhall),len(biochemist),len(EEE),len(OldE),len(OldM)))

    image_list = np.hstack((raymond, willsonhall, biochemist,EEE,OldE,OldM))
    label_list = np.hstack((label_raymond, label_willsonhall, label_biochemist,label_EEE,label_OldE,label_OldM))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    label_list = tf.one_hot(indices=label_list, depth=6)
    return image_list, label_list


# %%

def get_batch(image, label, image_W, image_H, batch_size, capacity):

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    sess = tf.Session()
    height = sess.run(tf.shape(image[:, 0][0]))
    width = sess.run(tf.shape(image[0, :][0]))

    if height > width:
        new_h = image_H
        new_w = new_h * width/height
    else:
        new_w = image_W
        new_h = new_w * height/width
    # if you want to test the generated batches of images, you might want to comment the following line.
    image = tf.image.resize_images(image, [int(new_h), int(new_w)])
    image = tf.image.resize_image_with_crop_or_pad(image, image_H, image_W)

    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=capacity)

    label_batch = tf.reshape(label_batch, [batch_size, 6])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch

