# This file is based on the idea of Kevin Xu's tensorflow tutorial
# Retrieved from https://github.com/kevin28520/My-TensorFlow-tutorials

import tensorflow as tf
import model

IMG_H = 150
IMG_W = 150
BATCH_SIZE = 1
N_CLASSES = 6
logs_train_dir = '/Users/kenanyang/PycharmProjects/CombineP6/logs/'


class Evaluation:
    def __init__(self):
        print('Evaluation Initialization')
        self.sess = tf.Session()
        self.x = tf.placeholder(tf.float32, shape=[1, IMG_H, IMG_W, 3])
        self.logit = model.inference(self.x, BATCH_SIZE, N_CLASSES, 1, 1)
        self.logit = tf.nn.softmax(self.logit)
        self.saver = tf.train.Saver()

        self.ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if self.ckpt and self.ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
        print('Evaluation Initialization Done')

    def get_one_image(self, page, image_H, image_W):
        '''Randomly pick one image from training data
        Return: ndarray
        '''

        file_path = "/Users/kenanyang/PycharmProjects/CombineP6/upload/" + str(page) + "/test.jpg"
        file = tf.cast(file_path, tf.string)
        image_contents = tf.read_file(file)
        image = tf.image.decode_jpeg(image_contents, channels=3)
        height = self.sess.run(tf.shape(image[:, 0][0]))
        width = self.sess.run(tf.shape(image[0, :][0]))

        if height > width:
            new_h = image_H
            new_w = new_h * width / height
        else:
            new_w = image_W
            new_h = new_w * height / width
        # if you want to test the generated batches of images, you might want to comment the following line.
        image = tf.image.resize_images(image, [int(new_h), int(new_w)])

        image = tf.image.resize_image_with_crop_or_pad(image, image_H, image_W)

        image = tf.image.per_image_standardization(image)

        return image

    def evaluate_one_image(self, page):
        import time
        start = time.time()
        image_array = self.get_one_image(page, IMG_H, IMG_W)
        image = tf.reshape(image_array, [-1, IMG_H, IMG_W, 3])
        prediction = self.sess.run(self.logit, feed_dict={self.x: self.sess.run(image)})

        max_index = tf.argmax(prediction, 1)

        end = time.time()
        duration = end - start
        if self.sess.run(max_index) == 0:
            #    print('This is raymond with possibility %.6f' % prediction[:, 0])
            return 'Like Diamond! This is raymond - run time: ' + str(duration)
        elif self.sess.run(max_index) == 1:
            #    print('This is willsonhall with possibility %.6f' % prediction[:, 1])
            return 'everybody all! This is Willsonhall - run time: ' + str(duration)
        elif self.sess.run(max_index) == 2:
            #   print('This is bio with possibility %.6f' % prediction[:, 2])
            return 'Test! Test ! This is biochemist - run time: ' + str(duration)
        elif self.sess.run(max_index) == 3:
            #    print('This is EEE with possibility %.6f' % prediction[:, 3])
            return 'High technology! This is EEE - run time: ' + str(duration)
        elif self.sess.run(max_index) == 4:
            #   print('This is oldE with possibility %.6f' % prediction[:, 4])
            return 'Dancing and Sing! This is Old Engineering - run time: ' + str(duration)
        elif self.sess.run(max_index) == 5:
            #    print('This is oldM with possibility %.6f' % prediction[:, 5])
            return 'Gee Gee Gee Gee! This is Old Metallurgy - run time: ' + str(duration)




