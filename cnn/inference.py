#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
import model as model
from reader import Cifar10Reader

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('epoch', 30, "訓練するEpoch数")
tf.app.flags.DEFINE_string('data_dir', './data/', "訓練データのディレクトリ")
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints/', "チェックポイントを保存するディレクトリ")

filenames = [os.path.join(FLAGS.data_dir, 'data_batch_%d.bin' %i) for i in range(1, 6)]


def main(argv=None):
    train_placeholder = tf.placeholder(tf.float32, shape=[32, 32, 3], name='input_image')
    image_node = tf.expand_dims(train_placeholder, 0)
    logits = model.inference(image_node)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        total_duration = 0

        for epoch in range(1, FLAGS.epoch + 1):
            start_time = time.time()

            for file_index in range(5):
                print('Epoch %d: %s' % (epoch, filenames[file_index]))
                reader = Cifar10Reader(filenames[file_index])

                for index in range(10000):
                    image = reader.read(index)

                    # 推論の実行
                    logits_value = sess.run([logits], feed_dict={
                        train_placeholder: image.byte_array,
                    })

                    if index % 1000 == 0:
                        print('[%d]: %r' % (image.label, logits_value))

                reader.close()

            duration = time.time() - start_time
            total_duration += duration

            print('epoch %d duration = %d sec' % (epoch, duration))

            tf.summary.FileWriter(FLAGS.checkpoint_dir, graph=sess.graph)

        print('Total duration = %d sec' % total_duration)


if __name__ == '__main__':
    tf.app.run()