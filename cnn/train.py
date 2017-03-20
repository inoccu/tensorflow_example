#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf
import model as model
from reader import Cifar10Reader

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('epoch', 30, "訓練するEpoch数")
tf.app.flags.DEFINE_string('data_dir', './data/', "訓練データのディレクトリ")
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints/', "チェックポイントを保存するディレクトリ")


# 損失関数を使って推論の結果と正解の誤差（損失）を求める
# logits: 推論の結果、label: 正解
def _loss(logits, label):
    labels = tf.cast(label, tf.int64)
    # 損失関数としてCross Entropy（交差エントロピー）を使用
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean


# 最適化アルゴリズム（Optimizer）を使ってパラメーターを更新する
def _train(total_loss, global_step):
    # 最適化アルゴリズムとして勾配降下法を使用
    # learning_rate: 学習率（学習率は人が決めるのでハイパーパラメーター）
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    grads = opt.compute_gradients(total_loss)
    train_op = opt.apply_gradients(grads, global_step=global_step)
    return train_op


filenames = [os.path.join(FLAGS.data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]


def main(argv=None):
    global_step = tf.Variable(0, trainable=False)

    train_placeholder = tf.placeholder(tf.float32, shape=[32, 32, 3], name='input_image')
    label_placeholder = tf.placeholder(tf.int32, shape=[1], name='label')

    image_node = tf.expand_dims(train_placeholder, 0)
    # 推論
    logits = model.inference(image_node)
    # 損失の計算
    total_loss = _loss(logits, label_placeholder)
    # パラメーターの更新
    train_op = _train(total_loss, global_step)

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

                    # 推論と学習（訓練）の実行
                    _, loss_value, logits_value = sess.run([train_op, total_loss, logits], feed_dict={
                        train_placeholder: image.byte_array,
                        label_placeholder: image.label
                    })

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

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