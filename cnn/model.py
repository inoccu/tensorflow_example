#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

NUM_CLASSES = 10


# 重みとバイアスのことを「モデルのパラメーター」という
# 学習によってパラメーターの値が変化する
# モデルそのものは現在のパラメーターを使って推論する役割しか持たない

def _get_weights(shape, stddev=1.0):
    # tf.get_variableのtrainableは初期値がTrue
    var = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
    return var


def _get_biases(shape, value=0.0):
    # tf.get_variableのtrainableは初期値がTrue
    var = tf.get_variable('biases', shape, initializer=tf.constant_initializer(value))
    return var


def inference(image_node):
    # conv1 畳み込み層1 画像から特徴量を抽出する
    with tf.variable_scope('conv1') as scope:
        # 5x5の大きさを持つ3チャンネルのフィルターを64枚
        weights = _get_weights(shape=[5, 5, 3, 64], stddev=1e-4)
        # 畳み込み tf.nn.conv2d
        # 5x5のフィルターをスライドさせてデータ全体をスキャン
        # 畳み込みで小さくなる値の分のパディングをあらかじめ付加（padding='SAME'）
        conv = tf.nn.conv2d(image_node, weights, [1, 1, 1, 1], padding='SAME')
        # バイアスは畳み込みに使ったフィルターの枚数と同じ値を指定する
        biases = _get_biases([64], value=0.1)
        # バイアスを加算
        bias = tf.nn.bias_add(conv, biases)
        # 活性化関数
        conv1 = tf.nn.relu(bias, name=scope.name)

    # pool1 プーリング層1
    # 畳み込み層で抽出した特徴量を圧縮
    # 微少な位置変化に対する応答不変性を得る
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # conv2 畳み込み層2
    with tf.variable_scope('conv2') as scope:
        weights = _get_weights(shape=[5, 5, 64, 64], stddev=1e-4)
        conv = tf.nn.conv2d(pool1, weights, [1, 1, 1, 1], padding='SAME')
        biases = _get_biases([64], value=0.1)
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)

    # pool2 プーリング層2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # 全結合するTensorは1次元配列である必要があるため、reshapeで平坦化（flatten）
    reshape = tf.reshape(pool2, [1, -1])
    dim = reshape.get_shape()[1].value

    # fc3 全結合層
    # 畳み込みとプーリングによって得られたすべての値を結合
    with tf.variable_scope('fc3') as scope:
        weights = _get_weights(shape=[dim, 384], stddev=0.04)
        biases = _get_biases([384], value=0.1)
        # 平坦化された前層の入力にtf.matmulを使って重みを乗算し、その結果にバイアスを加算
        fc3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # fc4 全結合層
    with tf.variable_scope('fc4') as scope:
        weights = _get_weights(shape=[384, 192], stddev=0.04)
        biases = _get_biases([192], value=0.1)
        fc4 = tf.nn.relu(tf.matmul(fc3, weights) + biases, name=scope.name)

    # output 出力層
    with tf.variable_scope('output') as scope:
        weights = _get_weights(shape=[192, NUM_CLASSES], stddev=1 / 192.0)
        biases = _get_biases([NUM_CLASSES], value=0.0)
        logits = tf.add(tf.matmul(fc4, weights), biases, name='logits')

    return logits
