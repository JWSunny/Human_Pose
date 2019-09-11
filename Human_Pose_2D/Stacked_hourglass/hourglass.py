# -*- coding: utf-8 -*-#
#-------------------------------------
# Name:         hourglass
# Description:  
# Author:       sunjiawei
# Date:         2019/9/6
#-------------------------------------

import tensorflow as tf
import tensorflow.contrib.layers as tcl
import time
import math
import datetime

class StackedHG2:
    def __init__(self, resolution_inp=256, channel=3, name='stackedhg'):
        self.name = name
        self.channel = channel
        self.resolution_inp = resolution_inp

    def res_blk(self, x, num_outputs, kernel_size, stride=1, scope=None):
        """
        參差单元，包含两个分支: 常规的深度分支和shortcut分支，
        深度分支(这里实现的是深层结构resnet50的參差单元结构) 由1个1x1卷积(通道降维)，1个3x3卷积，1个1x1卷积(通道升维) 串联组成，其中每个卷积后都做relu和batchnorm
        shortcut分支有两种情况：当參差单元的输入输出shape不一致时(stride=2)，shortcut包含一个1x1卷积，否则shortcut等于输入x
        输出为shortcut分支和深度分支的元素和(带relu)
        :param x: input tensor
        :param num_outputs: number channels of output
        :param kernel_size:
        :param stride:
        :param scope:
        :return:
        """
        with tf.variable_scope(scope, "resBlk"):
            arg_scope = tf.contrib.framework.arg_scope
            with arg_scope([tcl.conv2d],
                           activation_fn=tf.nn.relu,
                           normalizer_fn=tcl.batch_norm,
                           padding="SAME"):
                small_ch = num_outputs // 4

                conv1 = tcl.conv2d(x, small_ch, kernel_size=1, stride=stride)
                conv2 = tcl.conv2d(conv1, small_ch, kernel_size=kernel_size, stride=1)
                conv3 = tcl.conv2d(conv2, num_outputs, kernel_size=1, stride=1)

                shortcut = x
                if stride != 1 or x.get_shape()[-1] != num_outputs:
                    shortcut = tcl.conv2d(x, num_outputs, kernel_size=1, stride=stride, scope="shortcut")

                out = tf.add(conv3, shortcut)
                out = tf.nn.relu(out)
                return out

    def hour_glass(self, x, level, num_outputs, scope=None):
        """
        single hour glass network 升级版. 可看做一个递归过程: hg(n)的输入x经过两个分支:下采样分支和求和分支，
        求和分支是一个残差快(resblock), 下采样分支是一个 maxpool-resblock 串联 一个残差快[n=1时]或hg(n-1)，
        然后hg(n-1)经过 resblock-上采样 后会求和分支进行按元素相加，输出相加的结果
        :param x:input tensor
        :param level: times of down sampling, i.e., hg(n) n的最大值
        :param num_outputs: number of output channel
        :param scope:
        :return:
        """
        with tf.variable_scope(scope, 'hourglass'):
            add_branch = self.res_blk(x, num_outputs, 3, 1, scope='up1')

            down_sampling = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], 'VALID')
            down_sampling = self.res_blk(down_sampling, num_outputs, 3, 1, scope='low1')

            if level > 1:
                center = hour_glass1(down_sampling, level - 1, num_outputs, scope='low2')
            else:
                center = self.res_blk(down_sampling, num_outputs, 3, 1, scope='low2')

            up_sampling = self.res_blk(center, num_outputs, 3, 1, scope='low3')
            up_sampling = tf.image.resize_nearest_neighbor(up_sampling, tf.shape(up_sampling)[1:3] * 2,
                                                           name='upsampling')
            add_out = tf.add(add_branch, up_sampling)
        return add_out

    def __call__(self, x, stage=4, is_training=True):
        """
        堆叠多个HG。由基础网络，stage x HG 串联组成，
        基础网络是 1个7x7卷积，1个參差，1个池化，2个參差串联组成
        HG网络包括hourglass 和 post网络组成，hourglass 的输出经过1个參差，1个卷积-relu-bn, 1个卷积(1x1，)输出N_landmark个热度图
        第i(i>1)个HG的输入是(i-1)个HG 中3部分的元素和: 输入, 输出层out经1x1卷积， 输出out的上一层经过1x1卷积。

        :param x: input tensor [batch, 256,256,3]
        :param stage: int, number of hourglass to stack, default is 4
        :param is_training: bool, train of test
        :return:
        """
        with tf.variable_scope(self.name) as scope:
            arg_scope = tf.contrib.framework.arg_scope
            with arg_scope([tcl.batch_norm], is_training=is_training, scale=True):
                with arg_scope([tcl.conv2d],
                               activation_fn=None,
                               padding="SAME"):
                    base = tcl.conv2d(x, 64, kernel_size=7, stride=2,
                                      activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
                    base = self.res_blk(base, 128, 3, 1)
                    base = tcl.avg_pool2d(base, kernel_size=2, stride=2)
                    base = self.res_blk(base, 128, 3, 1)
                    base = self.res_blk(base, 256, 3, 1)

                    inputs = base
                    for i in range(0, stage):
                        with tf.variable_scope('hg%d' % i):
                            hg = self.hour_glass(inputs, 4, 256)
                            # post
                            top_hg = self.res_blk(hg, 256, 3, 1)
                            previous = tcl.conv2d(top_hg, 256, kernel_size=1, stride=1,
                                                  activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
                            out = tcl.conv2d(previous, 68, kernel_size=1, stride=1)

                            if i < stage - 1:
                                al = tcl.conv2d(out, 256, kernel_size=1, stride=1)
                                bl = tcl.conv2d(previous, 256, kernel_size=1, stride=1)
                                sum_ = tf.add(bl, inputs)
                                sum_ = tf.add(sum_, al)
                                inputs = sum_

                    return out


batch_size = 16
num_batches = 100


def time_tensorflow_run(session, target, feed, info_string):
    """
    calculate time for each session run
    :param session: tf.Session
    :param target: opterator or tensor need to run with session
    :param feed: feed dict for session
    :param info_string: info message for print
    :return:
    """
    num_steps_burn_in = 10  # 预热轮数
    total_duration = 0.0  # 总时间
    total_duration_squared = 0.0  # 总时间的平方和用以计算方差
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target, feed_dict=feed)

        duration = time.time() - start_time

        if i >= num_steps_burn_in:  # 只考虑预热轮数之后的时间
            if not i % 10:
                print('[%s] step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration

    mn = total_duration / num_batches  # 平均每个batch的时间
    vr = total_duration_squared / num_batches - mn * mn  # 方差
    sd = math.sqrt(vr)  # 标准差
    print('[%s] %s across %d steps, %.3f +/- %.3f sec/batch' % (datetime.now(), info_string, num_batches, mn, sd))


# test demo
def run_benchmark():
    """
    main function for test or demo
    :return:
    """
    with tf.Graph().as_default():
        image_size = 256  # 输入图像尺寸
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], dtype=tf.float32, stddev=1e-1))

        # method 0
        # prediction = hour_glass(images, 256, "hg")
        # prediction = hour_glass1(images, 3, 256, "hg")
        model = StackedHG2(image_size, 3)
        prediction = model(images, 4)
        fc = prediction

        params = tf.trainable_variables()

        for v in params:
            print(v)
        init = tf.global_variables_initializer()

        print("out shape ", prediction)
        sess = tf.Session()
        print("init...")
        sess.run(init)

        print("predict..")
        writer = tf.summary.FileWriter("./logs")
        writer.add_graph(sess.graph)
        time_tensorflow_run(sess, prediction, {}, "Forward")

        # 用以模拟训练的过程
        objective = tf.nn.l2_loss(fc)  # 给一个loss
        grad = tf.gradients(objective, params)  # 相对于loss的 所有模型参数的梯度

        print('grad backword')
        time_tensorflow_run(sess, grad, {}, "Forward-backward")
        writer.close()


if __name__ == '__main__':
    run_benchmark()