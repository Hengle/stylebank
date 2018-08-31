import tensorflow as tf
import tensorflow.contrib.slim as slim
import VGG19
import tensorflow.contrib.slim.nets as nets
import os
from config import *

STYLIZING = 0
AUTO_ENCODER = 1
training_branch = STYLIZING  # 0-stylizing 1-auto-encoder
BANK_KERNEL_SIZE = 3
vgg19 = VGG19.VGG19()

@slim.add_arg_scope
def conv_in(x, out_channel, kernel, stride=1, trainable=True):
    out = slim.conv2d(x, out_channel, kernel, stride, trainable=trainable,
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                      weights_regularizer=slim.l2_regularizer(0.0005))
    out = slim.instance_norm(out, trainable=trainable)
    return out


@slim.add_arg_scope
def conv_transpose_in(x, out_channel, kernel, stride=1, trainable=True):
    out = slim.conv2d_transpose(x, out_channel, kernel, stride, trainable=trainable)
    out = slim.instance_norm(out, trainable=trainable)
    return out


def encoder(input_data, training=True):
    """
    :param input_data:[batch,h,w,3]
    :param training:
    :return:feature maps [batch,nh,nw,128]
    """
    with slim.arg_scope([conv_in],
                        trainable=training,
                        ):
        # 仅调用一遍encoder？？如果是，可以用variable_scope来保证
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            net = conv_in(input_data, 32, 9)
            net = conv_in(net, 64, 3, 2)
            net = conv_in(net, 128, 3, 2)
    return net


def decoder(input_data, training=True):
    """
    :param input_data: feature maps [batch,h,w,128]
    :param training:
    :return: image [batch,origin_h,origin_w,3]
    """
    with slim.arg_scope([conv_transpose_in, slim.conv2d_transpose],
                        trainable=training):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            net = conv_transpose_in(input_data, 64, 3, 2)
            net = conv_transpose_in(net, 32, 3, 2)
            net = slim.conv2d_transpose(net, 3, 9, 1)
    return net


def bank(input_data, w, training=True):
    """
    :param input_data: [batch,h,w,128]
    :param w: weight list of the kernel
    :param training:
    :return:
    """
    with slim.arg_scope([conv_in],
                        trainable=training):
        # choose the relu output
        output = 0
        for i in range(STYLE_NUM):
            if w[i] != 0:
                with tf.variable_scope('bank_' + str(i), reuse=tf.AUTO_REUSE):
                    net = conv_in(input_data, 128, BANK_KERNEL_SIZE)
                    output += w[i] * net
        if isinstance(output, int):
            print("w is zero vector. bank output is an Int.")
            exit(1)
    return output

#
# class StyleBank:
#     """
#     main part: Encoder, Stylebank, Decoder
#     :key Two learning branch, E-D and E-SB-D
#
#     """
#     def __init__(self, stylenum, is_training=True):
#         '''
#         :param styles: a list of style include its name,matrix
#         '''
#         self.stylenum = stylenum
#         self.eTraining = is_training
#         self.dTraining = is_training
#         self.bTraining = is_training
#         foo = tf.zeros([1, 512, 512, 3], dtype=tf.float32)
#         _ = self.stylizing_branch(foo)  # initialize all variable
#
#     def encoder(self, input_data):
#         """
#         :param input_data:[batch,h,w,3]
#         :return:feature maps [batch,nh,nw,128]
#         """
#         with slim.arg_scope([conv_in],
#                             trainable=self.eTraining,
#                             ):
#             # 仅调用一遍encoder？？如果是，可以用variable_scope来保证
#             with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
#                 net = conv_in(input_data, 32, 9)
#                 net = conv_in(net, 64, 3, 2)
#                 net = conv_in(net, 128, 3, 2)
#         return net
#
#     def decoder(self, input_data):
#         """
#         :param input_data: feature maps [batch,h,w,128]
#         :return: image [batch,origin_h,origin_w,3]
#         """
#         with slim.arg_scope([conv_transpose_in, slim.conv2d_transpose],
#                             trainable=self.dTraining):
#             with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
#                 net = conv_transpose_in(input_data, 64, 3, 2)
#                 net = conv_transpose_in(net, 32, 3, 2)
#                 net = slim.conv2d_transpose(net, 3, 9, 1)
#         return net
#     '''
#     对于bank如何设置多分枝，根据一个外部的styleid来选择分枝，还有待考虑
#     '''
#     # def bank(self,input_data):
#     #     with slim.arg_scope([conv_in],
#     #                         trainable=self.bTraining):
#     #         for i in range(self.stylenum):
#     #             net = []
#     #             with tf.variable_scope('bank:' + str(i)):
#     #                 net.append(conv_in(input_data, 128, 3))
#     #     # 这里的初值检测需要补充
#     #     return net
#     def bank(self, input_data, w):
#         """
#         :param input_data: [batch,h,w,128]
#         :param w: weight list of the kernel
#         :return:
#         """
#         with slim.arg_scope([conv_in],
#                             trainable=self.bTraining):
#             # choose the relu output
#             output = 0
#             for i in range(self.stylenum):
#                 if w[i] != 0:
#                     with tf.variable_scope('bank_' + str(i), reuse=tf.AUTO_REUSE):
#                         net = conv_in(input_data, 128, BANK_KERNEL_SIZE)
#                         output += w[i] * net
#             if isinstance(output, int):
#                 print("bank output is an Int.")
#                 exit(1)
#         return output
#
#     def forward(self, input_data, w):
#         """
#         :param input_data: 4D-tensor
#         :param style_id: int not tensor
#         :param w: list of weights of style combination
#         :return stylized Image ,4D-tensor
#         """
#         encoded = self.encoder(input_data)
#         stylized = self.bank(encoded, w)
#         decoded = self.decoder(stylized)
#         return decoded
#
#
#     # banned 因为auto_encoder和decoder都是必将被引用的，而stlizing将重复创建stylenum次
#
#     def auto_encoder_branch(self, input_data):
#         encoded = self.encoder(input_data)
#         decoded = self.decoder(encoded)
#         return decoded
#
#     def stylizing_branch(self, input_data):
#         '''
#         :param input_data: 4D-tensor
#         :param style_id: int not tensor
#         :return stylized Image list ,4D-tensor list
#         '''
#         encoded = self.encoder(input_data)
#         decoded_s = []
#         for i in range(self.stylenum):
#             # 除了i标为1其他全为0的权重列表
#             w = [0] * 10
#             w[i] = 1
#             stylized = self.bank(encoded, w)
#             decoded_s.append(self.decoder(stylized))
#         return decoded_s
#
#     def set_fix_encoder(self, flag):
#         pass
#
#     def set_fix_decoder(self, flag):
#         pass
#




