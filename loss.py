import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import net
import io_process
from config import *

layer_dict = {'relu1_1': 'vgg_16/conv1/conv1_1', 'relu1_2': 'vgg_16/conv1/conv1_2',
              'relu2_1': 'vgg_16/conv2/conv2_1', 'relu2_2': 'vgg_16/conv2/conv2_2',
              'relu3_1': 'vgg_16/conv3/conv3_1', 'relu3_2': 'vgg_16/conv3/conv3_2', 'relu3_3': 'vgg_16/conv3/conv3_3',
              'relu4_1': 'vgg_16/conv4/conv4_1', 'relu4_2': 'vgg_16/conv4/conv4_2', 'relu4_3': 'vgg_16/conv4/conv4_3',
              'relu5_1': 'vgg_16/conv5/conv5_1', 'relu5_2': 'vgg_16/conv5/conv5_2', 'relu5_3': 'vgg_16/conv5/conv5_3'}


def vgg19(img):
    """
    :param img: raw image of type float32, shape [?,H,W,3]
    :return: vgg19 layers output
    """
    processed_img = img - MEAN_PIXEL
    layers = net.vgg19.forward(processed_img)
    return layers


class StyleCalculator:
    def __init__(self, style, save_path=None):
        """
        :param style: A Tensor of type float32, shape [1, H, W, 3]
        :param save_path:
        """
        self.style_grams = {}
        layers_output = vgg19(style)  # ends 是 dict
        for layer in STYLE_LAYERS:
            self.style_grams[layer] = gram(layers_output[layer])
        return

    def loss(self, gen_ends): # 如果用gen的话,求style和content需要计算两遍的gen lay。运算量变大
        """
        :param gen_ends: every layer's output of the vgg19.
        :return: gen's style loss
        """
        style_loss = 0
        for layer in STYLE_LAYERS:
            size = tf.size(gen_ends[layer])
            gen_gram = gram(gen_ends[layer])
            style_loss += tf.nn.l2_loss(gen_gram - self.style_grams[layer])
        return style_loss


class ContentCalculator:
    def __init__(self, content):
        """
        :param content: shape [batch, height, width, 3]
        """
        self.content = {}
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            layers_output = vgg19(content)
        for layer in CONTENT_LAYER:
            self.content[layer] = layers_output[layer]
        return

    def loss(self, gen_ends):
        """
        :param gen_ends: every layer's output
        :return: gen's content loss
        """
        content_loss = 0
        for layer in CONTENT_LAYER:
            _, height, width, channels = map(lambda i: i.value, self.content[layer].get_shape())
            size = height * width * channels
            content_loss += tf.nn.l2_loss(gen_ends[layer] - self.content[layer])
        return content_loss


def auto_encoder_loss(content, reconstructed):
    total_loss = tf.losses.mean_squared_error(content, reconstructed)
    return total_loss


def stylizing_loss(gen, content_cal, style_cal, content_w, style_w, tv_w):
    gen_ends = vgg19(gen)
    content_loss = content_w * content_cal.loss(gen_ends)
    style_loss = style_w * style_cal.loss(gen_ends)
    if tv_w != 0:
        tv_loss = tv_w * total_variation_loss(gen)
        total_loss = style_loss + content_loss + tv_loss
    else:
        total_loss = style_loss + content_loss
    return total_loss


# def loss(input_data, is_encoder, style_mat, style_id, style_w, content_w, tv_w=0):
#     """
#     :param input_data: training batch 4D-tensor [batch,h,w,3]
#     :param is_encoder: bool placeholder controlled by T and global_step
#     :param style_mat: style 4D-tensor [1,h,w,3]
#     :param style_id: style kernel id in bank
#     :param style_w: α
#     :param content_w: β
#     :param tv_w: γ
#     :return: total_loss
#     """
#     style_bank = net.StyleBank()
#     branch_1 = auto_branch_loss(input_data, style_bank)
#     branch_2 = stylize_branch_loss(input_data, style_bank, style_mat, style_id, style_w, content_w, tv_w)
#     total_loss = tf.where(is_encoder, branch_1, branch_2)
#     #
#     # # use T to judge
#     # if auto-encoder branch:
#     #     reconstructed_output = style_bank.auto_encoder_branch(input_data)
#     #     # calculate loss of encoder branch
#     #     # total_loss = auto_branch_loss(input_data,reconstructed_output) # MSE
#     #     total_loss = slim.losses.mean_squared_error(reconstructed_output,input_data)
#     # # elif stylized_branch:
#     # else:
#     #     style_cal = StyleCalculator(style_mat)
#     #     content_cal = ContentCalculator(input_data)
#     #
#     #     stylized_outputs = style_bank.stylizing_branch(input_data)
#     #     stylized_output = stylized_outputs[style_id]
#     #
#     #     # calculate loss of style_branch
#     #     style_loss = style_w * style_cal.loss(stylized_output)
#     #     content_loss = content_w * content_cal.loss(stylized_output)
#     #
#     #     if tv_w != 0:
#     #         tv_loss = tv_w * total_variation_loss(stylized_output)
#     #         total_loss = style_loss + content_loss + tv_loss  # and α β γ
#     #     else:
#     #         total_loss = style_loss + content_loss
#     #     # total_loss = stylize_branch_loss(input_data,style_cal,stylized_output) # and α β γ
#     return total_loss


def total_variation_loss(img):
    shape = img.get_shape()
    height = shape[1].value
    width = shape[2].value
    y = tf.slice(img, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1])) - tf.slice(img, [0, 1, 0, 0],
                                                                                        [-1, -1, -1, -1])
    x = tf.slice(img, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1])) - tf.slice(img, [0, 0, 1, 0],
                                                                                       [-1, -1, -1, -1])
    loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
    return loss


def gram(input_data):
    """
    :param input_data: layer output shapes [batch,height,width,channel]
    :return: [batch, channel, channel]
    Note: this function will change the input. Be sure that the origin input won't use again.
    """
    batch = input_data.shape[0].value
    # height = input_data.shape[1].value
    # weight = input_data.shape[2].value
    channel = input_data.shape[-1].value
    # height = tf.shape(input_data)
    features = tf.reshape(input_data, [batch, -1, channel])
    shape = tf.shape(features)
    size = shape[1] * shape[2]
    # size = features.shape[1].value * features.shape[2].value  # feature.shape[1]是Dimension类型，不是int类型
    gram_matrix = tf.matmul(features, features, transpose_a=True) / tf.cast(size, tf.float32)  # / (channel*h*w) 在后续l2_loss时会被平方
    return gram_matrix                                 # 可以试试/(channel * (h * w) **4) gram先除hw**2再用mse(gram1,gram2)
