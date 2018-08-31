import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from config import *
import io_process
import loss
import net
import numpy as np
from tensorflow.contrib.slim.python.slim.learning import train_step


# def train_step_fn(session, train_ops, *args, **kwargs):
#     step = train_step_fn.step
#     if step / T == T - 1:
#         train_op = train_ops[STYLE_NUM]
#         select = "auto-encoder"
#     else:
#         i = (step - step / T) % STYLE_NUM  #
#         train_op = train_ops[i]
#         select = "bank_kernel" + i
#     total_loss, should_stop = train_step(session, train_op, *args, **kwargs)
#     print("global_step:", step)
#     print("%s selected" % select)
#     print("total loss:", total_loss)
#
#     train_step_fn.step += 1
#     return [total_loss, should_stop]

def var_check(var_list, var_scope):
    print(var_scope)
    print("length:", len(var_list))
    for v in var_list:
        print(v)
    print('')
    return


def train():
    print("style number:", STYLE_NUM)
    # Create the model and specify the losses...
    with tf.variable_scope("input"):
        content_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, HEIGHT, WIDTH, 3], name="content")

    # style_bank = net.StyleBank(STYLE_NUM)
    with tf.variable_scope("control"):
        global_step = tf.Variable(0, trainable=False)
    # pre_processed_content = io_process.preprocess_for_train(content_placeholder, HEIGHT, WIDTH)
    lr = tf.train.exponential_decay(LR, global_step, DECAY_STEP, DECAY_RATE)
    optimizer = tf.train.AdamOptimizer(lr)
    train_ops = []
    losses = []


    ''' build style bank net'''
    with tf.variable_scope("style_bank"):
        # encoder
        encoded_content = net.encoder(content_placeholder, training=TRAINING)
        # bank --- stylizing branch
        decoded_s = []
        for i in range(STYLE_NUM):
            # 除了i标为1其他全为0的权重列表
            w = [0] * STYLE_NUM
            w[i] = 1
            stylized = net.bank(encoded_content, w, training=TRAINING)
            decoded_s.append(net.decoder(stylized, training=TRAINING))
        # decoder --- auto-encoder branch
        decoded_content = net.decoder(encoded_content, training=TRAINING)

    stylebank_vars = slim.get_variables(scope='style_bank')
    var_check(stylebank_vars, 'stylebank_vars')

    '''branch losses and train ops'''
    # style branch loss
    style_cals = []
    content_cal = loss.ContentCalculator(content_placeholder)
    for i, style in enumerate(STYLE_DICT):
        print(i)
        image = io_process.load_img(style['load_path'])
        pre_processed_style_image = io_process.preprocess_for_style(image, STYLE_IMG_HEIGHT, STYLE_IMG_WIDTH)
        style_cals.append(loss.StyleCalculator(pre_processed_style_image))
        stylizing_loss = loss.stylizing_loss(decoded_s[i],
                                             content_cal,
                                             style_cals[i],
                                             STYLE_DICT[i]["style_weight"],
                                             STYLE_DICT[i]["content_weight"],
                                             STYLE_DICT[i]["tv_weight"]
                                             )
        losses.append(stylizing_loss)
        train_ops.append(optimizer.minimize(stylizing_loss, global_step=global_step))
    # auto_encoder  branch loss
    ae_loss = loss.auto_encoder_loss(content_placeholder, decoded_content)
    losses.append(ae_loss)
    print("losses len:", len(losses))
    train_ops.append(optimizer.minimize(ae_loss, global_step=global_step))

    # '''
    # variable check
    # '''
    # vgg_vars = tf.trainable_variables(scope='vgg_16')
    # # vgg_vars = slim.get_variables_to_restore(include=['vgg'])  slim的get_variables会获得Adam中用到的gamma和beta参数，是多余的# vgg16 variables to restore
    # var_check(vgg_vars, 'vgg_vars')  # 应该只有32个
    # # adam_vars = slim.get_variables_to_restore(include=['Adam'])
    # # var_check(adam_vars,'adam_vars')
    # vs = tf.trainable_variables(scope='vgg_16')
    # var_check(vs, 'trainable_vars')
    # # model restore
    # vgg_restorer = tf.train.Saver(vgg_vars)
    # # global_step restore

    # batch iterator
    next_path_batch = io_process.get_imgpath_batch_iterator(BATCH_SIZE, EPOCH)  # a batch of path
    print('Session begin')
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./improved_graph2', sess.graph)
        sess.run(tf.global_variables_initializer())
        print('Tensorflow variables initiation done')
        # # vgg_restorer.restore(sess, VGG16_PATH)
        # # print("VGG Model restored")
        # print('Start training')
        # for step in range(MAX_ITERATION):
        #     print("step:", step)
        #     next_batch = sess.run(next_path_batch)  # get one batch
        #     print(next_batch)
        #     for k, _ in enumerate(next_batch):
        #         print(next_batch[k])
        #         next_batch[k] = str(next_batch[k])[2:-1]
        #         next_batch[k] = io_process.get_train_img(next_batch[k])
        #         next_batch[k] = sess.run(next_batch[k])
        #         next_batch[k] = io_process.preprocess_for_train(next_batch[k], HEIGHT, WIDTH)
        #         io_process.save_img(next_batch[k], str(step) + '_' + str(k) + '.jpg', sess)  # 可以考虑怎么用map简化代码
        #     next_batch = next_batch.tolist()
        #     next_batch = np.array(next_batch)
        #     fd = {content_placeholder: next_batch}
        #     if step % T == T - 1:
        #         # ae_loss
        #         select_id = STYLE_NUM
        #         select = "auto-encoder"
        #     else:
        #         # bank_loss
        #         select_id = (step - step / T) % STYLE_NUM  #
        #         select = "bank_kernel" + str(select_id)
        #     train_op = train_ops[select_id]
        #     # train_one_step
        #     sess.run(train_op, fd)
        #
        #     print("global_step:", step)
        #     print("%s selected" % select)
        #     if step % LOG_ITER:
        #         ls = sess.run(losses[select_id])
        #         print('loss:', ls)
        writer.flush()
        writer.close()
    return


if __name__ == '__main__':
    train()
