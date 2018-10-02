from  __future__ import print_function
import tensorflow as tf
from DataLoader import*
from OCRNet import*


num_epochs = 100

def train(session,  model, sampler, epoch):
  tensor_board_writer = tf.summary.FileWriter('tf_log', session.graph)
  for batch_idx, (captcha_imgs, labels, captcha_txt,) in enumerate(sampler):
      lbl_batch  = \
            to_sparse_tensor_lbls(labels)
      cpths_batch = \
           to_dense_tensor_imgs(captcha_imgs, 32, 80)
      summary, step, cost, num_err_symbls, eval_lbls = \
                 model.train(session, cpths_batch, lbl_batch, tensor_board_writer)
      line = 'Train: epoch: {} batch num : {} batch loss: {}  num error symbols: {}'.format(
                  epoch, batch_idx, cost, num_err_symbls)
      print(line)
      for idx , cptch in enumerate(eval_lbls):
          line = 'Train: captcha: {} ----> eval: {}'.format(captcha_txt[idx], cptch)
          print(line)

def test(session, model, sampler, epoch):
    tensor_board_writer = tf.summary.FileWriter('tf_log', session.graph)
    for batch_idx, (captcha_imgs, labels, captcha_txt,) in enumerate(sampler):
        lbl_batch = \
            to_sparse_tensor_lbls(labels)
        cpths_batch = \
            to_dense_tensor_imgs(captcha_imgs, 32, 80)
        summary, step, cost, num_err_symbls, eval_lbls = \
            model.test(session, cpths_batch, lbl_batch, tensor_board_writer)
        line = 'Test: epoch : {} batch num : {} batch loss: {}  num error symbols: {}'.format(
            epoch, batch_idx, cost, num_err_symbls)
        print(line)
        for idx, cptch in enumerate(eval_lbls):
            line = 'Test: captcha: {} ----> eval: {}'.format(captcha_txt[idx], cptch)
            print(line)

set_and_print_setted_args(LearningParameters)
is_training_mode = True
train_model  = OCRNet(is_training_mode, LearningParameters)
train_sampler, valid_sampler = \
    load_captcha_dates('D:\cptchdataset', 'train', anno_txt='anno.txt', valid_size=20000, batch_size=50)
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    for epoch in range(num_epochs):
       train(session, train_model, train_sampler, epoch)
       test(session, train_model,  valid_sampler, epoch)
