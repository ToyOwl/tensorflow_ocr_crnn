import tensorflow as tf
import tensorflow.nn  as nn
import tensorflow.contrib.layers as layers
import tensorflow.contrib as tfcontrib
from datetime import datetime
from utils import*
import numpy as np
import os

# ---------------------convolutional_layer_type ----filter_size-kernel_size_stride--normalization
convolutional_layer_params = [ ['convolutional' ,   64,          3,            1,    False],
#----------------------pooling_layer_type----kernel_width---kernel_heights__stride
                               ['pooling',        2,                 2,                2],
                               ['convolutional',  128, 3, 1,  True],
                               ['pooling',        2, 2, 2],
                               ['convolutional',  128, 2, 1,  True],
                               ['convolutional',  64,  2, 2,  False]]

#--------original--paper-acrhitecture------------------------------------------------------------------------
#1  ['convolutional' , 64,  3,  1, False],
#2  ['pooling',2, 2, 2],
#3  ['convolutional', 128, 3, 1,  True],
#4  ['pooling',2, 2, 2],
#5  ['convolutional',  256, 3, 1,  False],
#6  ['convolutional',  256, 3, 1,  False]
#7  ['pooling',  1, 2, 2]
#8  ['convolutional',  512, 3, 1,  True],
#9  ['convolutional',  512, 3, 1,  True],
#10 ['pooling',        1, 2, 2],
#11 ['convolutional',  512, 2, 1, False]
class OCRNet(object):
  def __init__(self, is_train_mode, num_classes, image_size, hidden_size, moment=.09, scale= .999,
               out_keep_prob=.7, decay_steps=1e+06, decay_rate=.96, initial_learning_rate =1e-03, save_dir ='tmp',
               model_prefix = 'ocr-dir', logging_dir='tf-log',save_steps = 100, max_to_keep=100, n_hours_for_save=.5):

      self.mode = is_train_mode
      self.num_classes = num_classes
      self.hidden_size = hidden_size
      self.out_keep_prob = out_keep_prob
      self.decay_steps   = decay_steps
      self.decay_rate = decay_rate
      self.beta_1 = moment
      self.beta_2 = scale
      self.initial_learning_rate = initial_learning_rate


      self.inputs = tf.placeholder(tf.float32, [None, image_size[0], image_size[1], 1])

      self.labels = tf.sparse_placeholder(tf.int32)

      self.save_steps = save_steps
      self.save_dir = save_dir
      self.logging_dir = logging_dir
      self.model_prefix = model_prefix
      self.max_to_keep  = max_to_keep
      self.n_hours_for_save = n_hours_for_save

      self._build_graph()

  def __init__(self, is_train_mode, learning_parameters= None):
      self.mode          = is_train_mode
      self.num_classes   = learning_parameters.num_classes
      self.hidden_size   = learning_parameters.num_hidden
      self.out_keep_prob = learning_parameters.out_keep_probability
      self.decay_steps   = learning_parameters.decay_steps
      self.decay_rate    = learning_parameters.decay_rate
      self.beta_1        = learning_parameters.moment
      self.beta_2        = learning_parameters.scale
      self.initial_learning_rate\
                         = learning_parameters.initial_learning_rate
      self.save_dir      = learning_parameters.save_dir
      self.save_steps    = learning_parameters.save_steps
      self.logging_dir   = learning_parameters.logging_dir
      self.model_prefix  = learning_parameters.model_prefix
      self.max_to_keep   = learning_parameters.max_to_keep
      self.n_hours_for_save\
                         = learning_parameters.n_hours_for_save

      self.inputs = tf.placeholder(tf.float32,
                                   [None, learning_parameters.image_height, learning_parameters.image_width, 1])
      self.labels = tf.sparse_placeholder(tf.int32)

      self._build_graph()
  def train(self, session, img_batch, lbs_batch, tensor_board_writer=None):
      feed = {self.inputs: img_batch,
              self.labels: lbs_batch}
      summary, dense_encoded, batch_loss, step, error_symbols, _ = session.run(
          [self.merge_summary, self.dense_encoded, self.loss, self.global_step, self.num_error_symbols,
           self.train_op], feed)
      decoded_symbols = \
          OCRNet.get_decoded_symbols(dense_encoded)
      if tensor_board_writer is not None:
         tensor_board_writer.add_summary(summary, step)
      if step % self.save_steps == 0:
         self.saver.save(session, os.path.join(self.save_dir, self.model_prefix), global_step=step)

      return summary, step, batch_loss, error_symbols, decoded_symbols

  def test(self, session, img_batch, lbs_batch, tensor_board_writer=None):
      feed = {self.inputs: img_batch,
              self.labels: lbs_batch}
      summary, dense_encoded, batch_loss, step, error_symbols  = session.run(
          [self.merge_summary, self.dense_encoded, self.loss, self.global_step,  self.num_error_symbols], feed)
      decoded_symbols = \
          OCRNet.get_decoded_symbols(dense_encoded)
      if tensor_board_writer is not None:
          tensor_board_writer.add_summary(summary, step)

      return summary, step, batch_loss, error_symbols, decoded_symbols

  def num_layers(self):
      return len(convolutional_layer_params)

  def sequence_representation(self, layer_data):
      batch, height, width, features =\
          layer_data.get_shape().as_list()
      layer_data = tf.transpose(layer_data, perm=[0, 2, 1, 3])
      layer_data = tf.reshape(layer_data, [tf.shape(layer_data)[0], width, height*features])
      sequence_lengths = tf.fill([tf.shape(layer_data)[0]], width)
      return layer_data, sequence_lengths, batch

  def get_decoded_symbols(dense_encoded):
      lbl_batch =[]
      for idx, encd_symbols in enumerate(dense_encoded):
          evl_symbols =''
          for evl_symbl in encd_symbols:
              evl_symbols += encode_chars[evl_symbl]
          lbl_batch.append(evl_symbols)
      return lbl_batch

  def _build_graph(self):
       self._rnn_layers(self._conv_layers(), self.mode)
       self._transcription_layer()
       self._logging()



  def _conv_layer(self, layer_name, is_training_mode,
                  in_data, filter_size, kernel_size, stride, with_normalization):
      with tf.variable_scope(layer_name):
          if with_normalization:
            activation_fn = None
          else:
            activation_fn = nn.relu

          layer = layers.conv2d(in_data, filter_size, kernel_size,  (stride, stride),
                        padding='same', activation_fn=activation_fn,  weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                biases_initializer=tf.zeros_initializer())
          if with_normalization:
            layer = tf.layers.batch_normalization(layer, axis=3,
                         training= is_training_mode,  name='normalization_'+layer_name)
            layer = tf.nn.relu(layer)
          return layer

  def _rnn_layer(self, name, is_train_mode, in_sequence, in_lengths):
      with tf.variable_scope(name):
          fw_cell = tfcontrib.rnn.LSTMCell(self.hidden_size, state_is_tuple=True,
                                           initializer=tf.truncated_normal_initializer(stddev=.001))
          bw_cell = tfcontrib.rnn.LSTMCell(self.hidden_size, state_is_tuple=True,
                                           initializer=tf.truncated_normal_initializer(stddev=.001))
          if is_train_mode:
              fw_cell = tfcontrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=self.out_keep_prob)
              bw_cell = tfcontrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=self.out_keep_prob)

          output, _ = nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, in_sequence,
                                 sequence_length=in_lengths, time_major=False, dtype=tf.float32)
          output = tf.concat(output, 2)
          return output

  def _rnn_layers(self, in_data, is_train_mode):
      #build recurrent layers
      seq, self.lengths, batches = \
         self.sequence_representation(in_data)
      seq = self._rnn_layer(
          'rnn_layer_1', is_train_mode, seq, self.lengths)
      seq = self._rnn_layer(
          'rnn_layer_2', is_train_mode, seq, self.lengths)

      seq = tf.reshape(seq, [-1, self.hidden_size])

      #inear projections of outputs by an LSTM
      w_out = tf.get_variable(name='w_out', shape=[self.hidden_size, self.num_classes],
                dtype=tf.float32, initializer=tf.glorot_normal_initializer())
      b_out = tf.get_variable(name='b_out', shape=self.num_classes, dtype=tf.float32,
                              initializer=tf.constant_initializer())
      seq = nn.xw_plus_b(seq, w_out, b_out)

      self.logints = tf.reshape(seq, [tf.shape(in_data)[0], -1, self.num_classes])
      self.logints = tf.transpose(self.logints, (1, 0, 2))


  def _conv_layers(self):
      #build convolutional part
      in_data = self.inputs
      num_layers = len(convolutional_layer_params)
      for lyr_idx in range(0, num_layers):
          layer_type = convolutional_layer_params[lyr_idx][0]
          if   layer_type == 'convolutional':
              n_features, krnl_size, stride, \
                 use_normalize = convolutional_layer_params[lyr_idx][1:]
              in_data = \
                  self._conv_layer(layer_type + '_' + str(lyr_idx), self.mode, in_data, n_features, \
                                   kernel_size=krnl_size, stride=stride, with_normalization=use_normalize)
          elif layer_type == 'pooling':
              f_h, f_w, pstride = \
                   convolutional_layer_params[lyr_idx][1:]
              in_data = self._pool_layer(in_data, layer_type + '_' + str(lyr_idx),
                                         f_h, f_w, pstride, pstride)
      return in_data

  def _transcription_layer(self):
      self.global_step = tf.train.get_or_create_global_step()

      self.loss = nn.ctc_loss(self.labels, self.logints,
                     self.lengths, time_major=True, ignore_longer_outputs_than_inputs=True)
      self.loss = tf.reduce_mean(self.loss)

      tf.summary.scalar('ctc-loss', self.loss)
      #need dependency
      ## https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
      self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(self.update_ops):
          self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate,
                                                          self.global_step, self.decay_steps, self.decay_rate,
                                                          staircase=True)
          self.optimazier = tf.train.AdamOptimizer(beta1=self.beta_1, beta2=self.beta_2,
                                                   learning_rate=self.learning_rate)
          self.train_op= self.optimazier.minimize(self.loss, global_step=self.global_step)
      # beam search decoding on the logits given in input
      self.sparse_encoded, self.lg_probability = tf.nn.ctc_beam_search_decoder(self.logints,
                    self.lengths, merge_repeated=False)

      self.labels_64 = tf.cast(self.labels, tf.int64)

      #Levenshtein distance between sequences
      levenshtein_loss = \
          tf.edit_distance(self.sparse_encoded[0], self.labels_64, normalize=False)
      #error symbols for batch
      self.num_error_symbols = \
           tf.count_nonzero(levenshtein_loss, 0)
      tf.summary.scalar('num-err-symbols', self.num_error_symbols)
      #batch accuracy
      batch_len_64 = tf.cast(tf.shape(self.inputs)[0], dtype=tf.int64)
      sub = tf.subtract(batch_len_64, self.num_error_symbols)
      self.accuracy =  tf.divide(sub, batch_len_64)
      tf.summary.scalar('accuracy', self.accuracy)
      self.dense_encoded = tf.sparse_tensor_to_dense(self.sparse_encoded[0], default_value=space_idx)



  def _pool_layer(self, in_data, layer_name, pool_width, pool_height, stride_width, stride_height):
      with tf.variable_scope(layer_name):
          return tf.layers.max_pooling2d(in_data,
                                         [pool_height, pool_width], [stride_height, stride_width], name=layer_name)
  def _logging(self):
      self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=self.n_hours_for_save, max_to_keep=self.max_to_keep)
      self.merge_summary = tf.summary.merge_all()
