from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse

char_set= 'abcdefghijklmnopqrstuvwxyz0123456789'

space_token = ''
space_idx = len(char_set)

encode_chars = dict(enumerate(char_set))
encode_chars[space_idx] =space_token

decode_chars = { value : key for key, value in encode_chars.items()}


def to_sparse_tensor_lbls(lblsequences):
    #create tuple of indices, values, dense_shape
    indices = []
    values = []

    for n, seq in enumerate(lblsequences):
        for idx, symbl in enumerate(seq):
            indices.append([n, idx])
            values.append(symbl)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64)
    shape = np.asarray([len(lblsequences), indices[len(lblsequences) - 1, 1]])
    return indices, values, shape

def to_dense_tensor_imgs(imgs, width, height, chanels =1):
    return np.array(imgs).reshape(len(imgs), width,
                                        height, chanels)



class LearningParameters:
    num_classes  = len(char_set)+1
    initial_learning_rate\
                 = 1e-03
    num_epochs   = 2e+02
    batch_size   = 50
    image_width  = 80
    image_height = 32
    moment       = 9e-02
    scale        = .999
    decay_steps  = 1e+06
    decay_rate   = .96

    num_hidden   = 128
    out_keep_probability \
        = .7

    save_steps   = 1e+03
    max_to_keep  = 100
    n_hours_for_save\
                 = .5
    logging_dir  = 'tf-log'
    save_dir     = 'tmp'
    model_prefix = 'ocr-model'


command_args = argparse.ArgumentParser()
command_args.add_argument('--init_learning_rate',  dest='init_learning_rate',
                          default=LearningParameters.initial_learning_rate, type=float)
command_args.add_argument('--num_epocs',  dest='num_epochs',
                          default=int(LearningParameters.num_epochs),       type=int)
command_args.add_argument('--batch_size',  dest= 'batch_size',
                          default=LearningParameters.batch_size,            type=int)
command_args.add_argument('--moment',  dest='moment',
                          default=LearningParameters.moment,                type=float)
command_args.add_argument('--scale',   dest='scale',
                          default=LearningParameters.scale,                 type=float)
command_args.add_argument('--decay_step', dest='decay_steps',
                          default=LearningParameters.decay_steps,           type=int)
command_args.add_argument('--decay_rate', dest='decay_rate',
                          default=LearningParameters.decay_rate,            type=float)
command_args.add_argument('--num_hidden', dest='num_hidden',
                          default=LearningParameters.num_hidden,            type=int)
command_args.add_argument('--out_keep_probability', dest='out_keep_probability',
                          default=LearningParameters.out_keep_probability,  type=float)
command_args.add_argument('--save_steps', dest='save_steps',
                          default=LearningParameters.save_steps,            type=int)
command_args.add_argument('--max_to_keep',dest='max_to_keep',
                          default=LearningParameters.max_to_keep,           type=int)
command_args.add_argument('--n_hours_to_save', dest='n_hours_to_save',
                          default=LearningParameters.n_hours_for_save,      type=float)
command_args.add_argument('--logging_dir', dest='logging_dir',
                          default=LearningParameters.logging_dir,           type=str)
command_args.add_argument('--save_dir', dest='save_dir',
                          default=LearningParameters.save_dir,              type=str)
command_args.add_argument('--model_prefix', dest='model_prefix',
                          default=LearningParameters.model_prefix,          type=str)


def set_and_print_setted_args(learning_params):
    setted_args= command_args.parse_args()
    learning_params.num_epochs= setted_args.num_epochs
    learning_params.num_hidden= setted_args.num_hidden
    learning_params.out_keep_probability = setted_args.out_keep_probability
    learning_params.decay_steps = setted_args.decay_steps
    learning_params.decay_rate = setted_args.decay_rate
    learning_params.moment  = setted_args.moment
    learning_params.scale   = setted_args.scale
    learning_params.initial_learning_rate = setted_args.init_learning_rate
    learning_params.save_dir = setted_args.save_dir
    learning_params.save_steps = setted_args.save_steps
    learning_params.logging_dir = setted_args.logging_dir
    learning_params.model_prefix = setted_args.model_prefix
    learning_params.max_to_keep  = setted_args.max_to_keep
    learning_params.n_hours_for_save = setted_args.n_hours_to_save

    print('Num epocs: {},  batch size: {}, save_dir: {}, num save steps: {}, number hours to save : {}, logging dir: {},  model prefix: {}, max to keep: {}'.format(
        setted_args.num_epochs, setted_args.batch_size, setted_args.save_dir, setted_args.save_steps, setted_args.n_hours_to_save, setted_args.logging_dir,
        setted_args.model_prefix, setted_args.max_to_keep))
    print('RNN hidden size: {}, out keep probability: {}'.format(setted_args.num_hidden, setted_args.out_keep_probability))
    print('Training params. initial rate: {}, moment: {}, scale: {}, decay step: {}, decay rate: {}'.format(setted_args.init_learning_rate,
        setted_args.moment, setted_args.scale, setted_args.decay_steps, setted_args.decay_rate
    ))