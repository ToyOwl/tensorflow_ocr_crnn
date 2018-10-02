from __future__ import print_function
import os
import random
import string

from PIL import Image
from claptcha import Claptcha


root_dir = 'D:\cptchdataset'
train_dir = 'train'
test_dir  = 'test'
font  = 'C:/Windows/Fonts/arial.ttf'
train_file_name = 'anno.txt'
test_file_name  = 'test.txt'
num_samples = 3e+05
num_test_samples  = 1e+05
num_symbls        = 6

using_names = set()

def unique_nms(num_symbols, names):
    max_probs = int(1e+03)
    txt = ''.join(random.choice(string.ascii_lowercase +
        string.digits) for _ in range(num_symbls))
    prob = 0
    while  (txt in names) and (prob < max_probs):
        txt = ''.join(random.choice(string.ascii_lowercase +
                                    string.digits) for _ in range(num_symbls))
        prob += 1
    names.add(txt)
    return txt





if not os.path.exists(root_dir):
    os.makedirs(root_dir)

train_path = os.path.join(root_dir, train_dir)
test_path  = os.path.join(root_dir, test_dir)

os.makedirs(train_path)
os.makedirs(test_path)

annotation_file = \
    open(os.path.join(root_dir, train_file_name), 'w')
test_file    = \
    open(os.path.join(root_dir, test_file_name), 'w')
img_path =''
for idx in range(int(num_samples)):
    txt = unique_nms(num_symbls, using_names)
    c = Claptcha(txt, font, (80,  32), margin=(2,2),
         resample=Image.BICUBIC, noise=.3)
    if idx < (int(num_samples - num_test_samples) - 1):
      img_path = os.path.join(train_path, txt+'.png')
      annotation_file.write(txt + '.png')
      annotation_file.write('\n')
    else:
      img_path = os.path.join(test_path, txt+'.png')
      test_file.write(txt + '.png')
      test_file.write('\n')
    c.write(img_path)
