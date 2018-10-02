import numpy as np
from utils import *
import os
import cv2
import math
class CaptchaData:
  def __init__(self, data_dir, image_dir= 'train',annotation_file='anno.txt'):
    self.data_dir = data_dir
    self.image_dir = image_dir
    self.annotation_txt = os.path.join(data_dir, annotation_file)

    self.imgs = []
    self.lbls = []
    self.txt_lbls = []

    self.load_data()

    self.num_samples = len(self.imgs)

  def __len__(self):
      return self.num_samples

  def load_data(self):
      img_names = [
          line.rstrip('\n') for line in open(self.annotation_txt)]
      for  img_name  in img_names:
           img = cv2.imread(
               os.path.join(self.data_dir, self.image_dir,
                            img_name), cv2.IMREAD_GRAYSCALE)
           img = cv2.normalize(img, None, alpha=0, beta=1,
                    norm_type=cv2.NORM_MINMAX, dtype= cv2.CV_32F)
           self.imgs.append(img)
           self.lbls.append(
               [ decode_chars[smbl] for smbl in list(img_name.split('.')[0]) ] )
           self.txt_lbls.append(img_name.split('.')[0])


class BatchCaptchaDataSampler:
 def __init__(self, captcha_data_set, subset_ids, batch_size=10, shuffle=True):
   if batch_size <= 0:
       raise ValueError('batch size should be a positive integral value, '
                        'but got batch size={}'.format(batch_size))

   self.data_set = captcha_data_set
   self.batch_size = batch_size
   self.ids  = subset_ids
   self.shuffle = shuffle

 def __len__(self):
     return (len(self.ids) + self.batch_size -1) // self.batch_size
 def __iter__(self):
     if self.shuffle:
        np.random.shuffle(self.ids)
     batch = []
     for idx in self.ids:
         batch.append(idx)
         if len(batch) == self.batch_size:
             yield [self.data_set.imgs[idx] for idx in batch], \
                   [self.data_set.lbls[idx] for idx in batch], [self.data_set.txt_lbls[idx] for idx in batch]
             batch = []
     if len(batch) > 0:
            yield [self.data_set.imgs[idx] for idx in batch], \
                  [self.data_set.lbls[idx] for idx in batch], [self.data_set.txt_lbls[idx] for idx in batch]



def load_captcha_dates(data_dir, image_dir, anno_txt='anno.txt', valid_size =0, batch_size=20, shuffle=True):
    data_set = CaptchaData(data_dir, image_dir, anno_txt)
    num_samples = len(data_set)
    indicies = list(range(num_samples))
    if valid_size > 0:
        split = num_samples - valid_size
        train_sampler = BatchCaptchaDataSampler(data_set,
                                                indicies[:split], batch_size, shuffle)
        valid_sampler = BatchCaptchaDataSampler(data_set,
                                                indicies[split:], batch_size, shuffle)
        return train_sampler, valid_sampler
    else:
        return BatchCaptchaDataSampler(data_set, indicies, batch_size, shuffle)
