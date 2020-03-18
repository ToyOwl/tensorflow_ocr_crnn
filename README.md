# Распознавание текстовой 6-символьной капчи с использованием tensorflow


## Зависимости 
  1.Python 3.5</br>
  2.Tensorflow 1.10</br>
  3.OpenCV 3.2</br>
  4.Clpatcha</br>

## Архитектура сети 
Адаптация сети из статьи n End-to-End Trainable Neural Network for Image-based 
Sequence Recognition and Its Application to Scene Text Recognition([arXiv:1507.0571](https://arxiv.org/abs/1507.05717)).

| layer |  layer type |  dimension  |  kernel size   |   padding   |   stride   |
|:-----:|-------------|-------------|:--------------:|-------------|------------|
|   1   | convolution |     64      |       3        |    same     |     1      |
|   2   | max pooling |     64      |      2,2       |    valid    |     2      |
|   3   | convolution |     128     |       3        |    same     |     1      |
|   4   | max pooling |     128     |      2,2       |    valid    |     2      |
|   5   | convolution |     128     |       2        |    same     |     1      |
|   6   | convolution |     128     |       2        |    same     |     2      |
|   7   | convolution |     256     |                |    same     |     2      |
|   8   | convolution |     256     |                |    same     |     2      |

## Метрики
![batch accuracy](https://github.com/ToyOwl/tensorflow_ocr_crnn/blob/master/imgs/accuracy.PNG)
![ctc_loss](https://github.com/ToyOwl/tensorflow_ocr_crnn/blob/master/imgs/ctc_loss.PNG)
![num_error_symbols](https://github.com/ToyOwl/tensorflow_ocr_crnn/blob/master/imgs/num_error_symbols.PNG)
