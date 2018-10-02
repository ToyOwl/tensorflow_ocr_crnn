# Распознавание текстовой 6-символьной капчи с использованием tensorflow


## Зависимости 
  1.python 3.5</br>
  2.tensorflow 1.10</br>
  3.opencv 3.2</br>
  4 Clpatcha</br>

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

Метрики:
![batch accuracy](tensorflow_ocr_crnn/imgs/accuracy.PNG)

