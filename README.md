# Crossover-Net
## Crossbar-Net: *Regions with Convolutional Neural Network Segmentation*

Created by Qian Yu, Yinhuan Shi, Yefeng Zheng and Yang Gao at Nanjing University.


### Introduction
This work is designed for non-elongated tissues segmentation.
The code is Pytorch version. 

### Training sub-models
If you have prepared the the training patches according to the sampling strategy, you can run the train_vh.py to train the model. The modules\HV_net.py is designed for 20*100 and 100*20 Crossover-patch, and the modules\inbreast_net.py is designed for 68*340 and 340*68 Crossover-patch.
### Testing sub-models
Preparing the test patches and runing the test_loss.py. 
