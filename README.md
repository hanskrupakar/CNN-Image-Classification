# CNN-Image-Classification
This contains code for various curiosity-quenching models that were implemented because of wanting to understand the nature and behaviour of Neural Nets. Also are the models for AlexNet and ResNet architectures in TensorFlow.

CNN architectures implemented for Image Classification:

1. AlexNet
2. ResNet

AlexNet was implemented and achieved a 50% accuracy with CIFAR-100 on a very low-end GPU (2GB NVIDIA GeForce). The weight vectors had to be trimmed of size to prevent Out Of Memory (OOM) errors.

ResNet could not be trained due to lack of resources. Nevertheless, working code is present.

Other models implemented for Image Classification out of wanting to understand the working of RNNs:

1. RNN LSTM on Image Pixels, one row at a time
2. RNN LSTM on Global Image features: Col Histogram  and gist512, one row at a time.
3. RNN LSTM on Local Image features, one row at a time as sequential input. 
