# Deep Koalarization: Image Colorization using CNNs and Inception-Resnet-v2
By: Shengjie Lin, Yaoxuan Liu

MS Students from Columbia University

This project is a implementation of paper [*Deep Koalarization: Image Colorization using CNNs and Inception-ResNet-v2*](https://arxiv.org/abs/1712.03400) written by Federico Baldassarre, Diego González Morín, and Lucas Rodés-Guirao. Their project github page is [here](https://github.com/baldassarreFe/deep-koalarization/#readme).

## Abstract
The project aims to develop a new deep learning model
which combines a deep Convolutional Neural Network
trained from scratch with high-level features extracted
from the Inception-ResNet-v2 pre-trained model to
complete image colorization tasks. Students do a
comprehensive review of the paper, and then reproduce
the results of it by recreating the neural network and
architecture with Tensorflow and Python code. The results
show that our approach is able to successfully colorize
high-level image components such as the sky, the sea, the
tree, the ground, and the skin. And the performance highly
depends on the specific contents in the images. The
comparisons between our results and the results in the
original paper are fully discussed.
## Project overview
The model has a deep CNN architecture with Inception-ResNet-v2 pretrained on ImageNet dataset. The encoder stores the shape and edges of an image, the feature extractor extracts the content, and finally the decoder colorized the image. The model is trained on [places dataset](http://places2.csail.mit.edu/download.html). All the details can be found in the report. 
![](https://i.postimg.cc/J0kxmp3Z/our-net.png)
## Results
<img src="https://i.postimg.cc/W4Z1Gr7S/result2.png" width="700">
<img src="https://i.postimg.cc/fLjWFNp9/result.png" width="700">

## Using the codes
To **Train** the model, download the dataset (500MB) from [HERE](http://data.csail.mit.edu/places/places365/val_256.tar), place it in the Dataset folder. Run the codes in Training section of Jupyter notebook.  
To **Run** the trained model and predict your picture, download the model_weights(700MB) from [HERE](https://drive.google.com/file/d/1X3rKKbXVv5en_ztab_vdoRx4W1g57smq/view?usp=sharing) and run the codes in Prediction section of Jupyter notebook.  
