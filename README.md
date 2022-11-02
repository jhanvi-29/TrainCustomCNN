# Train Custom Convolutional Neural Network
CNN is already implemented in several packages, including TensorFlow and Keras. These libraries shield the programmer from specifics and merely provide an abstracted API to simplify life and prevent implementation complexity. However, in real life, these particulars might matter. The data analyst may occasionally need to review these particulars to boost efficiency.

In this repository, we develop the CNN from scratch using a custom dataset. Convolution (Conv for short), ReLU, and max pooling are the only layers that are used in this model to form a neural network. 

This repository will help you to train a custom CNN model with just 3 steps.

Follow these steps:
1. Clone the repository

2. Add your dataset and maintain following folder structure dataset/train dataset/test

   p.s: Here i have used 3 class classification model 
   
3. Create a folder "model_checkpoints" to save the checkpoints of model training.
   
4. Run trainModel.py  

   pass number of classes in your dataset 
   
   And here we go!  
   
Model checkpoints will be available under model_checkpoints folder
