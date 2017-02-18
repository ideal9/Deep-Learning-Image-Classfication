Github Repo:
https://github.com/wert23239/DeepLearningImageClassfication/edit/master/README.md

# DeepLearningImageClassfication
Basic MNIST and CIFAR010 Dataset

Call python keras.py with Tensorflow backend (python 3.5)
Use Train varaible to change between true and false


##The Network (Architecture) description
I started the model using PCA.

Then I used a 5*5 64 filter convultional layer

Next I used another 5*5 64 filter convultional layer except with LeakyRLu

Follow by a 2*2 Pooling layer

Then after dropout the date is flattend

Finally there is a Dense Layer of size 128

The optimizer is Atom(with built-in Expodential Update Law)

##MNIST
Scored:99%
Used PCA Compenets of 100

##CIFAR10
Scored:78%
1024 PCA

##Findings
Leaky worked very well

PCA doesn't do much with CIFAR10

MNIST is very easy to classify

The more complicated a NN the easier is it to mess-up the strcture
