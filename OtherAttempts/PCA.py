

#99%
#88% Benchmarks
# (W−F+2P)/S+1
# 3. Import libraries and modules
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, normalization
from keras.utils import np_utils
from keras.datasets import mnist,cifar10
from keras.layers.advanced_activations import LeakyReLU
from sklearn.decomposition import PCA
import matplotlib.image as mpimg
# def ReshapeX(X):
#     X_Axis
#     for image in X:
#         count=0
#         if(count>)
#     return New_X
def pca_cifar(X):
    X = X.astype('float32') / 255.
    X = X.reshape((len(X), np.prod(X.shape[1:])))
    print(X.shape)
    pca = PCA(n_components=899)
    pca.fit(X)
    X = pca.inverse_transform(pca.transform(X))
    X = X.reshape(X.shape[0],32, 32)
    print(X.shape)
    return X

# 4. Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X=X_train
print(X.shape)
X=np.transpose(X, (3,0, 1, 2))
print(X.shape)
temp1=0
temp2=0
temp3=0
for i in range(0):
     temp_X=X[i][:][:][:]
     temp_X=np.transpose(temp_X, (0,2,1))
     temp_X=pca_cifar(temp_X)
     if(i==0):
        temp1=temp_X
     elif(i==1):
        temp2=temp_X   
     else:
        temp3=temp_X
     #X[i]=pca_cifar(temp_X)
#X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)
#X=np.array([temp1,temp2,temp3])
print(X.shape)
X=np.transpose(X, (1,2,3,0))
print(X.shape)
img=X[0]
# print(img.shape)
# img=img[0]
# img=np.transpose(img)
# print(img.shape)
plt.imshow(img, interpolation='nearest')
plt.show()
#X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)

# X_train=X_train.reshape(X_train.shape[0],784)
# pca = PCA(n_components=256)
# pca.fit(X_train)
# print(X_train.shape)
# print(X_train.shape)
# X_train=ReshapeX(X_train)
# X_train = X_train.reshape(X_test.shape[0],1,28,28)
# #X_train = X_train.astype('float32')
# #X_train /= 255
# pixels = np.array(X_train[0], dtype='uint8')
# plt.imshow(pixels, cmap='gray')
# plt.show()
# X_test = X_test.reshape(X_test.shape[0], 3, 28,28)

# X_test = X_test.astype('float32')

# X_test /= 255
# print(X_test.shape)

# print(X_test.shape)
# X_train= X_train.reshape(X_train.shape[0],-1)
# X_test= X_test.reshape(X_train.shape[0],-1)
# pca = PCA(n_components=256)
# pca.fit(X_train)
# X_train = pca.transform(X_train) 

# X_test = pca.transform(X_test)


# PCA,Evals=CovarianceMatirx(X_train,256)
# Reconstusted=np.array(np.dot(PCA,(np.dot(PCA.T,A.T))))
# X_train=np.array(Reconstusted.T[666].reshape(16, 16))
# # 5. Preprocess input data PCA here
