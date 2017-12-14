# -*- coding: utf-8 -*-



import jyquickhelper
import pandas as pd
import matplotlib as plt
import urllib.request
import sklearn
import math
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from openpyxl import load_workbook
pd.set_option("display.max_columns",200)
import PIL
from PIL import Image
from PIL import ImageChops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.decomposition import PCA
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV


######################################
##############  DATASET ##############
######################################

X=pd.read_csv("C:/Users/woill/Desktop/ensae/self/Stanford/NN cars/save X.csv")
X=pd.DataFrame.transpose(X)
X=X.iloc[1:X.shape[1] , :]

y=pd.read_csv("C:/Users/woill/Desktop/ensae/self/Stanford/NN cars/save y.csv")
y.iloc[ :,1:y.shape[1]]

nb_pic_c=600; nb_pic_h=85; nb_pic_a=237; nb_pic_b=48
nb_a=237; nb_b=48; nb_c= 48+ 237+ 85; nb_h=85

Xa=X.iloc[0:nb_a,:]; Xb=X.iloc[nb_a: nb_a+nb_b,:]; Xc=X.iloc[nb_a + nb_b:nb_a + nb_b+ nb_c,:];
Xh=X.iloc[nb_a + nb_b + nb_pic_c :nb_a + nb_b + nb_pic_c + nb_h,:]
ya=y.iloc[0:nb_a,:]; yb=y.iloc[nb_a: nb_a+nb_b,:]; yc=y.iloc[nb_a + nb_b:nb_a + nb_b+ nb_c,:];
yh=y.iloc[nb_a + nb_b + nb_pic_c :nb_a + nb_b + nb_pic_c + nb_h,:]

X=pd.concat([Xa,Xb,Xc,Xh])
y=pd.concat([ya,yb,yc,yh])
print(X.shape)

## train test split:
X_train, X_test, y_train, y_test = train_test_split(X, y.iloc[:,1:5] , test_size = 0.15)


######################################
##############  MODEL ################
######################################

import tensorflow as tf
from __future__ import print_function


############################################
##############  initialization #############

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,    # 1 batch at a time 
                                       1, 1, # 1x1 slide
                                       1],   # in B&W
                        padding='SAME')
s_l= 5; s_h= 5
def max_pool_2x2(x,s_l= s_l, s_h= s_h):
    return tf.nn.max_pool(x, ksize=[1,     # batch
                                    5, 5,  # dim of the pooling window
                                    1],    # channel
                          strides=[1,      # batch
                                   s_l, s_h,   # dim of the stride sliding
                                   1],     # channel
                          padding='SAME')


res_h=300 ; res_l=250
x = tf.placeholder(tf.float32) # input
y_= tf.placeholder(tf.float32) # output


###################################################
##############  1st conv & pool layer #############

W_conv1 = weight_variable([2, 2, # 5x5 convolutionnal window
                           1,      # 1 batch at a time
                           32])    # 32 times
b_conv1 = bias_variable([32])


x_image = tf.reshape(x, [-1,           # 1 batch
                         res_l, res_h, # size of pic
                         1])           # in color(3) or not(1)


h_conv1 = tf.nn.relu(         # apply relu function to 
    conv2d(x_image, W_conv1)  # 1 batch of 5x5pic 32 times
    + b_conv1)                # remove bias
h_pool1 = max_pool_2x2(h_conv1) # take max of 2x2 pooling from previous convolution layer output
## now we have round up(res_h/stride) * round up(res_l/stride) pics


###################################################
##############  2nd conv & pool layer #############

## h_pool1.shape=(629, 84, 100, 32)
W_conv2 = weight_variable([2, 2, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, # output of previous pooling layer
                            W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
## now we have (res_h/stride_1/stride_2) * (res_l/stride_1/stride_2) pics


print(math.ceil(math.ceil(res_h/s_h)/s_h) * math.ceil(math.ceil((res_l/s_l)/s_l))*64)


####################################################
##############  1st full connect layer #############

W_fc1 = weight_variable([ math.ceil(math.ceil(res_h/s_h)/s_h) * math.ceil(math.ceil((res_l/s_l)/s_l))*64  # size of pics rounded * channels
                         , 1024])                            # nb of neurons in next layer
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1                                   # flattened
                                    , math.ceil(math.ceil(res_h/s_h)/s_h) * math.ceil(math.ceil((res_l/s_l)/s_l))*64 ])

h_fc1 = tf.nn.relu(                           # apply relu on
    tf.matmul(h_pool2_flat, W_fc1) + b_fc1)   # linear combination

                   
###############################################
##############  1st dropout layer #############

keep_prob = tf.placeholder(tf.float32) # the probability to keep a neuron's output
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # used to avoid overfitting on large networks


###########################################
##############  readout layer #############

W_fc2 = weight_variable([1024   # nb of neurons in previous layer
                         , 4])  # nb of class
b_fc2 = bias_variable([4])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 # linear combination



######################################
##############  TRAINING #############
######################################


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits( # transform distance into prob
        labels=y_                            # target 
        , logits=y_conv))                    # estimation
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy) # minimization algorithm
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # 0 or 1
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))     # mean of the errors


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())    
    xx_train1 = X_train.reset_index(drop =True) ; xx = np.array(xx_train1.index)
    yy_train1 =y_train.reset_index(drop =True)
    ac_te=[] ;ac_tr=[]
    
    for i in range(100):
        np.random.shuffle(xx)
        xx_train = xx_train1.iloc[xx[0:25]] # bootstrap size
        yy_train = yy_train1.iloc[xx[0:25]]

        train_step.run(feed_dict={x: np.array(xx_train).reshape(-1,75000)
                                  ,y_: np.array(yy_train).reshape(-1,4),keep_prob: 0.5})
    
        train_accuracy = accuracy.eval(feed_dict={x: np.array(X_train).reshape(-1,75000)
                                                      ,y_: np.array(y_train).reshape(-1,4),keep_prob: 1})
        test_accuracy = accuracy.eval(feed_dict={x: np.array(X_test).reshape(-1,75000)
                                                      ,y_: np.array(y_test).reshape(-1,4),keep_prob: 1})

        ac_te.append([test_accuracy]) ; ac_tr.append([train_accuracy])
        if i%10 == 0:
            print('step %d, training accuracy %g' % (i, train_accuracy))
            print('step %d, test accuracy %g' % (i, test_accuracy))    

print('test accuracy %g' % accuracy.eval(feed_dict={x: np.array(X_test).reshape(-1,75000)
                          ,y_: np.array(y_test).reshape(-1,4),keep_prob: 1}))




    
######################################
##############  BROUILLON ############
######################################












ac_te=[]
ac_te.append([1])





sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits( # transform distance into prob
        labels=y_                            # target 
        , logits=y_conv))                    # estimation
xx_train = X_train.reset_index(drop =True) ; xx = np.array(xx_train.index) ; np.random.shuffle(xx)
xx_train = xx_train.iloc[xx[0:50]]
yy_train=y_train.reset_index(drop =True)
yy_train = yy_train.iloc[xx[0:50]]

sess.run(cross_entropy,feed_dict={x: np.array(xx_train).reshape(-1,75000)
                          ,y_: np.array(yy_train).reshape(-1,4)                         ,keep_prob: 0.5})













sess=tf.InteractiveSession()
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits( # transform distance into prob
        labels=y_                            # target 
        , logits=y_conv))                    # estimation
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # minimization algorithm
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # 0 or 1
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))     # mean of the errors
xx_train = X_train.reset_index(drop =True) ; xx = np.array(xx_train.index) ; np.random.shuffle(xx)
xx_train = xx_train.iloc[xx[0:50]]
yy_train=y_train.reset_index(drop =True)
yy_train = yy_train.iloc[xx[0:50]]
sess.run(tf.global_variables_initializer())
train_step.run(feed_dict={x: np.array(xx_train).reshape(-1,75000)
                          ,y_: np.array(yy_train).reshape(-1,4)
                          ,keep_prob: 0.5})




























