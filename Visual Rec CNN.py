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
from matplotlib import cm
from sklearn.metrics import confusion_matrix


######################################
##############  DATASET ##############
######################################

def extrc_ph(res_h,res_l,nb_pic,pat):
    X=pd.DataFrame(columns=range(1,3*res_h*res_l+1),index=range(1,nb_pic+1))
    for i in range(1,nb_pic+1):
        im = Image.open(pat+"resultat_"+str(i)+".jpg")
        im_redi=im.resize((res_h , res_l),Image.ANTIALIAS)
        if len(im_redi.split())==3 :
            r,g,b = im_redi.split()
        else:
            r,g,b,fucknose = im_redi.split()
        rd=list(r.getdata()) ; gd=list(g.getdata()) ; bd=list(b.getdata())
        X.iloc[i-1,:]=np.concatenate(([rd,gd,bd]))
        if i==round(nb_pic/2) or  i==round(nb_pic/4) or  i==round(3*nb_pic/4):
            print(100*i/nb_pic)
    return(X)

res_h=224 ; res_l=224

pat = "C:/Users/woill/Downloads/Google Images/640 480/cars/"
nb_pic_c=370
yc=pd.DataFrame(columns=["cars","animals","bicycle","house"] ,index=range(1,nb_pic_c+1))
yc.iloc[:,0]=1 ; yc.iloc[:,1:4]=0
Xc=extrc_ph(res_h,res_l,nb_pic_c,pat)

pat = "C:/Users/woill/Downloads/Google Images/640 480/house/"
nb_pic_h=85
yh=pd.DataFrame(columns=["cars","animals","bicycle","house"] ,index=range(1,nb_pic_h+1))
yh.iloc[:,3]=1 ; yh.iloc[:,[0,2,1]]=0
Xh=extrc_ph(res_h,res_l,nb_pic_h,pat)

pat = "C:/Users/woill/Downloads/Google Images/640 480/animals/"
nb_pic_a=237 #237
ya=pd.DataFrame(columns=["cars","animals","bicycle","house"] ,index=range(1,nb_pic_a+1))
ya.iloc[:,1]=1 ; ya.iloc[:,[0,2,3]]=0
Xa=extrc_ph(res_h,res_l,nb_pic_a,pat)

pat = "C:/Users/woill/Downloads/Google Images/640 480/bicycle/"
nb_pic_b=48
yb=pd.DataFrame(columns=["cars","animals","bicycle","house"] ,index=range(1,nb_pic_b+1))
yb.iloc[:,2]=1 ; yb.iloc[:,[0,1,3]]=0
Xb=extrc_ph(res_h,res_l,nb_pic_b,pat)





X=pd.concat([Xa,Xb,Xc,Xh])
y=pd.concat([ya,yb,yc,yh])
X.shape

pd.DataFrame.transpose(X).to_csv("C:/Users/woill/Desktop/ensae/self/Stanford/NN cars/save X.csv")
y.to_csv("C:/Users/woill/Desktop/ensae/self/Stanford/NN cars/save y.csv")













X=pd.read_csv("C:/Users/woill/Desktop/ensae/self/Stanford/NN cars/save X.csv")
X=pd.DataFrame.transpose(X)
X=X.iloc[1:X.shape[1] , :]

y=pd.read_csv("C:/Users/woill/Desktop/ensae/self/Stanford/NN cars/save y.csv")
y.iloc[ :,1:y.shape[1]]

nb_pic_c=370; nb_pic_h=85; nb_pic_a=237; nb_pic_b=48
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
                                       1],   # in color
                        padding='SAME')

s_l= 2; s_h= 2
def max_pool_2x2(x,s_l= s_l, s_h= s_h, strides= [1,s_l,s_h,1]):
    return tf.nn.max_pool(x, ksize=[1,     # batch
                                    2, 2,  # dim of the pooling window
                                    1],    # channel
                          strides=strides,    # channel
                          padding='SAME')

res_h=224 ; res_l=224
x = tf.placeholder(tf.float32) # input
y_= tf.placeholder(tf.float32) # output

###################################################
##############  1st conv & pool layer #############

W_conv1_1 = weight_variable([4, 4, # 5x5 convolutionnal window
                           3,      # 1 batch at a time
                           8])    # 32 times
b_conv1_1 = bias_variable([8])


x_image = tf.reshape(x, [-1,           # 1 batch
                         res_l, res_h, # size of pic
                         3])           # in color(3) or not(1)

h_conv1_1 = tf.nn.relu(         # apply relu function to
    conv2d(x_image, W_conv1_1)  # 1 batch of 5x5pic 32 times
    + b_conv1_1)                # remove bias


## 2 conv before pooling:
W_conv1_2 = weight_variable([4, 4, # 5x5 convolutionnal window
                           8,      # 1 batch at a time
                           8])    # 32 times
b_conv1_2 = bias_variable([8])

h_conv1_2 = tf.nn.relu(
    conv2d(h_conv1_1, W_conv1_2)
    + b_conv1_2)

h_pool1 = max_pool_2x2(h_conv1_2) # take max of 2x2 pooling from previous convolution layer output
## now we have round up(res_h/stride) * round up(res_l/stride) pics
# sess=tf.Session();sess.run(tf.global_variables_initializer());hh=sess.run(h_pool2,feed_dict={x:X_train[0:3]})
# hh.shape

###################################################
##############  2nd conv & pool layer #############

## h_pool1.shape=(629, 84, 100, 32)
W_conv2 = weight_variable([3, 3,
                           8, 8])
b_conv2 = bias_variable([8])

h_conv2 = tf.nn.relu(conv2d(h_pool1, # output of previous pooling layer
                            W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2,strides = [1,
                                          2, 2,
                                          1])

## now we have (res_h/stride_1/stride_2) * (res_l/stride_1/stride_2) pics
print(math.ceil(math.ceil(res_h/s_h)/6) * math.ceil(math.ceil((res_l/s_l)/6))*64)

###################################################
##############  3rd conv & pool layer #############

## h_pool1.shape=(629, 84, 100, 32)
W_conv3_1 = weight_variable([3, 3,
                           8, 16])
b_conv3_1 = bias_variable([16])
h_conv3_1 = tf.nn.relu(conv2d(h_pool2, # output of previous pooling layer
                            W_conv3_1) + b_conv3_1)

## 2 conv layers
W_conv3_2 = weight_variable([3, 3,
                           16, 16])
b_conv3_2 = bias_variable([16])
h_conv3_2 = tf.nn.relu(conv2d(h_conv3_1, # output of previous pooling layer
                            W_conv3_2) + b_conv3_2)


h_pool3 = max_pool_2x2(h_conv3_2,strides = [1,
                                          2, 2,
                                          1])
## now we have (res_h/stride_1/stride_2/stride_3) * (res_l/stride_1/stride_2/stride_3) pics



###################################################
##############  4th conv & pool layer #############

## h_pool1.shape=(629, 84, 100, 32)
W_conv4 = weight_variable([3, 3,
                           16, 16])
b_conv4 = bias_variable([16])

h_conv4 = tf.nn.relu(conv2d(h_pool3, # output of previous pooling layer
                            W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4,strides = [1,
                                          2, 2,
                                          1])
## now we have (res_h/stride_1/stride_2/stride_3/stride_4) * (res_l/stride_1/stride_2/stride_3/stride_4) pics


###################################################
##############  5th conv & pool layer #############

## h_pool1.shape=(629, 84, 100, 32)
W_conv5 = weight_variable([3, 3,
                           16, 32])
b_conv5 = bias_variable([32])

h_conv5 = tf.nn.relu(conv2d(h_pool4, # output of previous pooling layer
                            W_conv5) + b_conv5)
h_pool5 = max_pool_2x2(h_conv5,strides = [1,
                                          2, 2,
                                          1])
## now we have (res_h/stride_1/stride_2/stride_3/stride_4/stride_5) * (res_l/stride_1/stride_2/stride_3/stride_4/stride_5) pics





dimm=math.ceil(math.ceil(math.ceil(math.ceil(math.ceil(res_h/s_h)/2)/2)/2)/2) * math.ceil(math.ceil(math.ceil(math.ceil(math.ceil(res_l/s_l)/2)/2)/2)/2) * 32
print(dimm)



####################################################
##############  1st full connect layer #############

W_fc1 = weight_variable([ dimm  # size of pics rounded * channels
                         , 1500])                            # nb of neurons in next layer
b_fc1 = bias_variable([1500])

h_pool2_flat = tf.reshape(h_pool5, [-1                                   # flattened
                                    , dimm ])

h_fc1 = tf.nn.relu(                           # apply relu on
    tf.matmul(h_pool2_flat, W_fc1) + b_fc1)   # linear combination


###############################################
##############  1st dropout layer #############

keep_prob = tf.placeholder(tf.float32) # the probability to keep a neuron's output
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # used to avoid overfitting on large networks


###########################################
##############  readout layer #############

W_fc2 = weight_variable([1500   # nb of neurons in previous layer
                         , 4])  # nb of class
b_fc2 = bias_variable([4])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 # linear combination



######################################
############ 1St TRAINING ############
######################################


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits( # transform distance into prob
        labels=y_                            # target
        , logits=y_conv))                    # estimation
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy) # minimization algorithm
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # 0 or 1
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))     # mean of the errors

import time
with tf.Session() as sess:
    #sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    eee=sess.run(W_conv5, feed_dict ={x: np.array(X_test).reshape(-1,224*224*3)
                          ,y_: np.array(y_test).reshape(-1,4),keep_prob: 1})
    ac_te=[] ;ac_tr=[]
    tic = time.time()
    for i in range(2001):

        X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train , test_size = 1-50/X_train.shape[0]) # bootstrap size
        #xx_train1 = X_train.reset_index(drop =True) ; xx = np.array(xx_train1.index);    yy_train1 =y_train.reset_index(drop =True)
        #np.random.shuffle(xx)
        #xx_train = xx_train1.iloc[xx[0:25]] # bootstrap size
        #yy_train = yy_train1.iloc[xx[0:25]]

        train_step.run(feed_dict={x: np.array(X_train1).reshape(-1,224*224*3)
                                  ,y_: np.array(y_train1).reshape(-1,4),keep_prob: 0.5})
        toc = time.time()

        #ac_te.append([test_accuracy]) ; ac_tr.append([train_accuracy])
        if i%60 == 0 or i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: np.array(X_train1).reshape(-1,224*224*3)
            ,y_: np.array(y_train1).reshape(-1,4),keep_prob: 1})
            #test_accuracy = accuracy.eval(feed_dict={x: np.array(X_test1).reshape(-1,75000)
            #,y_: np.array(y_test1).reshape(-1,4),keep_prob: 1})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            #print('step %d, test accuracy %g' % (i, test_accuracy))

        if i % 150 == 0 and i%1000==0 :
            confusion_mc=sess.run(tf.contrib.metrics.confusion_matrix(tf.argmax(y_conv, 1), tf.argmax(y_, 1)),
                                feed_dict = {x: np.array(X_train1).reshape(-1,224*224*3),
                                             y_: np.array(y_train1).reshape(-1,4),keep_prob: 1})

            df_cm = pd.DataFrame(confusion_mc,
                                 index =y_train1.columns, columns = y_train1.columns )
            plt.figure()
            sns.heatmap(df_cm, annot=True)
            plt.ylabel('True label %d ' %i)
            plt.xlabel('Predicted label')
            plt.show()

            #print(sess.run(cross_entropy,feed_dict={x: np.array(X_train).reshape(-1,75000)
            #,y_: np.array(y_train).reshape(-1,4),keep_prob: 1}))
            print((toc-tic)/60)
        se1=sess
    ddd=sess.run(W_conv5, feed_dict ={x: np.array(X_test).reshape(-1,224*224*3),
                                      y_: np.array(y_test).reshape(-1,4),keep_prob: 1})