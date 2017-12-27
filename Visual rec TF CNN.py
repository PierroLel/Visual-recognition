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
        im = Image.open(pat+"result_"+str(i)+".jpg")
        im_redi=im.resize((res_h , res_l),Image.ANTIALIAS)
        if len(im_redi.split())>=3 :
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

pat = "C:/Users/woillp/Downloads/Google Images/cars/"
nb_pic_c=100
yc=pd.DataFrame(columns=["cars","animals","bicycle","house"] ,index=range(1,nb_pic_c+1))
yc.iloc[:,0]=1 ; yc.iloc[:,1:4]=0
Xc=extrc_ph(res_h,res_l,nb_pic_c,pat)

pat = "C:/Users/woillp/Downloads/Google Images/house/"
nb_pic_h=100
yh=pd.DataFrame(columns=["cars","animals","bicycle","house"] ,index=range(1,nb_pic_h+1))
yh.iloc[:,3]=1 ; yh.iloc[:,[0,2,1]]=0
Xh=extrc_ph(res_h,res_l,nb_pic_h,pat)

pat = "C:/Users/woillp/Downloads/Google Images/animals/"
nb_pic_a=100 #237
ya=pd.DataFrame(columns=["cars","animals","bicycle","house"] ,index=range(1,nb_pic_a+1))
ya.iloc[:,1]=1 ; ya.iloc[:,[0,2,3]]=0
Xa=extrc_ph(res_h,res_l,nb_pic_a,pat)

pat = "C:/Users/woillp/Downloads/Google Images/bicycle/"
nb_pic_b=100
yb=pd.DataFrame(columns=["cars","animals","bicycle","house"] ,index=range(1,nb_pic_b+1))
yb.iloc[:,2]=1 ; yb.iloc[:,[0,1,3]]=0
Xb=extrc_ph(res_h,res_l,nb_pic_b,pat)


X=pd.concat([Xa,Xb,Xc,Xh])
y=pd.concat([ya,yb,yc,yh])
X.shape

pd.DataFrame.transpose(X).to_csv("C:/Users/woillp/save X 650.csv")
y.to_csv("C:/Users/woillp/save y 650.csv")

######################################
######################################

X=pd.read_csv("C:/Users/woillp/save X 650.csv")
X=pd.DataFrame.transpose(X)
X=X.iloc[1:X.shape[1] , :]

y=pd.read_csv("C:/Users/woillp/save y 650.csv")
y.iloc[ :,1:y.shape[1]]

nb_pic_c=100; nb_pic_h=100; nb_pic_a=100; nb_pic_b=100
nb_a=100; nb_b=100; nb_c= 100; nb_h=100

Xa=X.iloc[0:nb_a,:]; Xb=X.iloc[nb_a: nb_a+nb_b,:]; Xc=X.iloc[nb_a + nb_b:nb_a + nb_b+ nb_c,:];
Xh=X.iloc[nb_a + nb_b + nb_pic_c :nb_a + nb_b + nb_pic_c + nb_h,:]
ya=y.iloc[0:nb_a,:]; yb=y.iloc[nb_a: nb_a+nb_b,:]; yc=y.iloc[nb_a + nb_b:nb_a + nb_b+ nb_c,:];
yh=y.iloc[nb_a + nb_b + nb_pic_c :nb_a + nb_b + nb_pic_c + nb_h,:]

X=pd.concat([Xa,Xb,Xc,Xh])
y=pd.concat([ya,yb,yc,yh])

X.reset_index(drop=True, inplace= True)
X.dropna(inplace=True)
y.reset_index(drop=True, inplace= True)# ; y.dropna(y, inplace= True)
y=y.loc[X.index]

print(X.shape)

## train test split:

X_train, X_test, y_train, y_test = train_test_split(X, y.iloc[:,1:5] , test_size = 0.03, stratify = y.iloc[:,1:5])

scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

######################################
##############  MODEL ################
######################################

import tensorflow as tf

############################################
##############  initialization #############

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.09)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.5, shape=shape)
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

def avg_pool_2x2(x,s_l= s_l, s_h= s_h, strides= [1,s_l,s_h,1]):
    return tf.nn.avg_pool(x, ksize=[1,     # batch
                                    2, 2,  # dim of the pooling window
                                    1],    # channel
                          strides=strides,    # channel
                          padding='SAME')


res_h=224 ; res_l=224
x = tf.placeholder(tf.float32,name="x") # input
y_= tf.placeholder(tf.float32,name="y") # output

###################################################
##############  1st conv & pool layer #############

W_conv1_1 = weight_variable([2, 2, # 5x5 convolutionnal window
                           3,      # 1 batch at a time
                           8])    # 32 times
b_conv1_1 = bias_variable([8])
x_image = tf.reshape(x, [-1,           # 1 batch
                         res_l, res_h, # size of pic
                         3])           # in color(3) or not(1)

h_conv1_1 = tf.nn.relu(         # apply relu function to
    conv2d(x_image, W_conv1_1)  # 1 batch of 5x5pic 32 times
    + b_conv1_1)                # remove bias


## 4 3x3 conv before pooling:
W_conv1_2 = weight_variable([2, 2, # 5x5 convolutionnal window
                           8,      # 1 batch at a time
                           8])    # 32 times
b_conv1_2 = bias_variable([8])

h_conv1_2 = tf.nn.relu(
    conv2d(h_conv1_1, W_conv1_2)
    + b_conv1_2)

W_conv1_3 = weight_variable([3, 3, 8, 8]) ; b_conv1_3 = bias_variable([8])
h_conv1_3 = tf.nn.relu( conv2d(h_conv1_2, W_conv1_3) + b_conv1_3)

#W_conv1_4 = weight_variable([3, 3, 8, 8]) ; b_conv1_4 = bias_variable([8])
#h_conv1_4 = tf.nn.relu( conv2d(h_conv1_3, W_conv1_4) + b_conv1_4)

#tf.summary.histogram('W_conv1_3', W_conv1_3)

h_pool1 = max_pool_2x2(h_conv1_3) # take max of 2x2 pooling from previous convolution layer output
#tf.summary.histogram('h_pool1', h_pool1)
## now we have round up(res_h/stride) * round up(res_l/stride) pics
# sess=tf.Session();sess.run(tf.global_variables_initializer());hh=sess.run(h_pool2,feed_dict={x:X_train[0:3]})
# hh.shape

###################################################
##############  2nd conv & pool layer #############

W_conv2_1 = weight_variable([3, 3,
                           8, 8])
b_conv2_1 = bias_variable([8])

h_conv2_1 = tf.nn.relu(conv2d(h_pool1, # output of previous pooling layer
                            W_conv2_1) + b_conv2_1)

W_conv2_2 = weight_variable([3, 3, 8, 8]) ; b_conv2_2 = bias_variable([8])
h_conv2_2 = tf.nn.relu( conv2d(h_conv2_1, W_conv1_3) + b_conv2_2)

#W_conv2_3 = weight_variable([3, 3, 8, 8]) ; b_conv2_3 = bias_variable([8])
#h_conv2_3 = tf.nn.relu( conv2d(h_conv2_2, W_conv2_3) + b_conv2_3)

# pool :
h_pool2 = avg_pool_2x2(h_conv2_2,strides = [1,
                                          2, 2,
                                          1])

## now we have (res_h/stride_1/stride_2) * (res_l/stride_1/stride_2) pics

###################################################
##############  3rd conv & pool layer #############

W_conv3_1 = weight_variable([3, 3,
                           8, 16])
b_conv3_1 = bias_variable([16])
h_conv3_1 = tf.nn.relu(conv2d(h_pool2, # output of previous pooling layer
                            W_conv3_1) + b_conv3_1)

## 2 conv layers
W_conv3_2 = weight_variable([3, 3,  16, 16])   ;  b_conv3_2 = bias_variable([16])
h_conv3_2 = tf.nn.relu(conv2d(h_conv3_1,   W_conv3_2) + b_conv3_2)

#W_conv3_3 = weight_variable([3, 3, 16, 16]) ; b_conv3_3 = bias_variable([16])
#h_conv3_3 = tf.nn.relu( conv2d(h_conv3_2, W_conv3_3) + b_conv3_3)

h_pool3 = max_pool_2x2(h_conv3_2,strides = [1,
                                          2, 2,
                                          1])
## now we have (res_h/stride_1/stride_2/stride_3) * (res_l/stride_1/stride_2/stride_3) pics



###################################################
##############  4th conv & pool layer #############

W_conv4_1 = weight_variable([3, 3,
                           16, 16])
b_conv4_1 = bias_variable([16])

h_conv4_1 = tf.nn.relu(conv2d(h_pool3, # output of previous pooling layer
                            W_conv4_1) + b_conv4_1)

W_conv4_2 = weight_variable([3, 3, 16, 32]) ; b_conv4_2 = bias_variable([32])
h_conv4_2 = tf.nn.relu( conv2d(h_conv4_1, W_conv4_2) + b_conv4_2)

W_conv4_3 = weight_variable([3, 3, 32, 32]) ; b_conv4_3 = bias_variable([32])
h_conv4_3 = tf.nn.relu( conv2d(h_conv4_2, W_conv4_3) + b_conv4_3)

h_pool4 = avg_pool_2x2(h_conv4_3,strides = [1,
                                          2, 2,
                                          1])
## now we have (res_h/stride_1/stride_2/stride_3/stride_4) * (res_l/stride_1/stride_2/stride_3/stride_4) pics


###################################################
##############  5th conv & pool layer #############

W_conv5_1 = weight_variable([3, 3,  32,  32]);  b_conv5_1 = bias_variable([32])

h_conv5_1 = tf.nn.relu(conv2d(h_pool4, W_conv5_1) + b_conv5_1)

#W_conv5_2 = weight_variable([3, 3, 32, 32]) ; b_conv5_2 = bias_variable([32])
#h_conv5_2 = tf.nn.relu( conv2d(h_conv5_1, W_conv5_2) + b_conv5_2)

W_conv5_3 = weight_variable([3, 3, 32, 64]) ; b_conv5_3 = bias_variable([64])
h_conv5_3 = tf.nn.relu( conv2d(h_conv5_1, W_conv5_3) + b_conv5_3)


h_pool5 = max_pool_2x2(h_conv5_3,strides = [1,
                                          2, 2,
                                          1])
## now we have (res_h/stride_1/stride_2/stride_3/stride_4/stride_5) * (res_l/stride_1/stride_2/stride_3/stride_4/stride_5) pics


###################################################
##############  6th conv & pool layer #############

W_conv6_1 = weight_variable([3, 3,  64,  64]);  b_conv6_1 = bias_variable([64])

h_conv6_1 = tf.nn.relu(conv2d(h_pool5, W_conv6_1) + b_conv6_1)

W_conv6_2 = weight_variable([3, 3, 64, 64]) ; b_conv6_2 = bias_variable([64])
h_conv6_2 = tf.nn.relu( conv2d(h_conv6_1, W_conv6_2) + b_conv6_2)

W_conv6_3 = weight_variable([3, 3, 64, 64]) ; b_conv6_3 = bias_variable([64])
h_conv6_3 = tf.nn.relu( conv2d(h_conv6_2, W_conv6_3) + b_conv6_3)
#tf.summary.histogram('W_conv6_3', W_conv6_3)
## average pooling
h_pool6 = avg_pool_2x2(h_conv6_3,strides = [1,
                                          2, 2,
                                          1])


#dimm= math.ceil(math.ceil(math.ceil(math.ceil(math.ceil(math.ceil(res_h/s_h)/2)/2)/2)/2)/2) * math.ceil(math.ceil(math.ceil(math.ceil(math.ceil(math.ceil(res_l/s_l)/2)/2)/2)/2)/2) * 64

dimm= h_pool6.shape[1]*h_pool6.shape[2]*h_pool6.shape[3]
print(dimm)


####################################################
##############  1st full connect layer #############

#h_pool_fc1 = avg_pool_2x2(h_pool6,strides = [1,
#                                          2, 2,
#                                          1])

W_fc1 = weight_variable([ int(dimm)  # size of pics rounded * channels
                         , 1024])    # nb of neurons in next layer
b_fc1 = bias_variable([1024])
#tf.summary.histogram('W_fc1', W_fc1)
h_pool6_flat = tf.reshape(h_pool6, [-1             # flattened
                                    , int(dimm)])

h_fc1 = tf.nn.relu(                           # apply relu on
    tf.matmul(h_pool6_flat, W_fc1) + b_fc1)   # linear combination

####################################################
##############  2nd full connect layer #############

#see=tf.InteractiveSession()
#see.run(tf.global_variables_initializer())
#eee=see.run(h_pool6_flat,feed_dict={x:X_train1})

W_fc2 = weight_variable([ 1024  ,  500])   ;   b_fc2 = bias_variable([500])
h_fc2 = tf.nn.relu(                    # apply relu on
    tf.matmul(h_fc1, W_fc2) + b_fc2)   # linear combination


###############################################
##############  1st dropout layer #############

keep_prob = tf.placeholder(tf.float32) # the probability to keep a neuron's output
#h_fc1_drop = tf.nn.dropout(h_fc2, keep_prob) # used to avoid overfitting on large networks


###########################################
##############  readout layer #############

W_fc2 = weight_variable([500   # nb of neurons in previous layer
                         , 4])  # nb of class
b_fc2 = bias_variable([4])

y_conv = tf.matmul(h_fc2, W_fc2) + b_fc2 # linear combination



######################################
############ 1St TRAINING ############
######################################


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits( # transform distance into prob
        labels=y_                            # target
        , logits=y_conv+tf.constant(value=0.000001)))                    # estimation

#cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv + 1e-10))
#cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-8,1.0)))
#cross_entropy = tf.reduce_mean(
#    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#epsilon = tf.constant(value=0.000001)
#logits = y_conv + epsilon
#softmax = tf.nn.softmax(logits)
#cross_entropy = -tf.reduce_mean(tf.reduce_sum(y_ * tf.log(softmax),
#                                                 reduction_indices=[1]))

tf.summary.histogram('cross_entropy', cross_entropy)

train_step = tf.train.AdamOptimizer(learning_rate=0.001,
                                    beta1=0.9,
                                    beta2=0.999,
                                    epsilon=1e-06,).minimize(cross_entropy) # minimization algorithm
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # 0 or 1
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))     # mean of the errors
tf.summary.histogram('accuracy', accuracy)

import time
saver = tf.train.Saver()
X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_train, y_train , test_size = 128, stratify = y_train)
with tf.Session() as sess:
    #sess=tf.Session()
    writer = tf.summary.FileWriter("C:/Users/woillp/.spyder-py3/tmp/log", sess.graph)
    sess.run(tf.global_variables_initializer())
    tic = time.time()
    for i in range(1001):
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train_cv, y_train_cv ,
                                                                train_size = 32,       # bootstrap size
                                                                stratify = np.zeros(shape=y_train_cv.shape)+1/4)

        sess.run(train_step,feed_dict={x: np.array(X_train1).reshape(-1,224*224*3),
                                       y_: np.array(y_train1).reshape(-1,4),
                                       keep_prob: 1})

        if math.isnan(np.mean(sess.run(cross_entropy,feed_dict={x: np.array(X_train1).reshape(-1,224*224*3),
                                                         y_:np.array(y_train1).reshape(-1,4),
                                                         keep_prob: 1}))) :
            print('step %d, training accuracy' %i)
            print(np.sum(sess.run(h_conv1_1,feed_dict={x: np.array(X_train1).reshape(-1,224*224*3),
                                                     y_: np.array(y_train1).reshape(-1,4),keep_prob: 1})))
            print(np.sum(sess.run(h_conv1_2,feed_dict={x: np.array(X_train1).reshape(-1,224*224*3),
                                                        y_: np.array(y_train1).reshape(-1,4),keep_prob: 1})))
            print(np.sum(sess.run(h_conv1_3,feed_dict={x: np.array(X_train1).reshape(-1,224*224*3),
                                                        y_: np.array(y_train1).reshape(-1,4),keep_prob: 1})))
            print(np.sum(sess.run(y_conv,feed_dict={x: np.array(X_train1).reshape(-1,224*224*3),
                                                        y_: np.array(y_train1).reshape(-1,4),keep_prob: 1})))

        if i%60 == 0 or i%100 == 0 or i == 1 or i == 2 or i == 3:
            train_accuracy = accuracy.eval(feed_dict={x: np.array(X_train1).reshape(-1,224*224*3)
            ,y_: np.array(y_train1).reshape(-1,4),keep_prob: 1})
            #test_accuracy = accuracy.eval(feed_dict={x: np.array(X_test1).reshape(-1,75000)
            #,y_: np.array(y_test1).reshape(-1,4),keep_prob: 1})
            print('step %d, training accuracy %g' % (i, train_accuracy))

            if train_accuracy >0.75 or i%120 == 0  or i == 1 or i == 2 or i == 3:
            #with tf.Session() as sess :
            #    sess.run(tf.global_variables_initializer())
                confusion_mc=sess.run(tf.confusion_matrix( tf.argmax(y_, 1),tf.argmax(y_conv, 1)),
                                      feed_dict = {x: np.array(X_test_cv).reshape(-1,224*224*3),
                                                   y_: np.array(y_test_cv).reshape(-1,4),keep_prob: 1})
                df_cm = pd.DataFrame(confusion_mc,
                                     index =y_train1.columns, columns = y_train1.columns )
                plt.figure()
                sns.heatmap(df_cm, annot=True)
                plt.ylabel('True label %d ' %i)
                plt.xlabel('Predicted label')
                plt.show()
                print(np.median(sess.run(W_conv1_1)),np.mean(sess.run(W_conv1_1)),np.var(sess.run(W_conv1_1)))
                print(np.median(sess.run(W_conv3_2)),np.mean(sess.run(W_conv3_2)),np.var(sess.run(W_conv3_2)))
                print(np.median(sess.run(W_fc2)),np.mean(sess.run(W_fc2)),np.var(sess.run(W_fc2)))
                print((sess.run(cross_entropy,feed_dict={x: np.array(X_train1).reshape(-1,224*224*3),
                                                            y_: np.array(y_train1).reshape(-1,4),keep_prob: 1})))
            toc = time.time()
            print((toc-tic)/60)
        merge = tf.summary.merge_all()
        summary = sess.run(merge,feed_dict={x: np.array(X_train1).reshape(-1,224*224*3),
                                                y_:np.array(y_train1).reshape(-1,4),
                                                keep_prob: 1})
        writer.add_summary(summary, i)
            #print(sess.run(cross_entropy,feed_dict={x: np.array(X_train).reshape(-1,75000)
            #,y_: np.array(y_train).reshape(-1,4),keep_prob: 1}))

    save_path = saver.save(sess, "C:/Users/woillp/.spyder-py3/tmp/model.ckpt")
    #print('test accuracy %g' % accuracy.eval(feed_dict={x: np.array(X_test).reshape(-1,224*224*3),
    #                                                    y_: np.array(y_test).reshape(-1,4),keep_prob: 1}))
    writer.close()
    
    
