# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 19:52:18 2016

@author: sezan1992
"""
from os import listdir
from os.path import isfile,join
import numpy as np
import cv2
import tensorflow as tf

#Functions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def xbatchCreate(num_batch,trainNp):
    batch = []
    while num_batch in range(len(trainNp)):
        batch.append(trainNp[num_batch])
        num_batch= num_batch+120
    return np.float32(batch)

def ybatchCreate(num_batch,responseNp):
    batch = []
    while num_batch in range(len(responseNp)):
        batch.append(responseNp[num_batch])
        num_batch = num_batch+120
    return np.float32(batch)

def Gray2BW(im, r=127, b=127, g=127):

    imRGBA = cv2.cvtColor(im,cv2.COLOR_RGB2RGBA)    # Convert to RGBA
    
    
    data = np.array(imRGBA)   # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T # Temporarily unpack the bands for readability

    # Replace grey with white... (leaves alpha values alone...)
    grey_areas = (red == r) & (blue == b) & (green == g)
    data[..., :-1][grey_areas.T] = (255, 255, 255) # Transpose back needed
    

    
    
    
    #im2.show()
    
    return cv2.cvtColor(data,cv2.COLOR_RGBA2GRAY)

#Data Preparation
path = '/home/sezan92/BanglaDigit/'
TestPath = '/home/sezan92/BanglaML/Test/'
FolderNames = []
TestFolderNames = []
for i in range(10):
    FolderNames.append(path+str(i))
    TestFolderNames.append(TestPath+str(i))
trainData = []
responseData = []
testData=[]
testResponseData=[]

k=0

for folder in FolderNames:
    k=k+1
    for image in listdir(folder):
        img =cv2.imread(join(folder,image))
        img = cv2.resize(img,(32,32))
        trainData.append(img.flatten())
        responseData.append(k)
trainNp = np.float32(trainData)
responseNp = np.float32(responseData)
responseNpOH = np.zeros((responseNp.shape[0],responseData[-1]))       
k=0
#OneHot
for i in range(responseNpOH.shape[0]):
    np.put(responseNpOH[i],responseNp[i]-1,1)




all_data = np.concatenate((trainNp,responseNpOH),axis=1) 
np.random.shuffle(all_data)
#TensorFlow
sess = tf.InteractiveSession()
feature_cols = trainNp.shape[1]
num_labels=10
    
x = tf.placeholder(tf.float32, shape=[None,feature_cols])
y_ = tf.placeholder(tf.float32, shape=[None, num_labels])    
x_image = tf.reshape(x, [-1,32,32,3])
W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
    
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([4*4 * 128, feature_cols])
b_fc1 = bias_variable([feature_cols])

h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([feature_cols,num_labels])
b_fc2 = bias_variable([num_labels])
    
    
y_conv = (tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(0.0003).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

num_batch =0

while num_batch<2181:
  #batch = mnist.train.next_batch(50)
  if num_batch%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:all_data[num_batch:num_batch+10,0:feature_cols], 
                  y_:all_data[num_batch:num_batch+10,feature_cols:feature_cols+10],
                  keep_prob:1.0})
    print("step %d, training accuracy %g"%(num_batch, train_accuracy))
  train_step.run(feed_dict={x:all_data[num_batch:num_batch+10,0:feature_cols], 
                            y_:all_data[num_batch:num_batch+10,feature_cols:feature_cols+10], keep_prob: 0.5})
  num_batch = num_batch+10

