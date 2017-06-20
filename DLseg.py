#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 20:21:44 2017

@author: sezan92
"""
# %% Importing libraries
import dicom, lmdb, cv2, re, sys
import os, fnmatch, shutil, subprocess
from IPython.utils import io
import numpy as np
np.random.seed(1234)
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
import pickle
warnings.filterwarnings('ignore') # we ignore a RuntimeWarning produced from dividing by zero
imgsAll = []
labelsAll = []


print("\nSuccessfully imported packages, hooray!\n")
# %% Data Preparation
SAX_SERIES = {
    # challenge training
    "SC-HF-I-1": "0004",
    "SC-HF-I-2": "0106",
    "SC-HF-I-4": "0116",
    "SC-HF-I-40": "0134",
    "SC-HF-NI-3": "0379",
    "SC-HF-NI-4": "0501",
    "SC-HF-NI-34": "0446",
    "SC-HF-NI-36": "0474",
    "SC-HYP-1": "0550",
    "SC-HYP-3": "0650",
    "SC-HYP-38": "0734",
    "SC-HYP-40": "0755",
    "SC-N-2": "0898",
    "SC-N-3": "0915",
    "SC-N-40": "0944",
}

SUNNYBROOK_ROOT_PATH = "/home/sezan92/Documents/ImageSegmentation"

TRAIN_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                            "Sunnybrook Cardiac MR Database ContoursPart3",
                            "TrainingDataContours")
TRAIN_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        "challenge_training")
TEST_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                             "challange_validation")
# %% Functions
def shrink_case(case):
    toks = case.split("-")
    def shrink_if_number(x):
        try:
            cvt = int(x)
            return str(cvt)
        except ValueError:
            return x
    return "-".join([shrink_if_number(t) for t in toks])

class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r"/([^/]*)/contours-manual/IRCCI-expert/IM-0001-(\d{4})-icontour-manual.txt", ctr_path)
        self.case = shrink_case(match.group(1))
        self.img_no = int(match.group(2))
    
    def __str__(self):
        return "<Contour for case %s, image %d>" % (self.case, self.img_no)
    
    __repr__ = __str__

def load_contour(contour, img_path):
    filename = "IM-%s-%04d.dcm" % (SAX_SERIES[contour.case], contour.img_no)
    full_path = os.path.join(img_path, contour.case, filename)
    f = dicom.read_file(full_path)
    img = f.pixel_array.astype(np.int)
    ctrs = np.loadtxt(contour.ctr_path, delimiter=" ").astype(np.int)
    label = np.zeros_like(img, dtype="uint8")
    cv2.fillPoly(label, [ctrs], 1)
    return img, label

    
def get_all_contours(contour_path):
    
    contours = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(contour_path)
        for f in fnmatch.filter(files, 'IM-0001-*-icontour-manual.txt')]
    print("Shuffle data")
    np.random.shuffle(contours)
    print("Number of examples: {:d}".format(len(contours)))
    extracted = map(Contour, contours)
    return extracted

def export_all_contours(contours, img_path, lmdb_img_name, lmdb_label_name):        
    for lmdb_name in [lmdb_img_name, lmdb_label_name]:
        db_path = os.path.abspath(lmdb_name)
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
    batchsz = 100
    print("Processing {:d} images and labels...".format(len(contours)))

    imgs, labels = [], []
    for i in xrange(int(np.ceil(len(contours) / float(batchsz)))):
        batch = contours[(batchsz*i):(batchsz*(i+1))]
        if len(batch) == 0:
            break
        
        for idx,ctr in enumerate(batch):
            img, label = load_contour(ctr, img_path)
            imgs.append(img.flatten())
            labels.append(label.flatten())
            imgsAll.append(img)
            labelsAll.append(label)
            #if idx % 20 == 0:
                #print ctr
                #plt.imshow(img)
                #plt.show()
                #plt.imshow(label)
                #plt.show()
    return imgs,labels
# %% Tensorflow
inputs_notImage = tf.placeholder(tf.float32,name ="inputs_notImage",shape = (None,65536))
inputs_ = tf.reshape(inputs_notImage,[-1,256,256,1])
targets_notImage = tf.placeholder(tf.float32,name = 'targets',shape = (None,65536))
targets_ = tf.reshape(targets_notImage,[-1,256,256,1])
# 256 x 256
conv1= tf.layers.conv2d(inputs_,16,(5,5),2,padding= 'same',activation= tf.nn.relu)
# 256 x 256 x16
pool1=tf.layers.max_pooling2d(conv1,(2,2),2,padding='same' ) 
# 128 x 128 x16
conv2 = tf.layers.conv2d(pool1,8,(5,5),2,padding='same',activation= tf.nn.relu)
#128 x128x8
pool2 = tf.layers.max_pooling2d(conv2,(2,2),2,padding='same')
#64 x64 x8
conv3 = tf.layers.conv2d(pool2,8,(3,3),1,padding='same',activation=tf.nn.relu)
#64 x 64 x8
conv4 = tf.layers.conv2d(conv3,8,(3,3),1,padding = 'same',activation= tf.nn.relu)
# 64 x 64 x 8
drop = tf.layers.dropout(inputs = conv4,rate= 0.1)
# 64 x 64 x8
score_classes = tf.layers.conv2d(drop,8,(1,1),1,padding='same')
#64 x 64 x8
upsample = tf.image.resize_nearest_neighbor(score_classes,(256,256))
#256 x 256 x 16
logits = tf.layers.conv2d(upsample,1,(3,3),padding='same')

decoded = tf.nn.sigmoid(logits,name = 'decoded')

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(0.001).minimize(cost)
correct_pred = tf.equal(logits,targets_)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#%%  Testing
if __name__== "__main__":
    SPLIT_RATIO = 0.1
    print("Mapping ground truth contours to images...")
    ctrs = get_all_contours(TRAIN_CONTOUR_PATH)
    val_ctrs = ctrs[0:int(SPLIT_RATIO*len(ctrs))]
    train_ctrs = ctrs[int(SPLIT_RATIO*len(ctrs)):]
    print("Done mapping ground truth contours to images")
    print("\nBuilding LMDB for train...")
    trainImgs,trainLabels=export_all_contours(train_ctrs, TRAIN_IMG_PATH, "train_images_lmdb", "train_labels_lmdb")
    print("\nGot Train Data Set...")
    valImgs,valLabels=export_all_contours(val_ctrs, TRAIN_IMG_PATH, "val_images_lmdb", "val_labels_lmdb")
    print("\nGot Validation Data Set...")
    print("Starting Tensorflow magic")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        count =0
        batch_size =10
        epochs = 100
        for i in range(epochs):
            print "epoch " +str(i)  
            count =0
            while count in range(0,len(trainImgs)):
                feed = {inputs_notImage:trainImgs[count:count+batch_size],
                        targets_notImage:trainLabels[count:count+batch_size]}
                loss, _ = sess.run([cost, opt], feed_dict=feed)
                print "train Loss "+ str(loss) + "with epoch "+str(i)
                resultLabels=[]
                if i == 99:
                    if count==230:
                        for image in range(len(valImgs)):
                            print "Validation Image"
                            feed = {inputs_notImage:valImgs[image].reshape((1,65536)),
                                    }
                            label= sess.run(decoded,feed_dict=feed)
                            resultLabels.append(label.reshape((256,256)))
                            #print loss
                            #plt.figure
                            #plt.imshow(label.reshape((256,256)))
                            #plt.figure()
                            #plt.imshow(valLabels[image].reshape(256,256))
                            #plt.figure()
                count = count+batch_size
    pickle.dump(resultLabels,open('Epoch100_labels.pkl','wb'))
    pickle.dump(valImgs,open('ValidationImages.pkl','wb'))
    pickle.dump(valLabels,open('ValidationLabels.pkl','wb'))
    #%% Visualization
    pred = pickle.load(open('Epoch100_labels.pkl','r'))
    for i in range(len(valImgs)):
        plt.figure()
        plt.subplot(1,3,1)
        a1 = valImgs[i].reshape((256,256))
        a2 = a1
        a3 = a1
        plt.imshow(a1)
        plt.title("Image")
        plt.subplot(1,3,2)
        b =valLabels[i].reshape((256,256))
        for row in range(b.shape[0]):
            for col in range(b.shape[1]):
                if b[row][col]==1:
                    a2[row][col]=500
                plt.imshow(a2)
                plt.title("Labeled")
                plt.subplot(1,3,3)
                c = pred[i].reshape((256,256))
        for row in range(c.shape[0]):
            for col in range(c.shape[1]):
                if c[row][col]>0.25:
                    a3[row][col]=500
                plt.imshow(a3)
                plt.title("Prediction")              