#!/usr/bin/env python
# coding: utf-8

# # Project 1

# ## Filepaths & User-Configurable Variables: Modify all to match your environment. Any filepaths with VOCdevkit_2007 will need to be downloaded from the PASCAL VOC webpage at citations [10][15][16] in the report.

# In[1]:

hdf5_filepath = r"C:\Users\kwhel\OneDrive\Documents\School Material\Deep Learning\Python\Project 1\weight.hdf5"
original_annotation = r"C:\Users\kwhel\OneDrive\Documents\School Material\Deep Learning\Python\Final Project\Car Dataset\data\export/_annotations.csv"
annotation_filepath = r"C:\Users\kwhel\OneDrive\Documents\School Material\Deep Learning\Python\Final Project\Car Dataset\data\export/"
folder_train_name = r"C:\Users\kwhel\OneDrive\Documents\School Material\Deep Learning\Python\Final Project\Car Dataset\data\export/Train/"
folder_test_name = r"C:\Users\kwhel\OneDrive\Documents\School Material\Deep Learning\Python\Final Project\Car Dataset\data\export/Test/"
num_classes = 11
B = (2 * 5 + num_classes)
random_annotation_path = r'C:\Users\kwhel\OneDrive\Documents\School Material\Deep Learning\Python\Final Project\Car Dataset\data\export/Train/1478899731683939176_jpg.rf.8o0hGoLOKhpep5ntNRKi.txt'
train_jpg_wildcard = r'C:\Users\kwhel\OneDrive\Documents\School Material\Deep Learning\Python\Final Project\Car Dataset\data\export\Train\*.jpg'
train_txt_wildcard = r'C:\Users\kwhel\OneDrive\Documents\School Material\Deep Learning\Python\Final Project\Car Dataset\data\export\Train\*.txt'
test_jpg_wildcard = r'C:\Users\kwhel\OneDrive\Documents\School Material\Deep Learning\Python\Final Project\Car Dataset\data\export\Test\*.jpg'
test_txt_wildcard = r'C:\Users\kwhel\OneDrive\Documents\School Material\Deep Learning\Python\Final Project\Car Dataset\data\export\Test\*.txt'

# map each class to an index (class names taken from dataset source)
classes = {'car': 0, 'pedestrian': 1, 'trafficLight-Red': 2, 'trafficLight-Green': 3, 'truck': 4, 'trafficLight': 5, 'biker':6,
           'trafficLight-RedLeft':7, 'trafficLight-GreenLeft':8, 'trafficLight-Yellow':9, 'trafficLight-YellowLeft':10}



# ## Import Libraries

# ### Note: Prior to importing, or if any imports fail, please use pip install to install the corresponding libraries on your machine. 

# In[2]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator as IDG
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import matplotlib.patches as patches


### Preprocess Training Data (Self-Driving Car data)
# citation: https://github.com/AlexeyAB/Yolo_mark/issues/60
# dataset: https://www.kaggle.com/datasets/sshikamaru/udacity-self-driving-car-dataset

import pandas as pd
import os
import pybboxes
import itertools
import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

'''
## YOLO/Darknet format annotation creation; this section can be commented out if already performed on your local machine

# bring in the original csv file from the dataset, and make sure to skip blank lines
data = pd.read_csv(original_annotation, skip_blank_lines=True)

annotations = []

# go through each row in the dataFrame to append to the new annotation file
for _, row in data.iterrows():

    # get the filename
    filename = row["filename"]

    # get the image dimensions
    width = row["width"]
    height = row["height"]

    # get the class and bounding box information
    class_name = row["class"]
    xmin = row["xmin"]
    ymin = row["ymin"]
    xmax = row["xmax"]
    ymax = row["ymax"]

    # convert the class name to the correct index
    class_num = classes[class_name]

    # convert the bounding box information to YOLO format (source: https://stackoverflow.com/questions/56115874/how-to-convert-bounding-box-x1-y1-x2-y2-to-yolo-style-x-y-w-h)
    # using pybboxes library
    box = (xmin, ymin, xmax, ymax)
    yolo_box = pybboxes.convert_bbox(box, from_type="voc", to_type="yolo", image_size=(width, height))

    # get the new bounding box numbers
    x_center = yolo_box[0]
    y_center = yolo_box[1]
    w = yolo_box[2]
    h = yolo_box[3]

    # generate the new (yolo) annotation format
    new_annotation = f"{class_num} {x_center} {y_center} {w} {h}"

    # add the filename and yolo annotation to the list
    annotations.append((filename, new_annotation))

# group the annotations by filename (source: https://www.geeksforgeeks.org/python-itertools/#)
annotations = sorted(annotations, key=lambda x: x[0])
grouped_annotations = [(k, [v for _, v in g]) for k, g in itertools.groupby(annotations, key=lambda x: x[0])]

# write out the annotations to new text files in the YOLO format
for filename, annotation in grouped_annotations:

    # get the text file name by replacing the file extension
    txt_filename = filename.replace(".jpg", ".txt")

    # make sure the text filename has the relative path
    txt_filename = annotation_filepath + txt_filename

    # open the text file before writing to prevent weird looping issues
    f = open(txt_filename, 'w+')

    # write the new text file
    for line in annotation:
        f.write(line + "\n") 
        #print("Wrote file " + txt_filename)

##  write annotations for any jpgs that don't have items in them
files = os.listdir(annotation_filepath)

# look through each file
for file in files:
    if file.endswith(".jpg"):
        # get the name of the corresponding txt file
        txt_file = file[:-4] + ".txt"
        # check if the txt file does not exist
        if txt_file not in files:
            # make a txt file for it
            with open(os.path.join(annotation_filepath, txt_file), "w") as f:
                f.write("\n")

print("finished making annotations")



### Train/Val data aggregation and splitting 

## Make a list of jpg and txt paths (source: https://stackoverflow.com/questions/66579311/yolov4-custom-dataset-train-test-split)

## Train/Test image splitting; this section can be commented out if already performed on your local machine

# function to split images
def split_img_label(data_train,data_test,folder_train,folder_test):
    
    try:
        os.mkdir(folder_train)
        os.mkdir(folder_test)
    except:
        print("WARNING! Train and Test directories already exist. Delete them and try again.")
        exit()
    
    train_ind=list(data_train.index)
    test_ind=list(data_test.index)
    
    # Train folder
    for i in tqdm(range(len(train_ind))):
                
        src = data_train[train_ind[i]]

        # split filepath (source: https://www.geeksforgeeks.org/python-os-path-split-method/#)
        head_tail = os.path.split(src)
        src_filename = head_tail[1]
        src_path = head_tail[0]

        # split text of path
        src_no_extension = os.path.splitext(src_filename)
        src_no_extension = src_no_extension[0]

        src_img = src_path + '/' + src_no_extension + '.jpg'
        dst_img = folder_train + src_no_extension + '.jpg'

        # copy the image and text files to the destination folder
        shutil.copy(src_img, dst_img)

        # change file extension
        src_txt = src_path + '/' + src_no_extension + '.txt'
        dst_txt = folder_train + src_no_extension + '.txt'

        shutil.copy(src_txt, dst_txt)
    
    # Test folder
    for j in tqdm(range(len(test_ind))):
        
        src = data_test[test_ind[j]]

        # split filepath (source: https://www.geeksforgeeks.org/python-os-path-split-method/#)
        head_tail = os.path.split(src)
        src_filename = head_tail[1]
        src_path = head_tail[0]

        # split text of path
        src_no_extension = os.path.splitext(src_filename)
        src_no_extension = src_no_extension[0]

        src_img = src_path + '/' + src_no_extension + '.jpg'
        dst_img = folder_test + src_no_extension + '.jpg'

        # copy the image and text files to the destination folder
        shutil.copy(src_img, dst_img)

        # change file extension
        src_txt = src_path + '/' + src_no_extension + '.txt'
        dst_txt = folder_test + src_no_extension + '.txt'

        shutil.copy(src_txt, dst_txt)


# split train and test data
PATH = annotation_filepath
list_img=[img for img in os.listdir(PATH) if img.endswith('.jpg')==True]
list_txt=[img for img in os.listdir(PATH) if img.endswith('.txt')==True]

path_img=[]

for i in range (len(list_img)):
    path_img.append(PATH+list_img[i])
    
df=pd.DataFrame(path_img)

# split 
data_train, data_test, labels_train, labels_test = train_test_split(df[0], df.index, test_size=0.20, random_state=42)

# Function split 
split_img_label(data_train,data_test,folder_train_name,folder_test_name)

 


print("end dataset splitting")
'''



### Test plotting a bounding box from the annotations (make sure the annotations are good)
# source: https://blog.paperspace.com/train-yolov5-custom-data/

from PIL import Image, ImageDraw

class_id_to_name_mapping = dict(zip(classes.values(), classes.keys()))

def plot_bounding_box(image, annotation_list):
    annotations = np.array(annotation_list)
    w, h = image.size
    
    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]
    
    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0,y0), (x1,y1)))
        
        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])
    
    plt.imshow(np.array(image))
    plt.show()

# Get any random annotation file 
annotation_file = random_annotation_path
with open(annotation_file, "r") as file:
    annotation_list = file.read().split("\n")[:-1]
    annotation_list = [x.split(" ") for x in annotation_list]
    annotation_list = [[float(y) for y in x ] for x in annotation_list]

#Get the corresponding image file
image_file = annotation_file.replace("txt", "jpg")
assert os.path.exists(image_file)

#Load the image
image = Image.open(image_file)

#Plot the Bounding Box
plot_bounding_box(image, annotation_list)


### Prepare the Data for Ingestion into the Model

# get list of files with glob, source: https://www.geeksforgeeks.org/python-list-files-in-directory-with-extension/#

train_jpg = glob.glob(train_jpg_wildcard)
train_txt = glob.glob(train_txt_wildcard)
val_jpg = glob.glob(test_jpg_wildcard)
val_txt = glob.glob(test_txt_wildcard)


#train_data = tf.data.Dataset.from_tensor_slices((train_jpg, train_txt))
#val_data = tf.data.Dataset.from_tensor_slices((val_jpg, val_txt))

## Make a paired list and separate into input/outputs
# It is reductant but makes it more followable

train_data = [(x, y) for x, y in zip(train_jpg, train_txt)]
val_data = [(x, y) for x, y in zip(val_jpg, val_txt)]

X_train = [item[0] for item in train_data]
Y_train = [item[1] for item in train_data]

X_val = [item[0] for item in val_data]
Y_val = [item[1] for item in val_data]


#TODO process data into something that the model will ingest

def read(image_path,label):
    ## Takes in JPEG image filepath and corresponding label and outputs image with desired size and label matrix with responses
    image = cv.imread(image_path)
    image_h, image_w = image.shape[0:2]
    image = cv.resize(image, (448, 448))
    image = image / 255.

    label_matrix = np.zeros([7, 7, B])

    ## Updates needed: Read in label .txt
    ##                 Change bbox values from distinct(previous) to proprotional 
    ##                 Figure out what the last if statement is doing
               
    for l in label:
        l = l.split(',')
        l = np.array(l, dtype=int)
        xmin = l[0]
        ymin = l[1]
        xmax = l[2]
        ymax = l[3]
        cls = l[4]
        x = (xmin + xmax) / 2 / image_w
        y = (ymin + ymax) / 2 / image_h
        w = (xmax - xmin) / image_w
        h = (ymax - ymin) / image_h
        loc = [7 * x, 7 * y]
        loc_i = int(loc[1])
        loc_j = int(loc[0])
        y = loc[1] - loc_i
        x = loc[0] - loc_j

        if label_matrix[loc_i, loc_j, 24] == 0:
            label_matrix[loc_i, loc_j, cls] = 1
            label_matrix[loc_i, loc_j, 20:24] = [x, y, w, h]
            label_matrix[loc_i, loc_j, 24] = 1  # response

    return image, label_matrix

class My_Custom_Generator(keras.utils.Sequence) :
  ## Generator class that batches inputted images and labels for a dataset
  def __init__(self, images, labels, batch_size) :
    self.images = images
    self.labels = labels
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.images) / float(self.batch_size))).astype(int)
  
  def __getitem__(self, idx) :
    batch_x = self.images[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

    train_image = []
    train_label = []

    for i in range(0, len(batch_x)):
      img_path = batch_x[i]
      label_path = batch_y[i]
      image, label_matrix = read(img_path, label_path)
      train_image.append(image)
      train_label.append(label_matrix)
    return np.array(train_image), np.array(train_label)

# ## Custom final layer to reshape output into a desired image size<br>
# ### This is based on the classes of the dataset and parameters listed in the paper



class Output_Reshape(tf.keras.layers.Layer):
  def __init__(self, target_shape):
    super(Output_Reshape, self).__init__()
    self.target_shape = tuple(target_shape)
  
  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'target_shape': self.target_shape
    })
    return config
  
  def call(self, input):
    dim = [self.target_shape[0],self.target_shape[1]]
    # curious if num_classes as a global would be utilized here
    num_classes = 11
    num_boxes = 2
    idx1 = dim[0] * dim[1] * num_classes
    idx2 = idx1 + dim[0] * dim[1] * num_boxes
    class_probs = keras.backend.reshape(input[:,:idx1],(keras.backend.shape(input)[0],) + tuple([dim[0],dim[1],num_classes]))
    class_probs = keras.backend.softmax(class_probs)
    confs = keras.backend.reshape(input[:, idx1:idx2], (keras.backend.shape(input)[0],) + tuple([dim[0], dim[1], num_boxes]))
    confs = keras.backend.sigmoid(confs)
    boxes = keras.backend.reshape(input[:, idx2:], (keras.backend.shape(input)[0],) + tuple([dim[0], dim[1], num_boxes * 4]))
    boxes = keras.backend.sigmoid(boxes)
    outputs = keras.backend.concatenate([class_probs, confs, boxes])
    return outputs


# ## Build the Model

# In[23]:


model = keras.Sequential(
    [
        tf.keras.layers.Conv2D(filters=64,kernel_size=7,strides=1,activation=LeakyReLU(alpha=0.1),input_shape=(448,448,3)),
        tf.keras.layers.MaxPool2D(pool_size=2,strides=2),
        tf.keras.layers.Conv2D(filters=192,kernel_size=2,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.MaxPool2D(pool_size=2,strides=2),
        tf.keras.layers.Conv2D(filters=128,kernel_size=1,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.Conv2D(filters=256,kernel_size=3,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.Conv2D(filters=256,kernel_size=1,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.Conv2D(filters=512,kernel_size=3,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.MaxPool2D(pool_size=2,strides=2),
        tf.keras.layers.Conv2D(filters=256,kernel_size=1,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.Conv2D(filters=512,kernel_size=3,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.Conv2D(filters=256,kernel_size=1,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.Conv2D(filters=512,kernel_size=3,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.Conv2D(filters=256,kernel_size=1,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.Conv2D(filters=512,kernel_size=3,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.Conv2D(filters=256,kernel_size=1,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.Conv2D(filters=512,kernel_size=3,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.Conv2D(filters=512,kernel_size=1,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.Conv2D(filters=1024,kernel_size=3,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.MaxPool2D(pool_size=2,strides=2),
        tf.keras.layers.Conv2D(filters=512,kernel_size=1,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.Conv2D(filters=1024,kernel_size=3,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.Conv2D(filters=512,kernel_size=1,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.Conv2D(filters=1024,kernel_size=3,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.Conv2D(filters=1024,kernel_size=3,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.Conv2D(filters=1024,kernel_size=3,activation=LeakyReLU(alpha=0.1),strides=2),
        tf.keras.layers.Conv2D(filters=1024,kernel_size=3,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.Conv2D(filters=1024,kernel_size=3,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.Dense(units=1024,activation=LeakyReLU(alpha=0.1)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(units=1470,activation='sigmoid'),
    ]
)
model.add(Output_Reshape(target_shape=(7,7,B)))
model.summary()


# ### The model uses a custom loss function. Due to time constraints, we will utilize an implementation from citation [4] in the report. 

# In[24]:


import keras.backend as K


# In[25]:


def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2
    return xy_min, xy_max


# In[26]:


def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]
    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas
    return iou_scores


# In[27]:


def yolo_head(feats):
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))
    box_xy = (feats[..., :2] + conv_index) / conv_dims * 448
    box_wh = feats[..., 2:4] * 448
    return box_xy, box_wh


# In[28]:


def yolo_loss(y_true, y_pred):
    label_class = y_true[..., :num_classes]  # ? * 7 * 7 * 20
    label_box = y_true[..., num_classes:num_classes + 4]  # ? * 7 * 7 * 4
    response_mask = y_true[..., num_classes + 4]  # ? * 7 * 7
    response_mask = K.expand_dims(response_mask)  # ? * 7 * 7 * 1
    predict_class = y_pred[..., :num_classes]  # ? * 7 * 7 * 20
    predict_trust = y_pred[..., num_classes:num_classes + 2]  # ? * 7 * 7 * 2
    predict_box = y_pred[..., num_classes + 2:]  # ? * 7 * 7 * 8
    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])
    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    label_xy = K.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_wh = K.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2
    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
    predict_xy = K.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_wh = K.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)  # ? * 7 * 7 * 2 * 1 * 2, ? * 7 * 7 * 2 * 1 * 2
    iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 7 * 7 * 2 * 1
    best_ious = K.max(iou_scores, axis=4)  # ? * 7 * 7 * 2
    best_box = K.max(best_ious, axis=3, keepdims=True)  # ? * 7 * 7 * 1
    box_mask = K.cast(best_ious >= best_box, K.dtype(best_ious))  # ? * 7 * 7 * 2
    no_object_loss = 0.5 * (1 - box_mask * response_mask) * K.square(0 - predict_trust)
    object_loss = box_mask * response_mask * K.square(1 - predict_trust)
    confidence_loss = no_object_loss + object_loss
    confidence_loss = K.sum(confidence_loss)
    class_loss = response_mask * K.square(label_class - predict_class)
    class_loss = K.sum(class_loss)
    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])
    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
    box_mask = K.expand_dims(box_mask)
    response_mask = K.expand_dims(response_mask)
    box_loss = 5 * box_mask * response_mask * K.square((label_xy - predict_xy) / 448)
    box_loss += 5 * box_mask * response_mask * K.square((K.sqrt(label_wh) - K.sqrt(predict_wh)) / 448)
    box_loss = K.sum(box_loss)
    loss = confidence_loss + class_loss + box_loss
    return loss


# ### Custom learning schedule implemented from citation [20] in the report. 

# In[29]:


class CustomLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.
    Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """
    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


# In[30]:


LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (0, 0.01),
    (75, 0.001),
    (105, 0.0001),
]


# In[31]:


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr







# ## Train the Model

# In[32]:


from tensorflow.keras.callbacks import ModelCheckpoint


# In[33]:


model.compile(loss=yolo_loss,optimizer='adam')


#  Save the best output of the model for future loading

# In[34]:


weights = ModelCheckpoint('hdf5_filepath', save_best_only=True, monitor='val_loss', mode='min')





# Train the model below, converted to Markdown in this version since we are using weights. To train your own model, convert the below block to code.

model.fit(train_data,
          epochs = 3,
          verbose = 1,
          workers= 4,
          validation_data = val_data,
           callbacks=[
              CustomLearningRateScheduler(lr_schedule),
              weights
          ])

### Filepath to load the model with saved weights from a previous run

# In[35]:


#model.load_weights(hdf5_filepath)


# # Test the Model

# ### Make a prediction and display image

#  Load and reshape image

# In[36]:

exit()
test, label = read(test_image_bike_filepath,[])
test = test.reshape(1,448,448,3)


#  Prediction #1: Predict and show output image

# In[37]:


y_pred = model.predict(test)


# In[38]:


y_pred = y_pred.reshape(7,7,B)
y_pred = y_pred.T


# In[39]:


test_airplane, label = read(test_image_plane_filepath,[])
test_airplane = test_airplane.reshape(1,448,448,3)


#  Prediction #2: Predict and show output image

# In[40]:


y_pred2 = model.predict(test_airplane)


# In[41]:


y_pred2 = y_pred2.reshape(7,7,B)
y_pred2 = y_pred2.T


# # Show Bounding Boxes

# ### This section is from open-source code and guides from citations [2][5][18] in the report.

# In[42]:


# declare the class names and corresponding colors to be used for the bounding boxes
class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle']
model_colors = ["red", "blue", "green", "orange", "purple"]


# These test cases can be switched back and forth from plane to bike by commenting/uncommenting the appropriate block.

# In[49]:


"""
 # define test case: bike
raw_image, label = read(test_image_bike_filepath,[])
y_pred = y_pred
"""
# define test case: airplane
raw_image, label = read(test_image_plane_filepath,[])
y_pred = y_pred2


# In[50]:


# function to convert YOLO output into bounding boxes and class probabilities
def yolo_to_box(y_pred):
  
  # probability of an object being present in the cell
  p_c = y_pred[0]

  # coordinates and dimensions of the bounding box
  b_x, b_y, b_h, b_w = y_pred[1:5]

  # class probabilities for each object category
  c_1, c_2 = y_pred[5:]
  
  # index of the maximum class probability
  c_max = np.argmax([c_1, c_2])
    
  # get class name and color
  c_name = class_names[c_max]
  c_color = model_colors[c_max]

  # convert the coordinates
  x_start = int((b_x - b_w / 2) * raw_image.shape[1])
  y_start = int((b_y - b_h / 2) * raw_image.shape[0])
  x_end = int((b_x + b_w / 2) * raw_image.shape[1])
  y_end = int((b_y + b_h / 2) * raw_image.shape[0])

  return p_c, x_start, y_start, x_end, y_end, c_name, c_color


# Threshold for the probability of an object being present

# In[51]:


threshold = 0.7


# In[52]:


fig, ax = plt.subplots()
ax.imshow(raw_image)

for i in range(y_pred.shape[0]):
    
  for j in range(y_pred.shape[1]):

    # get the output vector for each cell
    y_cell = y_pred[i][j]

    # convert the output vector to bounding box information
    p_c, x_start, y_start, x_end, y_end, c_name, c_color = yolo_to_box(y_cell)
    
    # if the probability of an object meets the threshold
    if p_c > threshold:
      
      # create a rectangle patch for the bounding box
      rect = patches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, linewidth=2, edgecolor=c_color, facecolor="none")
      
      # add the patch and class name to the axis
      ax.add_patch(rect)
      ax.text(x_start, y_start - 10, c_name, color=c_color, fontsize=12)

# display the output with bounding boxes
plt.show()

