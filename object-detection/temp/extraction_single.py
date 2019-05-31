#IMPORTS
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed to display the images.
# %matplotlib inline
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import label_map_util
# from utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/home/ashish/Anomoly-detection/object_detection1/object_detection/inference_graph/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/ashish/Anomoly-detection/object_detection1/object_detection/training/labelmap.pbtxt'

NUM_CLASSES = 1

#Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

#Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#Helper code
def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape(
	  (im_height, im_width, 3)).astype(np.uint8)

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
def runTest(imgPath):
	targetDIR = "/home/ashish/Anomoly-detection/object-detection/aadhar_image_extracted/"
	path = "/home/ashish/Anomoly-detection/object_detection1/aadhar_image_extracted/test.jpg"
	# from tqdm import tqdm
	# PATH_TO_TEST_IMAGES_DIR = 'Images/run_test'
	# temp= []
	# for img in tqdm(os.listdir(PATH_TO_TEST_IMAGES_DIR)):
	# 	temp.append(img)
	TEST_IMAGE_PATHS = [imgPath]
	Tagetpath = [targetDIR]
	IMAGE_SIZE = (12, 8)


	with detection_graph.as_default():
	  	with tf.Session(graph=detection_graph) as sess:
		    # Definite input and output Tensors for detection_graph
		    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
		    # Each box represents a part of the image where a particular object was detected.
		    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
		    # Each score represent how level of confidence for each of the objects.
		    # Score is shown on the result image, together with the class label.
		    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
		    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
		    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
		    i =0
		    for image_path in TEST_IMAGE_PATHS:
		      image = Image.open(image_path)
		      # the array based representation of the image will be used later in order to prepare the
		      # result image with boxes and labels on it.
		      image_np = load_image_into_numpy_array(image)
		      image_np2 = load_image_into_numpy_array(image)
		      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		      image_np_expanded = np.expand_dims(image_np, axis=0)
		      # Actual detection.
		      (boxes, scores, classes, num) = sess.run(
		          [detection_boxes, detection_scores, detection_classes, num_detections],
		          feed_dict={image_tensor: image_np_expanded})
		      # Visualization of the results of a detection.
		    
		      vis_util.visualize_boxes_and_labels_on_image_array(
		          image_np,
		          np.squeeze(boxes),
		          np.squeeze(classes).astype(np.int32),
		          np.squeeze(scores),
		          category_index,
		          use_normalized_coordinates=True,
		          line_thickness=8)
		      print(image_np.shape)
		      ymin = int((boxes[0][0][0]*image_np.shape[0]))
		      xmin = int((boxes[0][0][1]*image_np.shape[1]))
		      ymax = int((boxes[0][0][2]*image_np.shape[0]))
		      xmax = int((boxes[0][0][3]*image_np.shape[1]))
		      Result = np.array(image_np2[ymin:ymax,xmin:xmax])
		      im = Image.fromarray(Result)
		      im.convert('RGB')
		      print("Successful")

		      cv2.waitKey(0)
	return (path,Result)
# 		      i+=1
	

# path2 = '/home/ashish/Anomoly-detection/Image-forgery-detection/dataset/Training_data/positive.16.jpeg'
# runTest(path2)
