
#Reference: 1. https://github.com/tensorflow/models/tree/master/research/object_detection
#           2. https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util

from matplotlib import pyplot as plt
from PIL import Image   #is required when using the model for static images.
import cv2

video = cv2.VideoCapture(0)

# This is needed.
sys.path.append("..")


# # Model preparation 

# We tested with  "SSD with Mobilenet & faster_rcnn_inception_v2_coco" model and faster_rcnn_inception_v2_coco_2018_01_28 using webcam. 

model = 'faster_rcnn_inception_v2_coco_2018_01_28'
model_tar = model + '.tar.gz'
download_url = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
path_to_model = model + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

#number of classes to be identified
NUM_CLASSES = 90


# Model gets downloaded

opener_web = urllib.request.URLopener()
opener_web.retrieve(download_url + model_tar, model_tar)
tarfile = tarfile.open(model_tar)
for file in tarfile.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tarfile.extract(file, os.getcwd())


# ## Loading this Tensorflow model into the memory

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(path_to_model, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category name

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# ## Helper code

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# Test on static images
#PATH_TO_TEST_IMAGES_DIR = 'test_images'
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
#IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    a=True;
    while a:
      ret, image_np = video.read()
      # Expand dimensions since the model expects images shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      
      # Each box represents a part of the image where a particular object was detected.
      detected_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      detected_scores = detection_graph.get_tensor_by_name('detection_scores:0')
      tensorimage = detection_graph.get_tensor_by_name('image_tensor:0')
      detected_classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      ( detected_classes,detected_boxes, detected_scores, num_detections) = sess.run(
          [ detected_classes,detected_boxes, detected_scores, num_detections],
          feed_dict={tensorimage: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(detected_boxes),
          np.squeeze(detected_classes).astype(np.int32),
          np.squeeze(detected_scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=4)

      cv2.imshow('Tensor Flow', cv2.resize(image_np, (800,600)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
