import os
import pathlib
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
import cv2
import numpy as np


# Import the object detection module.


from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile






import time
start_time = time.time()
tf.keras.backend.clear_session()
detection_model = tf.saved_model.load('models/research/object_detection/test_data/efficientdet_d0_coco17_tpu-32/saved_model/')
end_time = time.time()
elapsed_time = end_time - start_time
print('Elapsed time: ' + str(elapsed_time) + 's')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

category_index = {
    1: {'id': 1, 'name': '.'},
    2: {'id': 2, 'name': '1'},
    3: {'id': 3, 'name': '2'},
    4: {'id': 4, 'name': '3'},
    5: {'id': 5, 'name': '4'},
    6: {'id': 6, 'name': '5'},
    7: {'id': 7, 'name': '6'},
    8: {'id': 8, 'name': '7'},
    9: {'id': 9, 'name': '8'},
    10: {'id': 10, 'name': '9'},
    11: {'id': 11, 'name': '0'},
}


# For the sake of simplicity we will test on 2 images:

# In[14]:


# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.


# # Detection

# Load an object detection model:

# In[15]:


# model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
# detection_model = load_model(model_name)


# Check the model's input signature, it expects a batch of 3-color images of type uint8:

# In[16]:


#print(detection_model.signatures['serving_default'].inputs)


# And returns several outputs:

# In[17]:


detection_model.signatures['serving_default'].output_dtypes


# In[18]:


detection_model.signatures['serving_default'].output_shapes


# Add a wrapper function to call the model, and cleanup the outputs:

# In[19]:


def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   

  return output_dict





def run_inference():


  cap = cv2.VideoCapture(0)
  
  while True:
    # Read frame from camera
      ret,image_np = cap.read()
      image_np = np.array(image_np)
      
      image_np_expanded = np.expand_dims(image_np, axis=0)
      #output_dict = run_inference_for_single_image(detection_model, image_np_expanded)
    



      label_id_offset = 1
      image_np_with_detections = image_np.copy()
      input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
      detections, predictions_dict, shapes = detect_fn(input_tensor)
  

      vis_util.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
       detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

      #

    # Display output
      resized_image =  cv2.resize(image_np_with_detections, (800, 600))
      print("object_detection",cv2.imshow(resized_image))

      if cv2.waitKey(25) & 0xFF == ord('q'):
         break
      cap.release()
      cv2.destroyAllWindows()

run_inference()




   