## DATA EXTRACTION IN DIGITAL GAUGES WITHOUT HUMAN INTERVENTION.
![ ](readme_images/sample_image_2.jpeg)

## Custom Trained Object Detection model for extracting the data in digital gauges.

![ ](readme_images/sample_gif.gif)

## Project Scope:
The aim of the project is  to capture data from digital gauges without any  human intervention. We focused mainly on digital gauges related to retail,automotive,Electrical,Thermal power,oil & gas-retail,boilers etc. The idea is to build a computer vision model that would capture the data and store its readings in particular files for future tracking.


## Ideas and Approaches:
* Tried different approaches in order to satisfy the objective like used ocr space api ,but ocr space api did not perform well on seven segment display images.
* So the final method tried to reach the objective is Object detection.
* Object detection will take an image and identify and label specific objects within the image. For a complex image with multiple objects in view, object detection will provide a bounding box around each detected object, as well as a label identifying the class to which the object belongs. So made a custom object detection model using tensorflow object detection api which recognizes shown numbers as objects and performs its detections showing accuracy and class of the shown digits.
* Detection time is very fast and cool.

![ ](readme_images/detected_image_b.png)
* Making a Custom object detection model using Tensor Flow Object Detection Api which recognizes shown numbers as objects and performs its detections showing accuracy and class of the shown digits.
* Considering every digit and point as an object shown in the image ,labelling it with respected class and Training a custom trained object detection model using the best suited pretrained Cnn architecture which gives accurate results .





## Design and Architecure:
* Using EfficientDet Object detection model (SSD with EfficientNet-b0 + BiFPN feature extractor, shared box predictor and focal loss), trained on COCO 2017 dataset.
 

## Technology used:
* Deep Learning
* Computer vision
* Object Detection
* Tensor Flow object detection api 2



