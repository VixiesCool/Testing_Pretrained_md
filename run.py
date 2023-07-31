from Detector import *

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"  #fastest for video obj
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz"
imagePath = "object/mouse.jpg"
videoPath = "object/Scared.mp4"
threshold = 0.5

classFile = 'coco.names'
detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()
#detector.predictImage(imagePath,threshold)
detector.predictVideo(videoPath,threshold)