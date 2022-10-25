#! /usr/bin/env python3.7

from object_detection.srv import *
from object_detection.msg import Result

import os
import threading
import numpy as np

import ros_numpy
import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest
from std_msgs.msg import UInt8MultiArray

from mask_rcnn_ros import coco
from mask_rcnn_ros import utils
from mask_rcnn_ros import model as modellib
from mask_rcnn_ros import visualize

import tensorflow as tf


# Local path to trained weights file
ROS_HOME = os.environ.get('ROS_HOME', os.path.join(os.environ['HOME'], '.ros'))
COCO_MODEL_PATH = os.path.join(ROS_HOME, 'mask_rcnn_coco.h5')

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: CLASS_NAMES.index('teddy bear')
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class MaskRCNNNode(object):
    def __init__(self):
        self._cv_bridge = CvBridge()

        config = InferenceConfig()
        config.display()

        self._visualization = rospy.get_param('~visualization', True)

        # Create model object in inference mode.
        self._model = modellib.MaskRCNN(mode="inference", model_dir="",
                                        config=config)
        # Load weights trained on MS-COCO
        model_path = rospy.get_param('~model_path', COCO_MODEL_PATH)
        # Download COCO trained weights from Releases if needed
        if model_path == COCO_MODEL_PATH and not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)

        self._model.load_weights(model_path, by_name=True)
        self.graph = tf.get_default_graph()

        self._class_names = rospy.get_param('~class_names', CLASS_NAMES)

        self._last_msg = None
        self._cloud = None
        self._msg_lock = threading.Lock()

        self._class_colors = visualize.random_colors(len(CLASS_NAMES))

        self._publish_rate = rospy.get_param('~publish_rate', 100)

    
    def run(self, msg1):
        self._result_pub = rospy.Publisher('~result', Result, queue_size=1)
        vis_pub = rospy.Publisher('~visualization', Image, queue_size=1)
        # sub = rospy.Subscriber('~input', Image, self._image_callback, queue_size=1)
        # sub = rospy.Subscriber('/device_0/sensor_1/Color_0/image/data', Image, self._image_callback, queue_size=1)
        # rospy.Service('run_inference_maskrcnn', Mask_RCNN, self._image_callback)
        self._image_callback(msg1)
        rate = rospy.Rate(self._publish_rate)
        while not rospy.is_shutdown():
            if self._msg_lock.acquire(False):
                msg = self._last_msg #sensor image msg
                cloud = self._cloud # pointcloud2 msg
                self._last_msg = None

                self._msg_lock.release()
            else:
                rate.sleep()
                continue
            # print("beforeeeeeeeeeeeeeeeeeeeeeee msg check", type(msg), msg.height, msg.width )

            if msg is not None:
                # np_image = self._cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
                #trying ros_numpy
                
                np_image = ros_numpy.numpify(msg)
                # print("After msg type coversion",type(np_image), np_image.shape)

                # Run detection
                with self.graph.as_default():
                    results = self._model.detect([np_image], verbose=0)
                result = results[0]
                result_msg = self._build_result_msg(msg = msg, result = result, cloud = cloud)
                # self._result_pub.publish(result_msg) #Testing

                # Visualize results
                if self._visualization:
                    vis_image = self._visualize(result, np_image)
                    cv_result = np.zeros(shape=vis_image.shape, dtype=np.uint8)
                    cv2.convertScaleAbs(vis_image, cv_result)
                    # image_msg = self._cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
                    #trying ros_numpy
                    image_msg = ros_numpy.msgify(Image, cv_result, encoding='rgb8')
                    vis_pub.publish(image_msg)
                    # self._srvReturn(result_msg, image_msg)
                    print("Return await~~~~~~~~~~~~~~~~~~~~~~~~")
                    return Mask_RCNNResponse(result_msg, image_msg)


            rate.sleep()
    
    

    def _build_result_msg(self, msg, result, cloud):
        result_msg = Result()
        result_msg.header = msg.header
        result_msg.cloudRes = cloud
        # print('In Build result msg', type(result['rois']),result['rois'])
        for i, (y1, x1, y2, x2) in enumerate(result['rois']):
            box = RegionOfInterest()
            box.x_offset = np.asscalar(x1)
            box.y_offset = np.asscalar(y1)
            box.height = np.asscalar(y2 - y1)
            box.width = np.asscalar(x2 - x1)
            result_msg.boxes.append(box)

            class_id = result['class_ids'][i]
            result_msg.class_ids.append(class_id)

            class_name = self._class_names[class_id]
            result_msg.class_names.append(class_name)

            score = result['scores'][i]
            result_msg.scores.append(score)

            mask = result['masks'][:, :, i] * np.uint8(255)
            # img_msg = self._cv_bridge.cv2_to_imgmsg(mask, 'mono8')
            img_msg = ros_numpy.msgify(Image, mask, encoding='mono8')
            img_msg.header = msg.header
            result_msg.masks.append(img_msg)

        return result_msg


    def _visualize(self, result, image):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        axes = fig.gca()
        visualize.display_instances(image, result['rois'], result['masks'],
                                    result['class_ids'], CLASS_NAMES,
                                    result['scores'], ax=axes,
                                    class_colors=self._class_colors)
        fig.tight_layout()
        canvas.draw()
        result = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

        _, _, w, h = fig.bbox.bounds
        result = result.reshape((int(h), int(w), 3))
        return result

    def _image_callback(self, msg):
        rospy.logdebug("Get an image")
        if self._msg_lock.acquire(False):
            self._last_msg = msg.img_req
            self._cloud = msg.cloudReq
            self._msg_lock.release()

def main():
    rospy.init_node('mask_rcnn')
    
    node = MaskRCNNNode()
    rospy.Service('run_inference_maskrcnn', Mask_RCNN, node.run)
    rospy.spin()

if __name__ == '__main__':
    main()
