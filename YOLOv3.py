import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import tensorflow as tf
from absl import logging
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
from yolov3_tf2.models import YoloV3
from yolov3_tf2.utils import load_darknet_weights

def convert(weights='./data/yolov3.weights', output='./checkpoints/yolov3.tf'):
    yolo = YoloV3()
        
    yolo.summary()
    logging.info('model created')

    load_darknet_weights(yolo, weights, False)
    logging.info('weights loaded')

    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    tmp_out = yolo(img)
    logging.info('sanity check passed')

    yolo.save_weights(output)
    logging.info('weights saved')

def detect(image='./data/girl.png', output='./output.jpg', 
           classes='./data/coco.names', weights='./checkpoints/yolov3.tf'):
    yolo = YoloV3()

    yolo.load_weights(weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(classes).readlines()]
    logging.info('classes loaded')

    img = tf.image.decode_image(open(image, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img = transform_images(img, 416)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           scores[0][i].numpy(),
                                           boxes[0][i].numpy()))

    img = cv2.imread(image)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(output, img)
    logging.info('output saved to: {}'.format(output))

'''
flags.DEFINE_string('dataset', '', 'path to dataset')
flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'none',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 2, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
'''