#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import cv2

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

try:
    from .yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
    from .yolo3.utils import letterbox_image
    from .yolo3 import utils as Utils
except:
    from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
    from yolo3.utils import letterbox_image
    from yolo3 import utils as Utils

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.utils import multi_gpu_model

gpu_num = 1

from configparser import ConfigParser

THETA_DIRECTION=-1
# THETA_DIRECTION=+1

class YOLO(object):
    def __init__(self, config_file="keras_yolo3.cfg", config=None):
        if config is None:
            assert (os.path.isfile(config_file))
            config = ConfigParser()
            config.read(config_file)

        self.model_path = config.get("Eval", "ModelPath")  # model path or trained weights path
        self.anchors_path = config.get("Model", "anchors")
        self.classes_path = config.get("Model", "LabelsPath")

        self.score = 0.3
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416)  # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes, self.angles = self.generate()

        # self.draw_offsets = np.array([1, 1, -1, 1, -1, -1, 1, -1])

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
        boxes, scores, classes, angles = yolo_eval(self.yolo_model.output, self.anchors,
                                                   len(self.class_names), self.input_image_shape,
                                                   score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes, angles

    def detect_image(self, image, info=True):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        if info: print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes, out_angles = self.sess.run(
            [self.boxes, self.scores, self.classes, self.angles],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        if info: print('Found {} boxes and {} angles for {}'.format(len(out_boxes), len(out_angles), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300


        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            theta = out_angles[i]
            if info: print("Outputs box and theta", box, theta)

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box

            points, (cx, cy) = Utils.angledBox2RotatedBox(top, left, bottom, right, THETA_DIRECTION*theta)

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            if info: print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([cx, cy - label_size[1]])
            else:
                text_origin = np.array([cx, cy + 1])

            # My kingdom for a good redistributable image drawing library.
            # for i in range(thickness):
            #     draw.rectangle(
            #         [left + i, top + i, right - i, bottom - i],
            #         outline=self.colors[c])

            points = points.flatten()
            if info: print(points, theta)
            if info: print(left, top, right, bottom)

            # Makes sure the points are inside the image.
            _points = points
            xs, ys = _points[::1], _points[::2]
            xs[xs > image.size[0]] = image.size[0]
            ys[ys > image.size[1]] = image.size[1]
            xs[xs < 0] = 0
            ys[ys < 0] = 0

            if info: print(tuple(_points[:4]), _points[2:6], _points[4:8], _points.take(range(6, 10), mode='wrap'))
            if info: print(_points[:4], _points[2:6], _points[4:8], _points.take(range(6, 10), mode='wrap'))

            # Draws the box
            draw.line(tuple(_points[:4]), fill=self.colors[c], width=thickness)
            draw.line(tuple(_points[2:6]), fill=self.colors[c], width=thickness)
            draw.line(tuple(_points[4:8]), fill=self.colors[c], width=thickness)
            draw.line(tuple(_points.take(range(6, 10), mode='wrap')), fill=self.colors[c], width=thickness)

            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])

            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        if info: print("Detection done in", end - start)
        return image, len(out_classes)>0

    def segment_image(self, image, info=True):
        '''
        Simple segmentation by object detection and cropping outside the box.
        :param image:
        :return:
        '''
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        img = np.array(boxed_image, dtype='uint8')
        image_data = np.array(boxed_image, dtype='float32')

        # print(img.shape)

        shape = image_data.shape
        if info: print(shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes, out_angles = self.sess.run(
            [self.boxes, self.scores, self.classes, self.angles],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        if info: print('Found {} boxes and {} angles for {}'.format(len(out_boxes), len(out_angles), 'img'))

        total_mask = np.zeros(shape, dtype=np.uint8)
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            theta = out_angles[i]
            if info: print("Outputs box and theta", box, theta)

            label = '{} {:.2f}'.format(predicted_class, score)

            top, left, bottom, right = box

            # print(total_mask.shape)
            total_mask = cv2.bitwise_or(total_mask,
                                        Utils.getMaskByAngledBox(shape, top, left, bottom, right, THETA_DIRECTION*theta))

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            if info: print(label, (left, top), (right, bottom))

        end = timer()
        if info: print("Segmentation done in", end - start)

        # If there is no object detection return whole image.
        if total_mask.sum() == 0:
            return image

        img = cv2.bitwise_and(img, total_mask)

        # img[total_mask==0] = 255

        # Segment with detection
        return img

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image, _ = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()





import glob
from os.path import join
from os.path import split
from os import makedirs
def getImageFiles(input_folder, exts=(".jpg", ".gif", ".png", ".tga", ".tif"), recursive=False):
    files = []
    for ext in exts:
        files.extend(glob.glob(join(input_folder, '*%s' % ext), recursive=recursive))
    return files


def detect_img(yolo, config):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
            image = image.convert('RGB')
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image, _ = yolo.detect_image(image)
            r_image.save(os.path.split(img)[1])
            r_image.show()

    yolo.close_session()

def detect_imgs(yolo, config):
    input_folder, output_folder = config.get("Eval", "input_folder"), config.get("Eval", "output_folder")

    imgs = getImageFiles(input_folder)
    makedirs(output_folder, exist_ok=True)

    ret, d = 0, 0
    for i, img in enumerate(imgs,1):
        try:
            image = Image.open(img)
            image = image.convert('RGB')
        except:
            print('Open',img,' Error! Try again!')
            continue
        r_image, detected = yolo.detect_image(image, info=False)
        output_file = join(output_folder, split(img)[1])
        r_image.save(output_file)
        print('[%.2f]Saved'%(i/len(imgs)*100), output_file)

        ret+=1
        if detected:
            d+=1

    print("Image detection percentage:%.2f" %(d/ret*100))
    yolo.close_session()

def segment_imgs(yolo, config):
    input_folder, output_folder = config.get("Eval", "input_folder"), config.get("Eval", "output_folder")
    imgs = getImageFiles(input_folder)
    ret, d = 0, 0
    for i, img in enumerate(imgs,1):
        try:
            image = Image.open(img)
            image = image.convert('RGB')
        except:
            print('Open',img,' Error! Try again!')
            continue
        r_image, detected = yolo.segment_image(image, info=False)
        output_file = join(output_folder, split(img)[1])
        r_image.save(output_file)
        print('[%.2f]Saved'%(i/len(imgs)*100), output_file)

        ret+=1
        if detected:
            d+=1

    print("Image detection percentage:%.2f" %(d/ret*100))
    yolo.close_session()

def segment_img(yolo, config):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
            image = image.convert('RGB')
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image, _ = yolo.segment_image(image)
            r_image.save(os.path.split(img)[1])
            r_image.show()

    yolo.close_session()

if __name__ == '__main__':
    config_file = "keras_yolo3.cfg"
    assert (os.path.isfile(config_file))
    config = ConfigParser()
    config.read(config_file)
    mode = config.get("Eval", "mode")
    yolo = YOLO(config=config)

    modes = {'image_detect':detect_img,'image_segment':segment_img,
             'dataset_detect':detect_imgs,'dataset_segment':segment_imgs}

    modes[mode](yolo, config)


