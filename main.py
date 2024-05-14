import cv2 as cv
import numpy as np
import argparse
import sys
import os
import signal
import time
import logging

from senxor.utils import (data_to_frame, connect_senxor,
                          cv_render, RollingAverageFilter)

from Kalman_utils import Kalman_with_predict
# from STARK_utils import STARKFilter

parser = argparse.ArgumentParser()
parser.add_argument('--input_shape', action='append', default=[96,96], help='input shape')
parser.add_argument('--imgpath', type=str, default='selfie.jpg', help="image path")
parser.add_argument('--modelpath', type=str, default='yolov7_2cls_320test320nolmks_v4',
                    help="onnx filepath e.g., yolov7_2cls_320test320nolmks_v4 for no-lmks,"
                         " yolov7_2cls_320test320lmks_v6ModelV2, yolov7_2cls_320test320lmks_v5ModelV2 for lmks,"
                         "good_old/yolov7_2cls_lmks_v4 or good_old/bestestlmks2 or good_old/bestlmks4 for lmks,"
                         " yolov7NEW_v2") # bestest2, bestestlmks2, bestest, bestlmks4
parser.add_argument('--which_run', type=str, default='cv', help="do inference on which platform")
parser.add_argument('--confThreshold', default=0.7, type=float, help='class confidence')
parser.add_argument('--nmsThreshold', default=0.45, type=float, help='nms iou thresh')
parser.add_argument('--use_kpts', type=bool, default=False,  help='use keypoints or not')
parser.add_argument('--num_classes', type=int, default=2,  help='number of classes used at training')
parser.add_argument('--allign', type=bool, default=False,  help='use keypoints or not')
parser.add_argument('--continous_align', type=bool, default=True,  help='use keypoints or not')
parser.add_argument('--showrawwarped', type=bool, default=False,  help='use keypoints or not')
parser.add_argument('--use_std_mean', type=bool, default=False,  help='use mean and std for antispoof or not')
parser.add_argument('--use_min_max', type=bool, default=False,  help='use mean and std for antispoof or not')
parser.add_argument('--antispoof', type=bool, default=True,  help='use antispoofing or not')
parser.add_argument('--from_thermal', default=True, type=bool, help='nms iou thresh')
parser.add_argument('--use_both', default=False, type=bool, help='nms iou thresh')

parser.add_argument('--scale_before', type=bool, default=False,  help='scale before detection if true, else after')
parser.add_argument('--blur', type=bool, default=False,  help='use blur or not')
parser.add_argument('--which_process', type=int, default=2,  help='use processing 2 or 1 for thermal frame or not')
parser.add_argument('--use_input_ratio', type=bool, default=False,  help='use input to resize if true, '
                                                                    'else use output or not')

parser.add_argument('--clip_min_percent', default=0.15, type=float, help='percentage min clip for display')
parser.add_argument('--clip_max_percent', default=0.01, type=float, help='percentage max clip for display')
parser.add_argument('--clip_human', default=True, type=bool, help='percentage max clip for display')

parser.add_argument('--large_screen', default=True, type=bool, help='the model name')
parser.add_argument('--displayRK', default=False, type=bool,
                    help='if Raw+Kalman, will only display these Raw and Kalman, else will display (a) Kalman alone,\
                               (b) Kalman before + Gaussian, (c) Kalman before+cnn, (d) Kalman after+cnn')
parser.add_argument('--disp_scale', default=2, type=int, help='how to devide the image i.e., make it smaller')
parser.add_argument('--interpolation', default=cv.INTER_NEAREST, type=int,
                    help='how to devide the image i.e., make it smaller')
parser.add_argument('--put_FPS', default=True, type=bool, help='If show FPS for all processing')
parser.add_argument('--FPS_Divisor', default=2, type=int,
                    help='Which FPS divisor is used')  # 1 for mini cougar and 1 for cougar: for params below
parser.add_argument('--superes', default=True, type=bool,
                    help='If Super-resolution is used')  # False if not superes model if True ise smooth predict Kalman

# # Firmware Kalman
parser.add_argument('--firmware_Kalman', default=False, type=bool,
                    help='Use firmware Kalman')  # False for mini cougar and False for cougar: for FPS1
parser.add_argument('--firmware_Kalman_strength', default=0x64, type=int,
                    help='Use firmware Kalman')  # 100 for mini cougar and 100 for cougar: for FPS1

# # Firmware STARK if STARK, if not will be rolling avg
parser.add_argument('--use_STARK', default=True, type=bool,
                    help='Use firmware Kalman')  # True for mini cougar and cougar
parser.add_argument('--STARK_gradient', default=0x0A, type=int,
                    help='Stark gradient')  # 25 for mini cougar and 25 for cougar: for FPS1
parser.add_argument('--STARK_cuttoff', default=0x0A, type=int,
                    help='Stark type')  # 25 for mini cougar and 25 for cougar: for FPS1
# # Firmware Median
parser.add_argument('--use_firmware_median', default=False, type=bool,
                    help='Use firmware Kalman')  # True for mini cougar and cougar

# # Kalman python
parser.add_argument('--use_python_filter', default=True, type=bool, help='use filter if not used in firmare')
parser.add_argument('--strength_Kalman_python', default=25, type=int,
                    help='')  # 25 for mini cougar and 25 for cougar: for FPS1
parser.add_argument('--err_est_Kalman_python', default=2, type=int,
                    help='')  # 2 for mini cougar and 2 for cougar: for FPS1
parser.add_argument('--clip_temp', default=False, type=bool,
                    help='clip temparature or not?')  # False for mini cougar and False for cougar: for FPS1
parser.add_argument('--rolling_ave_stabilize', default=True, type=bool,
                    help='rolling ave stabilize or not?')  # True for mini cougar and True for cougar: for FPS1
parser.add_argument('--use_normalization', default=1, type=int,
                    help='Which models are used')  # 1 for 0 to 1 norm 2 otherwise
parser.add_argument('--depth_before', default=4, type=float, help='power Rolling average before Kalman')
parser.add_argument('--depth_after', default=4, type=float, help='power Rolling average before Kalman')
parser.add_argument('--depth', default=30, type=float,
                    help='power Rolling average stabilize Kalman')  # 30 for mini cougar and 30 for cougar: for FPS1
parser.add_argument('--pixel_err_std_Kalman_python', default=1, type=float,
                    help='err_est of Kalman')  # 2 for mini cougar and 1 for cougar: for FPS1
parser.add_argument('--ksize_Kalman_python', default=5, type=int,
                    help='ksize of Kalman predict blur')  # 5 for mini cougar and 5 for cougar: for FPS1
parser.add_argument('--sigma_Kalman_python', default=1, type=int,
                    help='sigma of Kalman predict blur')  # 1 for mini cougar and 1 for cougar: for FPS1
parser.add_argument('--scale_Kalman_python', default=0.2, type=int,
                    help='<= 0.5')  # 0.2 for mini cougar and 0.2 for cougar: for FPS1
parser.add_argument('--smooth_new_frame_Kalman_python', default=True, type=bool,
                    help='err_est of Kalman')  # True for mini cougar and True for cougar: for FPS1
parser.add_argument('--Kalman_which_predict', default='Box', type=str, help='what to use to predict Kalman gain')
parser.add_argument('--out_smooth', default=False, type=bool, help='smooth out or not')
parser.add_argument('--gain_which', default='stair-case', type=str, help='sigmoid-approx or stair-case')
parser.add_argument('--model_name', default=os.path.join('RFDB_WaveletFalse_frozen_graph22.pb'), type=str,
                    help='the model name')  # good 20, 22 and 30

# which device to run on
parser.add_argument('--cpu_or_gpu', default=1, type=int, help='0 for gpu+cpu, 1 for cpu only')


providers = [['CUDAExecutionProvider', 'CPUExecutionProvider'], ['CPUExecutionProvider']]


class YOLOv7_face:
    def __init__(self, path, conf_thres=0.2, iou_thres=0.5, which_run='tflite', shape=(96, 96),
                 num_classes=1, use_kpts=True):
        self.alignMat = None
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.which_run = which_run
        self.frame = None
        self.num_classes = num_classes
        if num_classes == 1:
            self.class_names = ['face']
        elif num_classes == 2:
            self.class_names = ['rgbface', 'Tface', 'RealFace']
        else:
            print(f'[ERORR]: num_classes must be in [1, 2], bus is instead: {num_classes}')
            exit(0)

        self.use_kpts = use_kpts

        # Initialize model
        if self.which_run == "onnx":
            import onnxruntime
            session_option = onnxruntime.SessionOptions()
            session_option.log_severity_level = 3
            # self.session = onnxruntime.InferenceSession(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.session = onnxruntime.InferenceSession(path + '.onnx', sess_options=session_option)
            model_inputs = self.session.get_inputs()
            self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
            self.output_names = [model_outputs.name for model_outputs in self.session.get_outputs()]
            print('inputs: ', self.input_names, '&   outputs: ', self.output_names)
            self.input_shape = model_inputs[0].shape
            self.input_height = int(self.input_shape[2])
            self.input_width = int(self.input_shape[3])

            model_outputs = self.session.get_outputs()
            self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

        elif self.which_run == 'tflite':
            import tensorflow as tf

            self.session = tf.lite.Interpreter(model_path=path + '.tflite')
            self.session.allocate_tensors()

            # Get input and output tensors.
            self.input_details = self.session.get_input_details()
            # print(str(self.model))
            print('inputs: ', str(self.input_details))
            self.output_details = self.session.get_output_details()
            print('outputs: ', str(self.output_details))
            self.input_shape = self.input_details[0]['shape']
            self.input_height = int(self.input_shape[2])
            self.input_width = int(self.input_shape[3])

        elif self.which_run == 'cv':

            # self.session = cv.dnn.readNet(path + '.onnx')
            self.session = cv.dnn.readNetFromONNX(path + '.onnx')
            self.input_shape = shape
            self.input_height = int(self.input_shape[1])
            self.input_width = int(self.input_shape[0])

    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_width, self.input_height
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_height, int(self.input_width / hw_scale)
                img = cv.resize(srcimg, (neww, newh), interpolation=cv.INTER_AREA)
                left = int((self.input_width - neww) * 0.5)
                img = cv.copyMakeBorder(img, 0, 0, left, self.input_width - neww - left, cv.BORDER_CONSTANT,
                                         value=(114, 114, 114))  # add border
            else:
                newh, neww = int(self.input_height * hw_scale), self.input_width
                img = cv.resize(srcimg, (neww, newh), interpolation=cv.INTER_AREA)
                top = int((self.input_height - newh) * 0.5)
                img = cv.copyMakeBorder(img, top, self.input_height - newh - top, 0, 0, cv.BORDER_CONSTANT,
                                         value=(114, 114, 114))
        else:
            img = cv.resize(srcimg, (self.input_width, self.input_height), interpolation=cv.INTER_AREA)
        return img, newh, neww, top, left

    def prepare_input(self, image):
        print(f"image.shape: {image.shape}")
        self.img_height, self.img_width = image.shape[:2]
        self.scale = np.array(
            [self.img_width / self.input_width, self.img_height / self.input_height] * 2, dtype=np.float32)

        # self.scale = np.array([self.input_width, self.input_height,
        #                        self.input_width, self.input_height], dtype=np.float32)

        # self.scale = np.array([self.img_width, self.img_height,
        #                        self.img_width, self.img_height], dtype=np.float32)

        input_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv.resize(input_img, (self.input_width, self.input_height))
        # input_img, newh, neww, top, left = self.resize_image(cv.cvtColor(image, cv.COLOR_BGR2RGB)) ###也可以使用保持高宽比resize的pad填充

        # Scale input pixel values to 0 to 1
        input_img = input_img.astype(np.float32) / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :]
        print(f"input_tensor.shape: {input_tensor.shape}")
        return input_tensor

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def detect(self, image, thermal=False, antispoof=False):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        if self.which_run == "onnx":
            outputs = self.session.run(self.output_names, {input_name: input_tensor for input_name in self.input_names})
        elif self.which_run == "tflite":
            self.session.set_tensor(self.input_details[0]['index'], input_tensor)
            self.session.invoke()

            outputs = [self.session.get_tensor(self.output_details[i]['index'])
                       for i in range(len(self.output_details))]

            if not self.use_kpts:
                outputs = [outputs[-1], outputs[0], outputs[1]]

        elif self.which_run == "cv":
            # input_tensor = cv.dnn.blobFromImage(image.astype(np.float32), scalefactor=1.0,
            #                                      size=(self.input_width, self.input_height), mean=0, swapRB=True)

            self.session.setInput(input_tensor)
            outputs = list(self.session.forward(self.session.getUnconnectedOutLayersNames()))

            # wrong positions, so swap
            # outputs = outputs[-1:] + outputs[:-1]
            outputs = [outputs[-1], outputs[0], outputs[1]]

        # print(len(outputs), outputs[0].shape, outputs[1].shape, outputs[2].shape)
        # exit(0)

        boxes, scores, kpts, classes = self.process_output(outputs, thermal=thermal, antispoof=antispoof)

        return boxes, scores, kpts, classes

    def _make_grid(self, nx=20, ny=20):
        yv, xv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(float)

    def _make_grid_np(self, nx=20, ny=20):
        yv, xv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((yv, xv), 2).reshape((1, 1, ny, nx, 2)).astype(float)

    def process_output(self, output, thermal=False, antispoof=False):
        if thermal and not antispoof:
            conf_threshold = self.conf_threshold # 0.1 for flir only 0.3 for others
        elif antispoof:
            conf_threshold = 0.3
        else:
            conf_threshold = 0.4
            # conf_threshold = self.conf_threshold + 0.1

        # print(len(output), output[0].shape, output[1].shape, output[2].shape)
        # # exit(0)

        # temp = np.concatenate([np.squeeze(output[0]).reshape((-1, 21)), np.squeeze(output[1]).reshape((-1, 21)),
        #                        np.squeeze(output[2]).reshape((-1, 21))], axis=0)
        # print(temp[:, 4])
        # exit(0)

        strides = [8, 16, 32]
        anchor0 = [4, 5, 6, 8, 10, 12]
        anchor1 = [15, 19, 23, 30, 39, 52]
        anchor2 = [72, 97, 123, 164, 209, 297]

        # anchor0 = [3,4,  5,6,  10,11]
        # anchor1 = [20,21,  33,38,  38,38]
        # anchor2 = [36,44,  54,46,  64,77]

        anchors = [anchor0, anchor1, anchor2]
        nl = len(anchors)  # number of detection layers
        na = len(anchors[0]) // 2  # number of anchors
        grid = [np.zeros(1)] * nl
        flip_test = False
        nkpt = 5  # number of keypoints
        nc = self.num_classes # number of classes

        if not self.use_kpts:
            nkpt = 0
        no_det = (nc + 5)  # number of outputs per anchor for box and class
        no_kpt = 3 * nkpt  ## number of outputs per anchor for keypoints
        no = no_det + no_kpt
        anchor_grid = np.array(anchors).astype(float).reshape(nl, -1, 2)  # shape(nl,na,2)
        anchor_grid = anchor_grid.reshape(nl, 1, -1, 1, 1, 2)  # shape(nl,1,na,1,1,2)
        # print(anchor_grid.shape)
        # # print(anchor_grid.reshape(1, na, 1, 1, 2).shape)
        # print(na, nl, no)
        # # exit(0)
        for i in range(na):
            # output[i] = output[i].astype(float), [4, 3], [3, 2]
            if self.use_kpts:
                x_det = self.sigmoid(output[i][..., :6+nc-1])
            else:
                x_det = self.sigmoid(output[i][..., :7])

                # x_det = output[i]
                # x_det[..., :6] = self.sigmoid(x_det[..., :6])
            # print(x_det[..., :7])
            # exit(0)
            # x_det = output[i][..., :6]

            if grid[i].shape[2:4] != output[i].shape[2:4]:
                # bs, na, ny, nx, no = output[i].shape
                bs, _, ny, nx, _ = output[i].shape
                # print(nx, ny)

                # original torch meshgrid
                # grid[i] = self._make_grid(nx, ny)
                # kpt_grid_x = grid[i][..., 0:1]
                # kpt_grid_y = grid[i][..., 1:2]

                # # # swap indices from original due to numpy's behavior
                grid[i] = self._make_grid_np(nx, ny)
                kpt_grid_x = grid[i][..., 0:1]
                kpt_grid_y = grid[i][..., 1:2]

            x_det[..., 0:2] = (x_det[..., 0:2] * 2. - 0.5 + grid[i]) * strides[i]
            x_det[..., 2:4] = (x_det[..., 2:4] * 2.) ** 2 * anchor_grid[i].reshape(1, na, 1, 1, 2)

            if self.use_kpts:
                x_kpt = output[i][..., 6+nc-1:]
                # x_kpt = self.sigmoid(output[i][..., 6:])

                # print(output[i].shape, x_kpt.shape, np.tile(kpt_grid_x, [1, 1, 1, 1, nkpt]).shape)
                # print(x_kpt[..., ::3].shape, x_kpt[..., 1::3].shape, x_kpt[..., 2::3].shape)
                # exit(0)

                x_kpt[..., 0::3] = (x_kpt[..., ::3] * 2. - 0.5 + np.tile(kpt_grid_x, [1, 1, 1, 1, nkpt])) * strides[i]  # xy
                x_kpt[..., 1::3] = (x_kpt[..., 1::3] * 2. - 0.5 + np.tile(kpt_grid_y, [1, 1, 1, 1, nkpt])) * strides[i]  # xy
                x_kpt[..., 2::3] = self.sigmoid(x_kpt[..., 2::3])

                output[i] = np.concatenate([x_det, x_kpt], axis=-1)

            else:
                output[i] = x_det

        #     print(output[i].shape)
        # exit(0)

        predictions = np.concatenate([np.squeeze(output[0]).reshape((-1, no)), np.squeeze(output[1]).reshape((-1, no)),
                                      np.squeeze(output[2]).reshape((-1, no))], axis=0)
        # print(predictions.shape)
        # print(predictions[..., 4].max())
        # exit(0)

        # # Filter out object confidence scores below threshold
        obj_conf = predictions[:, 4]
        # print(obj_conf)
        # exit(0)

        predictions = predictions[obj_conf > conf_threshold]
        obj_conf = obj_conf[obj_conf > conf_threshold]
        # print('obj_conf and scores check')
        # print(obj_conf, 'obj_conf', thermal)
        # print(predictions[:, 5], 'predictions_5', thermal)
        # print(predictions[:, 6], 'predictions_6', thermal)
        # print(predictions[:, -1], 'predictions_last', thermal)


        # Multiply class confidence with bounding box confidence
        # predictions[:, 5] = self.sigmoid(predictions[:, 5]) * obj_conf
        predictions[:, 5:5+nc] *= obj_conf[..., np.newaxis]

        # Get the scores
        scores = predictions[:, 5:5+nc]
        # print(scores, 'pred before', thermal)

        scores, cls = scores.max(axis=1, keepdims=False), np.argmax(scores, axis=1,  keepdims=False)
        # print(scores, cls)

        # Filter out the objects with a low score
        valid_scores = scores > conf_threshold
        predictions = predictions[valid_scores]
        scores = scores[valid_scores]
        cls = cls[valid_scores]
        # print(scores, 'pred after', thermal)

        # Get bounding boxes for each object
        boxes, kpts = self.extract_boxes(predictions)

        # print(boxes)
        # exit(0)

        indices = cv.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold,
                                   self.iou_threshold)
        # print(boxes)
        # exit(0)
        if self.use_kpts:
            return boxes[indices], scores[indices], kpts[indices], cls[indices]
        else:
            return boxes[indices], scores[indices], None, cls[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4] * self.scale

        # Convert boxes to xywh format
        boxes_ = np.copy(boxes)
        boxes_[..., 0] = boxes[..., 0] - boxes[..., 2] * 0.5
        boxes_[..., 1] = boxes[..., 1] - boxes[..., 3] * 0.5

        # kpts
        kpts = []
        if self.use_kpts:
            kpts = predictions[:, 6+self.num_classes-1:]  ###x1,y1,score1, ...., x5,y5,score5
            kpts *= np.tile(np.array([self.scale[0], self.scale[1], 1], dtype=np.float32), (1, 5))

        return boxes_, kpts

    def remap(self, data, new_range=(0, 255), curr_range=None, to_uint8=True):

        lo2, hi2 = new_range
        #
        if curr_range is None:
            lo1 = np.min(data)
            hi1 = np.max(data)
        else:
            lo1, hi1 = curr_range
        #
        # The relpos below represents the relative position of _data in the
        # current range.
        # We could potentially manipulate relpos by some function to
        # realise non-linear remapping
        relpos = (data - lo1) / float(hi1 - lo1)
        out = lo2 + relpos * (hi2 - lo2)
        #
        if to_uint8:
            return out.astype('uint8')
        else:
            return out.astype('float16')

    def process_thermal1(self, frame, clip_min=0.2, clip_max=0.05, clip_human=False,
                         Tmax=None, Tmin=None, scale=None, scale_before=True):
        Trange = Tmax - Tmin

        if Tmax is None or Tmin is None:
            self.Tmax, self.Tmin = frame.max(), frame.min()

        if clip_human:
            self.frame = frame.copy()
            self.frame[frame > 40] = 40
            self.frame[frame < 24] = 24
            frame = (frame + 9 * self.frame) / 10
        else:

            frame = np.clip(frame, Tmin + clip_min * Trange, Tmax - clip_max * Trange)

        if scale is None:
            scale = int(self.input_shape[0] // frame.shape[0])
        self.frame = frame.repeat(scale, axis=0).repeat(scale, axis=1)

        if scale_before:
            frame = self.remap(self.frame)
        else:
            frame = self.remap(frame)

        frame = np.stack([frame, frame, frame], axis=-1)

        return frame

    def process_thermal2(self, frame, clip_min=0.2, clip_max=0.05, clip_human=False, Tmax=None, Tmin=None, scale=None,
                         blur=False, use_input_ratio=False, interpolation=cv.INTER_NEAREST, scale_before=True):
        Trange = Tmax - Tmin

        if Tmax is None or Tmin is None:
            self.Tmax, self.Tmin = frame.max(), frame.min()

        if clip_human:
            self.frame = frame.copy()
            self.frame[frame > 40] = 40
            self.frame[frame < 23] = 23
            frame = (frame + 9*self.frame)/10

        else:
            self.frame = np.clip(frame, Tmin + clip_min * Trange, Tmax - clip_max * Trange)
            frame = (frame + 9 * self.frame) / 10

        frame = self.remap(frame)

        if scale is None:
            scale = int(self.input_shape[0] // frame.shape[0])

        if use_input_ratio:
            self.frame = cv.resize(frame, (self.input_shape[1] * scale, self.input_shape[0] * scale),
                               interpolation=interpolation)
        else:
            self.frame = cv.resize(frame, (frame.shape[1]*scale, frame.shape[0]*scale),
                               interpolation=interpolation)

        if scale_before:
            frame = self.remap(self.frame)
        else:
            frame = self.remap(frame)

        frame = np.stack([frame, frame, frame], axis=-1)

        if blur:
            frame = cv.blur(frame, (3,3))

        self.frame = self.remap(self.frame, (Tmin, Tmax), to_uint8=False)

        return frame

    def process_thermal(self, frame, clip_min=0.2, clip_max=0.05, clip_human=False, Tmax=None, Tmin=None, scale=None,
                         blur=False, which_process=2, use_input_ratio=False, scale_before=True):

        if which_process == 1:
            frame = self.process_thermal1(frame, clip_min=clip_min, clip_max=clip_max, scale_before=scale_before,
                                          clip_human=clip_human, Tmax=Tmax, Tmin=Tmin, scale=scale)

        else:
            frame = self.process_thermal2(frame, clip_min=clip_min, clip_max=clip_max, clip_human=clip_human,
                                          Tmax=Tmax, Tmin=Tmin, scale=scale, blur=blur,
                                          use_input_ratio=use_input_ratio, scale_before=scale_before,)

        return frame

    def process_rgb(self, rgbimg):
        rgbimg = cv.cvtColor(rgbimg, cv.COLOR_BGR2GRAY)
        return np.stack([rgbimg, rgbimg, rgbimg], axis=-1)

    def TooSlowboxblur(self, img, kernel_size=(3,3)):
        kernel = np.zeros(kernel_size)
        kernel.fill(1)
        kernel_sum = kernel.sum()
        height, width = img.shape
        filtered_sum = 0
        max_img_temp = img.max()
        min_img_temp = img.min()

        for x in range(1, width - 1):
            for y in range(1, height - 1):
                # print(f"x: {x}, y: {y}")
                for i in range(kernel.shape[0]):
                    for j in range(kernel.shape[1]):
                        # print(f"kernel_shape: {kernel.shape}, j: {j}, {i}")
                        index_y, index_x = y - 1 + j, x - 1 + i
                        if index_y >= height:
                            index_y = -1
                        if index_x >= width:
                            index_x = -1
                        filtered_sum += img[index_y, index_x] * kernel[i][j]

                new_pixel_r = filtered_sum / kernel_sum

                if max_img_temp > max_img_temp:
                    new_pixel_r = max_img_temp
                elif new_pixel_r < min_img_temp:
                    new_pixel_r = min_img_temp

                img[y, x] = new_pixel_r
                filtered_sum = 0

        return img

    def draw_detections(self, image, boxes, scores, kpts, classes, thermal=False, use_std_mean=False,
                        use_min_max=False, scale=(1, 1), hoffset=40, xoffset=15, antispoof_=False):
        return self.draw_kpts_and_boxes(image, boxes, scores, kpts, classes, thermal=thermal,
                                        use_std_mean=use_std_mean, use_min_max=use_min_max,
                                        hoffset=hoffset, xoffset=xoffset,scale=scale, antispoof_=antispoof_)

    def draw_kpts_and_boxes(self, image, boxes, scores, kpts=None, classes=None, thermal=False, antispoof_=False,
                            use_std_mean=False, use_min_max=False, hoffset=15, xoffset=15, scale=(1,1)):
        size = image.shape
        ratio = size[1] // 5, size[0] // 5
        use_kpts = True
        if kpts is None:
            kpts=[[0]*15]*len(boxes)
            use_kpts = False
        # print(use_kpts)

        if thermal:
            self.TfacesBool = False
            self.Tfacemask = np.zeros_like(image.copy())
            # self.frame = self.TooSlowboxblur(self.frame, kernel_size=(2,2))

        else:
            self.RGBFacebool = False
            self.RGBFaceMask = np.zeros((int(size[0]*scale[0]), int(size[1]*scale[1]), 3)).astype(np.uint8)

        for box, score, kp, cls in zip(boxes, scores, kpts, classes):
            x, y, w, h = box.astype(int)

            if thermal and self.frame is not None:

                Temparature = self.frame[y:y + h, x:x + w]

                hT, wT = Temparature.shape
                if hT < ratio[0] or wT < ratio[1]:
                    print("[INFO] Too far. Can't determine if human", hT, wT, size)
                    continue

                elif wT >= int(ratio[1]*4*.95) or hT >= int(ratio[0]*4*.95):
                    print("[INFO] Too close. Can't determine if human", hT, wT, size)
                    continue

                elif (hT > 1.5 * wT) or (wT > 1.5 * hT):  # hT > 1.62*wT or wT > 1.62*hT:
                    print("[INFO] Face aspect incorrect."
                          "Or Face not centered. Can't determine if human", hT, wT, size)
                    continue

                if use_std_mean:
                    std, mean = np.std(Temparature), Temparature.mean()
                    # print('====', std, mean, min, max, '====')  # fix std

                    if std == mean or mean > 150:
                        print("[INFO] Seems thermal data not passed as temparature, so don't trust algo"
                              "  min : %.2f, max: %.2f, score: %.2f" %(std, mean, score))
                    elif std < 0.25:
                        print("[INFO] Seems temp signature is not human.", std)
                        continue
                if use_min_max:
                    min, max = Temparature.min(), Temparature.max()
                    if max > 36.5 or min < 31:
                        print("[INFO] Seems too hot or too cold. Can't determine if human"
                              "  min : %.2f, max: %.2f, score: %.2f" %(min, max, score))
                        continue

                    elif max - min > 1.5:
                        print("[INFO] Seems Face Statistics has issues. Can't determine if human"
                              "  min : %.2f, max: %.2f, score: %.2f" %(min, max, score))
                        continue

                y1, y2, x1, x2 = y - hoffset, y + h + hoffset, x - xoffset, x + w + xoffset
                if y1 < 0:
                    y1 = 0
                if x1 < 0:
                    x1 = 0
                if y2 > size[0]:
                    y2 = size[0]
                if x2 > size[1]:
                    x2 = size[1]

                self.Tfacemask[y1:y2, x1:x2, :] += 1
                self.TfacesBool = True

            else:
                Temparature = image[y:y + h, x:x + w, 0]

                hT, wT = Temparature.shape
                if hT < ratio[0] or wT < ratio[1]:
                    print("[INFO] Too far. Can't determine if human", hT, wT, size)
                    continue

                elif wT >= int(ratio[1] * 4 * .95) or hT >= int(ratio[0] * 4 * .95):
                    print("[INFO] Too close. Can't determine if human", hT, wT, size)
                    continue

                # elif (hT > 1.4 * wT) or (wT > 1.4 * hT):  # hT > 1.62*wT or wT > 1.62*hT:
                #     print("[INFO] Face aspect incorrect."
                #           "Or Face not centered. Can't determine if human", hT, wT, size)
                #     continue

                scaley, scalex = scale
                y1, x1, w1, h1 = int(y * scaley), int(x * scalex), int(w * scalex), int(h * scaley)
                hoffset, xoffset = int(hoffset * scaley), int(xoffset * scalex)
                y1, y2, x1, x2 = y1 - hoffset, y1 + h1 + hoffset, x1 - xoffset, x1 + w1 + xoffset
                # y1, y2, x1, x2 = int(y1 * scale[1]), int(y2 * scale[1]), int(x1 * scale[0]), int(x2 * scale[0])
                if y1 < 0:
                    y1 = 0
                if x1 < 0:
                    x1 = 0
                if y2 > size[0]:
                    y2 = size[0] * scale[1]
                if x2 > size[1]:
                    x2 = size[1] * scale[0]
                y1, y2, x1, x2 = int(y1), int(y2), int(x1), int(x2)
                print(self.RGBFaceMask.shape)
                print(y1, y2, x1, x2)
                self.RGBFaceMask[y1:y2, x1:x2, :] += 1
                self.RGBFacebool = True

            # Draw rectangle
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=1)
            if antispoof_:
                label = self.class_names[-1]
            else:
                label = self.class_names[int(cls)]

            # labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # top = max(y1, labelSize[1])
            # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]),
            #                                                               top + baseLine), (255,255,255), cv.FILLED)

            cv.putText(image, label + ": %.2f"%(score), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)

            if use_kpts:
                for i in range(5):
                    cv.circle(image, (int(kp[i * 3]), int(kp[i * 3 + 1])), 1, (0, 255, 0), thickness=-1)

        if thermal:
                self.Tfacemask[self.Tfacemask > 0] = 255
        else:
            self.RGBFaceMask[self.RGBFaceMask > 0] = 255

        return image

    def allign_compute(self, imagestatsT, imagestatsRGB, planar=False, from_thermal_=True):
        # print("compute allign mat")
        maxarea = 0
        statsT = []
        _, boxes, scores, kpts, classes = imagestatsT
        if kpts is None:
            kpts = [0] * len(boxes)
        for box, score, kp, cls in zip(boxes, scores, kpts, classes):
            x, y, w, h = box.astype(int)
            area = w * h
            if area > maxarea:
                maxarea = area
                statsT = (x, y, w, h)
        # print('statsT: ', statsT, maxarea)

        maxarea = 0
        statsRGB = []
        _, boxes, scores, kpts, classes, scale = imagestatsRGB
        scaley, scalex = scale
        if kpts is None:
            kpts = [0] * len(boxes)
        for box, score, kp, cls in zip(boxes, scores, kpts, classes):
            x, y, w, h = box.astype(int)
            area = w * h
            if area > maxarea:
                maxarea = area
                statsRGB = (x, y, w, h)
        # print('statsRGB: ', statsRGB, maxarea)

        x, y, w, h = statsT
        # x, y, w, h = float(x), float(y), float(w), float(h)
        pointsT = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], np.float32)

        x, y, w, h = statsRGB
        x, y, w, h = x * scalex, y * scaley, w * scalex, h * scaley
        # print('scaled statsRGB: ', (x, y, w, h), w*h, 'scaley, scalex: ', scaley, scalex)
        pointsRGB = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], np.float32)

        # print(pointsRGB, pointsT)
        # exit(0)

        if not from_thermal_:
            temp = pointsRGB
            pointsRGB = pointsT
            pointsT = temp

        if planar:
            # get the affine transformation matrix
            M = cv.getAffineTransform(pointsT, pointsRGB)

        else:
            # get the perspective transformation matrix
            M = cv.getPerspectiveTransform(pointsT, pointsRGB)

        return M

    def allign_draw(self, imageT_, imageRGB_, warpmatrix_, planar=False,
                    Threshold=(28, 40), showwarped=False, antispoof=False,
                    use_both=False):

        warped_ = 0
        Twarped_ = 0
        if not antispoof and not use_both:
            self.Tnotoverlayed = imageT_.copy() # .astype(float)
            self.Tnotoverlayed[self.frame < Threshold[0]] = 0
            self.Tnotoverlayed[self.frame > Threshold[1]] = 0
        else:
            self.Tnotoverlayed = imageT_ # .astype(float)

        if planar:
            # apply the transformation
            height_, width_, _ = imageT_.shape
            if showwarped:
                notoverlayed_ = cv.warpAffine(imageT_, warpmatrix_, (width_, height_))
                warped_ = cv.addWeighted(notoverlayed_, 0.5, imageRGB_, 0.7, 0)

            self.Tnotoverlayed = cv.warpAffine(self.Tnotoverlayed, warpmatrix_, (width_, height_))
        else:
            # get the perspective transformation matrix
            # apply the transformation
            height_, width_, _ = imageT_.shape
            if showwarped:
                notoverlayed = cv.warpPerspective(imageT_, warpmatrix_, (width_, height_))
                warped_ = cv.addWeighted(notoverlayed, 0.5, imageRGB_, 0.7, 0)
            self.Tnotoverlayed = cv.warpPerspective(self.Tnotoverlayed, warpmatrix_, (width_, height_))

        if not antispoof:
            Twarped_ = cv.addWeighted(self.Tnotoverlayed, 0.7, imageRGB_, 0.7, 0)

        return self.Tnotoverlayed, Twarped_, warped_

    def get_objects(self, imgRGB):
        self.Tnotoverlayed[self.Tnotoverlayed > 0] = 1
        out = self.Tnotoverlayed*imgRGB
        self.Tnotoverlayed[self.Tnotoverlayed == 1] = 255
        return self.Tnotoverlayed, out

    def enhnace_thermal(self, imgT, imgobjects):
        edges = cv.Canny(imgobjects, 100, 200)
        edges[edges==True] = 255
        edges = np.stack([edges, edges, edges], axis=-1)
        img_disp = cv.addWeighted(imgT, 0.5, edges, 0.7, 0)

        return edges, img_disp

    def antispoof(self, image_disp_, continuos_align_=False, STATST_=[], STATSRGB_=[], planar_=False,
                  from_thermal_=True, interpolation=cv.INTER_NEAREST, use_both=False):
        # image_disp_ = [RGBimg, Thermalimg]
        if continuos_align_ or self.alignMat is None:
            if len(STATSRGB_[1]) and len(STATST_[1]):
                self.alignMat = YOLOv7_face_detector.allign_compute(STATST_, STATSRGB_, planar=planar_,
                                                                    from_thermal_=from_thermal_)

        if (self.TfacesBool) and (from_thermal_) and (self.alignMat is not None):
            TFace_ = image_disp_[1]
            image_disp_ = image_disp_[0]
            self.TfacesBool = False
            self.Tfacemask, _, _ =self.allign_draw(self.Tfacemask, image_disp_, self.alignMat, planar=False,
                                                   Threshold=None, showwarped=False, antispoof=True)

            if use_both:
                # faces_ = cv.applyColorMap(faces_, cv.COLORMAP_JET)
                faces_ = cv.cvtColor(image_disp_, cv.COLOR_BGR2GRAY)
                faces_ = np.stack([faces_, faces_, faces_], axis=-1)
                _, faces_, _ = self.allign_draw(TFace_, faces_, self.alignMat, planar=False,
                                                Threshold=None, showwarped=False, antispoof=False, use_both=use_both)
                faces_ = faces_ * self.RGBFaceMask * self.Tfacemask

            else:
                self.Tfacemask[self.Tfacemask > 0] = 1
                self.RGBFaceMask[self.RGBFaceMask > 0] = 1
                faces_ = image_disp_ * self.RGBFaceMask * self.Tfacemask
            # faces_ = self.process_rgb(faces_)

            if faces_.max() != 0:
                boxes_, scores_, kpts_, classes_ = self.detect(faces_, thermal=False,
                                                               antispoof=True)  # 2nd pass not needed
                # _, boxes_, scores_, kpts_, classes_, _ = STATST_[0]
                image_disp_ = self.draw_detections(image_disp_, boxes_, scores_, kpts_, classes_,
                                                   thermal=False, antispoof_=True)

            return image_disp_, faces_, self.alignMat

        elif (self.RGBFacebool) and (not from_thermal_) and (self.alignMat is not None):
            # print(self.RGBFaceMask.shape, self.Tfacemask.shape, image_disp_[0].shape, image_disp_[1].shape,
            #       STATSRGB_[0].shape, STATST_[0].shape, STATSRGB_[-1])
            # # exit(0)
            RGBFace_ = image_disp_[0]
            image_disp_ = image_disp_[1]
            shape_ = (image_disp_.shape[1], image_disp_.shape[0])

            self.RGBFacebool = False
            self.RGBFaceMask, _, _ = self.allign_draw(self.RGBFaceMask, image_disp_, self.alignMat, planar=False,
                                                      Threshold=None, showwarped=False, antispoof=True)

            if use_both:
                faces_ = cv.applyColorMap(image_disp_, cv.COLORMAP_JET)
                _, faces_, _ = self.allign_draw(RGBFace_, faces_, self.alignMat, planar=False,
                                                Threshold=None, showwarped=False, antispoof=False, use_both=use_both)
                faces_ = faces_ * self.RGBFaceMask * self.Tfacemask
            else:
                self.RGBFaceMask[self.RGBFaceMask > 0] = 1
                self.Tfacemask[self.Tfacemask > 0] = 1
                faces_ = image_disp_ * self.RGBFaceMask * self.Tfacemask

            # faces_ = self.process_rgb(faces_)
            image_disp_ = cv_render(image_disp_, resize=shape_, colormap='rainbow2', display=False,
                                    interpolation=interpolation)

            if faces_.max() != 0:
                boxes_, scores_, kpts_, classes_ = self.detect(faces_, thermal=False, antispoof=False)
                # _, boxes_, scores_, kpts_, classes_ = STATST_
                image_disp_ = self.draw_detections(image_disp_, boxes_, scores_, kpts_, classes_, thermal=True,
                                                   antispoof_=True)

            faces_ = cv_render(faces_, resize=shape_, colormap='rainbow2', display=False,
                               interpolation=interpolation)

            return image_disp_, faces_, self.alignMat

        else:
            faces_ = np.zeros_like(image_disp_[0])
            if from_thermal_:
                return image_disp_[0], faces_, faces_

            else:
                image_disp_ = cv_render(image_disp_[1], resize=1, colormap='rainbow2', display=False,
                                        interpolation=interpolation)
                return image_disp_, faces_, faces_

    def scale_contour(self, cont, scale, cntr=None):
        """Resize contour to a given scale"""
        # bring contour to origin
        if cntr is None:
            # find centre from bounding box
            x, y, w, h = cv.boundingRect(cont)
            cntr = int(x + w / 2.), int(y + h / 2.)
        # bring to origin
        cont_at_origin = cont - cntr
        # scale
        scaled_cont = cont_at_origin * scale
        # add new centre
        scaled_cntr = int(cntr[0] * scale), int(cntr[1] * scale)
        scaled_cont += scaled_cntr
        return scaled_cont

    def get_and_draw_contours(self, thermal_data, img_scale=3, colormap='rainbow2', interpolation=cv.INTER_LINEAR,
                              use_median_on_tresh=False, median_size=5, use_adaptive_thresh=False, ath_blockSize=97,
                              ath_C=-29, _delta=10, minArea=-4, contour_line_width=2, color='white'):
        if color == 'white':
            color = [255] * 3
        elif color == 'green':
            color = [0, 255, 0]
        elif color == 'yellow':
            color = [0, 255, 255]
        elif color == 'red':
            color = [0, 0, 255]
        elif color == 'dark':
            color = [50] * 3
        elif color == 'dark-gray':
            color = [169] * 3
        elif color == 'vdark':
            color = [3, 3, 3]

        # CVFONT = cv.FONT_HERSHEY_SIMPLEX
        # CVFONT_SIZE = 0.7

        if use_adaptive_thresh:
            binary = cv.adaptiveThreshold(thermal_data, 1, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv.THRESH_BINARY, blockSize=ath_blockSize, C=ath_C)
        else:
            # Otsu
            th_otsu, binary = cv.threshold(thermal_data, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

            # find otsu's threshold
            th_otsu += _delta
            th_otsu, binary = cv.threshold(thermal_data, th_otsu, 1, cv.THRESH_BINARY)
            #        print('Otsu threshold {}'.format(th_otsu, _delta))

        if use_median_on_tresh:
            binary = cv.medianBlur(binary, ksize=median_size)

        # get the contours
        contours, hierarchy = cv.findContours(binary, cv.RETR_TREE,
                                               cv.CHAIN_APPROX_SIMPLE)

        outputs = []
        # first identify warm-on-cold contours that have at least
        # minArea number of pixels
        for i, c in enumerate(contours):
            # rough estimate of area with sign, to distinguish b/w
            # hot on cold (-ve) and cold on hot (+ve)
            area = cv.contourArea(c, oriented=True)
            # work only with hot on cold contours, assuming hot is foreground
            if minArea is None or area < minArea:
                # create a filled mask for the current contour
                mask = np.zeros(thermal_data.shape, dtype='uint8')
                cv.drawContours(mask, contours, i, 1, cv.FILLED)

                outputs.append((c, mask))

        thermal_data = cv_render(self.remap(thermal_data), resize=img_scale, colormap=colormap,
                                 display=False, interpolation=interpolation)

        # contours, masks = zip(*outputs)

        for i, contour_mask in enumerate(outputs):
            contour, mask = contour_mask[0], contour_mask[1]
            # print(i, len(contour), len(mask))
            cv.drawContours(thermal_data, [self.scale_contour(contour, scale=img_scale)], contourIdx=0,
                             color=color, thickness=contour_line_width)

        return thermal_data

    def putFPS(self, images, FPS=0, color='white', thickness=1, scale=1):
        if color == 'white':
            color = [255] * 3
        elif color == 'green':
            color = [0, 255, 0]
        elif color == 'yellow':
            color = [0, 255, 255]
        elif color == 'red':
            color = [0, 0, 255]
        elif color == 'dark':
            color = [50] * 3
        elif color == 'dark-gray':
            color = [169] * 3
        elif color == 'vdark':
            color = [3, 3, 3]

        return cv.putText(images, "FPS: %.2f" % (FPS), (25, 25), cv.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    def scale_detections(self, detections):
        boxes, scores, kpts, classes, scale = detections
        scaley, scalex = scale
        if kpts is None:
            kpts = [[0]*15] * len(boxes)
        scaled_boxes = []
        scaled_kpts = []
        for box, score, kp, cls in zip(boxes, scores, kpts, classes):
            x, y, w, h = box
            x, y, w, h = x * scalex, y * scaley, w * scalex, h * scaley
            scaled_boxes.append([x,y, w, h])

            for i in range(5):
                kp[i * 3] = kp[i * 3]*scalex
                kp[i * 3 + 1] = kp[i * 3 + 1]*scaley
            scaled_kpts.append(kp)

        return np.array(scaled_boxes), scores, np.array(scaled_kpts), classes, scale


if __name__ == '__main__':
    args = parser.parse_args()

    # This will enable mi48 logging debug messages
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

    # Make the a global variable and use it as an instance of the mi48.
    # This allows it to be used directly in a signal_handler.
    global mi48

    # define a signal handler to ensure clean closure upon CTRL+C
    # or kill from terminal
    def signal_handler(sig, frame):
        """Ensure clean exit in case of SIGINT or SIGTERM"""
        logger.info("Exiting due to SIGINT or SIGTERM")
        mi48.stop()
        cv.destroyAllWindows()
        logger.info("Done.")
        sys.exit(0)


    # Define the signals that should be handled to ensure clean exit
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # ==============================
    # create an USB interface object
    # ==============================
    # try:
    #     ser = get_serial()
    # except IndexError:
    #     # if on WSL-1 (hack); apply similarly to other cases where
    #     # device may not be readily found by get_serial
    #     try:
    #         ser = serial.Serial('/dev/ttyS4')
    #     except OSError:
    #         ser = serial.Serial('/dev/ttyS3')
    # usb = USB_Interface(ser)

    # logger.debug('Connected USB interface:')
    # logger.debug(usb)

    # # Make an instance of the MI48, attaching USB for
    # # both control and data interface.
    # mi48 = MI48([usb, usb])

    # # Using the connect_senxor()
    # can try connect_senxor(src='/dev/ttyS3') or similar if default cannot be found
    mi48, connected_port, port_names = connect_senxor()
    ncols, nrows = mi48.fpa_shape

    # print out camera info
    camera_info = mi48.get_camera_info()
    logger.info('Camera info:')
    logger.info(camera_info)

    # set desired FPS
    if len(sys.argv) == 2:
        STREAM_FPS = int(sys.argv[1])
    else:
        STREAM_FPS = args.FPS_Divisor
    # mi48.set_fps(STREAM_FPS)

    # ensure we use max fps, regardless of the internal processing within the MI48
    mi48.set_frame_rate(STREAM_FPS)

    # see if filtering is available in MI48 and set it up
    # kalman_Strength = args.firmware_Kalman_strength
    # mi48.disable_filter(f1=True, f2=True, f3=True)
    # mi48.enable_filter(f1=args.firmware_Kalman, f2=args.use_STARK, f3=False, f3_ks_5=args.STARK_cuttoff)
    # mi48.set_filter_1(kalman_Strength)
    # mi48.set_filter_2(args.STARK_gradient)

    if ncols != 160:
        val = 0x00
        if args.use_STARK:
            val += 0x04
        if args.firmware_Kalman:
            val += 0x03
        if args.use_firmware_median:
            val += 0x40
        mi48.regwrite(0xD0, val)
        mi48.regwrite(0xD1, args.firmware_Kalman_strength)
        mi48.regwrite(0x20, 0x01) # use stark
        mi48.regwrite(0x25, 0x00) # no MMS
        # mi48.regwrite(0xD3, args.STARK_gradient)
        # mi48.regwrite(0xD9, args.STARK_cuttoff)

    else:
        val = 0x00
        if args.use_STARK:
            val += 0x04
        if args.firmware_Kalman:
            val += 0x03
        if args.use_firmware_median:
            val += 0x40
        mi48.regwrite(0xD0, 0x00)
        mi48.regwrite(0xD1, 0x00)
        mi48.regwrite(0x20, 0x00)
        mi48.regwrite(0x30, 0x00)
        mi48.regwrite(0x25, 0x00)   # disable MMS

    # Ensure sensitivity enhancement factor is 1. (e.g. sensitivity is as obtained during calibration)
    # mi48.set_distance_corr(0x64)

    # Assume radiometric blackbody, e.g. PCN-7, i.e. emissivity = 1.0 (100%)
    # mi48.set_emissivity(100)

    # initiate continuous frame acquisition
    with_header = True
    mi48.start(stream=True, with_header=with_header)

    # # change this to false if not interested in the image
    # GUI = True
    data, header = mi48.read()
    if data is None:
        logger.critical('NONE data received instead of GFRA')
        mi48.stop()
        sys.exit(1)
    data = data_to_frame(data, (ncols, nrows), hflip=False)[:,:-40]

    # Initialize YOLOv7_face object detector
    YOLOv7_face_detector = YOLOv7_face(os.path.join('.\model_stack',args.modelpath), conf_thres=args.confThreshold, iou_thres=args.nmsThreshold,
                                       which_run=args.which_run, num_classes=args.num_classes, use_kpts=args.use_kpts,
                                       shape=args.input_shape)
    print(f"which_run param: {YOLOv7_face_detector.which_run}")
    print(f"YOLOv7_face_detector.input_shape: {YOLOv7_face_detector.input_shape}")
    # exit(0)

    # change filter below
    if args.use_python_filter:
        # loading cnn model
        pb_model = None
        if args.Kalman_which_predict == 'CNN':
            pb_model_name = os.path.join(args.model_dir, args.model_name)
            print(f"loading 2 {pb_model_name} a frozen graph")
            pb_model = cv.dnn.readNetFromTensorflow(pb_model_name)
            print('Finished loading trained model: {}'.format(pb_model_name))

        Kalman_frame = Kalman_with_predict(frame0=data, update_which='l1', r_depth=args.depth,
                                           which_predict=args.Kalman_which_predict,
                                           cnn_model=pb_model, err=args.pixel_err_std_Kalman_python,
                                           smooth_new_frame=args.smooth_new_frame_Kalman_python,
                                           out_smooth=args.out_smooth,
                                           gain_scale=args.scale_Kalman_python, gain_which='stair-case',
                                           sigma=args.sigma_Kalman_python, ksize=3,
                                           use_normalization=args.use_normalization)

        # Kalman_frame = STRAFilter_original_v1(alpha=2.0, beta=2.0, lm_ks=(5, 5), lm_ad=12, lm_atype='ra',
        #                                       gain_type='stair-case', offset=0.1, scale=0.9, cutoff=2.5, top=1.0)

    # for clipping
    RA_min = RollingAverageFilter(15)
    RA_max = RollingAverageFilter(8)

    # for alignment
    alignMat = None

    cam = cv.VideoCapture(0)
    while True:
        start = time.time()

        # read rgb
        _, srcimg = cam.read()
        if srcimg is None:
            print("no rgb cam input")
            exit(0)

        # read Themal
        data, header = mi48.read()
        Tframe = data_to_frame(data, (ncols, nrows), hflip=False)[:,:-40]
        # process Thermal
        sorted_data = np.sort(data)
        Tmin, Tmax = RA_min(np.median(sorted_data[:16])), RA_max(np.median(sorted_data[-5:]))

        clipped = Tframe
        if args.use_python_filter:
            clipped = Kalman_frame(Tframe)

        # process Thermal
        clipped = YOLOv7_face_detector.process_thermal(clipped, clip_min=args.clip_min_percent, Tmin=Tmin,
                                                       Tmax=Tmax, clip_human=args.clip_human,
                                                       clip_max=args.clip_max_percent, scale=args.disp_scale,
                                                       which_process=args.which_process, blur=args.blur,
                                                       use_input_ratio=args.use_input_ratio,
                                                       scale_before=args.scale_before)
        print(f"clipped shape: {clipped.shape}")
        # Detect Objects thermal
        shape = (clipped.shape[1], clipped.shape[0])
        boxes, scores, kpts, classes = YOLOv7_face_detector.detect(clipped, thermal=True)
        # print(boxes, kpts)

        if not args.scale_before:
            # clipped = cv_render(clipped, resize=args.disp_scale, colormap='rainbow2',
            #                 display=False, interpolation=args.interpolation)

            clipped = cv.resize(clipped, (clipped.shape[1]*args.disp_scale,
                                           clipped.shape[0]*args.disp_scale),
                                 interpolation=args.interpolation)

            new_shape = (clipped.shape[1], clipped.shape[0])
            # scale_ratio = (new_shape[0]/shape[0], new_shape[1]/shape[1])
            scale_ratio = (args.disp_scale, args.disp_scale)
            boxes, scores, kpts, classes, scale = YOLOv7_face_detector.scale_detections((boxes, scores,
                                                                                         kpts, classes, scale_ratio))
            shape = new_shape
            # print(boxes, kpts)
            # exit(0)

        Tframe = clipped.copy()
        # frame = clipped.copy()

        # Draw detections thermal
        clipped = cv_render(clipped, resize=1, colormap='rainbow2',
                            display=False, interpolation=args.interpolation)

        clipped = YOLOv7_face_detector.draw_detections(clipped, boxes, scores, kpts, classes, thermal=True,
                                                       use_std_mean=args.use_std_mean, use_min_max=args.use_std_mean)

        # stats thermal for warping
        STATST = (clipped, boxes, scores, kpts, classes)

        # Detect Objects rgb
        srcimg = np.array([array[::-1] for array in srcimg])
        dstimg = srcimg.copy()
        # dstimg = YOLOv7_face_detector.process_rgb(dstimg)
        boxes, scores, kpts, classes = YOLOv7_face_detector.detect(dstimg, thermal=False)

        # Draw detections rgb
        scale_T2rgb = (shape[1]/srcimg.shape[0], shape[0]/srcimg.shape[1])
        dstimg = YOLOv7_face_detector.draw_detections(dstimg, boxes, scores, kpts, classes, thermal=False,
                                                      scale=scale_T2rgb) #, hoffset=20, xoffset=10)

        dstimg = cv.resize(dstimg, shape)

        # stats RGB for warping
        resize_srcimg = cv.resize(srcimg, shape)
        STATSRGB = (resize_srcimg, boxes, scores, kpts, classes, scale_T2rgb)

        # # antispoof:
        antispoof_dip = np.zeros_like(dstimg)
        facesExtracted = np.zeros_like(antispoof_dip)
        if args.antispoof:
            antispoof_dip, facesExtracted, _ = YOLOv7_face_detector.antispoof([resize_srcimg, Tframe],
                                                                              continuos_align_=args.continous_align,
                                                                              STATST_=STATST, STATSRGB_=STATSRGB,
                                                                              planar_=False, use_both=args.use_both,
                                                                              from_thermal_=args.from_thermal)

        disp = np.hstack([antispoof_dip, facesExtracted])

        # # warped
        if args.allign:
            warped = np.zeros_like(dstimg)
            notoverlayed = np.zeros_like(warped)
            thresholded = np.zeros_like(warped)
            objects = np.zeros_like(warped)
            edges = np.zeros_like(warped)
            enhanced = np.zeros_like(warped)
            if args.continous_align or alignMat is None:
                if len(STATSRGB[1]) and len(STATST[1]):
                    alignMat = YOLOv7_face_detector.allign_compute(STATST, STATSRGB, planar=False)
            if alignMat is not None:
                notoverlayed, Twarped, warped = YOLOv7_face_detector.allign_draw(STATST[0], STATSRGB[0], alignMat,
                                                                                 planar=False, Threshold=(29, 40),
                                                                                 showwarped=args.showrawwarped)
                if not args.showrawwarped:
                    warped = Twarped

                thresholded, objects = YOLOv7_face_detector.get_objects(STATSRGB[0])
                edges, img_disp = YOLOv7_face_detector.enhnace_thermal(STATST[0], objects)  # working as well

            mid1 = np.hstack([warped, notoverlayed])
            mid2 = np.hstack([objects, thresholded])
            bottom = np.hstack([enhanced, edges]) # working as well
            disp = np.vstack([mid1, mid2, bottom])

        # for display
        # print(dstimg.shape, clipped.shape, disp.shape)
        top = np.hstack([dstimg, clipped])
        disp = np.vstack([top, disp])

        winName = 'Deep learning object detection in ONNXRuntime'
        # print(dstimg.shape)

        if args.put_FPS:
            # calculate FPS
            end = time.time()
            totalTime = end - start
            fps = 1 / totalTime
            # print("FPS: ", fps)
            YOLOv7_face_detector.putFPS(disp, FPS=fps)

        # print(disp.shape)
        # cv.namedWindow(winName, 0)
        cv.imshow(winName, disp)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
    cam.release()
    cv.destroyAllWindows()