import numpy as np
import cv2
from senxor.utils import remap

class Kalman():
    """ Kalman implementation. Requires an initial frame i.e., frame0 must be pased as input
        For better results, frame0 can be an average of 2 successive frames or more.
        Else a very random interger is initialized as initial temparature reading.
        Performance is better when used after filter.
        mea_err: the error measured between new frame and previous estimated frame
        err_est: the mean estimated error. can be initialized randomly or using calibration data.
                 will decay over time.
        process_error: the mean of the std per pixel matrix obtained from calibration or a fixed
                       std for all pixels. If std for all pixels is used, then use the
                       mean gain rather than the gain per pixel matrix.
        update_which: Which loss to use to compute the mean_err (mean erro) t iteration t.
                      l1 and l2 are both implemented.
    """

    def __init__(self, frame0=None, err_est=10, process_err=10, update_which='l1',
                 sigmoid=False, stabilize=False, r_depth=10, gain_scale=0.25, pixel_err_std=0.25):

        # use l1 or l2 loss
        self.update_which = update_which

        # initial frame keeping or initialize complete random constant integer between 0 and 37 as initial value
        # and rely on python broadcasting to treat it as matrix for gain update
        if frame0 is None:
            self.est = np.ones((120, 120))
            self.min_temp = 16
            self.max_temp = 34
        else:
            self.est = frame0
            self.min_temp = self.est.min()
            self.max_temp = self.est.max()

        # initialize process error randomly
        self.err_est = err_est
        self.init_err = err_est

        # initialize process error
        self.process_err = process_err

        # if all models
        self.sigmoid = sigmoid
        self.gain_scale = gain_scale
        self.pixel_err_std = pixel_err_std

        # if stabilize
        self.r_depth = r_depth
        self.stabilize = stabilize

    def predict(self, controls, new_frame):
        # Estimate frame update from previous frame and control params
        # predict Estimated noise
        pass

    def update(self, new_frame):

        # if stabilize
        if self.stabilize:
            self.min_temp += 1. / self.r_depth * (self.est.min() - self.min_temp)
            self.max_temp += 1. / self.r_depth * (self.est.max() - self.max_temp)
            self.est = self.est.clip(self.min_temp, self.max_temp)
            new_frame = new_frame.clip(self.min_temp, self.max_temp)

        # calculate measured error from previous frame and new frame
        if self.update_which == 'l1':
            mea_err = np.abs(new_frame - self.est)
        elif self.update_which == 'l2':
            mea_err = (new_frame - self.est)**2

        if self.sigmoid:
            self.gain = self.gain_scale * (
                    1 + ((mea_err - self.pixel_err_std) / np.sqrt((1 + (mea_err - self.pixel_err_std) ** 2))))
            # calculate new estimate
            self.est = self.est + self.gain * (new_frame - self.est)
            # print(self.gain, mea_err.mean(), self.err_est, self.est.mean(), new_frame.mean())

        else:
            # get kalman gain. i.e., a parameter to determine
            # the percentage of the new pixel to keep and th old pixel to remove

            self.gain = (self.err_est) / (self.err_est + self.process_err + 1e-6)

            # calculate new estimate
            self.est = self.est + self.gain * (new_frame - self.est)
            # print(self.gain, mea_err.mean(), self.err_est, self.est.mean(), new_frame.mean())

            # update estimated error
            self.err_est = (1-self.gain)*self.err_est + self.err_est*self.init_err
            # print(self.gain.max(), self.gain.min(), self.gain.mean())
            # # exit(0)

        # print(self.est.min(), self.est.max())
        return self.est

    def __call__(self, new_frame, *args, **kwargs):
        return self.update(new_frame)


class Kalman_with_predict():
    """ Kalman implementation. Requires an initial frame i.e., frame0 must be pased as input
        For better results, frame0 can be an average of 2 successive frames or more.
        Else a very random interger is initialized as initial temparature reading.
        Performance is better when used after filter.
        mea_err: the error measured between new frame and previous estimated frame
        err_est: the mean estimated error. can be initialized randomly or using calibration data.
                 will decay over time.
        process_error: the mean of the std per pixel matrix obtained from calibration or a fixed
                       std for all pixels. If std for all pixels is used, then use the
                       mean gain rather than the gain per pixel matrix.
        update_which: Which loss to use to compute the mean_err (mean erro) t iteration t.
                      l1 and l2 are both implemented.
        see: https://github.com/Ugenteraan/Kalman-Filter-Scratch/blob/master/Kalman-Scratch-Implementation.ipynb
    """

    def __init__(self, frame0=None, update_which='l1', smooth_new_frame=False, which_predict='Median', ksize=3,
                 sigma=1, cnn_model=None, use_normalization=1, r_depth=4, err=1.5, gain_scale=0.35,
                 gain_which='stair-case', out_smooth=False, offset=0.1, scale=0.9):

        # should the new input be smooth before Kalman?
        self.smooth_new_frame = smooth_new_frame

        # use l1 or l2 loss
        self.update_which = update_which

        # initial frame keeping or initialize complete random constant integer between 0 and 37 as initial value
        # and rely on python broadcasting to treat it as matrix for gain update
        if frame0 is None:
            self.est = np.random.randint(low=0, high=37)
            self.min_temp = 16
            self.max_temp = 34
        else:
            self.est = frame0
            self.min_temp = frame0.min()
            self.max_temp = frame0.max()

        # initialize method of update for gain
        self.gain_which = gain_which
        if self.gain_which == 'stair-case':
            self.update_which = 'l1'

        # smooth output or raw improved output
        self.out_smooth = out_smooth

        # initialize rolling average depth
        self.r_depth = r_depth

        # initialize pixel err std for sigmoid
        self.pixel_err_std = err

        # initialize gain scale
        self.gain_scale = gain_scale

        #initialize counter
        self.count = 0

        # for poly
        self.offset = offset
        self.scale = scale

        # initialize prediction parameters
        assert which_predict in ['Median', 'Box', 'Gaussian', 'CNN', None]
        self.which_predict = which_predict
        self.ksize = ksize
        self.sigma = sigma

        if self.which_predict == 'CNN':
            self.model = cnn_model
            self.use_normalization = use_normalization

    def to_tensor(self, img):
        if img.ndim == 2:
            return img[np.newaxis, ..., np.newaxis]

    def from_tensor(self, img):
        return np.squeeze(img)

    def cnn_filter(self, datas):
        """
        Run the normalised data through the noise cancellation CNN filter
        """
        # if use_normalization keep range for norm/renorm
        # model works with [0,1]
        try:
            # try a Keras model
            x = self.to_tensor(datas)
            y = self.model.predict(x)
            output = self.from_tensor(y)
        except AttributeError:
            # assume it is the cv.dnn_Net object
            x = cv2.dnn.blobFromImage(datas.astype(np.float32), scalefactor=1.0,
                                     size=(datas.shape[1], datas.shape[0]), mean=0, swapRB=False)
            # model.setInput(x, name='input')
            self.model.setInput(x)
            # y = model.forward(outputName='subtract_1/sub')
            y = self.model.forward()
            output = np.squeeze(y, axis=(0, 1))
        return output

    def run_cnn(self, frames, min_temp, max_temp):
        '''plural names used for dinstingtion, are singular'''
        if self.use_normalization == 1:
            frames = remap(frames, curr_range=(min_temp, max_temp),
                           new_range=(0, 1), to_uint8=False)
            frames = self.cnn_filter(frames)
            frames = remap(frames, curr_range=(frames.min(), frames.max()),
                           new_range=(min_temp, max_temp), to_uint8=False)

        elif self.use_normalization == 2:
            frames = remap(frames, curr_range=(min_temp, max_temp),
                           new_range=(-1, 1), to_uint8=False)
            frames = self.cnn_filter(frames)
            frames = remap(frames, curr_range=(frames.min(), frames.max()),
                           new_range=(min_temp, max_temp), to_uint8=False)

        else:
            frames = self.cnn_filter(frames)
            frames = remap(frames, curr_range=(frames.min(), frames.max()),
                           new_range=(min_temp, max_temp), to_uint8=False)

        return frames

    def remap_tensor(self, data, new_range=(0, 1), axis_totake=(1, 2)):
        lo2, hi2 = new_range

        hi = np.max(data, axis=axis_totake, keepdims=True)[0]
        lo = np.min(data, axis=axis_totake, keepdims=True)[0]
        data -= lo
        data /= (hi - lo)
        data = lo2 + data * (hi2 - lo2)

        return data

    def predict(self):

        # Estimate frame update from previous frame and control params

        if self.which_predict == 'Median':
            self.est = remap(cv2.medianBlur(remap(self.est, curr_range=(self.min_temp, self.max_temp)),
                                            ksize=self.ksize),
                             new_range=(self.min_temp, self.max_temp), to_uint8=False)
        elif self.which_predict == 'Gaussian':
            self.est = remap(cv2.GaussianBlur(remap(self.est, curr_range=(self.min_temp, self.max_temp)),
                                              (self.ksize, self.ksize), sigmaX=self.sigma),
                             new_range=(self.min_temp, self.max_temp), to_uint8=False)
        elif self.which_predict == 'Box':
            self.est = remap(cv2.blur(remap(self.est, curr_range=(self.min_temp, self.max_temp)),
                                      (self.ksize, self.ksize)),
                             new_range=(self.min_temp, self.max_temp), to_uint8=False)

        elif self.which_predict == 'CNN':
            self.est = self.run_cnn(self.est, self.min_temp, self.max_temp)

    def predict_new_frame(self, new_frame):
        # Estimate frame update from previous frame and control params
        max_temp = self.max_temp
        min_temp = self.min_temp

        if self.which_predict == 'Median':
            new_frame = remap(cv2.medianBlur(remap(new_frame, curr_range=(min_temp, max_temp)),
                                             ksize=self.ksize))
            new_frame = remap(new_frame, curr_range=(new_frame.min(), new_frame.max()),
                           new_range=(min_temp, max_temp), to_uint8=False)
        elif self.which_predict == 'Gaussian':
            new_frame = remap(cv2.GaussianBlur(remap(new_frame, curr_range=(min_temp, max_temp)),
                                              (self.ksize, self.ksize), sigmaX=self.sigma))
            new_frame = remap(new_frame, curr_range=(new_frame.min(), new_frame.max()),
                              new_range=(min_temp, max_temp), to_uint8=False)
        elif self.which_predict == 'Box':
            new_frame = remap(cv2.blur(remap(new_frame, curr_range=(min_temp, max_temp)),
                                              (self.ksize, self.ksize)))
            new_frame = remap(new_frame, curr_range=(new_frame.min(), new_frame.max()),
                              new_range=(min_temp, max_temp), to_uint8=False)

        elif self.which_predict == 'CNN':
            new_frame = self.run_cnn(new_frame, min_temp, max_temp)

        return new_frame

    def update(self, new_frame):

        # # update roll average
        # self.min_temp += 1. / self.r_depth * (new_frame.min() - self.min_temp)
        # self.max_temp += 1. / self.r_depth * (new_frame.max() - self.max_temp)

        # no roll avg on new frame
        self.min_temp = new_frame.min()
        self.max_temp = new_frame.max()

        # # uptate new and estimate temp from roll avg
        # self.est.clip(self.min_temp, self.max_temp)
        # new_frame.clip(self.min_temp, self.max_temp)
        # print('a', new_frame.min(), new_frame.max(), self.min_temp, self.max_temp, self.est.min(), self.est.max())

        # predict current frame and process noise
        if self.count ==0:
            self.predict()
            self.count += 1
        if self.smooth_new_frame:
            new_frame_smooth = self.predict_new_frame(new_frame=new_frame)

        # calculate new actual error and new frame
        if self.update_which == 'l1':
            mea_err = np.abs(new_frame_smooth - self.est)
        elif self.update_which == 'l2':
            mea_err = (new_frame_smooth - self.est)**2

        if self.gain_which == 'sigmoid-approx':
            self.gain = self.gain_scale * (
                        1 + ((mea_err - self.pixel_err_std) / np.sqrt((1 + (mea_err - self.pixel_err_std) ** 2))))

        elif self.gain_which == 'normalizing':
            mea_err = mea_err + self.gain_scale
            self.gain = mea_err/(mea_err + self.pixel_err_std)

        elif self.gain_which == 'stair-case':
            # self.gain = 1/(1 + np.exp(-mea_err))
            self.gain = np.full(mea_err.shape, 0.8, dtype=float)
            self.gain[(mea_err <=2.0)] = 0.01
            self.gain[(mea_err > 2.0) & (mea_err <= 3.0)] = 0.1
            self.gain[(mea_err > 3.0) & (mea_err <= 5.0)] = 0.2
            self.gain[(mea_err > 5.0) & (mea_err <= 6.0)] = 0.3
            self.gain[(mea_err > 6.0) & (mea_err <= 10.)] = 0.5
            # self.gain[(mea_err > 3.0) & (mea_err <= 8.0)] = 0.7
            # # # self.gain[(mea_err > 5.0) & (mea_err <= 6.0)] = 0.5q
            # # # self.gain[(mea_err > 6.0) & (mea_err <=7.0)] = 0.6
            # # # self.gain[(mea_err > 7.0) & (mea_err <= 8.0)] = 0.7
            # # # self.gain[(mea_err > 8.0) & (mea_err <= 9.0)] = 0.8

        elif self.gain_which == 'poly':
            # self.gain = np.zeros_like(mea_err)
            self.gain = np.full((mea_err.shape), self.offset*self.scale)
            mea_err = (mea_err - self.pixel_err_std)*self.gain_scale
            self.gain[(mea_err > 0) & (mea_err <=1)] = (3*mea_err[(mea_err > 0) & (mea_err <=1)]**2
                                        - 2*mea_err[(mea_err > 0) & (mea_err <=1)]**3
                                        + self.offset)*self.scale
            self.gain[(mea_err>1)] = (1+self.offset)*self.scale

        # print(self.gain.max(), self.gain.min())
        # print(mea_err.max(), mea_err.min(), np.median(mea_err), mea_err.mean())

        # calculate new estimate
        if self.out_smooth:
            self.est = self.est + self.gain * (new_frame_smooth - self.est)
        else:
            self.est = self.est + self.gain * (new_frame - self.est)

        # print('b', new_frame.min(), new_frame.max(), self.min_temp, self.max_temp, self.est.min(), self.est.max())
        return self.est

    def __call__(self, new_frame, *args, **kwargs):
        return self.update(new_frame)
