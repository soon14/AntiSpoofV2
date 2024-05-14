import numpy as np
import cv2 as cv
from senxor.utils import remap, RollingAverageFilter, TrueAverageFilter
from functools import partial

def sigmoid(x, alpha, beta):
    return 1. / (1 + np.exp(-alpha * (x - beta)))

def smoothstep(x, alpha, beta, delta, quad, cube):
    _x = -alpha * (x - beta)
    x2 = _x * _x
    x3 = x2 * _x
    res = quad * x2 + cube * x3 + delta
    res[res > 0.85] = 0.85
    return res

class STARKFilter:
    def __init__(self, param):
        """
        Spatio-Temporal Advanced Rolling Kernel Filter
        Usage:
            frame_filter = STARKFilter()
            ...
            filtered = frame_filter(input_data)
        """
        print('STARK parameters: ', param)
        # local mean can have different kernel size, e.g. 3x3 or 5x5 pixels
        lm_ks = param['lm_ks']
        # local mean can have a temporal average: true or rolling
        lm_atype = param.get('lm_atype', 'ra')
        # local mean temporal average can have various depth 
        lm_ad = param['lm_ad']
        # describe the sigmoid which controls the gain based on the difference
        # between a historical value (of or around) a pixel, and the
        # instantaneous local mean at the pixel
        # alpha and beta and delta are response acceleration, noise barrier
        # and minimum fraction of frame update
        if param.get('sigmoid_type', 'sigmoid') == 'sigmoid':
            alpha = param['alpha']
            beta  = param['beta']
            self.sigmoid = partial(sigmoid, alpha=alpha, beta=beta)
        if param['sigmoid'] in ['smoothstep', 'cube']:
            alpha = param['alpha']
            beta  = param['beta']
            delta = param['delta']
            quad  = param['quad']
            cube  = param['cube']
            self.sigmoid = partial(smoothstep, alpha=alpha, beta=beta, delta=delta,
                                  quad=quad, cube=cube)
        # local mean kernel size
        self.lm_ks = lm_ks
        # temporal average filter on the local mean
        # if we have a temporal average of the local mean, then we use this
        # against the instantaneous mean to predict motion
        if lm_ad > 0:
            if lm_atype == 'ra':
                self.lm_av = RollingAverageFilter(lm_ad)
            if lm_atype == 'ta':
                self.lm_av = TrueAverageFilter(lm_ad)
            self.get_diff = self.diff_lmav
        else:
            self.get_diff = self.diff_self
        # STARK filter output. Init to 0 int, so as to avoid figuring out the shape
        self.av = 0

    def diff_self(self, lm):
        return lm - self.av

    def diff_lmav(self, lm):
        """
        Compute the difference between instantaneous local mean and the historical
        average of the local mean. Then update the historical average of the local mean.
        """
        diff = lm - self.lm_av.av
        return diff

    def update(self, new):
        minmax = new.min(), new.max()
        # the following allows `new` to be 3-channel image (cv.MAT of uint8)
        # such a matrix needs no remapping and thus STARK can be applied to
        # visual camera input as well
        if len(new.shape) == 2:
            new_u8 = remap(new)
        else:
            new_u8 = new
        # smooth input and convert it back to temperature:
        # we must explicitly specify both current and new range
        # we can work with uint for new_lm and lm, but it becomes more difficult to
        # set beta correctly
        new_lm = cv.blur(new_u8, self.lm_ks)
        # remap from uint to temp only if one-channel 3D
        if len(new.shape) == 2:
            new_lm = remap(new_lm, curr_range=(0,255), new_range=minmax, to_uint8=False)
        self.x = self.get_diff(new_lm)
        self.gamma = self.sigmoid(np.abs(self.x))
        gamma_p = smoothstep(np.abs(self.x), 0.3, 0.0, 0.02, 0.8, -0.9 )
        gamma_s = sigmoid(np.abs(self.x), 2.0, 2.0)
        try:
            print(f'{new[31,40]:5.2f}, {new_lm[31,40]:5.2f}, {self.av[31,40]:5.2f}',
                  f'{self.x[31,40]:5.2f}, {self.gamma[31,40]:5.2f},'
                  f' {gamma_p[31,40]:5.2f}, {gamma_s[31,40]:5.2f}')
        except TypeError:
            # at first frame, self.av is just an int = 0
            print(f'{new[31,40]:5.2f}, {new_lm[31,40]:5.2f}, {self.av:5.2f}',
                  f'{self.x[31,40]:5.2f}, {self.gamma[31,40]:5.2f},'
                  f' {gamma_p[31,40]:5.2f}, {gamma_s[31,40]:5.2f}')
        self.av += self.gamma * (new - self.av)
        # try to cure the noisy pixels by replacing them with the blurred
        # self.av[self.gamma > 0.9] = 0.5 * (self.av[self.gamma > 0.9] +\
        #                                    new_lm[self.gamma > 0.9])
        self.new_lm = new_lm
        # update the local mean average last
        if self.get_diff == self.diff_lmav: self.lm_av(new_lm)

    def _quick_update(self, new):
        minmax = new.min(), new.max()
        # the following allows `new` to be 3-channel image (cv.MAT of uint8)
        # such a matrix needs no remapping and thus STARK can be applied to
        # visual camera input as well
        if len(new.shape) == 2:
            new_u8 = remap(new)
        else:
            new_u8 = new
        # smooth input and convert it back to temperature:
        # we must explicitly specify both current and new range
        # we can work with uint for new_lm and lm, but it becomes more difficult to
        # set beta correctly
        new_lm = cv.blur(new_u8, self.lm_ks)
        # remap from uint to temp only if one-channel 3D
        if len(new.shape) == 2:
            new_lm = remap(new_lm, curr_range=(0,255), new_range=minmax, to_uint8=False)
        self.x = new_lm - self.av
        # sigmoid on the difference
        self.gamma = self.sigmoid(self.x)
        gamma_p = smoothstep(self.x, 0.3, 0.0, 0.02, 0.8, -0.9)
        gamma_s = sigmoid(self.x, 2.0, 2.0)
        print(f'{np.abs(self.x[31,40]):5.2f}, {self.gamma[31,40]:5.2f}, {gamma_p[31,40]:5.2f}, {gamma_s[31,40]:5.2f}')
        self.av += self.gamma * (new - self.av)

    def __call__(self, new):
        self.update(new)
        return self.av