import math
import numpy as np

class IntegralImage(object):

    #initialization of class variables
    def __init__(self, img):
        self.shape = (img.shape[0] + 1, img.shape[1] + 1)
        self.img = img
        self.img_sq = img * img
        # integral image to be calculated
        self.integ_img = np.ones(self.shape)
        self.integ_img_sq = np.ones(self.shape) #normalized
        # memo that indicates if this position is already calculated
        self.memo = np.zeros(self.shape)
        self.variance = 0.0
        self.get()
        self.set_variance() #setting variance function

    def get_integral_image(self):
        return self.integ_img, self.variance

    # Calculate value of each pixel
    def calc(self, x, y, sq=False):
        if x == 0 or y == 0:
            return 0
        # if already calc, return value
        if self.memo[x][y] == 1:
            if not sq:
                return self.integ_img[x][y]
            else:
                return self.integ_img_sq[x][y]
        else:
            cummulative = self.calc(x-1, y, sq) + self.calc(x, y-1, sq) - self.calc(x-1, y-1, sq)
            if not sq:
                cummulative += self.img[x-1][y-1]
            else:
                cummulative += self.img_sq[x-1][y-1]
            self.memo[x][y] = 1
            return cummulative

    def get(self):
        # Get the integral image with additional rows/cols of 0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.int_img[i][j] = self.calc(i, j)
        # Get the squared integral image with additional rows/cols of 0
        self.memo = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.integ_img_sq[i][j] = self.calc(i, j, sq=True)

    def set_variance(self):
        N = (self.shape[0] - 1) * (self.shape[1] - 1) # number of pixels
        m = self.int_img[-1][-1] / N # mean
        sum_sq = self.integ_img_sq[-1][-1] # sum of x^2
        self.variance = (sum_sq / N) - math.pow(m, 2)

# get summed value over a rectangle
def get_sum(integ_img, top_left, bottom_right):
    top_left = (top_left[1], top_left[0])
    bottom_right = (bottom_right[1], bottom_right[0])
    # must swap the tuples since the orientation of the coordinate system is different
    if top_left == bottom_right:
        return integ_img[top_left]
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])
    
    return integ_img[bottom_right] + integ_img[top_left] - integ_img[bottom_left] - integ_img[top_right]