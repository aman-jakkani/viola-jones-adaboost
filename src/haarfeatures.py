import numpy as np 
from enum import Enum 
from src.integralimage import IntegralImage as II
from src.integralimage import get_sum

class feat_type(Enum):
    #.value will access tuple
    TWO_VERTICAL = (1, 2)
    TWO_HORIZONTAL = (2, 1)
    THREE_VERTICAL = (1, 3)
    THREE_HORIZONTAL = (3, 1)
    FOUR = (2, 2)

class HaarLikeFeature(object):
    #selecting haar features
    # h_j(x) = 1 if p_j*f_j(x) < p_j*threshold
    # h_j(x) = 0 otherwise
    def __init__(self, feat_type, pos, width, height, threshold, parity, weight=1):
        self.type = feature_type
        self.top_left = pos
        self.bottom_right = (pos[0]+width, pos[1]+height)
        self.width = width
        self.height = height
        self.threshold = threshold
        self.parity = parity
        self.weight = weight

    def calc_score(self, integ_img):
        score, white, grey = 0, 0, 0
        if self.type == feat_type.TWO_VERTICAL:
            white += get_sum(integ_img, self.top_left, 
                (int(self.top_left[0] + self.width), int(self.top_left[1] + self.height / 2)))
            grey += get_sum(integ_img, (self.top_left[0], 
                int(self.top_left[1] + self.height / 2)), self.bottom_right)

        elif self.type == feat_type.TWO_HORIZONTAL:
            white += get_sum(integ_img, self.top_left,
                (int(self.top_left[0] + self.width/2), self.top_left[1] + self.height))
            grey += get_sum(integ_img,
                (int(self.top_left[0] + self.width/2), self.top_left[1]), self.bottom_right)
            
        elif self.type == feat_type.THREE_VERTICAL:
            white += get_sum(integ_img, self.top_left,
                (self.bottom_right[0], int(self.top_left[1] + self.height / 3)))
            grey += get_sum(integ_img, (self.top_left[0], int(self.top_left[1] + self.height / 3)), 
                (self.bottom_right[0], int(self.top_left[1] + 2 * self.height / 3)))
            white += get_sum(integ_img, (self.top_left[0],
                int(self.top_left[1] + 2 * self.height / 3)), self.bottom_right)

        elif self.type == feat_type.THREE_HORIZONTAL:
            white += get_sum(integ_img, self.top_left,
                (self.bottom_right[0], int(self.top_left[1] + self.height / 3)))
            grey += get_sum(integ_img, (self.top_left[0], int(self.top_left[1] + self.height / 3)),
                (self.bottom_right[0], int(self.top_left[1] + 2 * self.height / 3)))
            white += get_sum(integ_img, (self.top_left[0], int(
                self.top_left[1] + 2 * self.height / 3)), self.bottom_right)

        elif self.type == feat_type.FOUR:
            white += get_sum(integ_img, self.top_left,
                (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)))
            grey += get_sum(integ_img, (int(self.top_left[0] + self.width / 2), self.top_left[1]),
                (self.bottom_right[0], int(self.top_left[1] + self.height / 2)))
            grey += get_sum(integ_img, (self.top_left[0], int(self.top_left[1] + self.height / 2)),
                (int(self.top_left[0] + self.width / 2), self.bottom_right[1]))
            white += get_sum(integ_img, (int(self.top_left[0] + self.width / 2),
                int(self.top_left[1] + self.height / 2)), self.bottom_right)
            
        score = white - grey
        return score

    def get_vote(self, int_img):
        # get prediction of feature given integral image
        score = self.calc_score(int_img)
        return self.weight * (1 if score < self.parity * self.threshold else 0)
