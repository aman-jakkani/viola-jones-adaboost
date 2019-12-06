import numpy as np
import os

from src.integralimage import IntegralImage as II
from src.haarfeatures import HaarLikeFeature as haar
from src.haarfeatures import feat_type
#from src.utils import *

def _create_features(img_width, img_height, min_feat_width, max_feat_width, min_feat_height, max_feat_height):
    # function to create all possible features, returned as list
    haar_feats = list()
    # iterate according to types of rectangle features
    for each_feat in feat_type:
        # min of feature width is set in feat_type enum
        feat_start_width = max(min_feat_width, each_feat.value[0])
        for feat_width in range(feat_start_width, max_feat_width, each_feat.value[0]):
            # min of feature height is set in feat_type enum
            feat_start_height = max(min_feat_height, each_feat.value[1])
            for feat_height in range(feat_start_height, max_feat_height, each_feat.value[1]):
                # scan the whole image with sliding windows
                for i in range(img_width - feat_width):
                    for j in range(img_height - feat_height):
                        haar_feats.append(haar(each_feat, (i,j), feat_width, feat_height, 0, 1)) 
                        haar_feats.append(haar(each_feat, (i,j), feat_width, feat_height, 0, -1)) 
    return haar_feats

def _get_feature_vote(feature, img):
    return feature.get_vote(img)