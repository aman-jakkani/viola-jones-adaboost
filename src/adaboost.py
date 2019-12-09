import numpy as np
import os
from multiprocessing import cpu_count, Pool
from functools import partial
from src.haarfeatures import HaarLikeFeature as haar
from src.haarfeatures import feat_type
from src.utils import *

haar_feats = list()

def _create_features(img_width, img_height, min_feat_width, max_feat_width, min_feat_height, max_feat_height):
    threshold = 0.4
    if(len(haar_feats) == 0):
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
                            haar_feats.append(haar(each_feat, (i,j), feat_width, feat_height, threshold, 1)) 
                            haar_feats.append(haar(each_feat, (i,j), feat_width, feat_height, threshold, -1)) 
    return haar_feats

def _get_feature_vote(feature, img):
    return feature.get_vote(img)

def learn(pos_int_img, neg_int_img, num_classifiers=-1, min_feat_width=1, max_feat_width=-1, min_feat_height=1, max_feat_height=-1):
    # adaboost learning algorithm with variable number of rounds/classifiers
    #return list of features, one feature per round/classifier

    num_pos = len(pos_int_img)
    num_neg = len(neg_int_img)
    num_imgs = num_pos + num_neg
    img_height, img_width = pos_int_img[0].shape

    # initialize weights and labels
    pos_weights = np.ones(num_pos) * 1. / (2 * num_pos) # w = 1/2m
    neg_weights = np.ones(num_neg) * 1. / (2 * num_neg) # w = 1/2l
    weights = np.hstack((pos_weights, neg_weights)) 
    labels = np.hstack((np.ones(num_pos), np.zeros(num_neg)))  

    # training images list
    images = pos_int_img + neg_int_img 

    print("\ncreating haar-like features")
    features = _create_features(img_width, img_height, min_feat_width, max_feat_width, min_feat_height, max_feat_height)
    num_features = len(features)
    print('done. %d features were created!' % num_features)
    """
    num_feat1= 0
    num_feat2= 0 
    num_feat3 = 0
    num_feat4= 0 
    num_feat5 = 0
    for each in features:
        if(each.type == feat_type.TWO_VERTICAL):
            num_feat1 +=1
        if(each.type == feat_type.TWO_HORIZONTAL):
            num_feat2 +=1
        if(each.type == feat_type.THREE_VERTICAL):
            num_feat3 +=1
        if(each.type == feat_type.THREE_HORIZONTAL):
            num_feat4 +=1
        if(each.type == feat_type.FOUR):
            num_feat5 +=1
    print('%d type 1 features were created' % num_feat1)
    print('%d type 2 features were created' % num_feat2)
    print('%d type 3 features were created' % num_feat3)
    print('%d type 4 features were created' % num_feat4)
    print('%d type 5 features were created' % num_feat5)
    """
    feature_index = list(range(num_features)) # save manipulation of data

    # default number of classifiers
    num_classifiers = num_features if num_classifiers == -1 else num_classifiers

    print("\ncalculating scores for images")

    votes = np.zeros((num_imgs, num_features))

    # parallelize the execution of a function across multiple input values
    pool = Pool(processes=3)
    # get all votes for each image and each feature (quite time-consuming so we use pool, even then)
    for i in range(num_imgs):
        votes[i, :] = np.array(list(pool.map(partial(_get_feature_vote, img=images[i]), features)))
    
    # select classifiers
    classifiers = list() # list of HaarLikeFeature objects

    print("\nselecting classifiers")

    for _ in range(num_classifiers):
        
        class_errors = np.zeros(len(feature_index)) 

        # normalize weights (w_t)
        weights *= 1. / np.sum(weights)

        # select the best classifier based on the weighted error
        for f in range(len(feature_index)):
            f_idx = feature_index[f]
            #uncomment below for false negative error while training
            #err = sum(map(lambda img_idx: weights[img_idx] if (labels[img_idx] != votes[img_idx, f_idx] and labels[img_idx] == 1.0) else 0, range(num_imgs)))
            #uncomment below for false positive error while training
            #err = sum(map(lambda img_idx: weights[img_idx] if labels[img_idx] == 0 and votes[img_idx, f_idx] == 1 else 0, range(num_imgs)))
            #uncomment below for empirical error while training
            err = sum(map(lambda img_idx: weights[img_idx] if labels[img_idx] != votes[img_idx, f_idx] else 0, range(num_imgs)))
            class_errors[f] = err
        #debug : print(class_errors)
        # get the best feature (with the smallest error)
        min_error_idx = np.argmin(class_errors) 
        best_error = class_errors[min_error_idx]
        best_feature_idx = feature_index[min_error_idx]
        # set alpha and add to classifier list
        best_feature = features[best_feature_idx]
        alpha = .5 * np.log((1 - best_error) / best_error)
        best_feature.weight = alpha
        classifiers.append(best_feature)

        def new_weights(best_error):
            return np.sqrt((1 - best_error) / best_error)

        # update image weights based on misclassification
        weights_map = map(lambda img_idx: weights[img_idx] * new_weights(best_error) if labels[img_idx] != votes[img_idx, best_feature_idx] else weights[img_idx] * 1, range(num_imgs))
        weights = np.array(list(weights_map))

        # remove feature (a feature cannot be selected twice)
        feature_index.remove(best_feature_idx)

    print("\nselected the classifiers, ending Adaboost")

    return classifiers