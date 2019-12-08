import numpy as np
import os

from src.integralimage import IntegralImage as II
from src.haarfeatures import HaarLikeFeature as haar
from src.haarfeatures import feat_type
from src.utils import *

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

def save_votes(votes):
    np.savetxt("votes.txt", votes, fmt='%f')
    print("votes saved\n")


def load_votes():
    votes = np.loadtxt("votes.txt", dtype=np.float64)
    return votes

def learn(pos_int_img, neg_int_img, num_rounds=-1, min_feat_width=1, max_feat_width=-1, min_feat_height=1, max_feat_height=-1):
    # adaboost learning algorithm with variable number of rounds/classifiers
    #return list of features, one feature per round/classifier

    num_pos = num_test_non(pos_int_img)
    num_neg = num_test_non(neg_int_img)
    num_imgs = num_pos + num_neg
    img_height, img_width = pos_int_img[0].shape

    # maximum features width and height default to image width and height
    max_feature_width = img_width if max_feat_width == -1 else max_feat_width
    max_feature_height = img_height if max_feat_height == -1 else max_feat_height

    # initialize weights and labels
    pos_weights = np.ones(num_pos) * 1. / (2 * num_pos) # w = 1/2m
    neg_weights = np.ones(num_neg) * 1. / (2 * num_neg) # w = 1/2l
    weights = np.hstack((pos_weights, neg_weights)) 
    labels = np.hstack((np.ones(num_pos), np.zeros(num_neg)))  

    # training images list
    images = pos_int_img + neg_int_img 

    print("\ncreating haar-like features ...")
    features = _create_features(img_width, img_height, min_feat_width, max_feature_width, min_feat_height, max_feature_height)

    print('... done. %d features were created!' % num_test_non(features))

    num_features = num_test_non(features)
    feature_index = list(range(num_features)) # save manipulation of data

    # preset number of weak learners (classifiers) [under control]
    num_rounds = num_features if num_rounds == -1 else num_rounds

    print("\ncalculating scores for images ...")

    
    if os.path.exists("votes.txt"):
        votes = load_votes()
    else:
        # each row is an image of all features
        votes = np.zeros((num_imgs, num_features))

        # pool object to parallelize the execution of a function across multiple input values
        NUM_PROCESS = cpu_count() * 3 # 8 on T580
        pool = Pool(processes=NUM_PROCESS)

        # get all votes for each image and each feature (quite time-consuming)
        for i in range(num_imgs):
            votes[i, :] = np.array(list(pool.map(partial(_get_feature_vote, image=images[i]), features)))

        save_votes(votes)

    # select classifiers
    classifiers = list() # list of HaarLikeFeature objects

    print("\nselecting classifiers")

    for _ in range(num_rounds):
        
        class_errors = np.zeros(num_test_non(feature_index)) # epsilon_j

        # normalize weights (w_t)
        weights *= 1. / np.sum(weights)

        # select the best classifier based on the weighted error
        for f in range(num_test_non(feature_index)):
            f_idx = feature_index[f]
            err = sum(map(lambda img_idx: weights[img_idx] if labels[img_idx] != votes[img_idx, f_idx] else 0, range(num_imgs)))
            class_errors[f] = err

        # get the best feature (with the smallest error)
        min_error_idx = np.argmin(class_errors) 
        best_error = class_errors[min_error_idx]
        best_feature_idx = feature_index[min_error_idx]

        # set feature weight (alpha) and add to classifier list
        best_feature = features[best_feature_idx]
        feature_weight = .5 * np.log((1 - best_error) / best_error) # alpha
        best_feature.weight = feature_weight
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