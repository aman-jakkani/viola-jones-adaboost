import numpy as np
from src.haarfeatures import feat_type
from functools import partial
import os
from PIL import Image

def ensemble_vote(int_img, classifiers):
    """
    Classifies given integral image (numpy array) using given classifiers, i.e.
    if the sum of all classifier votes is greater 0, image is classified
    positively (1) else negatively (0). The threshold is 0, because votes can be
    +1 or -1.
    """
    return 1 if sum([c.get_vote(int_img) for c in classifiers]) >= 0 else 0


def ensemble_vote_all(int_imgs, classifiers):
    """
    Classifies given list of integral images (numpy arrays) using classifiers
    """
    vote_partial = partial(ensemble_vote, classifiers=classifiers)
    return list(map(vote_partial, int_imgs))


def load_images(path):
    imgs = []
    for _file in os.listdir(path):
        if _file.endswith('.png'):
            img_arr = np.array(Image.open((os.path.join(path, _file))), dtype=np.float64)
            img_arr /= img_arr.max()
            imgs.append(img_arr)
    return imgs

def count_rate(pos_images, neg_images, classifiers):
    # return: False positive and False negative rate
    num_pos = len(pos_images)
    num_neg = len(neg_images)

    # True positives
    correct_pos_image = sum(ensemble_vote_all(pos_images, classifiers))
    # False negatives 
    incorrect_pos_image = num_pos - correct_pos_image
    
    # False positives
    incorrect_neg_image = sum(ensemble_vote_all(neg_images, classifiers))
    # True negatives
    correct_neg_image = num_neg - incorrect_neg_image
    TP, FN, FP, TN = correct_pos_image, incorrect_pos_image, incorrect_neg_image, correct_neg_image
    return TP, FN, FP, TN


def two_haar_equal(haar0, haar1):
    # check whether two haar like features are the same 
    # para haar0: haar features ready to write a json file
    # para haar1: haar features loaded from the json file
    if str(haar0.type.name) != str(haar1.type):
        return False
    elif haar0.top_left[0] != haar1.top_left[0] or haar0.top_left[1] != haar1.top_left[1]:
        return False
    elif haar0.width != haar1.width:
        return False
    elif haar0.height != haar1.height:
        return False
    elif haar0.threshold != haar1.threshold:
        return False
    elif haar0.parity != haar1.parity:
        return False
    elif int(haar0.weight) != int(haar1.weight):
        return False
    else:
        return True