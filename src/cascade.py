import numpy as np
import src.adaboost as AB
import src.haarfeatures as haar
import src.utils as utils
from multiprocessing import cpu_count, Pool


def cascaded_classifier(pos_train_int_imgs, neg_train_int_imgs, pos_test_int_imgs, neg_test_int_imgs, max_cascade=3):
    # train a cascade of classifier which need to meet the overall detection rates & FPR
    # preset parameters for detection and false positive
    d = .8 
    f = .5 
    F_target = .05 # target FPR for the cascaded classifier
    D_target = .80 # target TPR for the cascaded classifier
    D = np.ones(max_cascade, dtype=np.float64) 
    F = np.ones(max_cascade, dtype=np.float64)
    n = np.zeros(max_cascade) #classifiers in one cascade
    i = 0

    cascaded_classifiers = list()
    # parameters for AdaBoost 
    min_feature_height = 6
    max_feature_height = 8
    min_feature_width = 6
    max_feature_width = 8

    face_train, non_face_train = pos_train_int_imgs, neg_train_int_imgs

    #stages are added until tpr and fpr goals are met
    print("\ntraining the cascaded classifiers ...")
    while F[i] > F_target or D[i] < D_target:
        # end when reaching maximum number of cascades
        if i == max_cascade:
            break
        i += 1 
        F[i] = F[i - 1]
        D[i] = D[i - 1]

        print("\nadding features at cascade %d with FPR controlled" % i)
        # control the FPR for each cascade lower than 0.5
        while F[i] > f * F[i - 1] or D[i] < d * D[i - 1]:
        
            n[i] += 1 # increase the number of features in strong classifer
            print("\ntraining the strong classifier in each cascade")
            # train classifier using AdaBoost
            classifier = AB.learn(face_train, non_face_train, n[i], min_feature_width, max_feature_width, min_feature_height, max_feature_height)
            print("\n...done\tvalidating the strong classifier on the test set")

            # test classifier performance on the test images
            TP, FN, FP, TN = utils.count_rate(pos_test_int_imgs, neg_test_int_imgs, classifier)
            print("\nstatistics for this strong classifier as follows\nTP: %d\tFN: %d\tFP: %d\tTN: %d" % (TP, FN, FP, TN))
            
            D[i] = TP / (TP + FN) # update detection
            F[i] = FP / (FP + TN) # update false positive

        non_face_train = test_previous_cascade_classifier(neg_train_int_imgs, cascaded_classifiers)
        cascaded_classifiers.append(classifier)
    return cascaded_classifiers


def test_previous_cascade_classifier(neg_test_int_imgs, classifiers_list):
    # return all the misclassified images to decrease the false positive rate
    new_neg_imgs = list()
    for test_imgs in neg_test_int_imgs:
        for classifier in classifiers_list:
            if utils.ensemble_vote(neg_test_int_imgs, classifier) == 0:
                new_neg_imgs.append(test_imgs)
    return new_neg_imgs


def test_cascade_classifier(pos_test_int_imgs, neg_test_int_imgs, cascade, fp=-1):
    # check the cascaded classifier

    pre_pos, rej_pos, pre_neg, rej_neg = 0, 0, 0, 0
    TP, FN, FP, TN = pre_pos, rej_pos, pre_neg, rej_neg
    fp = np.array((range(11)))*10 if fp != -1 else []

    detected_face, detected_non_face = False, False
    for i in range(len(pos_test_int_imgs)):    
        #faces    
        for classifier in cascade:
            if utils.ensemble_vote(pos_test_int_imgs[i], classifier) == 0:
                FN += 1 #false negatives
                detected_face = True
                break
        TP += 1 if detected_face else 0
        #non faces
        for classifier in cascade:
            if utils.ensemble_vote(neg_test_int_imgs[i], classifier) == 1:
                FP += 1 # false positives
                detected_non_face = True
                break
        TN += 1 if detected_non_face else 0

        print("TP %d FN %d FP %d TN %d" % (TP, FN, FP, TN))
        detection_rate = TP/(TP+FN) if TP+FN != 0 else 0.0
        if FP in fp:
            print("detection rate: %f when false detections: %d" % (detection_rate, FP))