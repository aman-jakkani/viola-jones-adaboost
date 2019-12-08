import src.integralimage as II
import src.adaboost as AB
import src.utils as UT
import numpy as np 

if __name__ == "__main__":
    pos_training_path = 'dataset-1/trainset/faces'
    neg_training_path = 'dataset-1/trainset/non-faces'
    pos_testing_path = 'dataset-1/testset/faces'
    neg_testing_path = 'dataset-1/testset/non-faces'

    print('Loading training faces')
    faces_training = UT.load_images(pos_training_path)
    num_train_face = len(faces_training)
    print('done. ' + str(num_train_face) + ' faces loaded.\n\nLoading non faces')
    non_faces_training = UT.load_images(neg_training_path)
    num_train_non = len(non_faces_training)
    print('done. ' + str(num_train_non) + ' non faces loaded.\n')
    
    print('Loading test faces')
    faces_testing = UT.load_images(pos_testing_path)
    num_test_face = len(faces_testing)
    print('done. ' + str(num_test_face) + ' faces loaded.\n\nLoading test non faces')
    non_faces_testing = UT.load_images(neg_testing_path)
    num_test_non = len(non_faces_testing)
    print('done. ' + str(num_test_non) + ' non faces loaded.\n')

    # integral images
    faces_train_int_imgs, non_train_int_imgs = list(), list()
    #pos_dev_int_imgs, neg_dev_int_imgs = list(), list()
    faces_test_int_imgs, non_test_int_imgs = list(), list()
    faces_train_variance, non_train_variance = list(), list()

    print("\ngetting integral images ...")
    for i in range(num_train_face):
        int_img_pos, var_pos = II.IntegralImage(faces_training[i]).get_integral_image()
        faces_train_int_imgs.append(int_img_pos)
        faces_train_variance.append(var_pos)

    for j in range(num_train_non):
        int_img_neg, var_neg = II.IntegralImage(non_faces_training[j]).get_integral_image()
        non_train_int_imgs.append(int_img_neg)
        non_train_variance.append(var_neg)

    for k in range(num_test_face):
        int_img_pos_test, var_pos_test = II.IntegralImage(faces_testing[k]).get_integral_image()
        faces_test_int_imgs.append(int_img_pos_test)
        
    for l in range(num_test_non):
        int_img_neg_test, var_neg_test = II.IntegralImage(non_faces_testing[l]).get_integral_image()
        non_test_int_imgs.append(int_img_neg_test)

    print("\nintegral images obtained")
    num_classifiers = 2
    # For performance reasons restricting feature size
    min_feature_height = 6
    max_feature_height = 8
    min_feature_width = 6
    max_feature_width = 8
    # classifiers are haar like features
    classifiers = AB.learn(faces_train_int_imgs, non_train_int_imgs, num_classifiers, min_feature_width, max_feature_width, min_feature_height, max_feature_height)

    print('Testing selected classifiers..')
    correct_faces = 0
    correct_non_faces = 0
    correct_faces = sum(UT.ensemble_vote_all(faces_test_int_imgs, classifiers))
    correct_non_faces = num_test_non - sum(UT.ensemble_vote_all(non_test_int_imgs, classifiers))

    print('..done.\n\nResult:\n      Faces: ' + str(correct_faces) + '/' + str(num_test_face)
          + '  (' + str((float(correct_faces) / num_test_face) * 100) + '%)\n  non-Faces: '
          + str(correct_non_faces) + '/' + str(num_test_non) + '  ('
          + str((float(correct_non_faces) / num_test_non) * 100) + '%)')