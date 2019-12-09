import src.integralimage as II
import src.adaboost as AB
import src.utils as UT
import src.cascade as C
import numpy as np 

if __name__ == "__main__":
    pos_training_path = 'dataset-1/trainset/faces'
    neg_training_path = 'dataset-1/trainset/non-faces'
    pos_testing_path = 'dataset-1/testset/faces'
    neg_testing_path = 'dataset-1/testset/non-faces'

    print('Loading training faces..')
    faces_train = UT.load_images(pos_training_path)
    faces_train_int = list(map(II.to_integral, faces_train))
    print('..done. ' + str(len(faces_train)) + ' faces loaded.\n\nLoading non faces..')
    non_faces_train = UT.load_images(neg_training_path)
    non_faces_train_int = list(map(II.to_integral, non_faces_train))
    print('..done. ' + str(len(non_faces_train)) + ' non faces loaded.\n')

    num_classifiers = 3
    # For performance reasons restricting feature size
    min_feature_height = 6
    max_feature_height = 8
    min_feature_width = 6
    max_feature_width = 8
    
    #learn algorithm
    classifiers = AB.learn(faces_train_int, non_faces_train_int, num_classifiers, min_feature_height, max_feature_height, min_feature_width, max_feature_width)
    for n in range(len(classifiers)):
        print(classifiers[n].type, classifiers[n].top_left, classifiers[n].width, classifiers[n].height, classifiers[n].threshold)
    
    print('Loading test faces..')
    faces_test = UT.load_images(pos_testing_path)
    faces_test_int = list(map(II.to_integral, faces_test))
    print('..done. ' + str(len(faces_test)) + ' faces loaded.\n\nLoading test non faces..')
    non_faces_test = UT.load_images(neg_testing_path)
    non_faces_test_int = list(map(II.to_integral, non_faces_test))
    print('..done. ' + str(len(non_faces_test)) + ' non faces loaded.\n')
    
    print('Testing selected classifiers..')
    correct_faces = 0
    correct_non_faces = 0
    correct_faces = sum(UT.ensemble_vote_all(faces_test_int, classifiers))
    correct_non_faces = len(non_faces_test) - sum(UT.ensemble_vote_all(non_faces_test_int, classifiers))

    print('..done.\n\nResult:\n      Faces: ' + str(correct_faces) + '/' + str(len(faces_test))
          + '  (' + str((float(correct_faces) / len(faces_test)) * 100) + '%)\n  non-Faces: '
          + str(correct_non_faces) + '/' + str(len(non_faces_test)) + '  ('
          + str((float(correct_non_faces) / len(non_faces_test)) * 100) + '%)')
    
    cascader = C.cascaded_classifier(faces_train_int, non_faces_train_int, faces_test_int, non_faces_test_int)
    C.test_cascade_classifier(faces_test_int, non_faces_test_int, cascader)