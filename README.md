# viola-jones-adaboost

This is an implementation of face detection algprithm by Viola-Jones.

External packages used:
math: mathematical expressions and calculations
numpy: more complicated numerical applications like default stacks, arrays, etc.
os: operating system information
multiprocessing (pool): to utilize multithreading and reduce wait times for feature selection drastically
functools (partial): to run multiple input values separately in multithreading
pillow (Image): an image processing library 

How to set-up and run:
You will need to run pip install for the follow libraries:
'pip install pillow'
'pip install multiprocessing'
'pip install functools'

Then simply run the command 'python3 main.py' with the dataset-1 folder in the same directory.

You can adjust threshold values in the adaboost.py file, in the create features function.
You can change the number of rounds the AdaBoost runs in the main.py file, where commented, along with feature size limitations.
