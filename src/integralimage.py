import math
import numpy as np

def to_integral(img):
    # an index of -1 refers to the last row/column
    row_sum = np.zeros(img.shape)
    # need an additional column and row
    integ_image_arr = np.zeros((img.shape[0] + 1, img.shape[1] + 1))
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            row_sum[y, x] = row_sum[y-1, x] + img[y, x]
            integ_image_arr[y+1, x+1] = integ_image_arr[y+1, x-1+1] + row_sum[y, x]
    return integ_image_arr

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