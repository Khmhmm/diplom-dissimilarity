import cv2
import skimage

def cv_2_skimage(cv2_img):
    return skimage.util.img_as_float(cv2_img)

def skimage_2_cv(sk_img):
    return skimage.util.img_as_ubyte(sk_img)
