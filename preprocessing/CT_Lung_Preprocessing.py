import os
import numpy as np
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import exposure



class CTPreprocessing:
    def __init__(self, output_folder, removedG_folder):
        self.output_folder = output_folder
        self.removedG_folder = removedG_folder

    def check_fov(self, img, threshold=-975):
        copy_img = img.copy()
        copy_img = copy_img[25, :, :]
        width, height = copy_img.shape
        top_left_corner = np.mean(copy_img[0:5, 0:5])
        top_right_corner = np.mean(copy_img[0:5, width - 5:width])
        bottom_left_corner = np.mean(copy_img[height - 5:height, 0:5])
        bottom_right_corner = np.mean(copy_img[height - 5:height, width - 5:width])

    # Check if there is FOV in at least 3 corners
        return int(top_left_corner < threshold) + int(top_right_corner < threshold) + int(bottom_left_corner < threshold)\
           + int(bottom_right_corner < threshold) > 2


    def segment_kmeans(self, image, K=3, attempts=10):
        image_inv = 255 - image
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        vectorized = image_inv.flatten()
        vectorized = np.float32(vectorized) / 255

        ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts,
                                    cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center * 255)
        res = center[label.flatten()]
        result_image = res.reshape((image.shape))
        return result_image
    

    def chest_hole_filling(self, image):
        image = image.astype(np.uint8)
        filled_image = np.zeros_like(image)
        for i, slice in enumerate(image):
            all_objects, hierarchy = cv2.findContours(slice, cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)
        
            mask = np.zeros(slice.shape, dtype="uint8")
            area = [cv2.contourArea(object_) for object_ in all_objects]
            if len(area) == 0:
                continue
            index_contour = area.index(max(area))
            cv2.drawContours(mask, all_objects, index_contour, 255, -1)
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            filled_image[i, :, :] = mask

        return filled_image / 255


    def remove_gantry(self, image, segmented):
        gantry_mask = segmented * (segmented == np.amin(segmented))
        contours = self.chest_hole_filling(gantry_mask)
        removed = np.multiply(image, contours)
        return removed, contours


    def apply_CLAHE(self, image):
        norm_image = exposure.rescale_intensity(image, in_range='image', out_range=(-2000, 2000))
        norm_image[norm_image > -2000] = exposure.rescale_intensity(norm_image[norm_image > -2000], in_range='image', out_range=(-1000, 1000))

        kernelsize = np.array(
            (norm_image.shape[0] // 5, norm_image.shape[1] // 5,
            norm_image.shape[2] // 5))
        # CLAHE needs range of 0 to 1
        rescale_img = exposure.rescale_intensity(norm_image, in_range='image', out_range=(0, 1))
        img_CLAHE = exposure.equalize_adapthist(rescale_img, kernel_size=kernelsize)
    
        # Converting back to original range of input images
        img_CLAHE = exposure.rescale_intensity(img_CLAHE, in_range='image', out_range=(np.amin(norm_image), np.amax(norm_image)))
        img_CLAHE = sitk.GetImageFromArray(np.int16(norm_image))

        return img_CLAHE
    


   