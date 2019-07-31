import argparse
import cv2
import foolbox.foolbox as foolbox
import importlib
import inspect
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage
import sys

class Show_Results(object):

    def __init__(self, image, attack):
        self.image = image
        self.attack = attack
        self.data_path = os.path.join('./results',
                                      self.image,
                                      self.attack)
        self.alpha = 0.75

    def change_alpha(self, alpha):
        self.alpha = alpha/100.0

    def load_data(self):
        self.orig_image = skimage.io.imread(os.path.join(self.data_path,
                                                         "orig.png"))
        self.diff = np.load(os.path.join(self.data_path,
                                         "abs_diff.npy"))

        mult_factor = int(self.orig_image.max()/self.diff.max())
        self.diff_image = self.diff * mult_factor
        self.diff_image = self.diff_image.astype('uint8')

        self.hist_data = self.diff.flatten()
        self.hist_data *= 255.0
        hist_max = (int(self.hist_data.max()/10)+1)*10
        self.hist_name = args["attack"] + " Histogram"
        plt.hist(self.hist_data, range(hist_max))
        plt.yscale('log')
        plt.xlabel("Diff")
        plt.ylabel('Count')
        plt.title(self.hist_name)
        plt.savefig('temp.png')
        self.hist_image = cv2.imread("temp.png")

    def show_data(self):
        cv2.namedWindow("Original Image")
        cv2.namedWindow("Adversarial Overlay")
        cv2.namedWindow(self.hist_name)

        cv2.moveWindow("Original Image", 10,250)
        cv2.moveWindow("Adversarial Overlay", 1200,250)
        cv2.moveWindow(self.hist_name, 1600,250)

        cv2.createTrackbar('Alpha','Adversarial Overlay',
                           int(self.alpha*100),
                           100, self.change_alpha)

        while True:
            k = cv2.waitKey(1) &0xFF
            ref_img = self.orig_image.copy()
            cv2.addWeighted(self.diff_image,
                            self.alpha,
                            ref_img,
                            1.0-self.alpha,
                            0,ref_img)

            cv2.imshow("Original Image",self.orig_image[:,:,::-1])
            cv2.imshow("Adversarial Overlay",ref_img[:,:,::-1])
            cv2.imshow(self.hist_name, self.hist_image)

            if k == 27 or chr(k) == 'q':
                #save_results(args["attack"], args["image"])
                break
                
        cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test foolbox")
    parser.add_argument("-i", "--image", type=str,
                        default="tiger")
    parser.add_argument("-a", "--attack", type=str,
                        default="FGSM")
 
    args = vars(parser.parse_args())
    summary = Show_Results(args["image"],  args["attack"])
    summary.load_data()
    summary.show_data()
