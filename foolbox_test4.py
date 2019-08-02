import argparse
import cv2
import foolbox.foolbox as foolbox
import importlib
import inspect
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import skimage
import sys
import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg
from tensorflow_vgg.vgg16 import Vgg16


class Adversary_Details(object):

    def __init__(self, img_name, path, attack_name):
        self.img_name = img_name
        self.path = path
        self.image = self.load_image(os.path.join(path, img_name))
        self.alpha = 0.75
        
        self.synset = [l.strip() for l in open("./test_data/synset.txt").readlines()]

        self.images = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        self.vgg16_net = Vgg16()
        self.vgg16_net.build(self.images)
        
        poss_attacks = self.get_attacks()
        if attack_name in poss_attacks:
            self.attack_class = poss_attacks[attack_name]
        else:
            print (attack_name, "is unknown attack")
            sys.exit("\n\n")


    def get_attacks(self):
        attack_module = importlib.import_module("foolbox.foolbox.attacks")
        attack_mod_classes = [xx
                              for xx in inspect.getmembers(attack_module)
                              if inspect.isclass(xx[1])]
        temp_dict = {xx[0]:xx[1] for xx in attack_mod_classes}
        attack_dict = {xx:temp_dict[xx]
                       for xx in temp_dict
                       if (issubclass(temp_dict[xx], temp_dict['Attack']) and xx != 'Attack')}

        return attack_dict

        

    # returns image of shape [224, 224, 3]
    # [height, width, depth]
    def load_image(self, path):
        # load image
        img = skimage.io.imread(path)
        img = img / 255.0
        assert (0 <= img).all() and (img <= 1.0).all()
        # print "Original Image Shape: ", img.shape
        # we crop image from center
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
        # resize to 224, 224
        resized_img = skimage.transform.resize(crop_img, (224, 224))
        resized_img = np.asarray(resized_img, dtype=np.float32)
        return resized_img

    def change_alpha(self, alpha):
        self.alpha = alpha/100.0

def save_results(attack, image_name,
                 orig_img, adv_img,
                 diff_img, hist_img,
                 data_str):
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    image_name = image_name.split('.')[0]
    res_path = os.path.join(base_path, "results", image_name, attack)
    os.makedirs(res_path, exist_ok=True)

    for curr_img, out_file in [(orig_img,"orig.png"),
                              (adv_img, "adv.png"),
                              (diff_img, "diff.png"),
                              (hist_img, "hist.png")]:

        if int(curr_img.max()) <= 1.0:
            curr_img = 255.0 * curr_img
        curr_img = curr_img.astype('int16')
        cv2.imwrite(os.path.join(res_path,out_file), curr_img)
    with open (os.path.join(res_path, "info.txt"), 'w') as f:
        f.write(data_str)
    print ("Saved results to", res_path)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test foolbox")
    parser.add_argument("-p", "--path",
                        default= "./test_data/")
    parser.add_argument("-i", "--image", type=str,
                        default="tiger.jpeg")
    parser.add_argument("-a", "--attack", type=str,
                        default="FGSM")
 
    args = vars(parser.parse_args())
    detailer = Adversary_Details(args["image"], args["path"], args["attack"])
    
    with tf.Session() as session:
        model = foolbox.models.TensorFlowModel(detailer.images,
                                               detailer.vgg16_net.fc8,
                                               (0,1.0),
                                               detailer.vgg16_net)
        
        attack = detailer.attack_class(model) 
        
        pre_softmax = model.forward_one(detailer.image)
        conf = foolbox.utils.softmax(pre_softmax)
        idx = np.argmax(conf)
        category = ' '.join(detailer.synset[idx].split()[1:])

        adv_image = attack(detailer.image, idx)
        pre_softmax2 = model.forward_one(adv_image)
        conf2 = foolbox.utils.softmax(pre_softmax2)
        idx2 = np.argmax(conf2)
        category2 = ' '.join(detailer.synset[idx2].split()[1:])
 
        diff = np.abs(detailer.image-adv_image)
        hist_image = diff * 255.0
        hist_image = hist_image.astype('int16')
        hist = hist_image.flatten()
        hist_max = (int(hist.max()/10)+1)*10
        hist_name = args["attack"] + " Histogram"
        plt.hist(hist, range(hist_max))
        plt.yscale('log')
        plt.xlabel("Diff")
        plt.ylabel('Count')
        plt.title(hist_name)
        plt.savefig('temp.png')
        hist_img = cv2.imread("temp.png")
        print("Shape =", hist_img.shape)
        
        diff_max = diff.max()
        mult_factor = int(detailer.image.max()/diff.max())
        diff *= mult_factor
        
        print("Max Diff:", diff_max)
        print("Raw Image:", idx, category, conf[idx])
        print("Adv Image:", idx2, category2, conf2[idx2])
        diff_title_str = "Difference (X" + str(mult_factor) +")"
        data_str = "Max Diff: " + str(diff_max) + '\n'
        data_str += "Raw Image:" + str(idx) + " -- " + str(category) + " -- " + str(conf[idx]) + '\n'
        data_str += "Adv Image:" + str(idx2) + " -- " + str(category2) + " -- " + str(conf[idx2]) + '\n'
        data_str += "Mult Factor: " + str(mult_factor) + '\n'
        
        cv2.namedWindow("Original Image")
        cv2.namedWindow("Adversarial Image")
        cv2.namedWindow(diff_title_str)
        cv2.namedWindow("Adversarial Overlay")
        cv2.namedWindow(hist_name)

        cv2.moveWindow("Original Image", 10,250)
        cv2.moveWindow("Adversarial Image", 400,250)
        cv2.moveWindow(diff_title_str, 800,250)
        cv2.moveWindow("Adversarial Overlay", 1200,250)
        cv2.moveWindow(hist_name, 1600,250)

        cv2.createTrackbar('Alpha','Adversarial Overlay',
                           int(detailer.alpha*100),
                           100, detailer.change_alpha)

        save_results_active = False
        
        while True:
            k = cv2.waitKey(1) &0xFF
            ref_img = detailer.image.copy()
            cv2.addWeighted(diff,
                            detailer.alpha,
                            ref_img,
                            1.0-detailer.alpha,
                            0,ref_img)

            cv2.imshow("Original Image",detailer.image[:,:,::-1])
            cv2.imshow("Adversarial Image",adv_image[:,:,::-1])
            cv2.imshow(diff_title_str,diff[:,:,::-1])
            cv2.imshow("Adversarial Overlay",ref_img[:,:,::-1])
            cv2.imshow(hist_name, hist_img)

            if k == 27 or chr(k) == 'q':
                #save_results(args["attack"], args["image"])
                break
            elif chr(k) == 's' or chr(k) == 'S':
                print ("Saving")
                save_results(args["attack"], args["image"],
                             detailer.image[:,:,::-1],
                             adv_image[:,:,::-1],
                             diff[:,:,::-1],
                             hist_img[:,:,::-1],
                             data_str)
                
        cv2.destroyAllWindows()
        
