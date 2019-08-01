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

    def __init__(self, image_name, path):
        self.image_name = image_name
        self.path = path
        self.image = self.load_image(os.path.join(path, image_name))
        self.alpha = 0.75
        
        self.synset = [l.strip() for l in open("./test_data/synset.txt").readlines()]

        self.images = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        
        self.poss_attacks = self.get_attacks()

    def reset(self, curr_attack):
        self.alpha = 0.75
        self.vgg16_net = Vgg16()
        self.vgg16_net.build(self.images)

        self.model = foolbox.models.TensorFlowModel(self.images,
                                                    self.vgg16_net.fc8,
                                                    (0, 255),
                                                    self.vgg16_net)
        self.attack = self.poss_attacks[curr_attack](self.model)
        self.data_str = ""
        self.pre_softmax = self.model.forward_one(self.image)
        self.conf = foolbox.utils.softmax(self.pre_softmax)
        self.idx = np.argmax(self.conf)
        self.category = ' '.join(self.synset[self.idx].split()[1:])



    def make_attack(self):
        self.adv_image = self.attack(self.image, self.idx)
        self.pre_softmax2 = self.model.forward_one(self.adv_image)
        self.conf2 = foolbox.utils.softmax(self.pre_softmax2)
        self.idx2 = np.argmax(self.conf2)
        self.category2 = ' '.join(self.synset[self.idx2].split()[1:])

        
        self.diff = np.abs(self.image - self.adv_image)
        diff_max = self.diff.max()
        mult_factor = int(max(1, int(self.image.max()/self.diff.max())))
        self.diff_image = self.diff * mult_factor

        self.data_str = "   Max Diff: " + str(diff_max) + '\n'
        self.data_str += "   Raw Image:" + str(self.idx) + " -- " + str(self.category)
        self.data_str += " -- " + str(self.conf[self.idx]) + '\n'
        self.data_str += "   Adv Image:" + str(self.idx2) + " -- " + str(self.category2)
        self.data_str += " -- " + str(self.conf2[self.idx2]) + '\n'
        self.data_str += "   Mult Factor: " + str(mult_factor) + '\n'


    def make_histogram(self, curr_attack):
        self.hist_image = 255.0 * self.diff
        self.hist_image = self.hist_image.astype('int16')
        hist = self.hist_image.flatten()
        hist_max = (int(hist.max()/10)+1)*10
        hist_name = curr_attack + " Histogram"
        plt.hist(hist, range(hist_max))
        plt.yscale('log')
        plt.xlabel("Diff")
        plt.ylabel('Count')
        plt.title(hist_name)
        plt.savefig('temp.png')
        plt.clf()
        self.hist_image = cv2.imread("temp.png")

    def save_results(self, curr_attack):     
        base_path = os.path.dirname(os.path.abspath(__file__))
        image_name = self.image_name.split('.')[0]
        res_path = os.path.join(base_path, "results", image_name, curr_attack)
        os.makedirs(res_path, exist_ok=True)

        for curr_img, out_file in [(self.image[:,:,::-1],"orig.png"),
                                   (self.adv_image[:,:,::-1], "adv.png"),
                                   (self.diff_image[:,:,::-1], "diff.png"),
                                   (self.hist_image[:,:,::-1], "hist.png")]:
            if int(curr_img.max()) <= 1.0:
                curr_img = 255.0 * curr_img
            curr_img = curr_img.astype('int16')
            cv2.imwrite(os.path.join(res_path,out_file), curr_img)
        np.save(os.path.join(res_path, "abs_diff.npy"), self.diff)
            
        with open (os.path.join(res_path, "info.txt"), 'w') as f:
            f.write(self.data_str)
        print ("Saved results to", res_path)




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


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test foolbox")
    parser.add_argument("-p", "--path",
                        default= "./test_data/")
    parser.add_argument("-i", "--image", type=str,
                        default="tiger.jpeg")
 
    args = vars(parser.parse_args())
    det = Adversary_Details(args["image"], args["path"])
    attack_dict = det.get_attacks()
    attack_list = sorted(attack_dict.keys())
    num_attacks = len(attack_dict)

    with open("tracker.txt","w") as f:
        sys.stderr = f
        for ctr, curr_attack in enumerate(attack_list): #["FGSM", "BIM", "PGD", "NewtonFoolAttack"]:
            with tf.Session() as session:
        
                try:
                    out_string = curr_attack + ": " + str(ctr) + " of " + str(num_attacks) + " ..... "
                    f.write (out_string)
                    f.flush()
                    det.reset(curr_attack)
                    det.make_attack()
                    det.make_histogram(curr_attack)
                    det.save_results(curr_attack)
                    f.write("OK\n")
                    f.write(det.data_str + '\n')
                except:
                    f.write("FAILED\n\n")
                f.flush()


