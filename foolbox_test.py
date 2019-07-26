import argparse
import foolbox.foolbox as foolbox
import numpy as np
import os
from PIL import Image
import skimage
import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg
from tensorflow_vgg.vgg16 import Vgg16



# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test foolbox")
    parser.add_argument("-i", "--image", type=str,
                        default = "./test_data/tiger.jpeg")#kitten2.png")

    args = vars(parser.parse_args())
    image = load_image(args["image"])
    synset = [l.strip() for l in open("./test_data/synset.txt").readlines()]
                        
    
    images = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))

    logits2 = Vgg16()
    logits2.build(images)
    
    with tf.device('/GPU:0'):
        with tf.Session() as sess:
            batch = image.reshape((1, 224, 224, 3))
            feed_dict = {images: batch}

            vgg = Vgg16()
            with tf.name_scope("content_vgg"):
                vgg.build(images)

            [prob, fc8, fc7, fc6, conv5_1, conv4_1, conv3_1,
             conv2_1, conv1_1, bgr, rgb_sc, rgb] = sess.run([vgg.prob,
                                                             vgg.fc8,
                                                             vgg.fc7,
                                                             vgg.fc6,
                                                             vgg.conv5_1,
                                                             vgg.conv4_1,
                                                             vgg.conv3_1,
                                                             vgg.conv2_1,
                                                             vgg.conv1_1,
                                                             vgg.bgr,
                                                             vgg.rgb_scaled,
                                                             vgg.rgb], feed_dict=feed_dict)

            direct_idx = np.argmax(prob)
            direct_conf = prob[0,direct_idx]
            direct_category = ' '.join(synset[direct_idx].split()[1:])
                    
    with foolbox.models.TensorFlowModel(images, logits2.fc8, (0, 255), logits2) as model:

        raw_conf = model.forward_one(image)
        idx = np.argmax(raw_conf)
        raw_max = raw_conf.max()
        raw_conf2 = raw_conf-raw_max
        raw_conf2_exp = np.exp(raw_conf2)
        raw_conf_norm = 1./raw_conf2_exp.sum()
        conf = raw_conf2_exp * raw_conf_norm
        category = ' '.join(synset[idx].split()[1:])

    print("Direct Net :", direct_idx, direct_category, direct_conf)
    print("Foolbox Net:", idx, category, conf[idx])
