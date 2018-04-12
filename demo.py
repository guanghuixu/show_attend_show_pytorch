from core.vggnet import Vgg19
from resize import resize_image
import numpy as np
from PIL import Image
import os
from scipy import ndimage
import tensorflow as tf
# with open('./demo.jpg','r+b') as f:
#     with Image.open(f) as image:
#         demo_img = resize_image(image)
#         demo_img.save('./demo_img.jpg', demo_img.format)

vgg_model_path = './data/imagenet-vgg-verydeep-19.mat'
vggnet = Vgg19(vgg_model_path)
vggnet.build()
demo_img = np.array(ndimage.imread('./demo_img.jpg',mode='RGB')).astype(np.float32)
demo_img = demo_img.reshape([-1,224,224,3])
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    feats = sess.run(vggnet.features, feed_dict={vggnet.images: demo_img})

import main
best_val_loss = main.load_pickle(main.args.loss_log)
model = main.RUNModel(main.args, best_val_loss)
main.args.demo_feat = feats
main.model.test(save_sampled_captions=False,evaluate_score=False,generate_demo_sample=True)
