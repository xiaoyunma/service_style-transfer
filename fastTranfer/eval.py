# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from fastTranfer.preprocessing import preprocessing_factory
from fastTranfer import reader
from fastTranfer import model
import time
import os
from io import BytesIO
import cv2
import numpy as np

tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                           'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')
tf.app.flags.DEFINE_string("model_file", "models.ckpt", "")
tf.app.flags.DEFINE_string("image_file", "a.jpg", "")

FLAGS = tf.app.flags.FLAGS


# def main(_):
#
#     # Get image's height and width.
#     height = 0
#     width = 0
#     with open(FLAGS.image_file, 'rb') as img:
#         with tf.Session().as_default() as sess:
#             if FLAGS.image_file.lower().endswith('png'):
#                 image = sess.run(tf.image.decode_png(img.read()))
#             else:
#                 image = sess.run(tf.image.decode_jpeg(img.read()))
#             height = image.shape[0]
#             width = image.shape[1]
#     tf.logging.info('Image size: %dx%d' % (width, height))
#
#     with tf.Graph().as_default():
#         with tf.Session().as_default() as sess:
#
#             # Read image data.
#             image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
#                 FLAGS.loss_model,
#                 is_training=False)
#             image = reader.get_image(FLAGS.image_file, height, width, image_preprocessing_fn)
#
#             # Add batch dimension
#             image = tf.expand_dims(image, 0)
#
#             generated = model.net(image, training=False)
#             generated = tf.cast(generated, tf.uint8)
#
#             # Remove batch dimension
#             generated = tf.squeeze(generated, [0])
#
#             # Restore model variables.
#             '''
#             zhonyao
#             '''
#             saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
#             sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
#             # Use absolute path
#             FLAGS.model_file = os.path.abspath(FLAGS.model_file)
#             saver.restore(sess, FLAGS.model_file)
#
#             # Make sure 'generated' directory exists.
#             generated_file = 'generated/res.jpg'
#             if os.path.exists('generated') is False:
#                 os.makedirs('generated')
#
#             # Generate and write image data to file.
#             with open(generated_file, 'wb') as img:
#                 start_time = time.time()
#                 img.write(sess.run(tf.image.encode_jpeg(generated)))
#                 end_time = time.time()
#                 tf.logging.info('Elapsed time: %fs' % (end_time - start_time))
#
#                 tf.logging.info('Done. Please check %s.' % generated_file)

def main(image_file,category,model_file,loss_model='/vgg_16'):

    # Get image's height and width.
    height = 0
    width = 0
    img=image_file
    with tf.Session().as_default() as sess:
        if category.lower()=='png':
            image = sess.run(tf.image.decode_png(img.read()))
        else:
            image = sess.run(tf.image.decode_jpeg(img.read()))
        height = image.shape[0]
        width = image.shape[1]
    tf.logging.info('Image size: %dx%d' % (width, height))
    # with open(image_file, 'rb') as img:
    #     with tf.Session().as_default() as sess:
    #         if image_file.lower().endswith('png'):
    #             image = sess.run(tf.image.decode_png(img.read()))
    #         else:
    #             image = sess.run(tf.image.decode_jpeg(img.read()))
    #         height = image.shape[0]
    #         width = image.shape[1]
    # tf.logging.info('Image size: %dx%d' % (width, height))

    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:

            # Read image data.
            image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
                FLAGS.loss_model,
                is_training=False)
            image = reader.get_image(image_file, category,height, width, image_preprocessing_fn)

            # Add batch dimension
            image = tf.expand_dims(image, 0)

            generated = model.net(image, training=False)
            generated = tf.cast(generated, tf.uint8)

            # Remove batch dimension
            generated = tf.squeeze(generated, [0])

            # Restore model variables.
            '''
            zhonyao
            '''
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            # Use absolute path
            model_file = os.path.abspath(model_file)
            saver.restore(sess, model_file)

            # Make sure 'generated' directory exists.
            generated_file = 'fastTranfer/generated/res.jpg'
            if os.path.exists('generated') is False:
                os.makedirs('generated')

            # Generate and write image data to file.
            resultBytes=sess.run(tf.image.encode_jpeg(generated))
            img.write(resultBytes)
    return resultBytes
if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    # tf.app.run()
    img=open('img/test2.jpg','rb')
    imgByte=img.read()



    imgArray = cv2.imdecode(np.fromstring(imgByte, np.uint8), 1)  # 字节流转为numpy数组
    imgHeight, imgWidth = imgArray.shape[0], imgArray.shape[1]
    if imgHeight > 1080:
        imgHeight = 1080
    if imgWidth > 600:
        imgWidth = 600
    image = cv2.resize(imgArray, (imgWidth, imgHeight), interpolation=cv2.INTER_CUBIC)  # 对图像数组进行裁剪
    bImg=cv2.imencode('.jpg',image)[1]
    image_file=BytesIO()
    image_file.write(bImg)
    image_file.seek(0)
    resultBytes=main(image_file, 'jpg', 'models/mosaic.ckpt-done')
    # print(resultBytes)