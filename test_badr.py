from PIL import Image
import shutil
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
import time
from modules.models import RetinaFaceModel
from modules.utils import (set_memory_growth, load_yaml, draw_bbox_landm,
                           pad_input_image, recover_pad_output)
from Image_creator import process_func

tf.compat.v1.enable_eager_execution()

flags.DEFINE_string('cfg_path', './configs/retinaface_res50.yaml',
                    'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_float('iou_th', 0.4, 'iou threshold for nms')
flags.DEFINE_float('score_th', 0.5, 'score threshold for nms')
flags.DEFINE_float('down_scale_factor', 1.0, 'down-scale factor for inputs')
flags.DEFINE_boolean('webcam', False, 'get image source from webcam or not')

rootdir = os.getcwd()+'\\test_browse\\downloads\\'
outputpath = os.getcwd()+'\\test_browse\\output\\'


def create_folder_structure(rootdir, outputpath):
    for subdir, dirs, files in os.walk(rootdir):
        structure = os.path.join(outputpath, subdir[len(rootdir):])
        if not os.path.isdir(structure):
            os.mkdir(structure)


def create_images(rootdir, outputpath, model, cfg):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            try:
                if str(file).split('.')[-1] == 'jpg':
                    img_origin_path = os.path.join(subdir, file)
                    img_output_path = img_origin_path.replace(
                        rootdir, outputpath)
                    process_single_image(
                        img_origin_path, img_output_path, model, cfg)
                else:
                    path = os.path.join(subdir, file)
                    shutil.copy(path, path.replace(rootdir, outputpath))
            except:
                # manage excetions here
                print('exception')


def initialize():
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # define network
    model = RetinaFaceModel(cfg, training=False, iou_th=FLAGS.iou_th,
                            score_th=FLAGS.score_th)

    # load checkpoint
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()

    return model, cfg


def process_single_image(img_path, img_outputpath, model, cfg):

    if not os.path.exists(img_path):
        print(f"cannot find image path from {img_path}")
        exit()

    print("[*] Processing on single image {}".format(img_path))

    img_raw = cv2.imread(img_path)
    img_height_raw, img_width_raw, _ = img_raw.shape
    img = np.float32(img_raw.copy())

    if FLAGS.down_scale_factor < 1.0:
        img = cv2.resize(img, (0, 0), fx=FLAGS.down_scale_factor,
                         fy=FLAGS.down_scale_factor,
                         interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # pad input image to avoid unmatched shape problem
    img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

    # run model
    outputs = model(img[np.newaxis, ...]).numpy()

    # recover padding effect
    outputs = recover_pad_output(outputs, pad_params)

    X = []
    # draw and save results
    save_img_path = os.path.join(os.path.split(img_outputpath)[
        0], 'out_' + os.path.split(img_outputpath)[1])
    for prior_index in range(len(outputs)):
        x = draw_bbox_landm(img_raw, outputs[prior_index], img_height_raw,
                            img_width_raw)
        X.append(x)
        #cv2.imwrite(save_img_path, img_raw)
    #print(f"[*] save result at {save_img_path}")
    for index, face in enumerate(X):
        image = process_func(img_path, face)
        if(len(X) == 1):
            path = img_outputpath
        else:
            path = img_outputpath.replace(".jpg", '_'+str(index)+'.jpg')
        cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path, cvt)


def main(_argv):
    create_folder_structure(rootdir, outputpath)
    model, cfg = initialize()
    create_images(rootdir, outputpath, model, cfg)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
