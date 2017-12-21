#coding=utf-8
import tensorflow as tf
from nets.netbox import model
from nets.textbox_common import Matcher
import svt_data_loader as coco
import nets.configuration as c
from nets.configuration import layer_boxes, classes
from processing.visualization import *
import numpy as np
import nets.graph_components as tfc
import signal
import sys
import cv2
import colorsys
import time
import skimage.transform
import skimage.io as io
import pickle
flags = tf.app.flags
FLAGS = flags.FLAGS
from threading import Thread

class SSD:
    def __init__(self, model_dir=None, gpu_fraction=0.9):
      
        self.sess = tf.Session()
        self.imgs_ph, self.bn, self.output_tensors, self.pred_labels, self.pred_locs = model.model(self.sess)
        total_boxes = self.pred_labels.get_shape().as_list()[1]
        self.positives_ph, self.negatives_ph, self.true_labels_ph, self.true_locs_ph, self.total_loss, self.class_loss, self.loc_loss = \
            model.loss(self.pred_labels, self.pred_locs, total_boxes)
        out_shapes = [out.get_shape().as_list() for out in self.output_tensors]
        print('out_shapes=',out_shapes)
       
        c.out_shapes = out_shapes #保存卷积的长宽
      
        c.defaults = model.default_boxes(out_shapes)

        # variables in model are already initialized, so only initialize those declared after
        with tf.variable_scope("optimizer"):
            self.global_step = tf.Variable(0)
            self.lr_ph = tf.placeholder(tf.float32, shape=[])
            self.optimizer = tf.train.AdamOptimizer(1e-3).minimize(self.total_loss, global_step=self.global_step)
            init = tf.global_variables_initializer()
            self.sess.run(init)


        if model_dir is None:
            model_dir = FLAGS.model_dir

        ckpt = tf.train.get_checkpoint_state(model_dir)
        self.saver = tf.train.Saver()

        if ckpt and ckpt.model_checkpoint_path:
            #ckpt.model_checkpoint_path
            self.saver.restore(self.sess,'checkpoints/ckpt-12395')
            print("restored %s" % ckpt.model_checkpoint_path)
 # ########################### min_conf= 0.01
    def single_image(self, sample, min_conf=0.98, nms=0.25):
        resized_img = skimage.transform.resize(sample, (image_size, image_size),mode='reflect')
        pred_labels_f, pred_locs_f, step = self.sess.run([self.pred_labels, self.pred_locs, self.global_step],
                                                        feed_dict={self.imgs_ph: [resized_img], self.bn: False})
        boxes_, confidences_ = matcher.format_output(pred_labels_f[0], pred_locs_f[0])
        #resize_boxes(resized_img, sample, boxes_, scale=float(image_size))

        return postprocess_boxes(boxes_, confidences_, min_conf, nms)


def prepare_feed(matches):
    positives_list = []
    negatives_list = []
    true_labels_list = []
    true_locs_list = []

    for o in range(len(layer_boxes)):
        for y in range(c.out_shapes[o][2]):
            for x in range(c.out_shapes[o][1]):
                for i in range(layer_boxes[o]):
                    match = matches[o][x][y][i]

                    if isinstance(match, tuple): # there is a ground truth assigned to this default box
                        positives_list.append(1)
                        negatives_list.append(0)
                        true_labels_list.append(match[1]) #id
                        default = c.defaults[o][x][y][i]
                        #保存默认的 bounding box  左上角坐标
                        true_locs_list.append(calc_offsets(default, corner2centerbox(match[0])))
                    elif match == -1: # this default box was chosen to be a negative
                        positives_list.append(0)
                        negatives_list.append(1)
                        true_labels_list.append(classes) #saoga  background class
                        true_locs_list.append([0]*4)
                    else: # no influence for this training step
                        positives_list.append(0)
                        negatives_list.append(0)
                        true_labels_list.append(classes)  # background class
                        true_locs_list.append([0]*4)

    a_positives = np.asarray(positives_list)
    a_negatives = np.asarray(negatives_list)
    a_true_labels = np.asarray(true_labels_list)
    a_true_locs = np.asarray(true_locs_list)

    return a_positives, a_negatives, a_true_labels, a_true_locs

def draw_matches(I, boxes, matches, anns):
    I = np.copy(I) * 255.0

    for o in range(len(layer_boxes)):
        for y in range(c.out_shapes[o][2]):
            for x in range(c.out_shapes[o][1]):
                for i in range(layer_boxes[o]):
                    match = matches[o][x][y][i]
                    # None if not positive nor negative
                    # -1 if negative
                    # ground truth indices if positive
                    if match == -1:
                        coords = center2cornerbox(boxes[o][x][y][i])
                        draw_rect(I, coords, (255, 0, 0))
                    elif isinstance(match, tuple):
                        coords = center2cornerbox(boxes[o][x][y][i])
                        draw_rect(I, coords, (0, 0, 255))
   
    for gt_box, id ,tag in anns:
        draw_rect(I, gt_box, (0, 255, 0), 3)
        cv2.putText(I, tag, (int(gt_box[0] * image_size), int((gt_box[1] + gt_box[3]) * image_size)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    I = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imshow("matches", I)
    cv2.waitKey(1)

def basic_nms(boxes, thres=0.45):
    re = []

    def pass_nms(c, lab):
        for box_, conf_, top_label_ in re:
            if lab == top_label_ and calc_jaccard(c, box_) > thres:
                return False
        return True

    index = 0

    for box, conf, top_label in boxes:
        if top_label != classes and pass_nms(box, top_label):
            if min(box) < 0 or min(box)<0.2 or max(box)>1.0:
                continue
            print(box,conf,top_label) 

            re.append((box, conf, top_label))
            if len(re) >= 20:
                break
        index += 1
    
    return re
    #之前的min_conf=0.01, nms=0.45
def postprocess_boxes(boxes, confidences, min_conf=0.90, nms=0.95):
    filtered = []

    for box, conf, top_label in confidences:
        if conf >= min_conf:
            coords = boxes[box[0]][box[1]][box[2]][box[3]]
            coords = center2cornerbox(coords)
            
            filtered.append((coords, conf, top_label))

    return basic_nms(filtered, nms)


def draw_outputs(img, boxes, confidences, wait=1):
    I = img * 255.0

    #nms = non_max_suppression_fast(np.asarray(filtered_boxes), 1.00)
    #boxes坐标是 center坐标
    picks = postprocess_boxes(boxes, confidences)

    for box, conf, top_label in picks:#[filtered[i] for i in picks]:
        if top_label != classes:
            draw_ann(I, box, '', confidence=conf)
    I = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imshow("outputs", I)
    cv2.waitKey(wait)
   
def draw_outputs0(img, boxes, confidences, wait=1):
    I = img * 255.0
    picks = postprocess_boxes(boxes, confidences)

    for box, conf, top_label in picks:#[filtered[i] for i in picks]:
        if top_label != classes:

            c = colorsys.hsv_to_rgb(((top_label * 17) % 255) / 255.0, 1.0, 1.0)
            c = tuple([255*c[i] for i in range(3)])

            draw_ann(I, box, '', color=c, confidence=conf)
    return I

def start_train():
    ssd = SSD()

    t = time.time()

    def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        ssd.saver.save(ssd.sess, "%s/ckpt" % FLAGS.model_dir, step)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    #summary_writer = tf.train.SummaryWriter(FLAGS.model_dir)
    summary_writer = tf.summary.FileWriter(FLAGS.model_dir)
    box_matcher = Matcher()
   
    #加载train数据集
    train_loader=coco.SVT('./dataset/train.xml', './dataset/test.xml')
  
    train_batches = train_loader.create_batches(FLAGS.batch_size,dataset='train')
    while True:
        #print('come in')
        batch = train_batches.__next__()
        #batch = train_loader.get_batch()
        imgs, anns = train_loader.preprocess_batch(batch,augment=False)
        pred_labels_f, pred_locs_f, step = ssd.sess.run([ssd.pred_labels, ssd.pred_locs, ssd.global_step],
                                                        feed_dict={ssd.imgs_ph: imgs, ssd.bn: False})
        batch_values = [None for i in range(FLAGS.batch_size)]

        def match_boxes(batch_i):
            matches = box_matcher.match_boxes(pred_labels_f[batch_i], anns[batch_i])

            positives_f, negatives_f, true_labels_f, true_locs_f = prepare_feed(matches)

            batch_values[batch_i] = (positives_f, negatives_f, true_labels_f, true_locs_f)

            if batch_i == 0:
                boxes_, confidences_ = matcher.format_output(pred_labels_f[batch_i], pred_locs_f[batch_i])
                if FLAGS.display:
                    draw_outputs(imgs[batch_i], boxes_, confidences_)
                    draw_matches(imgs[batch_i], c.defaults, matches, anns[batch_i])
        for batch_i in range(FLAGS.batch_size):
            match_boxes(batch_i)
        #得到正负样本
        positives_f, negatives_f, true_labels_f, true_locs_f = [np.stack(m) for m in zip(*batch_values)]

        if step < 4000:
            lr = 8e-4
        elif step < 10000:#step < 180000:
            lr = 1e-3
        elif step < 24000:
            lr = 1e-4
        else:
            lr = 1e-5

        _, c_loss_f, l_loss_f, loss_f, step = ssd.sess.run([ssd.optimizer, ssd.class_loss, ssd.loc_loss, ssd.total_loss, ssd.global_step],
                                   feed_dict={ssd.imgs_ph: imgs, ssd.bn: True, ssd.positives_ph:positives_f, ssd.negatives_ph:negatives_f,
                                           ssd.true_labels_ph:true_labels_f, ssd.true_locs_ph:true_locs_f, ssd.lr_ph:lr})

        t = time.time() - t
        print("%i: %f (%f secs)" % (step, loss_f, t))
        t = time.time()

        tfc.summary_float(step, "loss", loss_f, summary_writer)
        tfc.summary_float(step, "class loss", c_loss_f, summary_writer)
        tfc.summary_float(step, "loc loss", l_loss_f, summary_writer)

        if step % 1000== 0:
            ssd.saver.save(ssd.sess, "%s/ckpt" % FLAGS.model_dir, step)

def evaluate_images():
    ssd = SSD()

    loader = coco.Loader(False)
    #test_batches = loader.create_batches(1, shuffle=True)
    test_batches = loader.create_batches(1, shuffle=True)
    # global i2name
    # i2name = loader.i2name
    while True:
        batch = test_batches.__next__()
        imgs, anns = loader.preprocess_batch(batch,augment=False)
        pred_labels_f, pred_locs_f, step = ssd.sess.run([ssd.pred_labels, ssd.pred_locs, ssd.global_step],
                                                        feed_dict={ssd.imgs_ph: imgs, ssd.bn: False})
        boxes_, confidences_ = matcher.format_output(pred_labels_f[0], pred_locs_f[0])
        I=draw_outputs0(imgs[0], boxes_, confidences_, wait=0)
        I = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imshow("outputs", I)
        if cv2.waitKey(0) == 27:
            break


def resize_boxes(resized, original, boxes, scale=1.0):
    scale_x = original.shape[1] / float(resized.shape[1]) * scale
    scale_y = original.shape[0] / float(resized.shape[0]) * scale

    for o in range(len(layer_boxes)):
        for y in range(c.out_shapes[o][2]):
            for x in range(c.out_shapes[o][1]):
                for i in range(layer_boxes[o]):
                    boxes[o][x][y][i][0] *= scale_x
                    boxes[o][x][y][i][1] *= scale_y
                    boxes[o][x][y][i][2] *= scale_x
                    boxes[o][x][y][i][3] *= scale_y

def get_image_detections(path):
    ssd = SSD()

    # global i2name
    # i2name = pickle.load(open("i2name.p", "rb"))
    print('path=',path)
    #cv2.namedWindow("outputs", cv2.WINDOW_NORMAL)
    sample = io.imread(path)[:, :, :3]

    #boxes_, confidences_,top_label= ssd.single_image(sample)
    re=ssd.single_image(sample)
    return re
    #return boxes_, confidences_,top_label

def evaluate_image(path):
    picks= get_image_detections(path)
    sample = io.imread(path)
    I = skimage.transform.resize(sample, (image_size, image_size),mode='reflect')
    I=I* 255.0

   
    for box, conf, top_label in picks:#[filtered[i] for i in picks]:
        if top_label != classes: 
            c = colorsys.hsv_to_rgb(((top_label * 17) % 255) / 255.0, 1.0, 1.0)
            c = tuple([255*c[i] for i in range(3)])
            draw_ann(I, box, '', color=c, confidence=conf)

    I = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imshow("outputs", I)
    cv2.waitKey(0)

if __name__ == "__main__":

    flags.DEFINE_string("model_dir", "checkpoints/", "model directory")#summaries/test0
    flags.DEFINE_integer("batch_size", 2, "batch size")
    flags.DEFINE_boolean("display", True, "display relevant windows")
    flags.DEFINE_string("mode", "image", "train, images, image, webcam")
    flags.DEFINE_string("image_path", "images/00_19.jpg", "path to image")

    if FLAGS.mode == "train":
        start_train()
    elif FLAGS.mode == "images":
        evaluate_images()
    elif FLAGS.mode == "image":
        evaluate_image(FLAGS.image_path)
