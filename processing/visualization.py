#coding=utf-8
import cv2
from nets.configuration import image_size
import numpy as np

def draw_rect(I, r, c, thickness=1):
    if abs(sum(r)) < 10: 
        cv2.rectangle(I, (int(r[0] * image_size), int(r[1] * image_size)),
                      (int((r[0] + max(r[2], 0)) * image_size), int((r[1] + max(r[3], 0)) * image_size)),
                      c, thickness)

def draw_ann(I, r, text, color=(255, 0, 255), confidence=-1):
    
    #脚点坐标
    draw_rect(I, r, color, 1)
    text_ = text
    if confidence >= 0:
        text_ += "%0.2f" % confidence

    cv2.putText(I, text_, (int(r[0] * image_size), int((r[1]) * image_size)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

def center2cornerbox(rect):
    return [rect[0] - rect[2]/2.0, rect[1] - rect[3]/2.0, rect[2], rect[3]]

def corner2centerbox(rect):
    return [rect[0] + rect[2]/2.0, rect[1] + rect[3]/2.0, rect[2], rect[3]]

def calc_intersection(r1, r2):
    left = max(r1[0], r2[0])
    right = min(r1[0] + r1[2], r2[0] + r2[2])
    bottom = min(r1[1] + r1[3], r2[1] + r2[3])
    top = max(r1[1], r2[1])

    if left < right and top < bottom:
        return (right - left) * (bottom - top)

    return 0

def calc_offsets(default, truth):
    return [truth[0] - default[0],
            truth[1] - default[1],
            truth[2] - default[2],
            truth[3] - default[3]]

def clip_box(r):
    return [r[0], r[1], max(r[2], 0.01), max(r[3], 0.01)]

def calc_jaccard(r1, r2):
    r1_ = clip_box(r1)
    r2_ = clip_box(r2)
    intersection = calc_intersection(r1_, r2_)
    union = r1_[2] * r1_[3] + r2_[2] * r2_[3] - intersection

    if union <= 0:
        return 0

    j = intersection / union

    return j

def calc_overlap(r1, host):
    intersection = calc_intersection(r1, host)
    return intersection / (1e-5 + host[2] * host[3])

def default2cornerbox(default, offsets):
    c_x = default[0] + offsets[0]
    c_y = default[1] + offsets[1]
    w = default[2] + offsets[2]
    h = default[3] + offsets[3]

    return [c_x - w/2.0, c_y - h/2.0, w, h]

