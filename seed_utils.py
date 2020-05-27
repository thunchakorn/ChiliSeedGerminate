import cv2
import numpy as np
import os
import copy
import tensorflow as tf
from tensorflow import keras
from itertools import combinations

def read_gen(img_folder_path, h_set = 1080):
	imgs_path = os.listdir(img_folder_path)
	print(imgs_path)
	for name in imgs_path:
		print(os.path.join(img_folder_path, name))
		img = cv2.imread(os.path.join(img_folder_path, name))
		h, w, _ = img.shape
		w_set = int(h_set*w/h)
		img_resize = cv2.resize(img, (w_set, h_set))
		yield img_resize, name

def thresholding(image, sat_thres = 50, hue_low = 15, hue_high = 35, vol_thres = None, open_size = 5, dilate_size = 7):
    img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
    img[img[...,1] < sat_thres] = 0
    img[img[...,0] < hue_low] = 0
    img[img[...,0] > hue_high] = 0
    if vol_thres is not None:
        img[img[...,2] < vol_thres] = 0
        
    img_mask = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    img_mask[img_mask > 0] = 255
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN,
                                np.ones((open_size,open_size), dtype = np.uint8))
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_DILATE,
                                np.ones((dilate_size,dilate_size), dtype = np.uint8))
    return img_mask

def sort_by_x(cnt):
    M = cv2.moments(cnt)
    return int(M['m10']/M['m00'])

def contour_box(image_mask, box_w = 128, box_h = 128, min_area_prop = 0.4):
    img_mask = image_mask.copy()
    cnt, _ = cv2.findContours( img_mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE )
    cnt = sorted(cnt, key = sort_by_x)
    bboxes =[]
    area = np.array([cv2.contourArea(i) for i in cnt])
    good_area = []
    min_area = area.mean()*min_area_prop
    for i in range(len(cnt)):
        M = cv2.moments(cnt[i])
        if area[i] > min_area:
            x_cen, y_cen = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
            x, y, w, h = int(x_cen - box_w/2), int(y_cen - box_h/2), int(box_w), int(box_h)
            bboxes.append([x, y, w, h])
            good_area.append(area[i])
    return bboxes, good_area

def overlap_area(box1, box2):  # returns None if rectangles don't intersect
    a = {'xmax': box1[0]+box1[2], 'xmin':box1[0], 'ymax':box1[1]+box1[3], 'ymin':box1[1]}
    b = {'xmax': box2[0]+box2[2], 'xmin':box2[0], 'ymax':box2[1]+box2[3], 'ymin':box2[1]}
    dx = min(a['xmax'], b['xmax']) - max(a['xmin'], b['xmin'])
    dy = min(a['ymax'], b['ymax']) - max(a['ymin'], b['ymin'])
    if (dx>=0) and (dy>=0):
        return dx*dy

def doOverlap(box1, box2): 
    
    # If one rectangle is on left side of other 
    if(box1[0] >= box2[0]+box2[2] or box2[0] >= box1[0]+box1[2]): 
        return False
  
    # If one rectangle is above other 
    if(box1[1] <= box2[1]-box2[3] or box2[1] <= box1[1]-box1[3]): 
        return False  
    
    return True

def seed2array(img, bboxes, size = 64):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
    seed_list = []
    for x, y, w, h in bboxes:
    	try:
    		seed_list.append(cv2.resize(img[y:y+h, x:x+w], (size, size)))
    	except:
    		print(x, y, w, h)
    seed_array = np.array(seed_list)
    return seed_array

def predict_seed(seed_array, model):
    output = model.predict(seed_array)
    return output.argmax(axis = 1), output

def prob_check(p, thres = 0.1):
    p_sortd = sorted(p)
    dif = abs(p_sortd[1]- p_sortd[2])
    if dif < thres:
        return True, dif
    else:
        return False, dif

def check_overlap_2(bboxes_o, area, classes, prob, thres = 0.25):
    overlap_pair = []
    checked_box = []
    checked_class = []
    checked_prob = []
    bboxes = copy.deepcopy(bboxes_o)
    w, h = bboxes[0][2], bboxes[0][3]
    for pair in combinations(bboxes, r=2):
        if doOverlap(pair[0], pair[1]):
            if overlap_area(pair[0], pair[1]) < w*h*thres:
                continue
            if classes[bboxes.index(pair[0])] == 2 or classes[bboxes.index(pair[1])] == 2:
                if classes[bboxes.index(pair[0])] == 2 and prob[bboxes.index(pair[0])].max() > prob[bboxes.index(pair[1])].max():
                    x = pair[0][0]
                    y = pair[0][1]
                    prob_class_pos = prob[bboxes.index(pair[0])]
                else:
                    x = pair[1][0]
                    y = pair[1][1]
                    prob_class_pos = prob[bboxes.index(pair[1])]

                checked_box.append([x,y,w,h])
                checked_class.append(2)
                checked_prob.append(prob_class_pos)
                overlap_pair.append(pair[0])
                overlap_pair.append(pair[1])
            else:
                most_confident = 0 if max(prob[bboxes.index(pair[0])]) > max(prob[bboxes.index(pair[1])]) else 1
                x = pair[most_confident][0]
                y = pair[most_confident][1]
                checked_box.append([x,y,w,h])
                checked_class.append(classes[bboxes.index(pair[most_confident])])
                checked_prob.append(prob[bboxes.index(pair[most_confident])])
                overlap_pair.append(pair[0])
                overlap_pair.append(pair[1])
    for b, c, p in zip(bboxes, classes, prob):
        if b not in overlap_pair:
            checked_box.append(b)
            checked_class.append(c)
            checked_prob.append(p)
    return checked_box, np.array(checked_class), np.array(checked_prob)

