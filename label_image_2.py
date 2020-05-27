import argparse
import os
from seed_utils import *
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-img_dir", help = 'path to directory of seed image', default = './seed_image')
parser.add_argument("-model_dir", help = 'path to model', default = './Model/base_model.h5')
parser.add_argument("-w", "--weight", help = "path to model's weight directory")
parser.add_argument("-train", help = "True or False, whether model learn from unsure", default = True)

args = parser.parse_args()


def main(img_dir, model_dir, weight):
    h_set = 512
    img_folder_path = './seed_train_image_0'
    imgs_path = sorted(os.listdir(img_folder_path))
    imgs_BGR = [cv2.imread(os.path.join(img_folder_path, i)) for i in imgs_path]
    des_dir = './seed_labeled'
    # สร้าง subfolder ย่อยข้างใน des_dir เป็น pos, neg, none
    pos = '1'
    neg = '2'
    none = '3'
    #img_gen = read_gen(img_folder_path)
    j = 0
    cv2.waitKey(0)
    
    while j < len(imgs_BGR):
        index = j
        img = copy.deepcopy(imgs_BGR[j])
        im_h, im_w, _ = img.shape
        w_set = int(h_set*im_w/im_h)
        img_thres = thresholding(img, sat_thres=57, hue_low = 12, hue_high = 33, dilate_size= 7, open_size=3)
        bboxes, area = contour_box(img_thres, 64, 64, 0.1)
        bboxes = check_overlap(bboxes, area,thres = 0.2)
        img_copy = img.copy()
        [cv2.rectangle(img_copy, (x,y), (x+w, y+h), (0,255,0), 2) for x,y,w,h in bboxes]
        [cv2.putText(img_copy, f'{i+1}',
                        (box[0],box[1]),
                        color = (255, 0 ,0),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 1,
                        thickness = 4) for i, box in enumerate(bboxes)]
        cv2.putText(img_copy, f'{len(bboxes)}',(50,50),
                    color = (0,0,255),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 2,
                    thickness = 4)
        cv2.imshow('all_box', cv2.resize(img_copy, (w_set, h_set)))
        i = 0
        while i < len(bboxes):
            x,y,w,h = bboxes[i]
            img_copy = img.copy()
            cv2.rectangle(img_copy, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img_copy, f'{j}_{i+1}',
                        (x,y),
                        color = (255, 0 ,0),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 1,
                        thickness = 4)
            img_copy = cv2.resize(img_copy, (w_set, h_set), interpolation = cv2.INTER_AREA)
            img_crop = img[y:y+h, x:x+w]
            cv2.imshow('seed', cv2.resize(img_crop, (128,128)))
            cv2.imshow('image', img_copy)
            key = cv2.waitKey(1)
            if key == ord('1'):
                cv2.imwrite(os.path.join(des_dir, 'pos','{0}_{1:03d}.png'.format(j,i)), img_crop)
                i += 1
            elif key == ord('2'):
                cv2.imwrite(os.path.join(des_dir, 'neg','{0}_{1:03d}.png'.format(j,i)), img_crop)
                i += 1
            elif key == ord('3'):
                cv2.imwrite(os.path.join(des_dir, 'none','{0}_{1:03d}.png'.format(j,i)), img_crop)
                i += 1
            elif key == 27:
                break
            elif key == ord('a'):
                i -= 1
            elif key == ord('s'):
                i += 1
            elif key == ord('f'):
                break
            elif key == ord('r'):
                break
            else:
                continue
        if key == 27:
            break
        elif key == ord('f'):
            j += 1
            continue
        elif key == ord('r'):
            j -= 1
    cv2.destroyAllWindows()
    print(j)
    k=j


if __name__ == '__main__':
    main(args.img_dir, args.model_dir, args.weight)