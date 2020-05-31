import argparse
import os
from seed_utils import *
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-img_dir", help = 'path to directory of seed image', default = './seed_image')

args = parser.parse_args()


def main(img_dir):
    h_set = 512
    img_gen = read_gen(img_dir)
    germ_dict = {1:'neg', 2:'none', 3:'pos'}
    print('for labeling seeds, please press key board to label each seed as follow:\n', germ_dict)
    des_dir = 'seed_train'
    if not os.path.exists(des_dir):
        os.makedirs('seed_train')
        os.makedirs(os.path.join(des_dir, 'pos'))
        os.makedirs(os.path.join(des_dir, 'neg'))
        os.makedirs(os.path.join(des_dir, 'none'))
    j = 0
    
    for img, name in img_gen:
        im_h, im_w, _ = img.shape
        w_set = int(h_set*im_w/im_h)
        img_thres = thresholding(img, sat_thres=57, hue_low = 12, hue_high = 33, dilate_size= 7, open_size=3)
        cv2.imshow('mask', cv2.resize(img_thres, (w_set, h_set)))
        bboxes, area = find_bbox(img_thres, 64, 64, 0.1)
        bboxes = check_overlap(bboxes, area,thres = 0.2)
        img_copy = img.copy()
        for (i, (x,y,w,h)) in enumerate(bboxes):
            cv2.rectangle(img_copy, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img_copy, f'{i+1}',
                        (x,y),
                        color = (255, 0 ,0),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 1,
                        thickness = 4)

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
                cv2.imwrite(os.path.join(des_dir, 'neg','{0}_{1:03d}.png'.format(j,i)), img_crop)
                i += 1
            elif key == ord('2'):
                cv2.imwrite(os.path.join(des_dir, 'none','{0}_{1:03d}.png'.format(j,i)), img_crop)
                i += 1
            elif key == ord('3'):
                cv2.imwrite(os.path.join(des_dir, 'pos','{0}_{1:03d}.png'.format(j,i)), img_crop)
                i += 1
            elif key == 27:
                break
            elif key == ord('f'):
                break
            else:
                continue
        if key == 27:
            break
        j += 1
    cv2.destroyAllWindows()
    print(j)


if __name__ == '__main__':
    main(args.img_dir)