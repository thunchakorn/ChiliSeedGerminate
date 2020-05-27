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
	model = keras.models.load_model(model_dir, compile = False)
	if weight:
		latest = tf.train.latest_checkpoint(weight)
		model.load_weights(latest)
	if not os.path.exists('Outputs'):
		os.makedirs('Outputs')
	h_set = 512
	record = []
	germ_dict = {1:'neg', 2:'none', 3:'pos'}
	print('for unsure seeds, please press key board to label each seed as follow:\n', germ_dict)
	germ_dict = {0:'neg', 1:'none', 2:'pos'}
	start = 0
	img_gen = read_gen(img_dir)
	j = start
	for img, name in img_gen:
		im_h, im_w, _ = img.shape
		w_set = int(h_set*im_w/im_h)
		img_thres = thresholding(img, sat_thres=57, hue_low = 12, hue_high = 33,
								 dilate_size= 7, open_size=3)
		cv2.imwrite(f'Outputs/contour_{name}', img_thres)
		bboxes, area = contour_box(img_thres, 64, 64, 0.1)
		img_copy = img.copy()
		seed_array = seed2array(img_copy, bboxes, size = 128)
		model.predict(seed_array)
		classes, prob  = predict_seed(seed_array, model)
		bboxes, classes, prob = check_overlap_2(bboxes, area, classes, prob,thres = 0.25)
		valid_box_count = 0
		i = 1
		for (box, output, p, a) in zip(bboxes, classes, prob, area):
			x, y, w, h = box
			check, min_dif_p = prob_check(p, 0.15)
			if check:
				cv2.imshow('unsure_big', cv2.resize(img_copy, (img_copy.shape[1]//3, img_copy.shape[0]//3)))
				cv2.imshow('unsure', img_copy[y:y+h, x:x+w])
				key = cv2.waitKey(0)
				while key not in [ord('1'), ord('2'), ord('3')]:
					key = cv2.waitKey(0)
				if key == ord('1'):
					output = 0
				elif key == ord('2'):
					output = 1
				elif key == ord('3'):
					output = 2
			if output == 0:
				cv2.putText(img_copy, f'{i}',
				        (x, y),
				        color = (0, 0 ,255),
				        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
				        fontScale = 1,
				        thickness = 2)
				cv2.rectangle(img_copy, (x,y), (x+w, y+h), (0,0,255), 2)
				valid_box_count += 1
				i += 1
			elif output == 2:
				cv2.putText(img_copy, f'{i}',
				        (x, y),
				        color = (0, 255 ,0),
				        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
				        fontScale = 1,
				        thickness = 2)
				cv2.rectangle(img_copy, (x,y), (x+w, y+h), (0,255,0), 2)
				valid_box_count += 1
				i += 1
			else:
				pass
				# cv2.rectangle(img_copy, (x,y), (x+w, y+h), (0, 255 ,255), 2)
				# for k, p_i in enumerate(p):
				# 	cv2.putText(img_copy, '{:.2f}'.format(p_i),
				# 		(x-10, y-30*k),
				# 		color = (0, 255 ,255),
				# 		fontFace = cv2.FONT_HERSHEY_SIMPLEX,
				# 		fontScale = 1,
				# 		thickness = 2)

		string = f'''{name} All:{valid_box_count} pos:{(classes == 2).sum()}
		neg:{(classes == 0).sum()}'''
		y0, dy = 50, 50
		for i, line in enumerate(string.split()):
			y = y0 + i*dy
			cv2.putText(img_copy, line,(50,y),
			            color = (0,255,255),
			            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
			            fontScale = 1,
			            thickness = 3)
	    
		cv2.imwrite(f'Outputs/output_{name}', img_copy)
		r = (f'{name}', valid_box_count, (classes == 2).sum(), (classes == 0).sum())
		record.append(r)
	    
	cv2.destroyAllWindows()
	record = pd.DataFrame(record, columns = ['image', 'all', 'pos', 'neg'])
	record.to_csv('Outputs/record.csv')

if __name__ == '__main__':
	main(args.img_dir, args.model_dir, args.weight)