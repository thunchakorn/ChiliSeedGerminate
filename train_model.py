import pandas as pd
import numpy as np
import argparse
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import Xception

def focal_loss(gamma=2., alpha=2):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed

parser = argparse.ArgumentParser()
parser.add_argument("-train_dir", help = 'path to directory of train image', default = './seed_train')


def main(seed_train_path):
	tf.random.set_seed(999)
	EPOCHS = 1000
	img_size = 128

	base_model = Xception(input_shape=(img_size, img_size,3), weights='imagenet',include_top=False)

	base_model.trainable = True

	classifier = Sequential([GlobalAveragePooling2D(),
	                         Dense(512, activation='relu'),
	                         Dropout(0.5),
	                         Dense(256, activation='relu'),
	                         Dropout(0.5),
	                         Dense(3, activation='softmax')])

	model = Sequential()
	model.add(base_model)
	model.add(classifier)
	model.compile(optimizer= Adam(lr = 0.0001) ,loss=focal_loss(alpha = 2),metrics=['accuracy'])
	model.summary()

	train_datagen = ImageDataGenerator(brightness_range = (0.2,1),
                                   rescale = 1./255,
                                   rotation_range=90,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   fill_mode = 'constant',
                                   validation_split = 0.2)

	training_set = train_datagen.flow_from_directory(seed_train_path,
                                                 target_size = (img_size, img_size),
                                                 batch_size = 32,
                                                 class_mode='categorical',
                                                 subset='training')

	validate_set = train_datagen.flow_from_directory(seed_train_path,
                                                 target_size = (img_size, img_size),
                                                 batch_size = 128,
                                                 class_mode='categorical',
                                                 subset='validation',
                                                 shuffle = False)


	early_stop = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss')
	h = model.fit(training_set,
                    steps_per_epoch = training_set.n // training_set.batch_size,
                    epochs = EPOCHS,
                    validation_data = valid_set,
                    validation_steps = valid_set.n // valid_set.batch_size,
                    callbacks=[model_checkpoint_callback, early_stop] )

	plt.plot(h.history['loss'])
	plt.plot(h.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('loss')
	plt.xlabel('epoch') 
	plt.legend(['train','valid'], loc='upper left')
	plt.show()

	score = model.evaluate(valid_set, verbose = 1)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	if not os.path.exists('Model'):
		os.makedirs('Model')
	model.save('Model/trained_model.h5')


if __name__ == '__main__':
	main(args.train_dir)