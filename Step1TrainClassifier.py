from model import get_testing_model
from model import get_classifier_attached_vgg
from keras.callbacks import TensorBoard
from keras import optimizers
import util
import time
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("-model", default="model.h5", help="path to original openpose h5 model file")
parse.add_argument("-data", default="train_classifier_layer_data", help="path to data folder")
parse.add_argument("-save", default="step1.h5", help="path to h5 file to save fine-tuned weights")
args = parse.parse_args()

tensorboard = TensorBoard(log_dir='step1_logs\{}'.format(time.time()))
num_iteration = 100


IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 1
CLASS_MODE = 'binary'

from_op = dict()
'''from_op['conv1_1'] = 'conv1_1'
from_op['conv1_2'] = 'conv1_2'
from_op['conv2_1'] = 'conv2_1'
from_op['conv2_2'] = 'conv2_2'
from_op['conv3_1'] = 'conv3_1'
from_op['conv3_2'] = 'conv3_2'
from_op['conv3_3'] = 'conv3_3'
from_op['conv3_4'] = 'conv3_4'
from_op['conv4_1'] = 'conv4_1'
from_op['conv4_2'] = 'conv4_2'
from_op['conv4_3_CPM'] = 'conv4_3_CPM'
from_op['conv4_4_CPM'] = 'conv4_4_CPM'
from_op['Mconv1_stage1_L1'] = 'Mconv1_stage1_L1'
'''
	
if __name__ == "__main__":
	# prepare training data
	train_data = util.prepare_data(args.data, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, CLASS_MODE)
	test_data = util.prepare_data(args.data, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, CLASS_MODE, shuffle=False)
	# load open pose trained weights
	open_pose_model = get_testing_model()
	open_pose_model.load_weights(args.model)
	
	# create dictionary of layers' name
	for layer in open_pose_model.layers:
		if layer.count_params() > 0:
			name = layer.name
			from_op[name] = name
	
	# create op10 model
	model = get_classifier_attached_vgg(5e-8)
	
	# transfer weights from trained open pose model to the new model
	for layer in model.layers:
		if layer.name in from_op:
			layer.set_weights(open_pose_model.get_layer(from_op[layer.name]).get_weights())
			layer.trainable = False
			print("Loaded layer: " + from_op[layer.name])
	
	for layer in model.layers:
		if layer.trainable == True and layer.count_params() > 0:
			print(layer.name)
	
	# define training mechanism
	model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=['accuracy'])
	
	# train classifier layers
	model.fit_generator(train_data, steps_per_epoch=train_data.n, epochs=num_iteration, callbacks=[tensorboard])
	
	#save model's weights to file
	model.save(args.save)
	#tensorboard.set_model(model)
	pred = model.predict_generator(test_data, steps=train_data.n)
	print(pred)
	score = model.evaluate_generator(test_data, steps=train_data.n)
	print(score)
