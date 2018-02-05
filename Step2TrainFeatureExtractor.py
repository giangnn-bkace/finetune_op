from keras.models import load_model
from keras.callbacks import TensorBoard
from keras import optimizers
import util
import time
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("-model", default="step1.h5", help="path to h5 model file of fine-tuned model by step 1 h5")
parse.add_argument("-data", default="train_feature_extracting_layer_data", help="path to data folder")
parse.add_argument("-test", default="test_data", help="path to test data folder")
parse.add_argument("-save", default="step2.h5", help="path to h5 file to save fine-tuned weights")
args = parse.parse_args()

tensorboard = TensorBoard(log_dir='step2_logs\{}'.format(time.time()))

num_iteration = 20
learning_rate = 1e-4

IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 1
CLASS_MODE = 'binary'


#lock = {'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3_CPM', 'conv4_4_CPM', 'dense_1', 'dense_2', 'dense_3', 'predictions'}
#open = ['Mconv1_stage1_L1', 'Mconv1_stage1_L2', 'Mconv2_stage1_L1', 'Mconv2_stage1_L2', 'Mconv3_stage1_L1', 'Mconv3_stage1_L2', 'Mconv1_stage4_L1', 'Mconv4_stage1_L2', 'Mconv5_stage1_L1', 'Mconv5_stage1_L2']
open = ['conv3_2']	
if __name__ == "__main__":
	# prepare training data
	train_data = util.prepare_data(args.data, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, CLASS_MODE)
	test_data = util.prepare_data(args.test, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, CLASS_MODE, shuffle=False)
	
	# load model
	model = load_model(args.model)
	# transfer weights from trained open pose model to the new model
	for layer in model.layers:
		layer.trainable = False
		if layer.name in open:
			layer.trainable = True
			print(layer.name)
	for layer in model.layers:
		if layer.trainable:
			print(layer.name)
	model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=learning_rate, momentum=0.9), metrics=['accuracy'])
	score = model.evaluate_generator(test_data, steps=test_data.n)
	print(score)
	# train classifier layers
	model.fit_generator(train_data, steps_per_epoch=train_data.n, epochs=num_iteration, callbacks=[tensorboard], validation_data=test_data, validation_steps=test_data.n)
	#save model's weights to file
	model.save(args.save)
	#tensorboard.set_model(model)
	pred = model.predict_generator(test_data, steps=test_data.n)
	print(pred)
	score = model.evaluate_generator(test_data, steps=test_data.n)
	print(score)