from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, clone_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, Adagrad
from keras import regularizers
from keras.utils import np_utils
import keras.backend as K
import keras
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report,jaccard_similarity_score
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score
import tensorflow as tf
#from datasets import get_dataset


import sys
import os
import numpy as np
import argparse
from os.path import isfile, join
from random import randint, shuffle
import time
import glob
import cv2

import deb
from keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy, sparse_accuracy_ignoring_last_label
from keras.initializers import RandomNormal

from densnet import DenseNetFCN


ap = argparse.ArgumentParser()
ap.add_argument('-s', '--source_weights', required=False, help="Path to weights file to load source model for training classification/adaptation")
ap.add_argument('-e', '--start_epoch', type=int,default=1, required=False, help="Epoch to begin training source model from")
ap.add_argument('-n', '--discriminator_epochs', type=int, default=10000, help="Max number of steps to train discriminator")
ap.add_argument('-l', '--lr', type=float, default=0.0001, help="Initial Learning Rate")
ap.add_argument('-f', '--train_discriminator', action='store_true', help="Train discriminator model (if TRUE) vs Train source classifier")
ap.add_argument('-a', '--discriminator_weights', help="Path to weights file to load discriminator")
ap.add_argument('-t', '--eval_source_classifier', default=None, help="Path to source classifier model to test/evaluate")
ap.add_argument('-d', '--eval_target_classifier', default=None, help="Path to target discriminator model to test/evaluate")
ap.add_argument('-sds', '--source_dataset', default="para", help="Path to source dataset")
ap.add_argument('-tds', '--target_dataset', default="acre", help="Path to target dataset")
ap.add_argument('-ting', '--testing', default=1, help="Path to target dataset")
ap.add_argument('-advval', '--adversarial_validating', type=int,default=0, help="Path to target dataset")
ap.add_argument('-sval', '--source_validating', type=int,default=0, help="0 for no validation, 1 for source  validation, 2 for target validation")

ap.add_argument('-ws', '--weights_save', type=bool,default=True, help="Save weights during source training")
ap.add_argument('-w', '--window_len', type=int,default=32, help="Save weights during source training")
ap.add_argument('-cln', '--class_n', type=int,default=3, help="Class number. 3 for wildfire, 3 for vaihingen")
ap.add_argument('-c', '--channel_n', type=int,default=6, help="Class number. 3 for wildfire, 3 for vaihingen")
ap.add_argument('-ibcknd', '--ignore_bcknd', type=int,default=1, help="Class number. 3 for wildfire, 3 for vaihingen")

ap.add_argument('-em', '--encoder_mode',default='basic', help="Gen mode. basic or densenet")

ap.add_argument('-tm', '--testing_mode',default=None, help="Testing mode can be 'for_loop'")
ap.add_argument('-vm', '--validating_mode',default=None, help="Validating mode can be 'loss'")

args = ap.parse_args()
deb.prints(args.testing_mode)
deb.prints(args.ignore_bcknd)
deb.prints(args.source_validating)
deb.prints(args.encoder_mode)
t0 = time.time()
class_n=args.class_n

conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02) # for batch normalization
channel_axis=-1 # Tensorflow backend
def conv2d(f, *a, **k):
	return Conv2D(f, kernel_initializer = conv_init, *a, **k)
def batchnorm():
	return BatchNormalization(momentum=0.9, axis=channel_axis, epsilon=1.01e-5,
								   gamma_initializer = gamma_init)    
def G(fn_generate, X):
	r = np.array([fn_generate([X[i:i+1]]) for i in range(X.shape[0])])
	#return r.swapaxes(0,1)[:,:,0] 
	return r 

def stats_print(x):
	print(np.min(x),np.max(x),np.average(x),x.dtype)
def batch_label_to_one_hot(im,class_n=3):
		#class_n=np.unique(im).shape[0]
		#deb.prints(class_n)
		im_one_hot=np.zeros((im.shape[0],im.shape[1],im.shape[2],class_n))
		#print(im_one_hot.shape)
		#print(im.shape)
		for clss in range(0,class_n):
			im_one_hot[:,:,:,clss][im[:,:,:]==clss]=1
		return im_one_hot
def label_to_one_hot(im,class_n=3):
		#class_n=np.unique(im).shape[0]
		#deb.prints(class_n)
		im_one_hot=np.zeros((im.shape[0],im.shape[1],class_n))
		#print(im_one_hot.shape)
		#print(im.shape)
		for clss in range(0,class_n):
			im_one_hot[:,:,clss][im[:,:]==clss]=1
		return im_one_hot

def domain_data_load(domain,validating=0,all_test=False,
	ignore_bcknd=1,testing_mode=None,class_n=3):
	deb.prints(domain['dataset'])
	path='../wildfire_fcn/src/patch_extract2/compact/'+domain['dataset']+'/'
	domain['train']={}
	domain['test']={}
	domain['train']['in']=np.load(path+"train_im.npy")
	domain['train']['label']=np.load(path+"train_label.npy")
	domain['test']['in']=np.load(path+"test_im.npy")
	domain['test']['label']=np.load(path+"test_label.npy")
	
	if all_test==True:
		domain['train']['in']=domain['test']['in'].copy()
		domain['train']['label']=domain['test']['label'].copy()
	if testing_mode=='for_loop':
		try:
			domain['test']['full_in']=np.load(path+"test_full_im.npy")
			domain['test']['full_label']=np.load(path+"test_full_label.npy")
			domain['test']['full_mask']=np.load(path+"test_full_mask.npy")
			domain['test']['full_label']=label_to_one_hot(domain['test']['full_label'],class_n=class_n)
			
			deb.prints(domain['test']['full_mask'].shape)
		except:
			print("No full images...")
			
	deb.prints(domain['train']['in'].shape)
	deb.prints(domain['train']['label'].shape)


	deb.prints(np.unique(domain['train']['label'],return_counts=True))
	deb.prints(np.unique(domain['test']['label'],return_counts=True))
	
	deb.prints(np.unique(domain['train']['label'],
		return_counts=True))
	domain['train']['label']=batch_label_to_one_hot(domain['train']['label'],class_n=class_n)
	domain['test']['label']=batch_label_to_one_hot(domain['test']['label'],class_n=class_n)
	deb.prints(domain['train']['label'].shape)

	if validating==1:
		domain['val']={}
		domain['val']['in']=np.load(path+"val_im.npy")
		domain['val']['label']=np.load(path+"val_label.npy")
		deb.prints(np.unique(domain['val']['label'],return_counts=True))	
		domain['val']['label']=batch_label_to_one_hot(domain['val']['label'],class_n=class_n)
	
		domain['val']['full_in']=np.load(path+"val_full_im.npy")
		domain['val']['full_label']=np.load(path+"val_full_label.npy")
		domain['val']['full_mask']=np.load(path+"val_full_mask.npy")
		domain['val']['full_label']=label_to_one_hot(domain['val']['full_label'],class_n=class_n)
			
			
	deb.prints(domain['train']['label'].shape[0])

	print("Estimating weights...")
	if domain['train']['label'].shape[0]>0:
		print("Class n° for weight estimation",class_n)
		domain['loss_weights']=loss_weights_estimate(domain,ignore_bcknd=ignore_bcknd,class_n=class_n)
	else:
		print("No training samples to estimate domain weights")
	return domain


def read_image(fn):
	img = np.load(fn)    
	return img   
def minibatch(data, batchsize):
	length = len(data)
	epoch = i = 0
	tmpsize = None    
	while True:
		size = tmpsize if tmpsize else batchsize
		if i+size > length:
			shuffle(data)
			i = 0
			epoch+=1        
		rtn = [read_image(data[j]) for j in range(i,i+size)] # Pick a batch
		i+=size
		tmpsize = yield epoch, np.float32(rtn) 

def folder_load(paths):
	files=[]
	deb.prints(len(paths))
	for path in paths:
		#print(path)
		files.append(np.load(path))
	return np.asarray(files)

#=============== METRICS CALCULATION ====================#
def ims_flatten(ims):
	return np.reshape(ims,(np.prod(ims.shape[0:-1]),ims.shape[-1])).astype(np.float32)

def probabilities_to_one_hot(vals):
	out=np.zeros_like(vals)
	out[np.arange(len(vals)), vals.argmax(1)] = 1
	return out
def metrics_get(data,ignore_bcknd=1,debug=1,
	only_one=None,mask=None,mask_id=None): #requires batch['prediction'],batch['label']
	
	# ==========================IMGS FLATTEN ==========================================#
	predictions = ims_flatten(data['prediction'])
	predictions=probabilities_to_one_hot(predictions)
	labels = ims_flatten(data['label']) #(self.batch['test']['size']*self.patch_len*self.patch_len,self.class_n

	deb.prints(predictions.shape)
	predictions = predictions.argmax(axis=1)
	labels = labels.argmax(axis=1)
	
	deb.prints(np.unique(labels,return_counts=True))   

	if mask is not None:
		mask=np.reshape(mask,-1)
		deb.prints(np.unique(mask,return_counts=True))

	if ignore_bcknd==1:
		print("Ignoring background...")
		predictions=predictions[labels>0]
		if mask is not None:
			mask=mask[labels>0]
		labels=labels[labels>0]
	
	if mask is not None:
			
		deb.prints(labels.shape)
		
		print("Metrics get: Applying mask...")
		predictions=predictions[mask==mask_id]
		labels=labels[mask==mask_id]

	deb.prints(np.unique(labels,return_counts=True))   

	print("predictions",predictions.shape)

	print(np.unique(predictions,return_counts=True))
	print(np.unique(labels,return_counts=True))

	print(predictions.shape,predictions.dtype)
	print(labels.shape,labels.dtype)

	metrics={}
	metrics['confusion_matrix']=confusion_matrix(labels,predictions)
	
	if only_one=='average_acc':
		acc=metrics['confusion_matrix'].diagonal()/np.sum(metrics['confusion_matrix'],axis=1)
		acc=acc[~np.isnan(acc)]
		metrics['average_acc']=np.average(acc)
		print("Acc",acc)
		print("AA",metrics['average_acc'])
	elif only_one=='f1_score_avg':
		metrics['f1_score']=f1_score(labels,predictions,average=None)
		metrics['f1_score_avg']=np.average(metrics['f1_score'])
		print("F1",metrics['f1_score'])
		print("F1_avg",metrics['f1_score_avg'])
	elif only_one=='iou':
		metrics['iou']=jaccard_similarity_score(labels,predictions)
		print("IOU",metrics['iou'])
	elif only_one=='kappa':
		metrics['kappa']=cohen_kappa_score(labels,predictions)
	elif only_one=='oa_aa':
		acc=metrics['confusion_matrix'].diagonal()/np.sum(metrics['confusion_matrix'],axis=1)
		acc=acc[~np.isnan(acc)]
		metrics['average_acc']=np.average(acc)
		
		metrics['overall_acc']=accuracy_score(labels,predictions)
		print("Acc",acc)
		print("AA",metrics['average_acc'])
		metrics['oa_aa']=np.average((metrics['overall_acc'],metrics['average_acc']))
		deb.prints(metrics['oa_aa'])
	else:
		metrics['f1_score']=f1_score(labels,predictions,average=None)
		metrics['f1_score_avg']=np.average(metrics['f1_score'])
		#metrics['f1_score_weighted']=f1_score(labels,predictions,average='weighted')
				
		metrics['overall_acc']=accuracy_score(labels,predictions)
		metrics['per_class_acc']=(metrics['confusion_matrix'].astype('float') / metrics['confusion_matrix'].sum(axis=1)[:, np.newaxis]).diagonal()
		print("acc",metrics['per_class_acc'])

		print(metrics['confusion_matrix'].sum(axis=1)[:, np.newaxis].diagonal())
		print(metrics['confusion_matrix'].diagonal())
		print(np.sum(metrics['confusion_matrix'],axis=1))
		acc=metrics['confusion_matrix'].diagonal()/np.sum(metrics['confusion_matrix'],axis=1)
		acc=acc[~np.isnan(acc)]
		metrics['average_acc']=np.average(acc)
		#metrics['iou']=jaccard_similarity_score(labels,predictions)
		metrics['kappa']=cohen_kappa_score(labels,predictions)
		metrics['precision']=precision_score(labels,predictions,average=None)
		metrics['recall']=recall_score(labels,predictions,average=None)
		print("kappa",metrics['kappa'])
		
		#print("IOU",metrics['iou'])
		print("Acc",acc)
		print("AA",metrics['average_acc'])
		print("OA",np.sum(metrics['confusion_matrix'].diagonal())/np.sum(metrics['confusion_matrix']))
		print("F1",metrics['f1_score'])
		print("F1_avg",metrics['f1_score_avg'])
		print("Precision",metrics['precision'],np.average(metrics['precision']))
		print("Recall",metrics['recall'],np.average(metrics['recall']))	
	print(metrics['confusion_matrix'])

	return metrics


def loss_weights_estimate(data,class_n,ignore_bcknd=1):
		unique,count=np.unique(data['train']['label'].argmax(axis=3),return_counts=True)
		if ignore_bcknd==1 and np.any(np.array(unique)==0):
			unique=unique[1:] # No bcknd
			count=count[1:].astype(np.float32)
			deb.prints(count)
		else:
			count=count.astype(np.float32)
		weights_from_unique=np.max(count)/count
		deb.prints(weights_from_unique)
		deb.prints(np.max(count))
		deb.prints(count)
		deb.prints(unique)
		loss_weights=np.zeros(class_n)
		for clss in range(0,class_n): # class 0 is bcknd. Leave it in 0
			
			if clss in unique:
				loss_weights[clss]=weights_from_unique[unique==clss]
			else:
				loss_weights[clss]=0
		deb.prints(loss_weights)

		#loss_weights[1:]=1
		deb.prints(loss_weights.shape)
		return loss_weights
class ADDA():
	def __init__(self, lr, window_len=32, channels=6, 
		class_n=3, encoder_mode='basic'):
		# Input shape
		self.encoder_mode=encoder_mode
		self.img_rows = window_len
		self.img_cols = window_len
		self.channels = channels
		self.img_shape = (self.img_rows, self.img_cols, self.channels)
		
		self.src_flag = False
		self.disc_flag = False
		
		self.discriminator_decay_rate = 3 #iterations
		self.discriminator_decay_factor = 0.5
		if self.encoder_mode=='basic':
			self.src_optimizer = Adam(lr, 0.5)
			self.tgt_optimizer = Adam(lr, 0.5)
		else:#		elif self.encoder_mode=='densenet':
			self.src_optimizer = Adagrad(0.01)
			self.tgt_optimizer = Adagrad(0.01)
		self.class_n=class_n
		self.source_weights_path='results/'

	def define_source_encoder(self, weights=None,model_return=False):
	
		#self.source_encoder = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=self.img_shape, pooling=None, classes=10)
		count=0
		conv2d_prefix="conv2d_e"
		self.source_encoder = Sequential()
		inp = Input(shape=self.img_shape)
		#self.encoder_mode=2
		if self.encoder_mode=='basic':
			
			x = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.img_shape, padding='same', name=conv2d_prefix+str(count))(inp)
			#x = batchnorm()(x, training=1)  
			##x = LeakyReLU(alpha=0.2)(x)
			count+=1
			x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name=conv2d_prefix+str(count))(x)
			#x = batchnorm()(x, training=1)  
			##x = LeakyReLU(alpha=0.2)(x)
			count+=1
			#x = MaxPooling2D(pool_size=(2, 2))(x)
			#x = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(x)
			x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', \
				name=conv2d_prefix+str(count))(x)
			#x = batchnorm()(x, training=1)  
			##x = LeakyReLU(alpha=0.2)(x)
			#x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
			#x = MaxPooling2D(pool_size=(2, 2))(x)
			#x = Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=self.img_shape, padding='same')(inp)
			#x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
			#x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
			#x = MaxPooling2D(pool_size=(2, 2))(x)
		else:# self.source_encoder=='densenet':
			x = DenseNetFCN(self.img_shape, nb_dense_block=2, growth_rate=16, dropout_rate=0.2,
					nb_layers_per_block=2, upsampling_type='deconv', classes=self.class_n, 
					activation='softmax', batchsize=32,input_tensor=inp,
					include_top=False)
		if model_return==False:
			self.source_encoder = Model(inputs=(inp), outputs=(x))
			
			self.src_flag = True
			
			if weights is not None:
				self.source_encoder.load_weights(weights, by_name=True)
		if model_return==True:
			return Model(inputs=(inp), outputs=(x))


	def define_target_encoder(self, weights=None):
		
		if not self.src_flag:
			self.define_source_encoder()
		
		with tf.device('/cpu:0'):
			self.target_encoder = clone_model(self.source_encoder)
		
		if weights is not None:
			self.target_encoder.load_weights(weights, by_name=True)
		
	def get_source_classifier(self, model=None, shape=None,weights=None, atomic=False):
		print("[@get_source_classifier]")
		#nb_classes=2
		# If atomic=False, returns both encoder+classifier. If true,
		# returns only the classifier 
		weight_decay=1E-4
		count=0
		conv2d_prefix="conv2d_c"
		deb.prints(self.class_n)

		if atomic==True:
			inp = Input(shape=shape)
			x = Conv2D(self.class_n, (1, 1), activation='softmax', padding='same', kernel_regularizer=l2(weight_decay),
						  use_bias=False, name=conv2d_prefix+str(count))(inp)

		else:			
			x = Conv2D(self.class_n, (1, 1), activation='softmax', padding='same', kernel_regularizer=l2(weight_decay),
						  use_bias=False, name=conv2d_prefix+str(count))(model.output)
		print(0.3)
		if atomic==True:
			source_classifier_model = Model(inputs=(inp),outputs=(x))
		else:
			source_classifier_model = Model(inputs=(model.input), outputs=(x))
		
		if weights is not None:
			print("Loading source weights")
			source_classifier_model.load_weights(weights,by_name=True)
	
		print(0.4)
		source_classifier_model.summary()
		print(0.5)
		return source_classifier_model

	def define_discriminator(self, shape, model_return=False):
		
		inp = Input(shape=shape)
		
		x = Flatten()(inp)
		x = Dense(128, activation=LeakyReLU(alpha=0.3), kernel_regularizer=regularizers.l2(0.01), name='discriminator1')(x)
		
		x = Dense(2, activation='sigmoid', name='discriminator2')(x)
		
		if model_return==False:
			self.disc_flag = True
			self.discriminator_model = Model(inputs=(inp), outputs=(x), name='discriminator')
		else:
			return Model(inputs=(inp), outputs=(x), name='discriminator')
	def define_discriminator(self,nc_in, ndf, max_layers=3, use_sigmoid=True):
		"""DCGAN_D(nc, ndf, max_layers=3)
		   nc: channels
		   ndf: filters of the first layer
		   max_layers: max hidden layers
		"""    
		input_a = Input(shape=(None, None, nc_in)) # Here I might put 128
		_ = input_a
		_ = conv2d(ndf, kernel_size=4, strides=2, padding="same", name = 'First') (_)
		_ = LeakyReLU(alpha=0.2)(_)
		
		for layer in range(1, max_layers):        
			out_feat = ndf * min(2**layer, 8)
			_ = conv2d(out_feat, kernel_size=4, strides=2, padding="same", 
					   use_bias=False, name = 'pyramid.{0}'.format(layer)             
							) (_)
			_ = batchnorm()(_, training=1)        
			_ = LeakyReLU(alpha=0.2)(_)
		
		out_feat = ndf*min(2**max_layers, 8)
		_ = ZeroPadding2D(1)(_)
		_ = conv2d(out_feat, kernel_size=4,  use_bias=False, name = 'pyramid_last') (_)
		_ = batchnorm()(_, training=1)
		_ = LeakyReLU(alpha=0.2)(_)
		
		# final layer
		_ = ZeroPadding2D(1)(_)
		_ = conv2d(1, kernel_size=4, name = 'final'.format(out_feat, 1), 
				   activation = "sigmoid" if use_sigmoid else None) (_)    
		return Model(inputs=[input_a], outputs=_)


	def tensorboard_log(self, callback, names, logs, batch_no):
		
		for name, value in zip(names, logs):
			summary = tf.Summary()
			summary_value = summary.value.add()
			summary_value.simple_value = value
			summary_value.tag = name
			callback.writer.add_summary(summary, batch_no)
			callback.writer.flush()
	
	def get_discriminator(self, model, weights=None):
		
		if not self.disc_flag:
			self.define_discriminator(model.output_shape[1:])
		
		disc = Model(inputs=(model.input), outputs=(self.discriminator_model(model.output)))
		
		if weights is not None:
			disc.load_weights(weights, by_name=True)
		
		return disc

	def source_model_train(self, model, data, batch_size=6, epochs=100, \
		save_interval=1, start_epoch=0,training=True,testing=1,
		weights_save=True,validating=0,patience=150,
		validation_data=None,ignore_bcknd=1,
		testing_mode=None, early_validating=True):
		self.training=training
		deb.prints(validating)
		batch_size=6
		if validating==1:
			print("Loading val data...")
			data['val']=validation_data.copy()
			deb.prints(data['val']['full_in'].shape)
		self.weights_save=weights_save
		# Define source data generator
		#batch_generator={}
		#batch_generator['in']=minibatch(data['in'], batch_size)
		#batch_generator['label']=minibatch(data['label'], batch_size)
		if self.training==True:
			loss_weighted=weighted_categorical_crossentropy(data['loss_weights'])
		else:
			loss_weighted='categorical_crossentropy'
		#model.compile(loss='categorical_crossentropy', optimizer=self.src_optimizer, metrics=['accuracy'])
		model.compile(loss=loss_weighted, optimizer=self.src_optimizer, metrics=['accuracy'])
		if not os.path.isdir('data'):
			os.mkdir('data')
		

		# saver = keras.callbacks.ModelCheckpoint('data/svhn_encoder_{epoch:02d}.hdf5', 
		#                                 monitor='val_loss', 
		#                                 verbose=1, 
		#                                 save_best_only=False, 
		#                                 save_weights_only=True, 
		#                                 mode='auto', 
		#                                 period=save_interval)

		# scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=10, verbose=0, mode='min')

		# if not os.path.isdir('data/tensorboard'):
		#     os.mkdir('data/tensorboard')
	
		# visualizer = keras.callbacks.TensorBoard(log_dir=os.path.join('data/tensorboard'), 
		#                                     histogram_freq=0, 
		#                                     write_graph=True, 
		#                                     write_images=False)
		
		
		batch={}
		count=0
		errSource=0
		epoch=0
		niter=100
		diplay_iters=200

		batch = {'train': {}, 'test': {}, 'val':{}}
		self.batch={'train':{},'test':{}, 'val':{}}
		self.metrics={'train':{},'test':{}, 'val':{}}
		self.batch['train']['size']=batch_size
		self.batch['test']['size']=5

		
		self.batch['train']['n'] = data['train']['in'].shape[0] // self.batch['train']['size']
		self.batch['train']['flag']=0

		self.batch['test']['n'] = data['test']['in'].shape[0] // self.batch['test']['size']
		self.batch['test']['flag']=0

		if validating==1:
			self.batch['val']['size']=batch_size
			self.batch['val']['n'] = data['val']['in'].shape[0] // self.batch['val']['size']
			self.batch['val']['flag']=0

			metric_most_important='loss'

			self.early_stop={'best':0,
					'count':0,
					'signal':False,
					'patience':patience,
					'minimize':False}
			deb.prints(self.batch['val']['n'])
			if metric_most_important=='loss':
				self.early_stop['minimize']=True
				self.early_stop['best']=9999
			print("Initializing validation...")

		deb.prints(self.batch['train']['n'])
		deb.prints(self.batch['test']['n'])

		if self.training==False:
			niter=1


		for epoch in range(niter):
			early_epoch=0
			print("Starting epoch...")
			
			self.metrics['test']['loss'] = np.zeros((1, 2))
			if training==True:
				self.metrics['train']['loss'] = np.zeros((1, 2))
				#=============================TRAIN LOOP=========================================#
				for batch_id in range(0, self.batch['train']['n']):
					
					idx0 = batch_id*self.batch['train']['size']
					idx1 = (batch_id+1)*self.batch['train']['size']

					batch['train']['in'] = data['train']['in'][idx0:idx1]
					batch['train']['label'] = data['train']['label'][idx0:idx1]

					self.metrics['train']['loss'] += model.train_on_batch(
						batch['train']['in'], batch['train']['label'])      # Accumulated epoch
					# ========= EARLY VALIDATION ==================
					if data['dataset']=='para' or data['dataset']=='acre':
						batch_interval=20
					else:
						batch_interval=2
					if early_validating==True and batch_id%batch_interval==0:
						deb.prints(batch_id)
						deb.prints(self.early_stop['best'])
						#metric_most_important='average_acc'
						#metric_most_important='loss'
						metrics_val,_=self.test_loop_source(data['val'],
							self.batch['val'],model,
							self.metrics['val'],
							ignore_bcknd=ignore_bcknd,
							metric_only_one=metric_most_important)						
						#metrics_val,_=self.test_loop_for(
						#	data['val'],
						#	self.batch['val'],
						#	model=model,
						#	ignore_bcknd=ignore_bcknd,
						#	mask_id=3,
						#	metric_only_one=metric_most_important)
						self.early_stop_check(metrics_val,epoch,
							most_important=metric_most_important,
							minimize=self.early_stop['minimize'])

						if self.early_stop['best_updated']==True:
							print("Saving weights...")
							#self.early_stop['best_predictions']=data['test']['prediction']
							model.save_weights('results_val/source_weights_'+data['dataset']+'.h5')
						if self.early_stop["signal"]==True:
							print("EARLY STOP EPOCH",epoch)
							assert 1==2


				# Average epoch loss
				self.metrics['train']['loss'] /= self.batch['train']['n']

				print(self.metrics['train']['loss'])


				# ================= SAVE WEIGHTS ===============
				if self.weights_save==True:
					model.save_weights(self.source_weights_path+'source_weights_'+data['dataset']+'.h5')
			else:
				print("Training was skipped")
			# ============ VAL LOOP ============================== #

			if validating==1 and early_validating==False:
				print("Validating...")
				most_important='average_acc'
				metrics_val,_=self.test_loop_for(
					data['val'],
					self.batch['val'],
					model=model,
					ignore_bcknd=ignore_bcknd,
					mask_id=3,
					metric_only_one=most_important) #val id

				#metrics_val,_=self.test_loop_source(data['val'],
				#	self.batch['val'],model,
				#	self.metrics['val'],
				#	ignore_bcknd=ignore_bcknd)
				#most_important='loss'

				self.early_stop_check(metrics_val,epoch,
					most_important=most_important)

			#==========================TEST LOOP================================================#
		
			if testing==1:
				print("Testing...")
				if testing_mode=='for_loop':
					metrics,data['test']['prediction']= \
						self.test_loop_for(data['test'],
							self.batch['test'],
							model=model,
							ignore_bcknd=ignore_bcknd)

				else:	
					deb.prints(ignore_bcknd)
					metrics,data['test']['prediction']=self.test_loop_source(
						data['test'],self.batch['test'],
						model,self.metrics['test'],
						ignore_bcknd=ignore_bcknd)

			else:
				print("Testing was skipped")
			
			print("Epoch={}".format(epoch)) 
			

			# ========================== EARLY STOP ===================
			if validating==1:
				if self.early_stop['best_updated']==True:
					print("Saving weights...")
					#self.early_stop['best_predictions']=data['test']['prediction']
					model.save_weights('results_val/source_weights_'+data['dataset']+'.h5')
				deb.prints(data['val']['full_in'].shape)
					
				print(self.early_stop['signal'])
				if self.early_stop["signal"]==True:
					print("EARLY STOP EPOCH",epoch)
					np.save("result_source/prediction.npy",data['test']['prediction'])
					np.save("result_source/labels.npy",data['test']['label'])
					break
	def layer_id_from_name_get(self,model,name):
		index = None
		for idx, layer in enumerate(model.layers):
			if layer.name == name:
				index = idx
				break
		return index		
	def early_stop_check(self,metrics,epoch,
		most_important='overall_acc',
		minimize=False):
		deb.prints(metrics[most_important])

		if minimize==False:
			condition=metrics[most_important]>self.early_stop['best']
		else:
			condition=metrics[most_important]<self.early_stop['best']		
		if condition and self.early_stop["signal"]==False:
			self.early_stop['best']=metrics[most_important]
			self.early_stop['count']=0
			print("Best metric updated")
			self.early_stop['best_updated']=True
			#data.im_reconstruct(subset='test',mode='prediction')
		else:
			self.early_stop['best_updated']=False
			self.early_stop['count']+=1
			deb.prints(self.early_stop['count'])
			if self.early_stop["count"]>=self.early_stop["patience"]:
				self.early_stop["signal"]=True
	def test_loop_source(self,data,batch,model,metrics,
		metric_only_one=None,batch_test_stats=True,
		ignore_bcknd=1):		

		metrics['loss']=np.zeros((1,2))
		data['prediction']=np.zeros_like(data['label'])

		if data['in'].shape[0] % batch['size'] != 0 and batch['flag']==0:
			batch['n'] += 1
			batch['flag']=1
		deb.prints(batch['n'])

		for batch_id in range(0, batch['n']):
			idx0 = batch_id*batch['size']
			if batch_id!=batch['n']-1:
				idx1 = (batch_id+1)*batch['size']
			else:
				idx1 = data['in'].shape[0]

			#batch['in'] = data['in'][idx0:idx1]
			#batch['label'] = data['label'][idx0:idx1]

			if batch_test_stats:
				metrics['loss'] += model.test_on_batch(
					data['in'][idx0:idx1], 
					data['label'][idx0:idx1])        # Accumulated epoch

			data['prediction'][idx0:idx1]=model.predict(
				data['in'][idx0:idx1],
				batch_size=batch['size'])
		metrics=metrics_get(data,debug=1,
			only_one=metric_only_one,
			ignore_bcknd=ignore_bcknd)
		return metrics,data['prediction']
	def test_loop(self,data,batch,fn_classify,G,
		metric_only_one=None,ignore_bcknd=1,
		loss=None):		

		data['prediction']=np.zeros_like(data['label'])
		deb.prints(data['prediction'].shape)
		batch['n'] = data['in'].shape[0] // batch['size']

		if data['in'].shape[0] % batch['size'] != 0:
			batch['n'] += 1

		deb.prints(batch['n'])

		for batch_id in range(0, batch['n']):
			idx0 = batch_id*batch['size']
			if batch_id!=batch['n']-1:
				idx1 = (batch_id+1)*batch['size']
			else:
				idx1 = data['in'].shape[0]
			data['prediction'][idx0:idx1]=np.squeeze(G(
				fn_classify, data['in'][idx0:idx1]))
		deb.prints(data['label'].shape)		
		if metric_only_one=='loss':
			metrics={'loss':loss}
		else:
			metrics=metrics_get(data,debug=1,
				only_one=metric_only_one,
				ignore_bcknd=ignore_bcknd)
			metrics['loss']=loss


		#self.early_stop_check(metrics_val,epoch,most_important='f1_score')
		return metrics,data['prediction']

	def test_loop_for(self,data,batch,
		model=None,metric_only_one=None,
		batch_test_stats=True,
		ignore_bcknd=1,fn_classify=None,G=None,
		window=32,overlap=0,mask_id=2,loss=None): #mask id defaults to test
		
		fname=sys._getframe().f_code.co_name

		im=data['full_in'].copy()
		#del data['full_in']
		mask=data['full_mask'].copy()
		deb.prints(np.unique(mask,return_counts=True))
		#del data['full_mask']
		label=data['full_label'].copy()
		#del data['full_label']

		deb.prints(window,fname)
		deb.prints(overlap,fname)
		print("STARTED PATCH EXTRACTION")
		#window= 256
		#overlap= 200
		patches_get={}
		h, w, channels = im.shape

		gridx = range(0, w - window, window - overlap)
		gridx = np.hstack((gridx, w - window))

		gridy = range(0, h - window, window - overlap)
		gridy = np.hstack((gridy, h - window))
		deb.prints(gridx.shape)
		deb.prints(gridy.shape)
		
		counter=0
		out={}
		out['prediction']=np.zeros_like(label).astype(np.float32)
		if model is not None:
			loss=0
		#======================== START IMG LOOP ==================================#
		for i in range(len(gridx)):
			for j in range(len(gridy)):
				#if counter % 10000000 == 0:
				#	deb.prints(counter,fname)
				xx = gridx[i]
				yy = gridy[j]
				patch = im[yy: yy + window, xx: xx + window,:]
				label_patch = label[yy: yy + window, xx: xx + window]
				out_patch=out['prediction'][yy: yy + window, xx: xx + window,:]
				mask_patch = mask[yy: yy + window, xx: xx + window]
				if np.all(mask_patch!=mask_id):
					continue
				if model is not None:
					# To-do: Prediction from more than 1
					prediction=model.predict(
						np.expand_dims(patch,axis=0),
						batch_size=1)
					loss+=model.test_on_batch(np.expand_dims(patch,axis=0),
						np.expand_dims(label_patch,axis=0))[0]
				else:
					prediction=G(fn_classify,
						np.expand_dims(patch,axis=0))

				prediction=np.squeeze(prediction)
					#deb.prints(prediction.shape)
				#deb.prints(prediction.shape)
				#deb.prints(prediction.dtype)

				prediction[out_patch!=0]=(prediction[out_patch!=0]+out_patch[out_patch!=0])/2
				out['prediction'][yy: yy + window, xx: xx + window,:]=np.squeeze(prediction)

				#	np.squeeze(prediction).argmax(axis=2).astype(np.uint8)

				counter=counter+1
		loss/=counter
#					out[:,yy: yy + window, xx: xx + window,:]					

		out['label']=label.copy()
		deb.prints(out['label'].shape)
		del label
		if metric_only_one=='loss':
			metrics={'loss':loss}
		else:
			metrics=metrics_get(out,debug=1,
				only_one=metric_only_one,
				ignore_bcknd=ignore_bcknd,
				mask=mask,mask_id=mask_id)
			metrics['loss']=loss
		return metrics,out['prediction']
	def discriminator_train(self, source,target, 
		source_weights=None, src_discriminator=None, 
		tgt_discriminator=None, epochs=2000, batch_size=6, 
		save_interval=1, start_epoch=0, num_batches=100,
		target_validating=1, patience=150, 
		early_validating=True, ignore_bcknd=1,
		testing_mode=None):   
		
		use_lsgan = True
		lrD = 2e-4
		#lrG = 2e-4
		lrG = 2e-4


		#lrD = lrG = 0.0001
		λ = 1

		source['encoder']=self.define_source_encoder(model_return=True)
		target['encoder']=self.define_source_encoder(model_return=True)
		C_label = Input(shape=source['train']['label'].shape[1::])
		classifier=self.get_source_classifier(shape=source['encoder'].output_shape[1:],atomic=True)

		#discriminator  = self.define_discriminator(source['encoder'].output_shape[1:],model_return=True)
		#discriminator = self.define_discriminator(self.channels, 64, use_sigmoid = not use_lsgan)
		#discriminator = self.define_discriminator(128, 64, use_sigmoid = not use_lsgan)
		discriminator = self.define_discriminator(128, 4, use_sigmoid = not use_lsgan)

		#target['discriminator']  = self.define_discriminator(target['encoder'].output_shape[1:],model_return=True)
		source['encoder'].summary()
		classifier.summary()
		discriminator.summary()

		if source_weights is not None:
			source['encoder'].load_weights(source_weights,by_name=True)
			classifier.load_weights(source_weights,by_name=True)
			target['encoder'].load_weights(source_weights,by_name=True)	
		
		weights = K.variable(source['loss_weights'])
			
		if use_lsgan:
			loss_fn = lambda output, target : K.mean(K.abs(K.square(output-target)))
		else:
			loss_fn = lambda output, target : -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target)) # Cross entropy

		loss_weighted = lambda output, target : -K.mean(K.log(output+1e-12)*target*weights) # Cross entropy

		def classifier_variables(encoder,classifier):
			encoder_in = encoder.inputs[0]
			encoder_out = encoder.outputs[0]
			classifier_out = classifier(encoder_out)
			fn_classify = K.function([encoder_in], [classifier_out])
			return {"encoder_in":encoder_in, "encoder_out":encoder_out, 
			"classifier_out":classifier_out, "fn_classify":fn_classify}

		source.update(classifier_variables(source['encoder'], classifier))
		target.update(classifier_variables(target['encoder'], classifier))


		def D_G_C_loss(discriminator, source_E_out, target_E_out, classifier, C_label, loss_weights): #here would go classifier_out
			C_out = classifier([target_E_out])
			source_D_out = discriminator([source_E_out])
			target_D_out = discriminator([target_E_out])
			source_D_loss = loss_fn(source_D_out, K.ones_like(source_D_out))
			target_D_loss = loss_fn(target_D_out, K.zeros_like(target_D_out))
			G_loss = loss_fn(target_D_out, K.ones_like(target_D_out)) # Fooling loss
			D_loss = source_D_loss + target_D_loss
			C_loss = loss_weighted(C_out, C_label)
			return D_loss, G_loss, C_loss					
		
		D_loss, G_loss, C_loss = D_G_C_loss(discriminator,source["encoder_out"],
			target["encoder_out"], classifier, C_label, 
			source['loss_weights'])

		#G_loss = G_loss
		# Add lambda when doing classification loss altogether

		D_weights = discriminator.trainable_weights
		G_weights = target['encoder'].trainable_weights
		C_weights = classifier.trainable_weights + target['encoder'].trainable_weights

		training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(D_weights,[],D_loss)
		netD_train = K.function([source['encoder_in'],target['encoder_in']],
								[D_loss],training_updates)
		training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(G_weights,[],G_loss)
		netG_train = K.function([target['encoder_in']],[G_loss],
								training_updates)
		training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(C_weights,[],C_loss)
		netC_train = K.function([target['encoder_in'],
								C_label],[C_loss],
								training_updates)
		netC_loss_view = K.function([target['encoder_in'],
								C_label],[C_loss])
		
		# ===================== Begin Training ========================= #
		source_testing=False
		target_testing=True
		errD_sum = errG_sum = 0


		batch = {'train': {}, 'test': {},'val':{}}
		self.batch={'train':{},'test':{},'val':{}}
		self.metricsG={'train':{},'test':{},'val':{}}
		self.metricsD={'train':{},'test':{},'val':{}}
		
		self.batch['train']['size']=batch_size
		self.batch['test']['size']=batch_size
		self.batch['val']['size']=batch_size
		
		smallest_sample_n_from_domains=source['train']['in'].shape[0] \
			if (source['train']['in'].shape[0]<target['train']['in'].shape[0]) \
			else target['train']['in'].shape[0]
		self.batch['train']['n'] = smallest_sample_n_from_domains // self.batch['train']['size']
		#self.batch['train']['n'] = target['train']['in'].shape[0] // self.batch['train']['size']

		deb.prints(self.batch['train']['n'])

		def data_random_permutation(data):
			idxs=np.random.permutation(data['in'].shape[0])
			data['in']=data['in'][idxs]
			data['label']=data['label'][idxs]
			return data
		self.batch['test']['n'] = source['test']['in'].shape[0] // self.batch['test']['size']		
		
		deb.prints(self.batch['test']['n'])
		training=True
		#target_validating=False
		
		# Early stop init ===========
		#metric_most_important='f1_score_avg'
		#metric_most_important='average_acc'
		#metric_most_important='kappa'
		#metric_most_important='oa_aa'
		metric_most_important='loss'

		self.early_stop={'best':0,
					'count':0,
					'signal':False,
					'patience':patience,
					'best_updated':False,
					'minimize':False}
		if metric_most_important=='loss':
			self.early_stop['minimize']=True
			self.early_stop['best']=9999


		self.metricsG['train']['loss'] = np.zeros((1, 2))
		self.metricsD['train']['loss'] = np.zeros((1, 2))
		deb.prints(self.metricsD['train']['loss'])
		for epoch in range(epochs):
			early_epoch=0

			if training==True:
				# ============ TRAIN LOOP ============================== #
				
				source['train']=data_random_permutation(source['train'])
				target['train']=data_random_permutation(target['train'])
				
				if self.metricsG['train']['loss'][0][0]>self.metricsD['train']['loss'][0][0]:
					D_training=False
				else:
					D_training=True
				deb.prints(D_training)
				self.metricsG['train']['loss'] = np.zeros((1, 2))
				if D_training==True:
					self.metricsD['train']['loss'] = np.zeros((1, 2))
				err_segmentation = np.zeros((1, 2))
				C_loss=0
				for batch_id in range(0, self.batch['train']['n']):
					idx0 = batch_id*self.batch['train']['size']
					idx1 = (batch_id+1)*self.batch['train']['size']

					errG = netG_train([target['train']['in'][idx0:idx1]])
					#errG = netG_train([target['train']['in'][idx0:idx1]])
					#errG = netG_train([target['train']['in'][idx0:idx1]])
					#errG = netG_train([target['train']['in'][idx0:idx1]])
					
					self.metricsG['train']['loss'] += errG

					if D_training==True:
						self.metricsD['train']['loss'] += netD_train([source['train']['in'][idx0:idx1],
									target['train']['in'][idx0:idx1]])
					C_loss+=netC_loss_view([target['val']['in'][idx0:idx1],
						target['val']['label'][idx0:idx1]])[0]
					#err_segmentation = netC_train([source['train']['in'][idx0:idx1],
					#	source['train']['label'][idx0:idx1]])
					#err_segmentation = netC_train([source['train']['in'][idx0:idx1],
					#	source['train']['label'][idx0:idx1]])
					#err_segmentation = netC_train([source['train']['in'][idx0:idx1],
					#	source['train']['label'][idx0:idx1]])

					# ============== IF EARLY VALIDATING ==============
					if source['dataset']=='para' or source['dataset']=='acre':
						batch_interval=20
					else:
						batch_interval=2
					if early_validating==True and batch_id%batch_interval==0:
						deb.prints(batch_id)
						deb.prints(self.early_stop['best'])
						deb.prints(C_loss)
						

						metrics_val,_=self.test_loop(target['val'],
							self.batch['val'],target['fn_classify'],G,
							metric_only_one=metric_most_important,
							ignore_bcknd=ignore_bcknd,
							loss=C_loss)

						#metrics_val,_=self.test_loop_for(
						#target['val'],
						#self.batch['val'],
						#fn_classify=target['fn_classify'],
						#G=G,
						#ignore_bcknd=ignore_bcknd,
						#loss=C_loss)
						deb.prints(metrics_val['loss'])
						self.early_stop_check(metrics_val,early_epoch,
							most_important=metric_most_important,minimize=self.early_stop['minimize'])
						C_loss=0

						if self.early_stop['best_updated']==True:
							print("BEST METRIC UPDATED")
							target['encoder'].save_weights('result_adv/target_encoder_best.h5')
							discriminator.save_weights('result_adv/discriminator_best.h5')
						if self.early_stop["signal"]==True:
							target['encoder'].load_weights('result_adv/target_encoder_best.h5')
							discriminator.load_weights('result_adv/discriminator_best.h5')
							print("Early stop; testing...")
							if testing_mode=='for_loop':
								metrics,prediction=self.test_loop_for(
									target['test'],
									self.batch['test'],
									fn_classify=target['fn_classify'],
									G=G,
									ignore_bcknd=ignore_bcknd)

							else:
								metrics,prediction=self.test_loop(target['test'],
									self.batch['test'],target['fn_classify'],G,
									ignore_bcknd=ignore_bcknd)

							print("EARLY STOP EPOCH",epoch,metrics)
							np.save("result_adv/"+target['dataset']+
								"_prediction.npy",prediction)
							np.save("result_adv/"+target['dataset']+
								"_label.npy",target['test']['label'])
							sys.exit()
							#break
						early_epoch+=1
						

					# ==================================================
				self.metricsG['train']['loss'] /= self.batch['train']['n'] 
				if D_training==True:
					self.metricsD['train']['loss'] /= self.batch['train']['n'] 
				
				print("Epoch: {}. G_loss: {}. D_loss: {}.".format(epoch,
					self.metricsG['train']['loss'],
					self.metricsD['train']['loss']))
				print("C_loss: {}".format(err_segmentation))
				target['encoder'].save_weights('target_encoder.h5')
				discriminator.save_weights('discriminator.h5')
			else:
				print("Skipped training")
			
			# ============ VAL LOOP ============================== #
			#deb.prints(target_validating)
			#if target_validating==1:		
			#	print("VALIDATING")	
			#	metrics_val=self.test_loop(target['val'],
			#		self.batch['val'],target['fn_classify'],G)
			#	self.early_stop_check(metrics_val,epoch)

			
			# ============ TEST LOOP ============================== #
			if target_validating==1:
				test_signal=self.early_stop['best_updated'] 
			else:
				test_signal=True

			if test_signal==True:
				print("Testing...")
				if target_testing==True:
					#if testing_mode=='for_loop':
					#	metrics,_=self.test_loop_for(target['test'],
					#		self.batch['test'],)

					#else:
					metrics,prediction=self.test_loop_for(
						target['test'],
						self.batch['test'],
						fn_classify=target['fn_classify'],
						G=G,
						ignore_bcknd=ignore_bcknd)
				if source_testing==True:						
					metrics,_=self.test_loop(source['test'],
						self.batch['test'],source['fn_classify'],G)


			#====================METRICS GET================================================#
			#deb.prints(idx1)
			print("Epoch={}".format(epoch))	

#			metric_most_important='average_acc'
			metric_most_important='average_acc'
						
			metrics_val,_=self.test_loop(target['val'],
				self.batch['val'],target['fn_classify'],G,
				metric_only_one=metric_most_important,
				ignore_bcknd=ignore_bcknd)
			#metrics_val,_=self.test_loop_for(target['val'],
			#	self.batch['val'],
			#	fn_classify=target['fn_classify'],
			#	G=G,
			#	metric_only_one=metric_most_important,
			#	ignore_bcknd=ignore_bcknd)
			
			self.early_stop_check(metrics_val,epoch,
				most_important=metric_most_important)

			if self.early_stop['best_updated']==True:
				print("BEST METRIC UPDATED")
				target['encoder'].save_weights('result_adv/target_encoder_best.h5')
				discriminator.save_weights('result_adv/discriminator_best.h5')
			if self.early_stop["signal"]==True:
				target['encoder'].load_weights('result_adv/target_encoder_best.h5')
				discriminator.load_weights('result_adv/discriminator_best.h5')
				print("Early stop; testing...")
				if testing_mode=='for_loop':
					metrics,prediction=self.test_loop_for(
						target['test'],
						self.batch['test'],
						fn_classify=target['fn_classify'],
						G=G,
						ignore_bcknd=ignore_bcknd)
			# =================== EARLY STOP CHECK ========================

			#if target_validating==1:
			#	if self.early_stop['best_updated']==True:
			#		self.early_stop['best_predictions']=target['test']['prediction']
			#		target['encoder'].save_weights('target_encoder_best.h5')
			#		discriminator.save_weights('discriminator_best.h5')
	#		#		.save_weights('weights_best.h5')
			#	print(self.early_stop['signal'])
			#	if self.early_stop["signal"]==True:
			#		print("EARLY STOP EPOCH",epoch,metrics)
			#		np.save("prediction.npy",self.early_stop['best_predictions'])
			#		np.save("labels.npy",target['test']['label'])
			#		break
		



	# Not being used
	def eval_source_classifier(self, model, data, batch_size=6, domain='Source'):
		

		model.compile(loss='categorical_crossentropy', optimizer=self.src_optimizer, metrics=['accuracy'])

		# ================= LOAD WEIGHTS ===============

		model.load_weights(self.source_weights_path + 'source_model' + \
			data['dataset'] + '.h5')
		#==========================TEST LOOP================================================#
		data['test']['prediction']=np.zeros_like(data['test']['label'])
		self.batch_test_stats=True

		for batch_id in range(0, self.batch['test']['n']):
			idx0 = batch_id*self.batch['test']['size']
			idx1 = (batch_id+1)*self.batch['test']['size']

			batch['test']['in'] = data['test']['in'][idx0:idx1]
			batch['test']['label'] = data['test']['label'][idx0:idx1]

			if self.batch_test_stats:
				self.metrics['test']['loss'] += model.test_on_batch(
					batch['test']['in'], batch['test']['label'])        # Accumulated epoch

			data['test']['prediction'][idx0:idx1]=model.predict(batch['test']['in'],batch_size=self.batch['test']['size'])

		#====================METRICS GET================================================#
		deb.prints(data['test']['label'].shape)     
		deb.prints(idx1)
		print("Epoch={}".format(epoch)) 
		
		# Average epoch loss
		self.metrics['test']['loss'] /= self.batch['test']['n']
		
		metrics=metrics_get(data['test'],debug=1)

		
		scores = model.evaluate_generator(src_datagen.flow(test_x[:10000], test_y[:10000]),10000)
		print('%s %s Classifier Test loss:%.5f'%(dataset.upper(), domain, scores[0]))
		print('%s %s Classifier Test accuracy:%.2f%%'%(dataset.upper(), domain, float(scores[1])*100))            
			
	def eval_target_classifier(self, source_model, target_discriminator, dataset='svhn'):
		
		self.define_target_encoder()
		model = self.get_source_classifier(self.target_encoder, source_model)
		model.load_weights(target_discriminator, by_name=True)
		model.summary()
		self.eval_source_classifier(model, dataset=dataset, domain='Target')
		 
if __name__ == '__main__':



	# ========= Define data sources =====================
	
	def load_data(file_pattern):

		def getKey(filename):
			file_text_name = os.path.splitext(os.path.basename(filename))  #you get the file's text name without extension
			file_last_num = os.path.basename(file_text_name[0]).split('patches')  #you get three elements, the last one is the number. You want to sort it by this number
			return int(file_last_num[-1])

		data=glob.glob(file_pattern)
		data=sorted(data,key=getKey)
		return data
		
	source={'dataset':args.source_dataset}
	target={'dataset':args.target_dataset}

	load_mode=2
	if load_mode==1:

		path='../wildfire_fcn/src/patch_extract2/patches/'
		source['mask'] = load_data(path+source['dataset']+"/mask/*.npy")
		target['mask'] = load_data(path+target['dataset']+"/mask/*.npy")

		source['in'] = load_data(path+source['dataset']+"/im/*.npy")
		target['in'] = load_data(path+target['dataset']+"/im/*.npy")
		source['label'] = load_data(path+source['dataset']+"/label/*.npy")
		target['label'] = load_data(path+target['dataset']+"/label/*.npy")


		print(source['mask'][0:3])
		print(source['in'][0:3])
		
		deb.prints(len(source['mask']))
		deb.prints(len(source['label']))
		deb.prints(len(source['in']))

		deb.prints(len(target['mask']))
		deb.prints(len(target['label']))
		deb.prints(len(target['in']))
		
		def train_test_split_from_mask(masks):
			ids_train=[]
			ids_test=[]
			
			count=0
			for _ in masks:
				mask=np.load(_)
				if np.all(mask==1):
					ids_train.append(count)
				elif np.all(mask==2):
					ids_test.append(count)
				count+=1
			return ids_train, ids_test    


		ids_train, ids_test = train_test_split_from_mask(source['mask'])
		deb.prints(len(ids_train))
		#deb.prints(ids_train.shape)
		#deb.prints(ids_train.dtype)
		#deb.prints(source['in'].shape)
		source['train']={}
		source['test']={}
		source['train']['in']=[source['in'][i] for i in ids_train]
		source['test']['in']=[source['in'][i] for i in ids_test]
		source['train']['label']=[source['label'][i] for i in ids_train]
		source['test']['label']=[source['label'][i] for i in ids_test]


		deb.prints(source['train']['label'][0:3])
		deb.prints(source['train']['in'][0:3])
		
		deb.prints(source['test']['label'][0:3])
		deb.prints(source['test']['in'][0:3])
		
		deb.prints(len(source['train']['in']))
		deb.prints(len(source['test']['in']))

		assert len(source['train']['in']) and len(source['test']['in'])



		source['train']['in']=folder_load(source['train']['in'])
		source['train']['label']=folder_load(source['train']['label'])
		if args.testing == 1:
			source['test']['in']=folder_load(source['test']['in'])
			source['test']['label']=folder_load(source['test']['label'])

	else:
		
		#source_validating=1 if args.source_validating==True else 0
		source=domain_data_load({"dataset":args.source_dataset},
			validating=args.source_validating,
			ignore_bcknd=args.ignore_bcknd,
			testing_mode=args.testing_mode,
			class_n=args.class_n)
			#testing_mode=args.testing_mode,

		target=domain_data_load({"dataset":args.target_dataset},
			validating=args.adversarial_validating,
			ignore_bcknd=args.ignore_bcknd,
			all_test=True,testing_mode=args.testing_mode,
			class_n=args.class_n)
		deb.prints(target['test']['full_in'].shape)
		deb.prints(source['test']['full_in'].shape)
		
	try:
		deb.prints(source['val']['in'].shape)
	except:
		print("No source val set")
	try:
		deb.prints(target['val']['in'].shape)
	except:
		print("No target val set")


	

	adda = ADDA(args.lr, args.window_len, args.channel_n,class_n=class_n,
		encoder_mode=args.encoder_mode)
	print(0.1)
	adda.define_source_encoder()
	print(0.2)
	# source
	source_model = adda.get_source_classifier(adda.source_encoder, 
		weights=args.source_weights)
	print(2)
	source_model.summary()
	if not args.train_discriminator:

		if args.eval_source_classifier is None:
			print("Training source classifier...")
			if args.source_validating==1:
				validation_data=target['val']
			else:
				validation_data=None
			adda.source_model_train(source_model, data=source, \
				start_epoch=args.start_epoch-1,
				testing=args.testing,weights_save=args.weights_save,
				validating=args.source_validating, 
				validation_data=validation_data,
				ignore_bcknd=args.ignore_bcknd) 
		else:
			adda.source_model_train(source_model, data=source, training=False, \
				start_epoch=args.start_epoch-1,weights_save=args.weights_save,
				ignore_bcknd=args.ignore_bcknd,testing_mode=args.testing_mode)
			
	adda.define_target_encoder(args.source_weights)
	
	if args.train_discriminator:
		adda.discriminator_train(epochs=args.discriminator_epochs, 
										source_weights=args.source_weights, 
										source=source, target=target, 
										start_epoch=args.start_epoch-1,
										target_validating=args.adversarial_validating,
										ignore_bcknd=args.ignore_bcknd,
										testing_mode=args.testing_mode)
	if args.eval_target_classifier is not None:
		adda.eval_target_classifier(args.eval_source_classifier, args.eval_target_classifier)
	
