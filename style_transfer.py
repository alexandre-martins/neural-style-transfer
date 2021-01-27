import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.compat.v1.keras import backend
from scipy.optimize import fmin_l_bfgs_b as opt
from scipy.misc import imsave
tf.logging.set_verbosity(tf.logging.ERROR)

# Building the neural style transfer class

class NeuralTransfer:
  def __init__(self, content, style, height=512, width=512):

    self.content_img = content
    self.style_img = style
    self.height = height
    self.width = width
    self.content_array = self._load_image(content)
    self.style_array = self._load_image(style)

    self.content_layers = 'block2_conv2'

    self.style_layers = ['block1_conv2', 
                         'block2_conv2', 
                         'block3_conv3', 
                         'block4_conv3', 
                         'block5_conv3']

    self._create_model()
    self._totalLoss()

    # Loading the backend keras session and initializing the variables
    self.sess = backend.get_session()
    self.sess.run(tf.global_variables_initializer())
  
  def _create_model(self):
    # Defining the random noise image
    self.combination_im = tf.Variable(tf.random_uniform((1, self.height, self.width, 3)))
    # Concatenating all tensors with respect to the first axis (1, ..., ..., ...)
    input_tensor = tf.concat([self.content_array, self.style_array, self.combination_im], 0)
    # Creating the vgg16 model and loading input tensor and pre-trained weights
    self.vgg16 = VGG16(input_tensor=input_tensor, include_top=False, weights='imagenet')

  def _contentLoss(self, content_img, combination_img):
    '''
    Reduce_sum simply is the sum of tensor's values in all dimensions.
    The function computes the direct squared difference between the content
    and the combination image
    '''
    return tf.reduce_sum(tf.square(content_img-combination_img))
  
  def _styleLoss(self, style, combination):
    h,w,d = style.get_shape()
    M = h.value*w.value
    N = d.value
    S=self._GramMatrix(style)
    C=self._GramMatrix(combination)
    
    return tf.reduce_sum(tf.square(S - C)) / (4. * (N ** 2) * (M ** 2))
  
  def _totalLoss(self):
    layers = dict([(layer.name,layer.output) for layer in self.vgg16.layers])
		
		# VGG16 Layers that will represent the content and style
    layer_content = layers[self.content_layers]
    layers_style = [layers[i] for i in self.style_layers]
    
    self.loss = tf.Variable(0.)

		# Content Loss
    self.loss = tf.add(self.loss, 0.01 * self._contentLoss(layer_content[0,:,:,:],layer_content[2,:,:,:]))
		
		# Style Loss
    for i in range(len(self.style_layers)):
      self.loss = tf.add(self.loss, 100 * self._styleLoss(layers_style[i][1,:,:,:],layers_style[i][2,:,:,:]))
  
  def _GramMatrix(self, x):
		# The first axis corresponds to the number of filters
    features=tf.keras.backend.batch_flatten(tf.transpose(x,perm=[2,0,1]))
    gram=tf.matmul(features, tf.transpose(features))
    
    return gram
    
  def _load_image(self, image):
    loaded_image = Image.open(image)
    resized_image = loaded_image.resize((self.height, self.width))
    array = np.asarray(resized_image, dtype='float32')
    reshaped_array = np.reshape(array, (1, self.height, self.width, 3))

    return reshaped_array

  def train(self, epochs):
    train_step = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                        var_list=[self.combination_im], 
                                                        method='L-BFGS-B',
                                                        options={'maxfun':20, 'iprint':-1})
    self.vgg16.load_weights('vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    
    for i in range(epochs):
      curr_loss = self.sess.run(self.loss)
      print("Iteration {0}, Loss: {1}".format(i,curr_loss))
      train_step.minimize(session=self.sess)
    
    finalOutput = self.sess.run(self.combination_im)
    best_image_reshaped = finalOutput.reshape((self.height, self.width, 3))
    best_image_reshaped = np.clip(best_image_reshaped, 0, 255).astype('uint8')
    if not os.path.exists('./outputs'):
    	os.makedirs('outputs')

    return imsave('./outputs/style_transfer_out.jpg', best_image_reshaped)
