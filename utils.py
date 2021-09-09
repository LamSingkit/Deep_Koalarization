import numpy as np
from skimage.io import imread
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
import skimage.transform 
import os
import tensorflow as tf
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input


class DataGenerator(tf.keras.utils.Sequence):
    'Generates image data in batches to prevent memory overload'
    '''
    image_filenames: List of all image file names
    batch_size: Desire batch size
    '''
    def __init__(self,image_filenames,directory,batch_size):
        'Initialization'
        self.image_filenames = image_filenames
        self.directory = directory
        self.batch_size = batch_size
        self.epoch_end()
    
    
    def __len__(self):
        return int(np.floor(len(self.image_filenames)/self.batch_size))
    
    
    def __getitem__(self,id):
        'Generate a batch of data'
        # Generate indexes for the batch
        idx = self.indexes[id*self.batch_size:(id+1)*self.batch_size]
        
        # Find list of image_filenames
        image_filenames_temp = [self.image_filenames[i] for i in idx]
        
        # Generate data
        X,y = self.generate_process_image(image_filenames_temp)
        
        return X,y
    def image_process(self,image):
        '''
        Input:
            image: np.array of RGB channel image
        Output:
            Encoder_image: size 224 x 224 x 1
            Extractor_image: size 299 x 299 x 3
            AB_channel: 224 x 224 x 2
        '''
        im = image.copy()
        im = rgb2lab(im)
        
        # centered and scaled image in order to have a interval of -1,1
        L_channel = im[:,:,0]-50; L_channel = L_channel/50
        A_channel = im[:,:,1]/128
        B_channel = im[:,:,2]/128
        
        # Encoder path
        Encoder_image = skimage.transform.resize(L_channel,(224,224,1),anti_aliasing = True)
        
        # Extractor path
        Extractor_image = rgb2gray(image)
        Extractor_image = gray2rgb(Extractor_image)
        Extractor_image = skimage.transform.resize(Extractor_image,(299,299,3),anti_aliasing = True)*255
        Extractor_image = preprocess_input(Extractor_image)
        
        
        # AB channel path
        A_channel = skimage.transform.resize(A_channel,(224,224,1))
        B_channel = skimage.transform.resize(B_channel,(224,224,1))
        AB_channel = np.concatenate([A_channel,B_channel],axis=-1)
               
        return Encoder_image,Extractor_image,AB_channel     
    
    
    def generate_process_image(self,image_filenames_temp):
        'Generates processed image data'
        # Initialization
        X_encoder = np.empty((self.batch_size,224,224,1))
        X_extractor = np.empty((self.batch_size,299,299,3))
        y_ab_channel = np.empty((self.batch_size,224,224,2))
        
        # Store the processed image into lists
        for i , filename in enumerate(image_filenames_temp):
            # Load and process
            im=np.array(imread(os.path.join(self.directory,filename)))
            encoder,extractor,ab_channel=self.image_process(im)
            
            # Store the data
            X_encoder[i,] = encoder
            X_extractor[i,] = extractor
            y_ab_channel[i,] = ab_channel
            
        return [X_encoder,X_extractor],y_ab_channel
    
    def epoch_end(self):
        'Updates the indexes when each epoch end'
        self.indexes = np.arange(len(self.image_filenames))
           
def process(image):
        '''
        Input:
            image: np.array of RGB channel image
        Output:
            Encoder_image: size 224 x 224 x 1
            Extractor_image: size 299 x 299 x 3
            AB_channel: 224 x 224 x 2
        '''
        im = image.copy()
        im = rgb2lab(im)
        
        # centered and scaled image in order to have a interval of -1,1
        L_channel = im[:,:,0]-50; L_channel = L_channel/50
        A_channel = im[:,:,1]/128
        B_channel = im[:,:,2]/128
        
        # Encoder path
        Encoder_image = skimage.transform.resize(L_channel,(224,224,1),anti_aliasing = True)
        
        # Extractor path
        Extractor_image = rgb2gray(image)
        Extractor_image = gray2rgb(Extractor_image)
        Extractor_image = skimage.transform.resize(Extractor_image,(299,299,3),anti_aliasing = True)*255
        Extractor_image = preprocess_input(Extractor_image)
        
        
        # AB channel path
        A_channel = skimage.transform.resize(A_channel,(224,224,1))
        B_channel = skimage.transform.resize(B_channel,(224,224,1))
        AB_channel = np.concatenate([A_channel,B_channel],axis=-1)
               
        return Encoder_image,Extractor_image,AB_channel     
            
            
            
            
            
            
            
            
            
            
            
            