# contains functions related to images

import numpy as np

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # resize and crop
    image.thumbnail([x * 256/min(image.size) for x in image.size])
    image = image.crop(((image.size[0]-224)/2, 
                        (image.size[1]-224)/2,
                         image.size[0]-(image.size[0]-224)/2,
                         image.size[1]-(image.size[1]-224)/2))
    
    # normalize the colors
    image = np.array(image) / 255
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # transpose to match pytorch expectations
    return np.transpose(image, (2, 0, 1))

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax