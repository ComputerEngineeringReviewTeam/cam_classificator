import math
from abc import abstractmethod

import PIL
import numpy as np
from PIL import Image, ImageFilter
from safetensors import torch
from torchvision.transforms import functional
import torch


CONTOUR = PIL.ImageFilter.CONTOUR
EDGE_ENHANCE = PIL.ImageFilter.EDGE_ENHANCE
MIN_FILTER = PIL.ImageFilter.MinFilter
MAX_FILTER = PIL.ImageFilter.MaxFilter
SHARPEN = PIL.ImageFilter.SHARPEN

#Abstract filter
class Filter:
    @abstractmethod
    def applyFilter(self, image):
        pass


#For use with built-in PIL library filters (PIL.ImageFilter.X)
class StandardPILFilter(Filter):
    def applyFilter(self, image):
        image.setImage(image.getImage().filter(self.PILfilter))
        return image

    def __init__(self, PILfilter):
        self.PILfilter = PILfilter


#A kernel filter, constructor takes a list of floats.
class MatrixFilter(Filter):

    def applyFilter(self, image):
        image.setImage(image.getImage().filter(PIL.ImageFilter.Kernel((self.matrixSize, self.matrixSize), self.matrixTuple, 1, 0)))
        return image

    def __init__(self, matrix):
        self.matrixTuple = tuple(matrix)
        self.matrixSize = int(math.sqrt(len(matrix)))
        return


#A single-pixel filter, constructor takes a lambda function - VERY slow
class PixelFilter(Filter):
    def applyFilter(self, image):
        imageNP = np.array(image.getImage())
        for x in range(np.shape(imageNP)[0]):
            for y in range(np.shape(imageNP)[1]):
                imageNP[x, y] = np.array(self.lambdaF(imageNP[x, y, 0], imageNP[x, y, 1], imageNP[x, y, 2]))
        image.setImage(PIL.Image.fromarray(imageNP.astype('uint8'), 'RGB'))
        return image

    def __init__(self, lambdaF):
        self.lambdaF = lambdaF
        return

#A filter operating on one channel of one pixel at a time.
class ChannelFilter(Filter):
    def applyFilter(self, image):
        imageNP = np.array(image.getImage())
        imageChannel = imageNP[:, :, self.channel]
        imageChannel = self.lambdaF(imageChannel)
        imageNP[:, :, self.channel] = imageChannel
        image.setImage(PIL.Image.fromarray(imageNP.astype('uint8'), 'RGB'))
        return image

    #Channels: 0 - R, 1 - G, 2 -B
    def __init__(self, channel, lambdaF):
        self.lambdaF = np.vectorize(lambdaF)
        self.channel = channel


#A set/pipeline of filters
class Filters:
    filters = []

    #Applies the set of filters to an image
    def applyFilters(self, image):
        for filter in self.filters:
            image = filter.applyFilter(image)
        return image

    #Adds a filter to the set (at the end)
    def addFilter(self, filter):
        self.filters.append(filter)
        return self

    #Adds a filter at any position in the pipeline
    def injectFilter(self, filter, position):
        self.filters.insert(position, filter)

    #Removes the filter from the specified position
    def removeFilter(self, position):
        self.filters.pop(position)

    #Replace the filter from the specified position with a new one
    def overwriteFilter(self, filter, position):
        self.filters[position] = filter

    #Return number of filters in the pipeline
    def size(self):
        return len(self.filters)


class Image:
    def __init__(self, image):
        self.imagePIL = image

    @classmethod
    def load(cls, path):
        return cls(PIL.Image.open(path, mode="r"))

    def getImage(self):
        return self.imagePIL

    def setImage(self, image):
        self.imagePIL = image
    def getTensor(self):
        return functional.pil_to_tensor(self.imagePIL).to(torch.float32)
