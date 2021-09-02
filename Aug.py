import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import math
from math import floor, ceil
import random

from scipy.ndimage import zoom
from scipy.ndimage import grey_erosion
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import shift

from skimage import transform
from skimage import util
from cv2 import Canny
import cv2

#from prettytable import PrettyTable
#from prettytable import MSWORD_FRIENDLY


def skew(image, skew_type='RANDOM', magnitude=0.5):#***************************************************8        
    #"TILT", "TILT_LEFT_RIGHT", "TILT_TOP_BOTTOM", "CORNER", "ALL"
    
    w, h = image.size

    x1 = 0
    x2 = h
    y1 = 0
    y2 = w

    original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]

    max_skew_amount = max(w, h)
    max_skew_amount = int(ceil(max_skew_amount * magnitude))
    skew_amount = random.randint(1, max_skew_amount)

    # Old implementation, remove.
    # if not self.magnitude:
    #    skew_amount = random.randint(1, max_skew_amount)
    # elif self.magnitude:
    #    max_skew_amount /= self.magnitude
    #    skew_amount = max_skew_amount

    if skew_type == "RANDOM":
        skew = random.choice(["TILT", "TILT_LEFT_RIGHT", "TILT_TOP_BOTTOM", "CORNER"])
    else:
        skew = skew_type

    # We have two choices now: we tilt in one of four directions
    # or we skew a corner.

    if skew == "TILT" or skew == "TILT_LEFT_RIGHT" or skew == "TILT_TOP_BOTTOM":

        if skew == "TILT":
            skew_direction = random.randint(0, 3)
        elif skew == "TILT_LEFT_RIGHT":
            skew_direction = random.randint(0, 1)
        elif skew == "TILT_TOP_BOTTOM":
            skew_direction = random.randint(2, 3)

        if skew_direction == 0:
            # Left Tilt
            new_plane = [(y1, x1 - skew_amount),  # Top Left
                         (y2, x1),                # Top Right
                         (y2, x2),                # Bottom Right
                         (y1, x2 + skew_amount)]  # Bottom Left
        elif skew_direction == 1:
            # Right Tilt
            new_plane = [(y1, x1),                # Top Left
                         (y2, x1 - skew_amount),  # Top Right
                         (y2, x2 + skew_amount),  # Bottom Right
                         (y1, x2)]                # Bottom Left
        elif skew_direction == 2:
            # Forward Tilt
            new_plane = [(y1 - skew_amount, x1),  # Top Left
                         (y2 + skew_amount, x1),  # Top Right
                         (y2, x2),                # Bottom Right
                         (y1, x2)]                # Bottom Left
        elif skew_direction == 3:
            # Backward Tilt
            new_plane = [(y1, x1),                # Top Left
                         (y2, x1),                # Top Right
                         (y2 + skew_amount, x2),  # Bottom Right
                         (y1 - skew_amount, x2)]  # Bottom Left

    if skew == "CORNER":

        skew_direction = random.randint(0, 7)

        if skew_direction == 0:
            # Skew possibility 0
            new_plane = [(y1 - skew_amount, x1), (y2, x1), (y2, x2), (y1, x2)]
        elif skew_direction == 1:
            # Skew possibility 1
            new_plane = [(y1, x1 - skew_amount), (y2, x1), (y2, x2), (y1, x2)]
        elif skew_direction == 2:
            # Skew possibility 2
            new_plane = [(y1, x1), (y2 + skew_amount, x1), (y2, x2), (y1, x2)]
        elif skew_direction == 3:
            # Skew possibility 3
            new_plane = [(y1, x1), (y2, x1 - skew_amount), (y2, x2), (y1, x2)]
        elif skew_direction == 4:
            # Skew possibility 4
            new_plane = [(y1, x1), (y2, x1), (y2 + skew_amount, x2), (y1, x2)]
        elif skew_direction == 5:
            # Skew possibility 5
            new_plane = [(y1, x1), (y2, x1), (y2, x2 + skew_amount), (y1, x2)]
        elif skew_direction == 6:
            # Skew possibility 6
            new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1 - skew_amount, x2)]
        elif skew_direction == 7:
            # Skew possibility 7
            new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2 + skew_amount)]

    if skew_type == "ALL":
        # Not currently in use, as it makes little sense to skew by the same amount
        # in every direction if we have set magnitude manually.
        # It may make sense to keep this, if we ensure the skew_amount below is randomised
        # and cannot be manually set by the user.
        corners = dict()
        corners["top_left"] = (y1 - random.randint(1, skew_amount), x1 - random.randint(1, skew_amount))
        corners["top_right"] = (y2 + random.randint(1, skew_amount), x1 - random.randint(1, skew_amount))
        corners["bottom_right"] = (y2 + random.randint(1, skew_amount), x2 + random.randint(1, skew_amount))
        corners["bottom_left"] = (y1 - random.randint(1, skew_amount), x2 + random.randint(1, skew_amount))

        new_plane = [corners["top_left"], corners["top_right"], corners["bottom_right"], corners["bottom_left"]]

    # To calculate the coefficients required by PIL for the perspective skew,
    # see the following Stack Overflow discussion: https://goo.gl/sSgJdj
    matrix = []

    for p1, p2 in zip(new_plane, original_plane):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(original_plane).reshape(8)

    perspective_skew_coefficients_matrix = np.dot(np.linalg.pinv(A), B)
    perspective_skew_coefficients_matrix = np.array(perspective_skew_coefficients_matrix).reshape(8)

    return image.transform(image.size, Image.PERSPECTIVE, perspective_skew_coefficients_matrix, resample=Image.BICUBIC)

def invert(image):#***************************************************8
    return ImageOps.invert(image)#invert

def brilho(image, min_factor=1, max_factor=1):#***************************************************8
    factor = np.random.uniform(min_factor, max_factor)
    image_enhancer_brightness = ImageEnhance.Brightness(image)
    return image_enhancer_brightness.enhance(factor)

def color(image, min_factor=1, max_factor=1):#***************************************************8
    factor = np.random.uniform(min_factor, max_factor)
    image_enhancer_color = ImageEnhance.Color(image)
    return image_enhancer_color.enhance(factor)

def contrast(image, min_factor=1, max_factor=1):#***************************************************8
    factor = np.random.uniform(min_factor, max_factor)
    image_enhancer_contrast = ImageEnhance.Contrast(image)
    return image_enhancer_contrast.enhance(factor)

def flip(image, top_bottom_left_right):#***************************************************8
    random_axis = random.randint(0, 1)
    if top_bottom_left_right == "LEFT_RIGHT":
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif top_bottom_left_right == "TOP_BOTTOM":
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    elif top_bottom_left_right == "RANDOM":
        if random_axis == 0:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        elif random_axis == 1:
            return image.transpose(Image.FLIP_TOP_BOTTOM)

def shear(image, max_shear_left = 0.8, max_shear_right = 0.7):#***************************************************8
    ######################################################################
    # Old version which uses SciKit Image
    ######################################################################
    # We will use scikit-image for this so first convert to a matrix
    # using NumPy
    # amount_to_shear = round(random.uniform(self.max_shear_left, self.max_shear_right), 2)
    # image_array = np.array(image)
    # And here we are using SciKit Image's `transform` class.
    # shear_transformer = transform.AffineTransform(shear=amount_to_shear)
    # image_sheared = transform.warp(image_array, shear_transformer)
    #
    # Because of warnings
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     return Image.fromarray(img_as_ubyte(image_sheared))
    ######################################################################
    
    width, height = image.size
    angle_to_shear = int(random.uniform((abs(max_shear_left)*-1) - 1, max_shear_right + 1))
    if angle_to_shear != -1: angle_to_shear += 1

    directions = ["x", "y"]
    direction = random.choice(directions)

    # We use the angle phi in radians later
    phi = math.tan(math.radians(angle_to_shear))

    if direction == "x":
        # Here we need the unknown b, where a is
        # the height of the image and phi is the
        # angle we want to shear (our knowns):
        # b = tan(phi) * a
        shift_in_pixels = phi * height

        if shift_in_pixels > 0:
            shift_in_pixels = math.ceil(shift_in_pixels)
        else:
            shift_in_pixels = math.floor(shift_in_pixels)

        # For negative tilts, we reverse phi and set offset to 0
        # Also matrix offset differs from pixel shift for neg
        # but not for pos so we will copy this value in case
        # we need to change it
        matrix_offset = shift_in_pixels
        if angle_to_shear <= 0:
            shift_in_pixels = abs(shift_in_pixels)
            matrix_offset = 0
            phi = abs(phi) * -1

        # Note: PIL expects the inverse scale, so 1/scale_factor for example.
        transform_matrix = (1, phi, -matrix_offset, 0, 1, 0)

        image = image.transform((int(round(width + shift_in_pixels)), height), Image.AFFINE, transform_matrix, Image.BICUBIC)

        image = image.crop((abs(shift_in_pixels), 0, width, height))

        return image.resize((width, height), resample=Image.BICUBIC)

    elif direction == "y":
        shift_in_pixels = phi * width

        matrix_offset = shift_in_pixels
        if angle_to_shear <= 0:
            shift_in_pixels = abs(shift_in_pixels)
            matrix_offset = 0
            phi = abs(phi) * -1

        transform_matrix = (1, 0, 0, phi, 1, -matrix_offset)
        image = image.transform((width, int(round(height + shift_in_pixels))), Image.AFFINE, transform_matrix, Image.BICUBIC)
        image = image.crop((0, abs(shift_in_pixels), width, height))
        return image.resize((width, height), resample=Image.BICUBIC)

def distort(image, grid_width=15, grid_height=15, magnitude=0.8):#**************************************************** ERRO COM COR
    grid_width = grid_width
    grid_height = grid_height
    magnitude = abs(magnitude)
    # TODO: Implement non-random magnitude.
    #randomise_magnitude = True

    w, h = image.size

    horizontal_tiles = grid_width
    vertical_tiles = grid_height

    width_of_square = int(floor(w / float(horizontal_tiles)))
    height_of_square = int(floor(h / float(vertical_tiles)))

    width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
    height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

    dimensions = []

    for vertical_tile in range(vertical_tiles):
        for horizontal_tile in range(horizontal_tiles):
            if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                dimensions.append([horizontal_tile * width_of_square,
                                   vertical_tile * height_of_square,
                                   width_of_last_square + (horizontal_tile * width_of_square),
                                   height_of_last_square + (height_of_square * vertical_tile)])
            elif vertical_tile == (vertical_tiles - 1):
                dimensions.append([horizontal_tile * width_of_square,
                                   vertical_tile * height_of_square,
                                   width_of_square + (horizontal_tile * width_of_square),
                                   height_of_last_square + (height_of_square * vertical_tile)])
            elif horizontal_tile == (horizontal_tiles - 1):
                dimensions.append([horizontal_tile * width_of_square,
                                   vertical_tile * height_of_square,
                                   width_of_last_square + (horizontal_tile * width_of_square),
                                   height_of_square + (height_of_square * vertical_tile)])
            else:
                dimensions.append([horizontal_tile * width_of_square,
                                   vertical_tile * height_of_square,
                                   width_of_square + (horizontal_tile * width_of_square),
                                   height_of_square + (height_of_square * vertical_tile)])

    # For loop that generates polygons could be rewritten, but maybe harder to read?
    # polygons = [x1,y1, x1,y2, x2,y2, x2,y1 for x1,y1, x2,y2 in dimensions]

    # last_column = [(horizontal_tiles - 1) + horizontal_tiles * i for i in range(vertical_tiles)]
    last_column = []
    for i in range(vertical_tiles):
        last_column.append((horizontal_tiles-1)+horizontal_tiles*i)

    last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

    polygons = []
    for x1, y1, x2, y2 in dimensions:
        polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

    polygon_indices = []
    for i in range((vertical_tiles * horizontal_tiles) - 1):
        if i not in last_row and i not in last_column:
            polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

    for a, b, c, d in polygon_indices:
        dx = random.randint(-magnitude, magnitude)
        dy = random.randint(-magnitude, magnitude)

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
        polygons[a] = [x1, y1,
                        x2, y2,
                        x3 + dx, y3 + dy,
                        x4, y4]

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
        polygons[b] = [x1, y1,
                        x2 + dx, y2 + dy,
                        x3, y3,
                        x4, y4]

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
        polygons[c] = [x1, y1,
                        x2, y2,
                        x3, y3,
                        x4 + dx, y4 + dy]

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
        polygons[d] = [x1 + dx, y1 + dy,
                        x2, y2,
                        x3, y3,
                        x4, y4]

    generated_mesh = []
    for i in range(len(dimensions)):
        generated_mesh.append([dimensions[i], polygons[i]])
    return image.transform(image.size, Image.MESH, generated_mesh, resample=Image.BICUBIC)

def center_zoom(image, min_factor=1, max_factor=1):#***************************************************8
    min_factor = min_factor
    max_factor = max_factor

    factor = round(random.uniform(min_factor, max_factor), 2)

    w, h = image.size

    image_zoomed = image.resize((int(round(image.size[0] * factor)),
                                    int(round(image.size[1] * factor))),
                                    resample=Image.BICUBIC)
    w_zoomed, h_zoomed = image_zoomed.size
    return image_zoomed.crop((floor((float(w_zoomed) / 2) - (float(w) / 2)),
                                floor((float(h_zoomed) / 2) - (float(h) / 2)),
                                floor((float(w_zoomed) / 2) + (float(w) / 2)),
                                floor((float(h_zoomed) / 2) + (float(h) / 2))))

def zoom_random(image, percentage_area, randomise):#***************************************************8
    if randomise:
        r_percentage_area = round(random.uniform(0.1, percentage_area), 2)
    else:
        r_percentage_area = percentage_area

    w, h = image.size
    w_new = int(floor(w * r_percentage_area))
    h_new = int(floor(h * r_percentage_area))

    random_left_shift = random.randint(0, (w - w_new))  # Note: randint() is from uniform distribution.
    random_down_shift = random.randint(0, (h - h_new))

    image = image.crop((random_left_shift, random_down_shift, w_new + random_left_shift, h_new + random_down_shift))
    return image.resize((w, h), resample=Image.BICUBIC)

def random_erasing(image, rectangle_area=0.3, repetitions=1):#***************************************************8
    
    for i in range (repetitions):
        w, h = image.size

        w_occlusion_max = int(w * rectangle_area)
        h_occlusion_max = int(h * rectangle_area)

        w_occlusion_min = int(w * 0.1)
        h_occlusion_min = int(h * 0.1)

        w_occlusion = random.randint(w_occlusion_min, w_occlusion_max)
        h_occlusion = random.randint(h_occlusion_min, h_occlusion_max)

        if len(image.getbands()) == 1:
            rectangle = Image.fromarray(np.uint8(  np.random.rand(w_occlusion, h_occlusion) * 255  ))
        else:
            rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion, len(image.getbands())) * 255))

        random_position_x = random.randint(0, w - w_occlusion)
        random_position_y = random.randint(0, h - h_occlusion)
        image.paste(rectangle, (random_position_x, random_position_y))
    
    return image

def shifts(image_in, horizontal_max=0.2, vertical_max=0.2, randomise=False, fill="nearest"):#*************************************************** erro colorida

    width, height = image_in.size
    
    image = np.array(image_in).astype('uint8')
    
    if randomise:
        horizontal_shift = random.uniform(-abs( horizontal_max * width ), abs( horizontal_max * width ))
        vertical_shift = random.uniform(-abs( vertical_max * height ), abs( vertical_max * height ))
    else:
        horizontal_shift = horizontal_max
        vertical_shift = vertical_max
    
    if image_in.mode == 'RGB':
        image = shift(image, [vertical_shift, horizontal_shift, 0],  cval=0, mode=fill)
    else:
        image = shift(image, [vertical_shift, horizontal_shift],  cval=0, mode=fill)
    
    return Image.fromarray(image)


def  rotation(image, max_left_rotation=90, max_right_rotation=90, fill='edge'): #constant`, `edge`, `wrap`, `reflect` or `symmetric`#***************************************************8
    max_left_rotation = -abs(max_left_rotation)   # Ensure always negative
    max_right_rotation = abs(max_right_rotation)  # Ensure always positive
    
    return transform.rotate(image,  random.uniform(max_left_rotation, max_right_rotation) , mode=fill)
'''
def edge(image, minVal=100, maxVal=100):#***************************************************8
    return  cv2.Canny(image, minVal=minVal, maxVal=maxVal)
'''

def edge(image):#***************************************************8
    return  cv2.Canny(image,100,100)

def gaussian(image, sig=1.0, fill='nearest' ):#***************************************************8
    return gaussian_filter(image, sigma=sig, order=0, cval=0.0, truncate=4.0, mode=fill)

def random_noise(image, mode1='s&p'):#***************************************************8
    return util.random_noise(image, mode=mode1, seed=None, clip=True)




# Databricks notebook source
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import math
from math import floor, ceil
import random
import pandas as pd
from IPython.display import display

from scipy.ndimage import zoom
from scipy.ndimage import grey_erosion
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import shift

from skimage import transform
from skimage import util

# vetor de string:

#cada elemento do vetor vc executa uma função/ parametro dela tbm
import random
from numpy import select

from numpy.random import laplace

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
#from tqdm import tqdm

import os
import pathlib
import random

# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
# from tensorflow.keras import layers, models, utils

from skimage import transform

from random import randint

from skimage import util

# COMMAND ----------

import Aug

# COMMAND ----------

class Operacao:
    #prob = 0
    def __init__(self, prob):
        
        if prob > 1 or  prob < 0:
            raise Exception('prob must be less than or equal to 1 and greater than or equal to 0')
        else:
            self.prob = prob

    def execute(self):
            raise NotImplementedError("abs class")
    def get_class_name(self):
        return self.__class__.__name__
    def get_dict_atrs(self):
        return self.__dict__

# COMMAND ----------

class Skew (Operacao):
    def __init__(self, prob, skew_type='RANDOM', magnitude=0.5):
        Operacao.__init__(self, prob)
        self.skew_type = skew_type
        self.magnitude = magnitude
    def execute(self, image):
        return skew(image, skew_type=self.skew_type, magnitude=self.magnitude)

# COMMAND ----------

class Invert (Operacao):
    def __init__(self, prob):
        Operacao.__init__(self, prob)
    def execute(self, image):
        return ImageOps.invert(image)

# COMMAND ----------

class Rotacao(Operacao):
    def __init__(self, prob, max_left_rotation=90, max_right_rotation=90, fill='edge'):
        Operacao.__init__(self, prob)
        self.max_left_rotation = max_left_rotation 
        self.max_right_rotation = max_right_rotation
        self.fill = fill #constant`, `edge`, `wrap`, `reflect` or `symmetric`
    def execute(self, image):    
        return Image.fromarray(np.uint8(rotation(np.array(image).astype('uint8'), max_left_rotation=self.max_left_rotation, max_right_rotation=self.max_right_rotation)*255 ))

# COMMAND ----------

class Zoom_Random(Operacao):
    def __init__(self, prob, percentage_area=1, randomise=False):
        Operacao.__init__(self, prob)
        self.percentage_area = percentage_area
        self.randomise       = randomise
    def execute(self, image):
        return zoom_random(image, self.percentage_area, self.randomise)

# COMMAND ----------

class Random_Noise(Operacao):
    def __init__(self, prob, mode='s&p'):
        Operacao.__init__(self, prob)
        self.mode = mode#gaussian,localvar,poisson,salt,pepper,s&p,speckle
    def execute(self, image):
        return Image.fromarray(np.uint8(random_noise(np.array(image).astype('uint8'))*255 ))

# COMMAND ----------

class Gaussian(Operacao):# argumentos-----------------------------------
    def __init__(self, prob, sig=2.0, fill='nearest'):
        Operacao.__init__(self, prob)
        self.sig=sig
        self.fill = fill
    def execute(self, image):
        return gaussian(image, sig=self.sig, fill=self.fill)

# COMMAND ----------

class Random_Erasing(Operacao):
    def __init__(self, prob, rectangle_area=0.4, repetitions=1):
        Operacao.__init__(self, prob)
        self.rectangle_area = rectangle_area
        self.repetitions = repetitions
    def execute(self, image):
        return random_erasing(image, rectangle_area = self.rectangle_area, repetitions=self.repetitions)
        

# COMMAND ----------

class Shift(Operacao):
    def __init__(self, prob, horizontal_max=0.2, vertical_max=0.2, randomise=False, fill='nearest'):
        Operacao.__init__(self, prob)
        self.horizontal_max = horizontal_max
        self.vertical_max = vertical_max
        self.randomise = randomise
        self.fill = fill
    def execute(self, image):        
        return shifts(image, horizontal_max=self.horizontal_max, vertical_max=self.vertical_max, randomise=self.randomise, fill=self.fill)

# COMMAND ----------

class Zoom(Operacao):
    def __init__(self, prob, min_factor=1, max_factor=2):
        Operacao.__init__(self, prob)
        self.min_factor = min_factor
        self.max_factor = max_factor
    def execute(self, image):
        return center_zoom(image, min_factor=self.min_factor, max_factor=self.max_factor)

# COMMAND ----------

class Distort(Operacao):
    def __init__(self, prob, grid_width=4, grid_height=4, magnitude=5):
        Operacao.__init__(self, prob)
        self.grid_width  = grid_width
        self.grid_height = grid_height
        self.magnitude   = magnitude
    def execute(self, image):
        return distort(image, grid_width=self.grid_width, grid_height=self.grid_height, magnitude=self.magnitude)

# COMMAND ----------

class Shear(Operacao):
    def __init__(self, prob, max_shear_left = 4, max_shear_right=4):
        Operacao.__init__(self, prob)
        self.max_shear_left  = max_shear_left
        self.max_shear_right = max_shear_right
    def execute(self, image):
        return shear(image,  max_shear_left=self.max_shear_left,  max_shear_right=self.max_shear_right)

# COMMAND ----------

class Flip(Operacao):
    def __init__(self, prob, top_bottom_left_right='RANDOM'):
        Operacao.__init__(self, prob)
        self.top_bottom_left_right  = top_bottom_left_right
    def execute(self, image):
        return flip(image,  top_bottom_left_right=self.top_bottom_left_right)

# COMMAND ----------

class Skew(Operacao):
    def __init__(self, prob, skew_type='RANDOM', magnitude=1):
        Operacao.__init__(self, prob)
        self.skew_type = skew_type
        self.magnitude = magnitude
    def execute(self, image):
        return skew(image,  skew_type=self.skew_type,  magnitude=self.magnitude)

# COMMAND ----------

class Grey_Erosion(Operacao):#******adicionar argumentos
    def __init__(self, prob):
        Operacao.__init__(self, prob)
    def execute(self, image):
        return Image.fromarray(np.uint8(grey_erosion(np.array(image).astype('uint8'),size=(3,3))*255 ))

# COMMAND ----------

'''class Edge(Operacao):#******adicionar argumentos
    def __init__(self, prob, minVal=100, maxVal=100):
        Operacao.__init__(self, prob)
        self.minVal=minVal
        self.maxVal=maxVal
    def execute(self, image):
        return Image.fromarray(np.uint8(Aug.edge(np.array(image).astype('uint8'), minVal=self.minVal, maxVal=self.maxVal)*255 ))'''
class Edge(Operacao):
    def __init__(self, prob):
        Operacao.__init__(self, prob)
    def execute(self, image):
        return Image.fromarray(np.uint8(edge(np.array(image).astype('uint8'))*255 ))

# COMMAND ----------

class Contrast(Operacao):
    def __init__(self, prob, min_factor=1, max_factor=1):
        Operacao.__init__(self, prob)
        self.min_factor = min_factor
        self.max_factor = max_factor
    def execute(self, image):
        return contrast(image, min_factor=self.min_factor, max_factor=self.max_factor)

# COMMAND ----------

class Color(Operacao):
    def __init__(self, prob, min_factor=1, max_factor=1):
        Operacao.__init__(self, prob)
        self.min_factor = min_factor
        self.max_factor = max_factor
    def execute(self, image):
        return color(image, min_factor=self.min_factor, max_factor=self.max_factor)

# COMMAND ----------

class Brilho(Operacao):
    def __init__(self, prob, min_factor=1, max_factor=1):
        Operacao.__init__(self, prob)
        self.min_factor = min_factor
        self.max_factor = max_factor
    def execute(self, image):
        return brilho(image, min_factor=self.min_factor, max_factor=self.max_factor)

# COMMAND ----------

class Pipe:
    def __init__(self):
        self.lista_de_operacoes = []
    def add(self, objeto):
        self.lista_de_operacoes.append(objeto)
    def remove(self, posicao = -1):
        self.lista_de_operacoes.pop(posicao)
    def replace(self, index, objeto):
        self.lista_de_operacoes[index] = objeto
    def operar(self, image:list, class_img:int = -1, string_class = None, vezes = 1):#Não retorna a imagem original | Se a imagem não for modificada não é adicionada
        image = Image.fromarray(image)
        aux_2 = []
        alterou = False
        for i in range(vezes):
            aux = image.copy()
            alterou = False #ta uma bosta-----------------------
            for operacao in self.lista_de_operacoes:
                
                if operacao.prob >= random.uniform(0,1):
                    aux = operacao.execute(aux)
                    alterou = True
                else:
                    continue
#--------------------------------------------------------------------------------------------ATENÇÃO            
            if alterou:
                # aux_2.append(aux)
                #aux_2.append( [      aux    , class_img] )
                if type(image) == type(aux):
                    aux_2.append([  np.array(aux).astype('uint8')  , class_img, string_class])
                else:
                    print('ERRO-------------------------------------------aux operar')
                #aux_2.append([  aux  ,class_img,string_class])  
                #alterou = False #isso aqui é o fino-----------------------------
            else:
                continue
        return aux_2
#----------------------------------------------------------------------------------------------ATENÇÃO
    def print_pipe(self):
        print(f"Operações:")
        '''
        aux = True
        for operacao in self.lista_de_operacoes:
            if aux:
                head = {"   Operação   ": operacao.get_class_name()}
                aux = False
            else:
                head = {"--------------": operacao.get_class_name()}
            atributos = operacao.get_dict_atrs()
            head.update(atributos)
            table = PrettyTable( head.keys() )
            table.set_style(MSWORD_FRIENDLY)
            table.add_row( head.values() )
            print(table)
            print()
        print()
        '''