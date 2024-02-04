#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import cv2
from matplotlib import image
import skimage
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
from PIL import Image


def MkDir(output_Path):
	output_directory = os.path.dirname(output_Path)
	if not os.path.exists(output_directory):
		os.makedirs(output_directory)
    
       
def Saveimg(size, filters, x_len, y_len, name):
    fig, axes = plt.subplots(y_len, x_len, figsize=size)
    for idx, ax in enumerate(axes.flatten()):
        ax.imshow(filters[idx], cmap='gray')
        ax.axis('off')
    plt.savefig(name)
    plt.close()

def gauss(sd, kernel, xd=None, yd=None):
	interval = kernel/2.5
	x, y = np.meshgrid(np.linspace(-interval, interval, kernel),
						np.linspace(-interval, interval, kernel))

	if xd is not None and yd is not None:
		grid = np.array([x.flatten(), y.flatten()])
		GFactor1 = [1 / np.sqrt(2 * 3.14 * 3*sd**2) * np.exp(-(grid[0,...]**2) / (2 * 3*sd**2)),
			       -1/np.sqrt(2 * 3.14 * 3*sd**2) * np.exp(-(grid[0,...]**2) / (2 * 3*sd**2))*(grid[0,...]/3*sd**2),
				   1 / np.sqrt(2 * 3.14 * 3*sd**2) * np.exp(-(grid[0,...]**2)/ (2 * 3*sd**2))*((grid[0,...]**2-3*sd**2)/3*sd**4)]
		GFactor2 = [1 / np.sqrt(2 * 3.14 * sd**2) * np.exp(-(grid[1,...]**2) / (2 * sd**2)),
			       -1/np.sqrt(2 * 3.14 * sd**2) * np.exp(-(grid[1,...]**2) / (2 * sd**2))*(grid[1,...]/sd**2),
				   1 / np.sqrt(2 * 3.14 * sd**2) * np.exp(-(grid[1,...]**2)/ (2 * sd**2))*((grid[1,...]**2-sd**2)/sd**4)]
		gauss = GFactor1[xd] * GFactor2[yd]
		filt = np.reshape(gauss, (kernel, kernel))
		return filt

	else:
		gauss = 1 / (2 * np.pi * sd**2) * np.exp(-(x**2 + y**2) / (2 * sd**2))
		return gauss

def laplace(sd, kernel):
      filter = np.array([[0,1,0],[1,-4,1],[0,1,0]])
      gaussian = gauss(sd, kernel)
      log = cv2.filter2D(gaussian,-1,filter)
      return log

def dog(sd, kernel, y_sobel):
      dog_filters = []
      orientations = np.linspace(0,360,16)
      for i in sd:
            gauss_kernel = gauss(i, kernel)
            sobel_convolve = cv2.filter2D(gauss_kernel, -1,  y_sobel)
            for j in orientations:
                  filter = skimage.transform.rotate(sobel_convolve, j)
                  dog_filters.append(filter)
      return dog_filters

def LeungMalik(sd, kernel):
    LMfilters = []
    orientations = np.linspace(0, 180, 6)

    def rotate_and_append(gauss_kernel, orientations):
        return [skimage.transform.rotate(gauss_kernel, j) for j in orientations]

    for i in sd:
        gauss_kernel_1st_order = gauss(i, kernel, 0, 1)
        LMfilters.extend(rotate_and_append(gauss_kernel_1st_order, orientations))

        gauss_kernel_2nd_order = gauss(i, kernel, 0, 2)
        LMfilters.extend(rotate_and_append(gauss_kernel_2nd_order, orientations))
    for i in sd:
        LMfilters.append(laplace(i, kernel))
    for i in sd:  
         LMfilters.append(laplace(3 * i, kernel))  
    for i in sd:
         LMfilters.append(gauss(i, kernel))

    return LMfilters


def gabor_filters(sigma , theta, Lambda, psi, gamma, number):
	filters = []
	orientations = np.linspace(90,270,number)
	for i in range(0,len(sigma)):
		x = np.ceil(max(1, max(abs(3 * sigma[i] * np.cos(theta)), abs(3 * float(sigma[i])/gamma * np.sin(theta)))))
		y = np.ceil(max(1, max(abs(3 * sigma[i] * np.sin(theta)), abs(3 * float(sigma[i])/gamma * np.cos(theta)))))
		(x, y) = np.meshgrid(np.arange(-x, y + 1), np.arange(-y, y + 1))
		x_theta = x * np.cos(theta) + y * np.sin(theta)
		y_theta = -x * np.sin(theta) + y * np.cos(theta)
		gabor_kernel = np.exp(-.5 * (x_theta ** 2 / sigma[i] ** 2 + y_theta ** 2 / float(sigma[i])/gamma ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
		for j in range(0, number):
			filter = skimage.transform.rotate(gabor_kernel, orientations[j])
			filters.append(filter)
	return filters
    
def texton_filter(image, filter_bank):
	map = np.array(image)
	for i in range(0, len(filter_bank)):
		filter = np.array(filter_bank[i])
		filter = cv2.filter2D(image,-1, filter)
		map = np.dstack((map, filter))
	return map


def texton(image, dog_F, lm_F, gabor_F, clusters):
	size = image.shape
	t_map_dog = texton_filter(image, dog_F)
	t_map_lm = texton_filter(image, lm_F)
	t_map_gabor = texton_filter(image, gabor_F)
	t_map = np.dstack((t_map_dog, t_map_lm, t_map_gabor))
	total_filters = t_map.shape[2]
	l = size[0]
	b = size[1]
	t = np.reshape(t_map, ((l*b),total_filters))    
	pred = KMeans(n_clusters=clusters, random_state=0).fit(t).predict(t)
	pred_ = np.reshape(pred, (l,b))
	return pred_

def brightness(image, clusters):       
	image = np.array(image)
	size = image.shape
	image_ = np.reshape(image, ((size[0]*size[1]),1))
	kmeans = KMeans(n_clusters=clusters, random_state=0).fit(image_)
	p = kmeans.predict(image_)                               
	p_ = np.reshape(p, (size[0],size[1]))
	
	return p_

def color(image, clusters): 
	image = np.array(image)
	size = image.shape
	image_ = np.reshape(image, ((size[0]*size[1]),size[2]))
	kmeans = KMeans(n_clusters=clusters, random_state=0).fit(image_)
	pred = kmeans.predict(image_)                               
	pred_ = np.reshape(pred, (size[0],size[1]))
	return pred_

def masks(scales):
	half_discs = []
	angles = [0, 180, 30, 210, 45, 225, 60, 240, 90, 270, 120, 300, 135, 315, 150, 330]
	no_of_disc = len(angles)
	for radius in scales:
		kernel_size = 2*radius + 1
		cc = radius
		kernel = np.zeros([kernel_size, kernel_size])
		for i in range(radius):
			for j in range(kernel_size):
				a = (i-cc)**2 + (j-cc)**2
				if a <= radius**2:
					kernel[i,j] = 1
		
		for i in range(0, no_of_disc):
			mask = skimage.transform.rotate(kernel, angles[i])
			mask[mask<=0.5] = 0
			mask[mask>0.5] = 1
			half_discs.append(mask)
	return half_discs

def Gradients(map, bins, filters):
	g = np.array(map)
	i = 0
	while i < len(filters)-1:    
		SquareDistance = Square_Distance(map, bins, filters[i], filters[i+1])
		g = np.dstack((g, SquareDistance)) 
		i += 2
	gradient = np.mean(g, axis = 2)
	return gradient

def Square_Distance(map, bins, mask, inv_mask):
	SquareDistance = map*0
	for i in range(0, bins):
		dis = np.zeros_like(map)
		dis[map == i] = 1
		a = cv2.filter2D(dis, -1, mask)
		b = cv2.filter2D(dis, -1, inv_mask)
		SquareDistance = SquareDistance + ((a - b)**2)/(a + b + 0.01)
	SquareDistance = SquareDistance/2
	return SquareDistance

    

def main():
	# """
	# Generate Difference of Gaussian Filter Bank: (DoG)
	# Display all the filters in this filter bank and save image as DoG.png,
	# use command "cv2.imwrite(...)"
	# """
	filter = np.array([[-1,0,1],
					[-4,0,4],
					[-1,0,1]])
	Oriented_Dog_filter = dog([3,5], 49, filter)
	pathdog = 'output/filters/OrientedDogFilter.png' 
	Saveimg((16,2), Oriented_Dog_filter, 16, 2, pathdog)

	# """
	# Generate Leung-Malik Filter Bank: (LM)
	# Display all the filters in this filter bank and save image as LM.png,
	# use command "cv2.imwrite(...)"
	# """
	LeungMalikFilter = LeungMalik([1, np.sqrt(2), 2, 2*np.sqrt(2)],49)
	pathlm = 'output/filters/LeungMalik.png'
	Saveimg((12,4), LeungMalikFilter, 12, 4,pathlm)
	# """
	# Generate Gabor Filter Bank: (Gabor)
	# Display all the filters in this filter bank and save image as Gabor.png,
	# use command "cv2.imwrite(...)"
	# """
	gaborFilter =gabor_filters([3,5,7,9,12], 0.25,1,1,1,8)
	pathgabor = 'output/filters/gabor.png'
	MkDir(pathgabor)
	Saveimg((8,5), gaborFilter, 8, 5, pathgabor)
	# """
	# Generate Half-disk masks
	# Display all the Half-disk masks and save image as HDMasks.png,
	# use command "cv2.imwrite(...)"
	# """
	hd = masks([5,10,15])
	hdpath = 'output/filters/HDmask.png'
	Saveimg((8,6),hd, 8,6,hdpath)
    
	for i in range(1,11):
		I = image.imread('../BSDS500/Images/' + str(i) + '.jpg')
		BWI = np.dot(I[...,:3], [0.2989, 0.5870, 0.1140])
		maps = []
		grads = []
		comparison = []
		TextonMap = texton(I, Oriented_Dog_filter, LeungMalikFilter, gaborFilter, 64)
		textonpred = 3*TextonMap
		cm = plt.get_cmap('gist_rainbow')
		textoncpred = cm(textonpred)
		maps.append(textoncpred)
		plt.imshow(textoncpred)
		plt.savefig('output/TextonMap/' + str(i) + '.png')
		plt.close()
		BrightMap = brightness(BWI,16)
		maps.append(BrightMap)
		plt.imshow(BrightMap, cmap = 'gray')
		plt.savefig('output/BrightMap/' + str(i) +'.png')
		plt.close()
		ColorMap = color(I,16)
		ColorPred = 30*ColorMap
		ColorCPred = cm(ColorPred)
		maps.append(ColorCPred)
		plt.imshow(ColorCPred)
		plt.savefig('output/ColorMap/' + str(i) +'.png')
		plt.close()
		Saveimg((12,6), maps, 3, 1, 'output/maps/' + str(i) + '.png')
		TextonGrad = Gradients(BrightMap, 64, hd)
		grads.append(TextonGrad)
		plt.imshow(TextonGrad)
		plt.savefig('output/TextonGrad/' + str(i) + '.png')
		plt.close()
		BrightnessGrad = Gradients(BrightMap, 16, hd)
		grads.append(BrightnessGrad)
		plt.imshow(BrightnessGrad)
		plt.savefig('output/BrightnessGrad/' + str(i) + '.png')
		plt.close()
		ColorGrad = Gradients(ColorMap,16,hd)
		grads.append(ColorGrad)
		plt.imshow(ColorGrad)
		plt.savefig('output/ColorGrad/' + str(i) + '.png')
		plt.close()
		Saveimg((12,6), grads, 3, 1, 'output/gradients/' + str(i) + '.png')
		cannyImg = image.imread('../BSDS500/CannyBaseline/' + str(i) + '.png')
		comparison.append(cannyImg)
		sobelImg = image.imread('../BSDS500/SobelBaseline/' + str(i) + '.png')
		comparison.append(sobelImg)
		result = (1/3)*(TextonGrad + BrightnessGrad + ColorGrad) * (0.5*cannyImg+ 0.5*sobelImg)
		comparison.append(result)
		plt.imshow(result, 'gray')
		plt.savefig('output/results/' + str(i) + '.png')
		plt.close()
		Saveimg((12, 6), comparison, 3, 1, 'output/comparison/' + str(i) + '.png')
          
          
if __name__ == '__main__':
    main()
              
          
          
          
          
          
          
		  
          
          



	# """
	# Generate Texton Map
	# Filter image using oriented gaussian filter bank
	# """


	# """
	# Generate texture ID's using K-means clustering
	# Display texton map and save image as TextonMap_ImageName.png,
	# use command "cv2.imwrite('...)"
	# """


	# """
	# Generate Texton Gradient (Tg)
	# Perform Chi-square calculation on Texton Map
	# Display Tg and save image as Tg_ImageName.png,
	# use command "cv2.imwrite(...)"
	# """


	# """
	# Generate Brightness Map
	# Perform brightness binning 
	# """


	# """
	# Generate Brightness Gradient (Bg)
	# Perform Chi-square calculation on Brightness Map
	# Display Bg and save image as Bg_ImageName.png,
	# use command "cv2.imwrite(...)"
	# """


	# """
	# Generate Color Map
	# Perform color binning or clustering
	# """


	# """
	# Generate Color Gradient (Cg)
	# Perform Chi-square calculation on Color Map
	# Display Cg and save image as Cg_ImageName.png,
	# use command "cv2.imwrite(...)"
	# """


	# """
	# Read Sobel Baseline
	# use command "cv2.imread(...)"
	# """


	# """
	# Read Canny Baseline
	# use command "cv2.imread(...)"
	# """


	# """
	# Combine responses to get pb-lite output
	# Display PbLite and save image as PbLite_ImageName.png
	# use command "cv2.imwrite(...)"
	# """
    





 


