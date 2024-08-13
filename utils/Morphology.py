# Measure features from segmentations

# Authors: Mirko Pavicic (main), John Lagergren (minor changes)
# Created: 2023-03-29
# Last modified: 2023-12-20
# Version: 1.7

# Updates with respect last verision
# added multiprocessing for faster processing
# saves results separately for each image, then merges them into a single file

# 1.2
# image name parser was removed to allow multiplatform usage

# 1.3
# area_convex was included from regionprops function.

# circularity, roundness and aspect_ratio were included
# formulas were taken from imageJ.

# 1.4
# added rotated rectangle features

# 1.5
# added RMS

# 1.6
# remove hidden files from segmentations list

# 1.7
# multiprocessing was added to speed up the process
# saves results separately for each image, then merges them into a single file

# Example usage:
# MODALITY=rgb2
# SCRIPT=/mnt/DGX01/Personal/lagergrenj/codebase/projects/appl/scripts/mirko/morphology_analysis_v1.7.py
# SEGDIR=/mnt/DGX01/Personal/lagergrenj/codebase/projects/appl/data/level_2/cbi-poplar/$MODALITY/segmentations/
# RESULTSDIR=/mnt/DGX01/Personal/lagergrenj/codebase/projects/appl/data/level_2/cbi-poplar/$MODALITY/
# NJOBS=64
# python $SCRIPT --segdir $SEGDIR --resultsdir $RESULTSDIR --njobs $NJOBS

##### Print header #######

print(
'------------------------------------------------------\n'+
'           Feature extractor v1.7\n' +
'------------------------------------------------------\n')

# Load skimage package
import skimage
from skimage import measure
import pandas as pd
import numpy as np
import os
import argparse
import time
import cv2
import math
from multiprocessing import Pool

###### Create arguments #########
parser = argparse.ArgumentParser(description = 'Compute shape descriptors on segmented binary images')
parser.add_argument('--segdir', type=str, default='./segmentations/', help='Path to folder containing segmented binary images')
parser.add_argument('--resultsdir', type=str, default='./', help='Path to where results should be stored. It defaults to your current directory')
parser.add_argument('--njobs', type=int, default=1, help='Number of jobs to run in parallel. Default = 1')

args = parser.parse_args()

# Pass arguments to objects
# Define segmentations path
segmentation_path = args.segdir
# Define results path
results_dir = args.resultsdir
njobs = args.njobs


# get start time
start_time = time.time()


##### Print arguments #####
print(
'Arguments:\n\n' +
'Segmentations folder: ' + str(segmentation_path) + '\n' +
'Results folder: ' + str(results_dir) + '\n'
)

# Check if morphology folder exists
if not os.path.exists(results_dir + 'morphology/'):
    # Print creating folder if not exists
    print('Creating ' + results_dir + 'morphology/ folder')
    # Make folder
    os.makedirs(results_dir + 'morphology/')
else:
    # Print folder already exists if exists
    print(results_dir + 'morphology/ folder already exists')

# Get segmentations filenames
segmentations_file_names = os.listdir(segmentation_path)

# remove hidden files
segmentations_file_names = [x for x in segmentations_file_names if not x.startswith('.')]

# Get number of images
print('\nNumber of images in input folder: ' + str(len(segmentations_file_names)) + '\n')

# Print action message
print('Measuring phenotypes\n')

######## Start for loop ########

# Define function to extract features from segmentations
def extract_features(img):

    # print action message
    print('Processing ' + img)
    # import segmentation
    segm = skimage.io.imread(segmentation_path + img)

    # if segm has 1 channels,
    if len(segm.shape) != 3:
        # save segmentation as gray image
        gray_image = segm
    else:
        # Convert the RGB image to grayscale
        gray_image = skimage.color.rgb2gray(segm)

    # Threshold the grayscale image to create a binary image
    binary_image = (gray_image > 0.5) * 255

    ######## Rotated rectangle ########

    # Find the contours of the binary mask
    contours = measure.find_contours(binary_image, 0.5)

    # Change the data type to float32
    contours = contours[0].astype(np.float32)

    # Calculate the minimum bounding rectangle
    rect = cv2.minAreaRect(contours)

    # get box points
    box = cv2.boxPoints(rect)

    # Create function to calculate width, height, hypotenuse and area of a rectangle
    def rect_prop(box_coords):
        '''
        Function to calculate width, height, hypotenuse and area of a rectangle
        '''
        # start empty list
        prop = []

        # extract euclidean distances between points
        for i in range(1,4):
            # extract x and y coordinates
            x = box[:,0]
            y = box[:,1]

            # compute euclidean distance
            dist = np.sqrt((x[0]-x[i])**2 + (y[0]-y[i])**2)

            # append to list
            prop.append(dist)

        # sort list
        prop.sort()

        # compute width, height, hypotenuse and area
        width = prop[0]
        height = prop[1]
        hypotenuse = prop[2]
        area = width*height

        # create dictionary compatible with pandas
        prop = {'rr_width': width, 'rr_height': height, 'rr_hypotenuse': hypotenuse, 'rr_area': area}

        # return dictionary
        return prop

    # apply function to extract features from rotated rectangle
    rot_rect = rect_prop(box)

    ######## Extract extra features using regions prop ########

    # Create feature list
    feature_list = [
    'area', 'bbox', 'area_bbox', 'eccentricity', 'equivalent_diameter_area',
    'euler_number', 'extent', 'feret_diameter_max', 'axis_major_length', 'axis_minor_length',
    'orientation', 'perimeter', 'perimeter_crofton', 'solidity', 'area_convex',
    'axis_major_length', 'axis_minor_length']
    
    # Extract mask features
    features = measure.regionprops_table(binary_image, properties=(feature_list))

    ######## Start dictionary and convert to data frame ########
    
    # convert segmentation name to dictionary
    features_dic = {"image_name" : img}
    
    # append features_dic at the beggining of features dictionary
    features_dic.update(features)

    # append rot_rect dictionary to features_dic
    features_dic.update(rot_rect)
    
    # Convert to dataframe
    features_df = pd.DataFrame(features_dic)

    ######## Compute extra features ########
    
    # Compute extra features
    features_df = (features_df
    # Width
     .assign(bb_width = features_df['bbox-3'] - features_df['bbox-1'])
    # Height
     .assign(bb_height = features_df['bbox-2'] - features_df['bbox-0'])
    # Circularity
     .assign(circularity = (4*np.pi*features_df['area'])/(features_df['perimeter']**2))
    # Aspect ratio
     .assign(aspect_ratio = features_df['axis_major_length']/features_df['axis_minor_length'])
    # Roundness
     .assign(roundness = (4*features_df['area'])/(np.pi*features_df['axis_major_length']**2))
    # rms
     .assign(rms = 2*(math.sqrt((0.5*features_df['axis_major_length'])**2-(0.5*features_df['axis_minor_length'])**2))/features_df['axis_major_length'])
     .drop(columns=['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']))
    
    # Reorder columns
    features_df = features_df[['image_name', 'area', 'area_bbox', 'area_convex', 
                               'bb_width', 'bb_height', 'rr_area', 'rr_width', 'rr_height', 
                               'rr_hypotenuse', 'perimeter', 'perimeter_crofton', 'equivalent_diameter_area', 
                               'euler_number', 'feret_diameter_max', 'axis_major_length', 'axis_minor_length', 
                               'orientation', 'aspect_ratio', 'circularity', 'eccentricity', 
                               'extent', 'rms', 'roundness', 'solidity']]
    
    # Export data
    features_df.to_csv(results_dir + 'morphology/' + img.split('.')[0] + '_morphology.csv', index=False)

# run function in parallel
if __name__ == '__main__':
    with Pool(njobs) as p:
        p.map(extract_features, segmentations_file_names)

# Merge all dataframes into a single dataframe
csv_files = os.listdir(results_dir + 'morphology/')
csv_files = sorted([x for x in csv_files if '.csv' in x])
df = pd.concat([pd.read_csv(results_dir + 'morphology/' + x) for x in csv_files])
df.to_csv(results_dir + 'results_morphology.csv', index=False)

# Print action message
print('Done\n')

# Get end time
end_time = time.time()
total_time = end_time - start_time
time_per_image = total_time/len(segmentations_file_names)

# Print ellapsed time
print(f'Elapsed time: {total_time:.2f} seconds')
print(f'Total images: {len(segmentations_file_names)} images')
print(f'Time / image: {time_per_image:.4f} seconds')
    