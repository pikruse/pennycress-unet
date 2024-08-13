# Extract color features

# Authors: Mirko Pavicic (main), John Lagergren (minor changes)
# Created: 2023-05-03
# Last modified: 2023-12-20
# Version: 1.4

# 1.2
# segmentation prefix removed
# Peter saved masks as RGB instead binary
# A correction was added in line 102

# 1.3
# Remove hidden files from input list
# Add error handling for corrupted images

# 1.4
# Added multiprocessing for faster processing
# applying segmentation BEFORE color analysis for faster processing
# saves results separately for each image, then merges them into a single file

# Example usage:
# MODALITY=rgb2
# SCRIPT=/mnt/DGX01/Personal/lagergrenj/codebase/projects/appl/scripts/mirko/color_analysis_v1.4.py
# INPUTDIR=/mnt/DGX01/Personal/lagergrenj/codebase/projects/appl/data/level_1/cbi-poplar/$MODALITY/
# SEGDIR=/mnt/DGX01/Personal/lagergrenj/codebase/projects/appl/data/level_2/cbi-poplar/$MODALITY/segmentations/
# RESULTSDIR=/mnt/DGX01/Personal/lagergrenj/codebase/projects/appl/data/level_2/cbi-poplar/$MODALITY/
# NJOBS=64
# python $SCRIPT --inputdir $INPUTDIR --segdir $SEGDIR --resultsdir $RESULTSDIR --njobs $NJOBS

##### Print header #######
print(
'------------------------------------------------------\n'+
'           Color analysis v1.4\n' +
'------------------------------------------------------\n')

# Load skimage package
from skimage import io, color
import pandas as pd
import numpy as np
import os
import argparse
import time
from multiprocessing import Pool

###### Create arguments #########
parser = argparse.ArgumentParser(description = 'Extract color features')
parser.add_argument('--inputdir', type=str, default='./input/', help='Path to folder containing input images. It defaults to ./input folder in your current directory')
parser.add_argument('--segdir', type=str, default='./segmentations/', help='Path to folder containing segmented binary images')
parser.add_argument('--resultsdir', type=str, default='./', help='Path to where results should be stored. It defaults to your current directory')
parser.add_argument('--njobs', type=int, default=1, help='Number of jobs to run in parallel. Default = 1')

args = parser.parse_args()

# Pass arguments to objects
# input dir
inputdir = args.inputdir
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
'Input folder: ' + str(inputdir) + '\n' +
'Segmentations folder: ' + str(segmentation_path) + '\n' +
'Results folder: ' + str(results_dir) + '\n'
)

# Check if color_analysis folder exists
if not os.path.exists(results_dir + 'color_analysis/'):
    # Print creating folder if not exists
    print('Creating ' + results_dir + 'color_analysis/ folder')
    # Make folder
    os.makedirs(results_dir + 'color_analysis/')
else:
    # Print folder already exists if exists
    print(results_dir + 'color_analysis/ folder already exists')

# create error file
with open(results_dir + 'color_error_images.txt', 'w') as f:
    f.write('')

# Get input filenames
image_file_names = sorted(os.listdir(inputdir))

# remove hidden files
image_file_names = [x for x in image_file_names if not x.startswith('.')]

# Get segmentations filenames
segmentations_file_names = os.listdir(segmentation_path)

# remove hidden files
segmentations_file_names = [x for x in segmentations_file_names if not x.startswith('.')]

# Get number of images
print('Number of images in input folder: ' + str(len(image_file_names)))

# Get number of segmentations
print('Number of segementations in input folder: ' + str(len(segmentations_file_names)) + '\n')

# Print action message
print('Measuring color\n')

# Start for loop

# Define function to extract features from segmentations
def extract_features(img):

    # print action message
    print('Processing ' + img)

    # Import image with error handling
    try:
        im = io.imread(inputdir + img)
    except:
        print('Error: could not read image ' + img)
        # Save error image name to an error file
        with open(results_dir + 'color_error_images.txt', 'a') as f:
            f.write(img + '\n')
        return

    # import segmentation
    if 'masked_' in img:
        segmentation = io.imread(segmentation_path + img.replace('masked_', 'segmentation_'))
    else:
        segmentation = io.imread(segmentation_path + 'segmentation_' + img)

    # if segm has 1 channels,
    if len(segmentation.shape) != 3:
        # save segmentation_pathentation as gray image
        gray_image = segmentation
    else:
        # Convert the RGB image to grayscale
        gray_image = skimage.color.rgb2gray(segmentation)
    
    # Convert to binary
    mask_binary = gray_image > 0.5

    # apply mask to image, PNG images has an extra alpha channel that's why I used indexes
    im = im[:, :, 0:3][mask_binary] # shape: [N, 3]
  
    # Convert to lab and hsb
    lab_img = color.rgb2lab(im)
    hsb_img = color.rgb2hsv(im)
    
    # Split the image into its color channels
    red_channel = im[..., 0]   # Extract the red channel
    green_channel = im[..., 1] # Extract the green channel
    blue_channel = im[..., 2]  # Extract the blue channel
    
    # Split the LAB image into its color channels
    L_channel = lab_img[..., 0]  # Extract the L channel
    A_channel = lab_img[..., 1]  # Extract the A channel
    B_channel = lab_img[..., 2]  # Extract the B channel
    
    # Split the HSB image into its color channels
    H_channel = hsb_img[..., 0]  # Extract the H channel
    S_channel = hsb_img[..., 1]  # Extract the S channel
    V_channel = hsb_img[..., 2]  # Extract the V channel
    

    ######## Extract only plant pixels present in segmentation #######
    
    # This is lazy coding but functional
    # Each operation does:
    # 1. Get pixels that are present only in the binary mask and convert to dataframe
    # 2. Add channel and color space columns
    
    ## RGB
    red_df = pd.DataFrame(red_channel, columns=['pixel_intensity'])
    red_df['channel'], red_df['color_space']  = 'red', 'RGB'
    
    green_df = pd.DataFrame(green_channel, columns=['pixel_intensity'])
    green_df['channel'], green_df['color_space']  = 'green', 'RGB'
    
    blue_df = pd.DataFrame(blue_channel, columns=['pixel_intensity'])
    blue_df['channel'], blue_df['color_space']  = 'blue', 'RGB'
    
    ## HSB
    H_df = pd.DataFrame(H_channel, columns=['pixel_intensity'])
    H_df['channel'], H_df['color_space']  = 'hue', 'HSB'
    
    S_df = pd.DataFrame(S_channel, columns=['pixel_intensity'])
    S_df['channel'], S_df['color_space']  = 'saturation', 'HSB'
    
    V_df = pd.DataFrame(V_channel, columns=['pixel_intensity'])
    V_df['channel'], V_df['color_space']  = 'brightness', 'HSB'
    
    ## LAB
    L_df = pd.DataFrame(L_channel, columns=['pixel_intensity'])
    L_df['channel'], L_df['color_space']  = 'L', 'LAB'
    
    A_df = pd.DataFrame(A_channel, columns=['pixel_intensity'])
    A_df['channel'], A_df['color_space']  = 'A', 'LAB'
    
    B_df = pd.DataFrame(B_channel, columns=['pixel_intensity'])
    B_df['channel'], B_df['color_space']  = 'B', 'LAB'
    
    ## Concatenate
    color_df = pd.concat([red_df,green_df,blue_df, H_df, S_df, V_df, L_df, A_df, B_df])
    
    # Group by 'Group1' and 'Group2', and get the mean and standard deviation of 'Value'
    color_result = color_df.groupby(['color_space', 'channel']).agg({'pixel_intensity': ['mean', 'std']})
    
    # unnest the column labels
    color_result.columns = ["_".join(x) for x in np.ravel(color_result.columns)]
    
    # convert index to columns
    color_result = color_result.reset_index()
    
    # Create new name
    color_result['new_name'] = color_result['color_space'] + '_' + color_result['channel']
    
    # drop columns
    color_result = color_result.drop(columns=['color_space', 'channel'])
    
    # rename columns
    color_result = color_result.rename(columns={'pixel_intensity_mean':'mean', 'pixel_intensity_std':'std'})
    
    # Convert to long format
    color_result_long = pd.melt(color_result, 
            id_vars=['new_name'], 
            value_vars=['mean', 'std'], 
            var_name='variable', 
            value_name='value')
    
    # Create unique feature names
    color_result_long['unique_name'] = 'color_'+color_result_long['new_name']+'_'+color_result_long['variable']
    
    # drop columns
    color_result_long = color_result_long.drop(columns=['new_name', 'variable'])
    
    # add id
    color_result_long['image_name'] = img
    
    # Convert to long format
    color_result_final = color_result_long.pivot(index = 'image_name', columns='unique_name', values='value').reset_index()
    
    # Export data
    color_result_final.to_csv(results_dir + 'color_analysis/' + img.split('.')[0] + '_color_analysis.csv', index=False)

# run function in parallel
if __name__ == '__main__':
    with Pool(njobs) as p:
        p.map(extract_features, image_file_names)

# Merge all dataframes into a single dataframe
csv_files = os.listdir(results_dir + 'color_analysis/')
csv_files = sorted([x for x in csv_files if '.csv' in x])
df = pd.concat([pd.read_csv(results_dir + 'color_analysis/' + x) for x in csv_files])
df.to_csv(results_dir + 'results_color_analysis.csv', index=False)

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
