#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 11:23:05 2018

@author: Tilemachos Bontzorlos

This is a sample code to experiment and transform the Singapore Maritime 
Dataset (SMD) .mat object detection files into a CSV format for further 
processing. This is a legacy script that creates some CSV files required for
the data analysis that takes place in the Jupyter Notebooks.

Dataset available here: https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset

If this dataset is used please cite it as:

D. K. Prasad, D. Rajan, L. Rachmawati, E. Rajabaly, and C. Quek, 
"Video Processing from Electro-optical Sensors for Object Detection and 
Tracking in Maritime Environment: A Survey," IEEE Transactions on Intelligent 
Transportation Systems (IEEE), 2017. 
"""

from scipy.io import loadmat
from os import listdir
from os.path import isfile, join

#PATH_TO_GT_FILES = "VIS_Onshore/ObjectGT" # here set the path to the ground truth .mat files
PATH_TO_GT_FILES = "NIR/ObjectGT"

class Frame:
    """
    This is a class to save the data for each video frame
    """
    csv_list = []
    csv_list_initialized = False
    
    def __init__(self, frame, image_name, bb, objects, motion, distance):
        """
        Parameters
        ----------
            frame : the frame number of the video. (string or int)
            image_name : the name of the image (for identification). (string)
            bb : bounding box coordinates of the objects. This is an array. 
                Each line is the bb of an object and corresponds to 
                [x_min,y_min,width,height]. See the dataset webpage for more
                info.
            objects : the type of objects. (array)
            motion : of the objects are moving or not. (array)
            distance : distance of each objects. (array)
            
        """
        self.frame = frame
        self.image_name = image_name
        self.bb = bb
        self.objects = objects
        self.motion = motion
        self.distance = distance
        self.csv_list_initialized = False
        
        if self.frame == 'MVI_0799_VIS_OB_frame80.jpg':
            print("=================================")
            print(self.objects)
        
    def generate_list_as_csv(self, integer_bb=False):
        """
        Tranform the frame data into a list of cvs entries. Each entry is of
        the form:
            
            [<video_name>_<frame_number>, 
            x_min, 
            y_min, 
            object width, 
            object height, 
            type of object, 
            distance of object, 
            type of motion of object]
        
        Parameters
        ----------
        integer_bb : should the bounding box coordinates be integers? (boolean)
                        Default is False.
        """
        self.csv_list = []
        number_of_objects = len(self.objects) # get the total number of objects
        
        # objects is a list in a list. To avoid problems with " len([[]]) -> 1 " that sanity chack should be used.
        if len(self.objects[0]) > 0:
            for i in range(number_of_objects):
                # avoid possible bad entries / there is one in MVI_1613_VIS_frame0.jpg
                if (int(self.objects[i][0])) != 0:
                    if integer_bb:
                        entry = self.image_name + ',' \
                                    + str(int(self.bb[i,0])) + ',' \
                                    + str(int(self.bb[i,1])) + ',' \
                                    + str(int(self.bb[i,2])) + ',' \
                                    + str(int(self.bb[i,3])) + ',' \
                                    + str(self.objects[i][0]) + ',' \
                                    + str(self.distance[i][0]) + ',' \
                                    + str(self.motion[i][0])
                    else:
                        entry = self.image_name + ',' \
                                    + str(self.bb[i,0]) + ',' \
                                    + str(self.bb[i,1]) + ',' \
                                    + str(self.bb[i,2]) + ',' \
                                    + str(self.bb[i,3]) + ',' \
                                    + str(self.objects[i][0]) + ',' \
                                    + str(self.distance[i][0]) + ',' \
                                    + str(self.motion[i][0])
                    self.csv_list.append(entry)
            
        self.csv_list_initialized = True
            
    def get_list_as_csv(self):
        if not self.csv_list_initialized:
            self.generate_list_as_csv() # create list with float bb
        return self.csv_list
    
def generate_gt_files_dict(path_to_gt_files):
    """
    Creates a dictionary with all the ground truth files location.
    
    Parameters
    ----------
    path_to_gt_files : the path to the ground truth files. (string)
    
    Returns
    -------
    object_gt_files_dict : dictionary in the form:
        (key:value) -> (<video_name>:<video_path>)
    """
    object_gt_files_dict = {}
    for f in listdir(path_to_gt_files):
        if isfile(join(path_to_gt_files, f)):
            object_gt_files_dict[f.split('.')[0].replace('_ObjectGT','')] = join(path_to_gt_files, f)
        
    return object_gt_files_dict
    

def load_mat_files_in_dict(path):
    """
    Loads all the .mat files of the Singapore Maritime Dataset. It converts
    each frame of the .mat files into a Frame class instance and then adds it
    into a dictionary called "frames".
    
    Parameters
    ----------
    path : the path where the .mat files are located. (string)
    
    Returns
    -------
    frames : a dictionary of the form:
            (key:value) -> (<video_name>_<frame_number>:<Frame class instance>)
    """
    frames = {}
    object_gt_files_dict = generate_gt_files_dict(path)
    
    for key in object_gt_files_dict.keys():
        file_name = object_gt_files_dict[key]
        
        gt = loadmat(file_name)
        
        # get the number of frames
        frames_number = len(gt['structXML'][0])
        
        # process data for each frame
        for i in range(frames_number):
            image_name = file_name.split('/')[-1].replace('_ObjectGT.mat','') + ('_frame%d.jpg' % i)
            bb = gt['structXML'][0]['BB'][i]
            objects = gt['structXML'][0]['Object'][i]
            
            # WE HAVE A BAD ENTRY HERE!!!!
#            if (image_name == 'MVI_1613_VIS_frame0.jpg'):
#                print('=================================')
#                print(gt['structXML'][0]['Object'][i])
#                print(key)
#                print(frames_number)
#                print(object_gt_files_dict[key])
                
            motion = gt['structXML'][0]['Motion'][i]
            distance = gt['structXML'][0]['Distance'][i]
            frame = Frame(i, image_name, bb, objects, motion, distance)
            frames[image_name] = frame
        
    return frames
    
def get_all_gt_files_in_csv(path, integer_bb=False):
    """
    Create a list with ALL frames object instance in csv format. Each frame has
    multiple objects. With this function we split each object as a separate
    entry as a csv value in a list.
    
    Parameters
    ----------
    path : the path where the .mat files are located. (string)
    integer_bb : should the bounding box coordinates be integers? (boolean)
                        Default is False.
                        
    Returns
    -------
    object_list : list of csv entries. Each entry is of the form:
            [<video_name>_<frame_number>, 
            x_min, 
            y_min, 
            object width, 
            object height, 
            type of object, 
            distance of object, 
            type of motion of object]
    """
    object_list = []
    frames = load_mat_files_in_dict(path)
    for key in frames.keys():
        frame = frames[key]
        
        frame.generate_list_as_csv(integer_bb)
        object_list_part = frame.get_list_as_csv()        
            
        # append part list of objects to full list of objects
        object_list = object_list + object_list_part
            
        
    print("Total objects in the dataset: ", len(object_list))
    
    return object_list
            
    
frame_list = get_all_gt_files_in_csv(PATH_TO_GT_FILES, False)

#frames = load_mat_files_in_dict(PATH_TO_GT_FILES)

# write to file
#with open("frames.csv", 'w', newline='') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     wr.writerow(frame_list)
     
with open('objects_nir_2.txt', 'w') as f:
    for item in frame_list:
        f.write("%s\n" % item)
     
     
#object_gt_files_dict = generate_gt_files_dict(PATH_TO_GT_FILES)
#print(object_gt_files_dict)
#
#file_name = PATH_TO_GT_FILES + '/MVI_1478_VIS_ObjectGT.mat'
#
#gt = loadmat(file_name)
#
## get the number of frames
#frames_number = len(gt['structXML'][0])
#
#
#
## process data for each frame
#frames = {}
#for i in range(frames_number):
#    image_name = file_name.split('/')[-1].replace('_ObjectGT.mat','') + ('_frame%d.jpg' % i)
#    bb = gt['structXML'][0]['BB'][i]
#    objects = gt['structXML'][0]['Object'][i]
#    motion = gt['structXML'][0]['Motion'][i]
#    distance = gt['structXML'][0]['Distance'][i]
#    frame = Frame(i, image_name, bb, objects, motion, distance)
#    frames[image_name] = frame
#
#
#test = frames['MVI_1478_VIS_frame67.jpg']
#test.generate_list_as_csv()
#test.generate_list_as_csv_integer()

#print(gt.items())