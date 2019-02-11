#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 11:23:05 2018

@author: Tilemachos Bontzorlos

This is a sample code to experiment and transform the Singapore Maritime 
Dataset (SMD) .mat object detection files into a CSV and VOC XML format for 
further processing.

Dataset available here: https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset

If this dataset is used please cite it as:

D. K. Prasad, D. Rajan, L. Rachmawati, E. Rajabaly, and C. Quek, 
"Video Processing from Electro-optical Sensors for Object Detection and 
Tracking in Maritime Environment: A Survey," IEEE Transactions on Intelligent 
Transportation Systems (IEEE), 2017. 

USAGE:
    
python load_mat_into_csv_xml.py -i <path of unzipped SMD dataset> -o <folder to output/save csv and xml files> -f <path of the train/test folders that contain the frames>
    
example:
    
python load_mat_into_csv_xml.py -i /home/tbontz2s/singapore_dataset -o /home/tbontz2s/git/tensorflow/workspace/training_demo/images -f /home/tbontz2s/git/tensorflow/workspace/training_demo/images
"""

from scipy.io import loadmat
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import argparse


GT_FILES_PATHS_LIST = ["NIR/ObjectGT", "VIS_Onshore/ObjectGT", "VIS_Onboard/ObjectGT"]

class Frame:
    """
    This is a class to save the data for each video frame
    """
    csv_list = []
    csv_list_initialized = False
    classes_dict = {
            1:"Ferry",
            2:"Buoy",
            3:"Vessel/ship",
            4:"Speed boat",
            5:"Boat",
            6:"Kayak",
            7:"Sail boat",
            8:"Swimming person",
            9:"Flying bird/plane",
            10:"Other"
            }
    
    def __init__(self, frame, image_name, bb, objects, motion, distance, image_path='', xml_path=''):
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
            image_path : path of the file (without the filename). (string)
            xml_path : path to save the generated xml file. (string)
            
        """
        self.frame = frame
        self.image_name = image_name
        self.bb = bb
        self.objects = objects
        self.motion = motion
        self.distance = distance
        self.image_path = image_path
        self.xml_path = xml_path
        self.csv_list_initialized = False
        self.xml_initialized = False
        
    def convert_frame_to_csv(self, integer_bb=False):
        """
        Tranform the frame data into a list of cvs entries. Each entry is of
        the form:
            
            ('filename', 
            'width', 
            'height',
            'class', 
            'xmin', 
            'ymin', 
            'xmax', 
            'ymax')
            
        This form is suitable for generating tensorflow records.
        
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
                        entry = (self.image_name,
                                    int(self.bb[i,2]),
                                    int(self.bb[i,3]),
                                    self.objects[i][0],
                                    int(self.bb[i,0]),
                                    int(self.bb[i,1]),
                                    int(self.bb[i,0] + self.bb[i,2]),
                                    int(self.bb[i,1] + self.bb[i,3])
                                )
                    else:
                        entry = (self.image_name,
                                    self.bb[i,2],
                                    self.bb[i,3],
                                    self.objects[i][0],
                                    self.bb[i,0],
                                    self.bb[i,1],
                                    self.bb[i,0] + self.bb[i,2],
                                    self.bb[i,1] + self.bb[i,3]
                                )
                    self.csv_list.append(entry)
            
        self.csv_list_initialized = True
        
    def convert_frame_to_VOC_xml(self, integer_bb=False):
        """
        Converts the frame data into the VOC XML format.
        Code based on:
        https://dataturks.com/help/ibbx_dataturks_to_pascal_voc_format.php
        
        Parameters
        ----------            
        integer_bb : should the bounding box coordinates be integers? (boolean)
                        Default is False.
        
        Returns
        -------
        xml : the corresponding VOC XML representation of the frame data. (string)
        """
        folder_name = self.image_path.split('/')[-1]
        file_path = os.path.join(self.image_path, self.image_name)
        xml = ''
        
        xml = "<annotation>\n<folder>" + folder_name + "</folder>\n"
        xml = xml + "<filename>" + self.image_name +"</filename>\n"
        xml = xml + "<path>" + file_path +"</path>\n"
        xml = xml + "<source>\n\t<database>Unknown</database>\n</source>\n"
        xml = xml + "<size>\n"
        xml = xml +     "\t<width>" + str(1920) + "</width>\n"
        xml = xml +    "\t<height>" + str(1080) + "</height>\n"
        xml = xml +    "\t<depth>"+str(3)+"</depth>\n"
        xml = xml +  "</size>\n"
        xml = xml + "<segmented>Unspecified</segmented>\n"
        
        
        number_of_objects = len(self.objects) # get the total number of objects
        
        # objects is a list in a list. To avoid problems with " len([[]]) -> 1 " that sanity chack should be used.
        if len(self.objects[0]) > 0:
            for i in range(number_of_objects):
                # avoid possible bad entries / there is one in MVI_1613_VIS_frame0.jpg
                if (int(self.objects[i][0])) != 0:
                    xml = xml  + self._get_xml_for_bbx(self.objects[i][0], self.bb[i,:], integer_bb)
                    
        xml = xml + "</annotation>"
        
        self.xml = xml
        
        
    def _get_xml_for_bbx(self, label, bb_data, integer_bb=False):
        """
        Creates the VOC XML representation for an object in an image.
        Code based on:
        https://dataturks.com/help/ibbx_dataturks_to_pascal_voc_format.php
        
        Parameters
        ----------
        label : the integer label corresponding to the object class. (int)
        
        bb_data : the bounding box data in the list format:
            [xmin, ymin, width, height]
            
        integer_bb : should the bounding box coordinates be integers? (boolean)
                        Default is False.
        
        Returns
        -------
        xml : the corresponding object entry in xml value. (string)
        """
        xmin = bb_data[0]
        xmax = bb_data[0] + bb_data[2]
        ymin = bb_data[1]
        ymax = bb_data[1] + bb_data[3]
        
        if integer_bb:
            xmin = int(xmin)
            xmax = int(xmax)
            ymin = int(ymin)
            ymax = int(ymax)
        
        xml = "<object>\n"
        xml = xml + "\t<name>" + str(self._convert_class_int_to_string(label)) + "</name>\n"
        xml = xml + "\t<pose>Unspecified</pose>\n"
        xml = xml + "\t<truncated>Unspecified</truncated>\n"
        xml = xml + "\t<difficult>Unspecified</difficult>\n"
        xml = xml + "\t<occluded>Unspecified</occluded>\n"
        xml = xml + "\t<bndbox>\n"
        xml = xml +     "\t\t<xmin>" + str(xmin) + "</xmin>\n"
        xml = xml +     "\t\t<xmax>" + str(xmax) + "</xmax>\n"
        xml = xml +     "\t\t<ymin>" + str(ymin) + "</ymin>\n"
        xml = xml +     "\t\t<ymax>" + str(ymax) + "</ymax>\n"
        xml = xml + "\t</bndbox>\n"
        xml = xml + "</object>\n"
        
        return xml
    
    def _convert_class_int_to_string(self, class_int):
        """
        TODO: write
        """
        return self.classes_dict[class_int]
            
    def get_frame_as_csv(self):
        if not self.csv_list_initialized:
            self.convert_frame_to_csv() # create list with float bb
        return self.csv_list
    
    def save_frame_as_xml(self):
        """
        Saves the frame in the VOC xml represenation in the specified path set
        in the constructor.
        
        Skips the generation if image or xml path are empty.
        """
        if not self.image_path.strip():
            print('There was no valid path set for the image. Skipping xml generation.')
            return
        
        if not self.xml_path.strip():
            print('There was no valid path set for the xml. Skipping xml generation.')
            return
        
        if not self.xml_initialized:
            self.convert_frame_to_VOC_xml()
            
        filename = os.path.join(self.xml_path, self.image_name.split('.')[0] + '.xml')
        with open(filename, 'w') as file:
            file.write(self.xml)
            
    
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
            ('filename', 
            'width', 
            'height',
            'class', 
            'xmin', 
            'ymin', 
            'xmax', 
            'ymax')
    """
    object_list = []
    frames = load_mat_files_in_dict(path)
    for key in frames.keys():
        frame = frames[key]
        
        frame.convert_frame_to_csv(integer_bb)
        object_list_part = frame.get_frame_as_csv()        
            
        # append part list of objects to full list of objects
        object_list = object_list + object_list_part
            
        
    print("Total objects in the dataset: ", len(object_list)) # TODO: maybe remove or rephrase?
    
    return object_list


def get_gt_files_in_csv(path, frames_tuple, integer_bb=False):
    """
    Create a list with frames object instances that are included in frames_list
    in csv format. Each frame has multiple objects. With this function we split 
    each object as a separate entry as a csv value in a list.
    
    Parameters
    ----------
    path : the path where the .mat files are located. (string)
    
    frames_tuple : tuple of frames names for train and test.
                    (train_frames, test_frames)
    
    integer_bb : should the bounding box coordinates be integers? (boolean)
                        Default is False.
                        
    Returns
    -------
    Tuple of lists with objects in the train and test frames.
        (object_list_train,object_list_test)
    
    Each list is a list of csv entries. Each entry is of the form:
            ('filename', 
            'width', 
            'height',
            'class', 
            'xmin', 
            'ymin', 
            'xmax', 
            'ymax')
    """
    train_frames, test_frames = frames_tuple
    
    object_list_train = []
    object_list_test = []
    frames = load_mat_files_in_dict(path)
    for key in frames.keys():
        if key in train_frames:
            frame = frames[key]
            
            frame.convert_frame_to_csv(integer_bb)
            object_list_part = frame.get_frame_as_csv()        
            
            object_list_train = object_list_train + object_list_part
        elif key in test_frames:
            frame = frames[key]
            
            frame.convert_frame_to_csv(integer_bb)
            object_list_part = frame.get_frame_as_csv()        
            
            object_list_test = object_list_test + object_list_part
            
            
        
    print("Total train objects produced: ", len(object_list_train))
    print("Total test objects produced: ", len(object_list_test))
    
    return (object_list_train,object_list_test)

def get_generated_frames_dict(paths):
    '''
    Helper function to get the generated frames from the videos in the train 
    and test dataset.
    
    Parameters
    ----------
    
    paths : a tuple with the train and test path. (train_path,test_path)
    
    Returns
    -------
    
    tuple of train frames and test frames list
    '''
    train_path, test_path = paths
    
    train_frames = [frame for frame in os.listdir(train_path)
                                if isfile(os.path.join(train_path, frame))]
    
    test_frames = [frame for frame in os.listdir(test_path)
                                if isfile(os.path.join(test_path, frame))]


    return (train_frames, test_frames)
    
        
def generate_split_dataset_csv_xml(path, frames_tuple, paths_list, integer_bb=False):
    """
    Loads all the .mat files of the Singapore Maritime Dataset. 
    It loads the .mat data for each frame included into the train/test dataset 
    and converts it into a Frame class instance. 
    Each frame is added in a "frames" dictionary as a Frame object.
    Each Frame object data is also converted into CSV format and added into the 
    corresponding list of train/test frames.
    The code also invokes the generation and file write of the VOC XML files 
    for each Frame instance.
    
    Parameters
    ----------
    path : the path where the .mat files are located. (string)
    
    frames_tuple : tuple of frames names for train and test.
                    (train_frames, test_frames)
                    
    paths_list : list of the paths for images and annotations.
        [images_train_path, images_test_path, xml_annotations_train_path, xml_annotations_test_path]
    
    integer_bb : should the bounding box coordinates be integers? (boolean)
                        Default is False.
    
    Returns
    -------
    frames : a dictionary of the form:
            (key:value) -> (<video_name>_<frame_number>:<Frame class instance>)
            
    object_list_train : list with objects in the train frames.
    
            This is a list of csv entries. Each entry is of the form:
                    ('filename', 
                    'width', 
                    'height',
                    'class', 
                    'xmin', 
                    'ymin', 
                    'xmax', 
                    'ymax')
                
    object_list_test : list with objects in the test frames.
    
            This is a list of csv entries. Each entry is of the form:
                    ('filename', 
                    'width', 
                    'height',
                    'class', 
                    'xmin', 
                    'ymin', 
                    'xmax', 
                    'ymax')
    """
    train_frames, test_frames = frames_tuple
    images_train_path, images_test_path, xml_annotations_train_path, xml_annotations_test_path = paths_list
    frames = {}
    object_list_train = []
    object_list_test = []
    object_gt_files_dict = generate_gt_files_dict(path)
    
    for key in object_gt_files_dict.keys():
        file_name = object_gt_files_dict[key]
        
        gt = loadmat(file_name)
        
        # get the number of frames
        frames_number = len(gt['structXML'][0])
        
        # process data for each frame
        for i in range(frames_number):
            image_name = file_name.split('/')[-1].replace('_ObjectGT.mat','') + ('_frame%d.jpg' % i)
            
            if image_name in train_frames:
                bb = gt['structXML'][0]['BB'][i]
                objects = gt['structXML'][0]['Object'][i]
                motion = gt['structXML'][0]['Motion'][i]
                distance = gt['structXML'][0]['Distance'][i]
                frame = Frame(i, image_name, bb, objects, motion, distance, images_train_path, xml_annotations_train_path)
                frames[image_name] = frame
                object_list_part = frame.get_frame_as_csv()
                object_list_train = object_list_train + object_list_part
                frame.save_frame_as_xml()
                
            elif image_name in test_frames:
                bb = gt['structXML'][0]['BB'][i]
                objects = gt['structXML'][0]['Object'][i]
                motion = gt['structXML'][0]['Motion'][i]
                distance = gt['structXML'][0]['Distance'][i]
                frame = Frame(i, image_name, bb, objects, motion, distance, images_test_path, xml_annotations_test_path)
                frames[image_name] = frame
                object_list_part = frame.get_frame_as_csv()
                object_list_test = object_list_test + object_list_part
                frame.save_frame_as_xml()
        
    return frames, object_list_train, object_list_test

   

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample TensorFlow SMD MAT-to-CSV-XML converter")
parser.add_argument("-i",
                    "--inputDir",
                    help="Path to the folder where the unziped GT files are stored",
                    type=str)
parser.add_argument("-o",
                    "--outputDir",
                    help="Name of output directory", type=str)
parser.add_argument("-f",
                    "--framesDir",
                    help="Directory that has the train and test folder with the frames", type=str)
args = parser.parse_args()

if(args.inputDir is None):
    args.inputDir = os.getcwd()
if(args.outputDir is None):
    args.outputDir = args.inputDir# + "/labels.csv"
if(args.framesDir is None):
    args.framesDir = os.getcwd()

assert(os.path.isdir(args.inputDir))
assert(os.path.isdir(args.outputDir))
assert(os.path.isdir(args.framesDir))

images_train_path = os.path.join(args.framesDir, 'train')
images_test_path = os.path.join(args.framesDir, 'test')
xml_annotations_train_path = os.path.join(args.framesDir, 'train_annotations')
xml_annotations_test_path = os.path.join(args.framesDir, 'test_annotations')

# generate the xml folders if they are not there
if not os.path.isdir(xml_annotations_train_path):
    os.mkdir(xml_annotations_train_path)
if not os.path.isdir(xml_annotations_test_path):
    os.mkdir(xml_annotations_test_path)


train_frames, test_frames =get_generated_frames_dict(
                        [os.path.join(args.framesDir, 'train'),
                         os.path.join(args.framesDir, 'test')])

# generate tuple of frames and list of paths
frames_tuple = (train_frames, test_frames)
paths_list = [images_train_path, images_test_path, xml_annotations_train_path, xml_annotations_test_path]


objects_list_train = []
objects_list_test = []
for mat_file in GT_FILES_PATHS_LIST:
    _, object_list_train_temp, object_list_test_temp = generate_split_dataset_csv_xml(os.path.join(args.inputDir, mat_file), frames_tuple, paths_list, integer_bb=False)
    #_, object_list_train_temp, object_list_test_temp = get_gt_files_in_csv(
    #        os.path.join(args.inputDir, mat_file), (train_frames, test_frames), False)
    objects_list_train = objects_list_train + object_list_train_temp
    objects_list_test = objects_list_test + object_list_test_temp
    
print('Total objects in train dataset: ', len(objects_list_train))
print('Total objects in test dataset: ', len(objects_list_test))

column_name = ['filename', 'width', 'height',
                'class', 'xmin', 'ymin', 'xmax', 'ymax']

objects_train_df = pd.DataFrame(objects_list_train, columns=column_name)
objects_test_df = pd.DataFrame(objects_list_test, columns=column_name)

objects_train_df.to_csv(args.outputDir + '/train_labels.csv', index=None)
objects_test_df.to_csv(args.outputDir + '/test_labels.csv', index=None)
print('Successfully converted mat to csv.')
