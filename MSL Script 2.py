# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 16:46:12 2022

@author: Daphne Titsing
"""



#-----------------------------------------------------------------------------
#                           LOADING PACKAGES
#-----------------------------------------------------------------------------
#FREQUENCY BANDS
#12-17
#17-20
#21-29

import os
import numpy as np
import mne
import pip
import pandas as pd
from mne.event import define_target_events 
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs, corrmap
from mne.connectivity import spectral_connectivity 
from mne.datasets import sample 
from mne.viz import plot_sensors_connectivity 
import matplotlib.pyplot as plt
import seaborn as sns
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.viz.utils import center_cmap
import numpy as np
from mne import create_info, EpochsArray
from mne.baseline import rescale
from mne.time_frequency import (tfr_multitaper, tfr_stockwell, tfr_morlet,
                                tfr_array_morlet)
from mne.viz import centers_to_edges
import os.path as op
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch

   
#-----------------------------------------------------------------------------#
#                               6-KEY CONDITION
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
#                               PARTICIPANT 1
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_1_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part1_SeqL_ERD_6_B1.fif'
Part_1_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part1_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p1 = mne.io.read_raw_fif(Part_1_1, preload = True) 
 

#--------B5--------#
raw5_p1 = mne.io.read_raw_fif(Part_1_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p1, _ = mne.events_from_annotations(raw_p1, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p1, _ = mne.events_from_annotations(raw5_p1, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p1 = np.copy(events_p1)
laststimpositionright_b1_p1 = np.copy(events_p1)
feedbackleft_b1_p1 = np.copy(events_p1)
feedbackright_b1_p1 = np.copy(events_p1)
leftresponse_b1_p1 = np.copy(events_p1)
rightresponse_b1_p1 = np.copy(events_p1)
lastresponseleft_b1_p1 = np.copy(events_p1)
lastresponseright_b1_p1 = np.copy(events_p1)
preparationleft_b1_p1 = np.copy(events_p1)
preparationright_b1_p1 = np.copy(events_p1)
nogob1p1 = np.copy(events_p1)
#--------B5-------#
laststimpositionleft_b5_p1 = np.copy(eventsB5_p1)
laststimpositionright_b5_p1 = np.copy(eventsB5_p1)
feedbackleft_b5_p1 = np.copy(eventsB5_p1)
feedbackright_b5_p1 = np.copy(eventsB5_p1)
leftresponse_b5_p1 = np.copy(eventsB5_p1)
rightresponse_b5_p1 = np.copy(eventsB5_p1)
lastresponseleft_b5_p1 = np.copy(eventsB5_p1)
lastresponseright_b5_p1 = np.copy(eventsB5_p1)
preparationleft_b5_p1 = np.copy(eventsB5_p1)
preparationright_b5_p1 = np.copy(eventsB5_p1)
nogob5p1 = np.copy(eventsB5_p1)


#print to see if event times are correct

print(events_p1)

#--------B1-------#


#last  stimulus position left (34)
laststimpositionleft_b1_p1 = mne.pick_events(laststimpositionleft_b1_p1, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p1 = mne.merge_events(laststimpositionleft_b1_p1, [5, 6, 7, 8], 34, replace_events=True)
laststimpositionleft_b1_p1 = np.delete(laststimpositionleft_b1_p1, [77, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#last  stimulus position right (35)
laststimpositionright_b1_p1 = mne.pick_events(laststimpositionright_b1_p1, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p1 = mne.merge_events(laststimpositionright_b1_p1, [9, 10, 11, 12], 35, replace_events=True)
laststimpositionright_b1_p1 = np.delete(laststimpositionright_b1_p1, [47, 83, 137, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)


#feedback left (36)
feedbackleft_b1_p1 = mne.pick_events(feedbackleft_b1_p1, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p1 = mne.merge_events(feedbackleft_b1_p1, [25,26], 36, replace_events=True)
feedbackleft_b1_p1 = mne.event.shift_time_events(feedbackleft_b1_p1, 36, -1.000, 500)
feedbackleft_b1_p1 = np.delete(feedbackleft_b1_p1, [4, 6, 7, 8, 10, 13, 15, 17, 20, 21, 22, 23, 32, 34, 35, 36, 37, 39, 40, 41, 44, 45, 46, 47], axis=0)



#feedback right (37)
feedbackright_b1_p1 = mne.pick_events(feedbackright_b1_p1, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p1 = mne.merge_events(feedbackright_b1_p1, [25,26], 37, replace_events=True)
feedbackright_b1_p1 = mne.event.shift_time_events(feedbackright_b1_p1, 37, -1.000, 500)
feedbackright_b1_p1 = np.delete(feedbackright_b1_p1, [0, 1, 2, 3, 5, 9, 11, 12, 14, 16, 18, 19, 24, 25, 26, 27, 28, 29, 30, 31, 33, 38, 42, 43], axis=0)

#left response (38) 
leftresponse_b1_p1 = mne.pick_events(leftresponse_b1_p1, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p1 = mne.merge_events(leftresponse_b1_p1, [14, 15, 16, 17], 38, replace_events=True)

#right response(39)
rightresponse_b1_p1 = mne.pick_events(rightresponse_b1_p1, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p1 = mne.merge_events(rightresponse_b1_p1, [18, 19, 20, 21], 39, replace_events=True)

#last response left(40)
lastresponseleft_b1_p1 = mne.pick_events(lastresponseleft_b1_p1, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p1 = mne.merge_events(lastresponseleft_b1_p1, [5, 6, 7, 8], 40, replace_events=True)

#last response right(41)
lastresponseright_b1_p1 = mne.pick_events(lastresponseright_b1_p1, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p1 = mne.merge_events(lastresponseright_b1_p1, [9, 10, 11, 12], 41, replace_events=True)


#preperation left (42) 
preparationleft_b1_p1 = mne.pick_events(preparationleft_b1_p1, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p1 = mne.merge_events(preparationleft_b1_p1, [5, 6, 7, 8], 42, replace_events=True)
preparationleft_b1_p1 = mne.event.shift_time_events(preparationleft_b1_p1, 42, 1.500, 500)
preparationleft_b1_p1  = np.delete(preparationleft_b1_p1 , [77, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)



#preperation right (43) 
preparationright_b1_p1 = mne.pick_events(preparationright_b1_p1, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p1 = mne.merge_events(preparationright_b1_p1, [9, 10, 11, 12], 43, replace_events=True)
preparationright_b1_p1 = mne.event.shift_time_events(preparationright_b1_p1, 43, 1.500, 500)
preparationright_b1_p1 = np.delete(preparationright_b1_p1, [47, 83, 137, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)


#nogo (44)
nogob1p1 = mne.pick_events(nogob1p1, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p1 = mne.merge_events(nogob1p1, [24], 44, replace_events=True)
#--------B5-------#


#last  stimulus position left (45)
laststimpositionleft_b5_p1 = mne.pick_events(laststimpositionleft_b5_p1, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p1 = mne.merge_events(laststimpositionleft_b5_p1, [5, 6, 7, 8], 45, replace_events=True)
laststimpositionleft_b5_p1 = np.delete(laststimpositionleft_b5_p1, [119, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#last  stimulus position right (46)
laststimpositionright_b5_p1 = mne.pick_events(laststimpositionright_b5_p1, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p1 = mne.merge_events(laststimpositionright_b5_p1, [9, 10, 11, 12], 46, replace_events=True)
laststimpositionright_b5_p1 = np.delete(laststimpositionright_b5_p1, [5, 65, 125, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)

#feedback (47)
feedbackleft_b5_p1 = mne.pick_events(feedbackleft_b5_p1, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p1 = mne.merge_events(feedbackleft_b5_p1, [25,26], 47, replace_events=True)
feedbackleft_b5_p1 = mne.event.shift_time_events(feedbackleft_b5_p1, 47, -1.000, 500)
feedbackleft_b5_p1 = np.delete(feedbackleft_b5_p1, [0, 1, 3, 4, 10, 11, 14, 15, 16, 17, 18, 21, 25, 26, 27, 31, 34, 35, 37, 38, 39, 40, 41, 46], axis=0)


#feedback (48)
feedbackright_b5_p1 = mne.pick_events(feedbackright_b5_p1, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p1 = mne.merge_events(feedbackright_b5_p1, [25,26], 48, replace_events=True)
feedbackright_b5_p1 = mne.event.shift_time_events(feedbackright_b5_p1, 48, -1.000, 500)
feedbackright_b5_p1 = np.delete(feedbackright_b5_p1, [2, 5, 6, 7, 8, 9, 12, 13, 19, 20, 22, 23, 24, 28, 29, 30, 32, 33, 36, 42, 43, 44, 45, 47], axis=0)


#left response (49) 
leftresponse_b5_p1 = mne.pick_events(leftresponse_b5_p1, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p1 = mne.merge_events(leftresponse_b5_p1, [14, 15, 16, 17], 49, replace_events=True)

#right response(50)
rightresponse_b5_p1 = mne.pick_events(rightresponse_b5_p1, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p1 = mne.merge_events(rightresponse_b5_p1, [18, 19, 20, 21], 50, replace_events=True)

#last response left(51)
lastresponseleft_b5_p1 = mne.pick_events(lastresponseleft_b5_p1, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p1 = mne.merge_events(lastresponseleft_b5_p1, [5, 6, 7, 8], 51, replace_events=True)

#last response right(52)
lastresponseright_b5_p1 = mne.pick_events(lastresponseright_b5_p1, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p1 = mne.merge_events(lastresponseright_b5_p1, [9, 10, 11, 12], 52, replace_events=True)


#preperation left (53) 
preparationleft_b5_p1 = mne.pick_events(preparationleft_b5_p1, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p1 = mne.merge_events(preparationleft_b5_p1, [5, 6, 7, 8], 53, replace_events=True)
preparationleft_b5_p1 = mne.event.shift_time_events(preparationleft_b5_p1, 53, 1.500, 500)
preparationleft_b5_p1 = np.delete(preparationleft_b5_p1,  [119, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#preperation right (54) 
preparationright_b5_p1 = mne.pick_events(preparationright_b5_p1, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p1 = mne.merge_events(preparationright_b5_p1, [9, 10, 11, 12], 54, replace_events=True)
preparationright_b5_p1 = mne.event.shift_time_events(preparationright_b5_p1, 54, 1.500, 500)
preparationright_b5_p1 = np.delete(preparationright_b5_p1, [5, 65, 125, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)

#nogo (55)
nogob5p1 = mne.pick_events(nogob5p1, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p1 = mne.merge_events(nogob5p1, [24], 55, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p1 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p1 = {'laststimleft':34,'laststimright':35, 'feedbackL':36, 'feedbackR':37, 'leftres':38, 'rightres':39, 'lastresleft':40, 'lastresright':41, 'prepleft':42, 'prepright':43, 'nogo': 44,}
event_dictB5_p1 = {'laststimleft':45,'laststimright':46, 'feedbackL':47, 'feedbackR':48, 'leftres':49, 'rightres':50, 'lastresleft':51, 'lastresright':52, 'prepleft':53, 'prepright':54, 'nogo': 55,}



#merging events togeher into one event list

finalB1_p1 = np.concatenate((laststimpositionleft_b1_p1, laststimpositionright_b1_p1, feedbackleft_b1_p1, feedbackright_b1_p1, leftresponse_b1_p1, rightresponse_b1_p1, lastresponseleft_b1_p1, lastresponseright_b1_p1, preparationleft_b1_p1, preparationright_b1_p1, events_p1, nogob1p1), axis=0)
finalB5_p1 = np.concatenate((laststimpositionleft_b5_p1, laststimpositionright_b5_p1, feedbackleft_b5_p1, feedbackright_b5_p1, leftresponse_b5_p1, rightresponse_b5_p1, lastresponseleft_b5_p1, lastresponseright_b5_p1, preparationleft_b5_p1, preparationright_b5_p1, events_p1, nogob5p1), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p1, event_id=event_dictB1_p1, 

                         sfreq=raw_p1.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p1, event_id=event_dictB5_p1, 

                         sfreq=raw5_p1.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 

#-----------------------------------------------------------------------------#
#                               PARTICIPANT 2
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_2_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part2_SeqL_ERD_6_B1.fif'
Part_2_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part2_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p2 = mne.io.read_raw_fif(Part_2_1, preload = True) 
 

#--------B5--------#
raw5_p2 = mne.io.read_raw_fif(Part_2_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p2, _ = mne.events_from_annotations(raw_p2, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p2, _ = mne.events_from_annotations(raw5_p2, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p2 = np.copy(events_p2)
laststimpositionright_b1_p2 = np.copy(events_p2)
feedbackleft_b1_p2 = np.copy(events_p2)
feedbackright_b1_p2 = np.copy(events_p2)
leftresponse_b1_p2 = np.copy(events_p2)
rightresponse_b1_p2 = np.copy(events_p2)
lastresponseleft_b1_p2 = np.copy(events_p2)
lastresponseright_b1_p2 = np.copy(events_p2)
preparationleft_b1_p2 = np.copy(events_p2)
preparationright_b1_p2 = np.copy(events_p2)
nogob1p2 = np.copy(events_p2)
#--------B5-------#
laststimpositionleft_b5_p2 = np.copy(eventsB5_p2)
laststimpositionright_b5_p2 = np.copy(eventsB5_p2)
feedbackleft_b5_p2 = np.copy(eventsB5_p2)
feedbackright_b5_p2 = np.copy(eventsB5_p2)
leftresponse_b5_p2 = np.copy(eventsB5_p2)
rightresponse_b5_p2 = np.copy(eventsB5_p2)
lastresponseleft_b5_p2 = np.copy(eventsB5_p2)
lastresponseright_b5_p2 = np.copy(eventsB5_p2)
preparationleft_b5_p2 = np.copy(eventsB5_p2)
preparationright_b5_p2 = np.copy(eventsB5_p2)
nogob5p2 = np.copy(eventsB5_p2)


#print to see if event times are correct

print(events_p2)

#--------B1-------#


#last  stimulus position left (56)
laststimpositionleft_b1_p2 = mne.pick_events(laststimpositionleft_b1_p2, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p2 = mne.merge_events(laststimpositionleft_b1_p2, [5, 6, 7, 8], 56, replace_events=True)
laststimpositionleft_b1_p2 = np.delete(laststimpositionleft_b1_p2, [77, 83, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#last  stimulus position right (57)
laststimpositionright_b1_p2 = mne.pick_events(laststimpositionright_b1_p2, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p2 = mne.merge_events(laststimpositionright_b1_p2, [9, 10, 11, 12], 57, replace_events=True)
laststimpositionright_b1_p2 = np.delete(laststimpositionright_b1_p2, [0, 1, 2, 3, 4, 6, 7, 8,	9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 74, 75, 76, 77, 78, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 93, 94, 95, 96, 97, 99, 100,	101, 102, 103, 105, 106, 107, 108, 109, 111, 112, 113, 114,	115, 117, 118, 119,	120, 121, 123, 124,	125, 126, 127, 130,	131, 132, 133, 134, 136, 137, 138, 139,	140, 142, 143, 144,	145, 146, 148, 149,	150, 151, 152, 154,	155, 156, 157, 158, 129, 80, 72, 73, 129, 86, 128], axis=0)			


#feedback left (58)
feedbackleft_b1_p2 = mne.pick_events(feedbackleft_b1_p2, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p2 = mne.merge_events(feedbackleft_b1_p2, [25,26], 58, replace_events=True)
feedbackleft_b1_p2 = mne.event.shift_time_events(feedbackleft_b1_p2, 58, -1.000, 500)
feedbackleft_b1_p2 = np.delete(feedbackleft_b1_p2, [2, 3, 9, 13, 14, 16, 17, 19, 20, 21, 22, 23, 24, 25, 31, 32, 33, 35, 40, 41, 44, 45, 46, 47], axis=0)


#feedback right (59)
feedbackright_b1_p2 = mne.pick_events(feedbackright_b1_p2, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p2 = mne.merge_events(feedbackright_b1_p2, [25,26], 59, replace_events=True)
feedbackright_b1_p2 = mne.event.shift_time_events(feedbackright_b1_p2, 59, -1.000, 500)
feedbackright_b1_p2 = np.delete(feedbackright_b1_p2, [0, 1, 4, 5, 6, 7, 8, 10, 11, 12, 15, 18, 26, 27, 28, 29, 30, 34, 36, 37, 38, 39, 42, 43], axis=0)


#left response (60) 
leftresponse_b1_p2 = mne.pick_events(leftresponse_b1_p2, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p2 = mne.merge_events(leftresponse_b1_p2, [14, 15, 16, 17], 60, replace_events=True)

#right response(61)
rightresponse_b1_p2 = mne.pick_events(rightresponse_b1_p2, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p2 = mne.merge_events(rightresponse_b1_p2, [18, 19, 20, 21], 61, replace_events=True)

#last response left(62)
lastresponseleft_b1_p2 = mne.pick_events(lastresponseleft_b1_p2, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p2 = mne.merge_events(lastresponseleft_b1_p2, [5, 6, 7, 8], 62, replace_events=True)

#last response right(63)
lastresponseright_b1_p2 = mne.pick_events(lastresponseright_b1_p2, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p2 = mne.merge_events(lastresponseright_b1_p2, [9, 10, 11, 12], 63, replace_events=True)


#preperation left (64) 
preparationleft_b1_p2 = mne.pick_events(preparationleft_b1_p2, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p2 = mne.merge_events(preparationleft_b1_p2, [5, 6, 7, 8], 64, replace_events=True)
preparationleft_b1_p2 = mne.event.shift_time_events(preparationleft_b1_p2, 64, 1.500, 500)
preparationleft_b1_p2 = np.delete(preparationleft_b1_p2, [77, 83, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#preperation right (65) 
preparationright_b1_p2 = mne.pick_events(preparationright_b1_p2, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p2 = mne.merge_events(preparationright_b1_p2, [9, 10, 11, 12], 65, replace_events=True)
preparationright_b1_p2 = mne.event.shift_time_events(preparationright_b1_p2, 65, 1.500, 500)
preparationright_b1_p2 = np.delete(preparationright_b1_p2, [0, 1, 2, 3, 4, 6, 7, 8,	9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 74, 75, 76, 77, 78, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 93, 94, 95, 96, 97, 99, 100,	101, 102, 103, 105, 106, 107, 108, 109, 111, 112, 113, 114,	115, 117, 118, 119,	120, 121, 123, 124,	125, 126, 127, 130,	131, 132, 133, 134, 136, 137, 138, 139,	140, 142, 143, 144,	145, 146, 148, 149,	150, 151, 152, 154,	155, 156, 157, 158, 129, 80, 72, 73, 129, 86, 128], axis=0)			

#nogo (66)
nogob1p2 = mne.pick_events(nogob1p2, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p2 = mne.merge_events(nogob1p2, [24], 66, replace_events=True)
#--------B5-------#


#last  stimulus position left (67)
laststimpositionleft_b5_p2 = mne.pick_events(laststimpositionleft_b5_p2, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p2 = mne.merge_events(laststimpositionleft_b5_p2, [5, 6, 7, 8], 67, replace_events=True)
laststimpositionleft_b5_p2 = np.delete(laststimpositionleft_b5_p2, [47, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#last  stimulus position right (68)
laststimpositionright_b5_p2 = mne.pick_events(laststimpositionright_b5_p2, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p2 = mne.merge_events(laststimpositionright_b5_p2, [9, 10, 11, 12], 68, replace_events=True)
laststimpositionright_b5_p2 = np.delete(laststimpositionright_b5_p2, [24, 86, 123, 18, 73, 80, 117, 0, 1, 2, 3, 4, 6, 7, 8,	9, 10, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71, 74, 75, 76, 77, 78, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 93, 94, 95, 96, 97, 99, 100,	101, 102, 103, 105,	106, 107, 108, 109, 111, 112, 113, 114,	115, 118, 119, 120,	121, 122, 124, 125,	126, 127, 128, 130,	131, 132, 133, 134, 136, 137, 138, 139,	140, 142, 143, 144,	145, 146, 148, 149,	150, 151, 152, 154,	155, 156, 157, 158, 160, 161, 162, 163,	164 ], axis=0)

#feedback (69)
feedbackleft_b5_p2 = mne.pick_events(feedbackleft_b5_p2, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p2 = mne.merge_events(feedbackleft_b5_p2, [25,26], 69, replace_events=True)
feedbackleft_b5_p2 = mne.event.shift_time_events(feedbackleft_b5_p2, 69, -1.000, 500)
feedbackleft_b5_p2 = np.delete(feedbackleft_b5_p2, [0, 1, 3, 6, 8, 10, 11, 12, 14, 16, 17, 19, 28, 33, 34, 35, 36, 40, 41, 42, 43, 45, 46, 47], axis=0)

#feedback (70)
feedbackright_b5_p2 = mne.pick_events(feedbackright_b5_p2, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p2 = mne.merge_events(feedbackright_b5_p2, [25,26], 70, replace_events=True)
feedbackright_b5_p2 = mne.event.shift_time_events(feedbackright_b5_p2, 70, -1.000, 500)
feedbackright_b5_p2 = np.delete(feedbackright_b5_p2, [2, 4, 5, 7, 9, 13, 15, 18, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 37, 38, 39, 44], axis=0)

#left response (71) 
leftresponse_b5_p2 = mne.pick_events(leftresponse_b5_p2, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p2 = mne.merge_events(leftresponse_b5_p2, [14, 15, 16, 17], 71, replace_events=True)

#right response(72)
rightresponse_b5_p2 = mne.pick_events(rightresponse_b5_p2, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p2 = mne.merge_events(rightresponse_b5_p2, [18, 19, 20, 21], 72, replace_events=True)

#last response left(73)
lastresponseleft_b5_p2 = mne.pick_events(lastresponseleft_b5_p2, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p2 = mne.merge_events(lastresponseleft_b5_p2, [5, 6, 7, 8], 73, replace_events=True)

#last response right(74)
lastresponseright_b5_p2 = mne.pick_events(lastresponseright_b5_p2, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p2 = mne.merge_events(lastresponseright_b5_p2, [9, 10, 11, 12], 74, replace_events=True)


#preperation left (75) 
preparationleft_b5_p2 = mne.pick_events(preparationleft_b5_p2, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p2 = mne.merge_events(preparationleft_b5_p2, [5, 6, 7, 8], 75, replace_events=True)
preparationleft_b5_p2 = mne.event.shift_time_events(preparationleft_b5_p2, 75, 1.500, 500)
preparationleft_b5_p2 = np.delete(preparationleft_b5_p2, [47, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)


#preperation right (76) 
preparationright_b5_p2 = mne.pick_events(preparationright_b5_p2, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p2 = mne.merge_events(preparationright_b5_p2, [9, 10, 11, 12], 76, replace_events=True)
preparationright_b5_p2 = mne.event.shift_time_events(preparationright_b5_p2, 76, 1.500, 500)
preparationright_b5_p2 = np.delete(preparationright_b5_p2, [24, 86, 123, 18, 73, 80, 117, 0, 1, 2, 3, 4, 6, 7, 8,	9, 10, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71, 74, 75, 76, 77, 78, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 93, 94, 95, 96, 97, 99, 100,	101, 102, 103, 105,	106, 107, 108, 109, 111, 112, 113, 114,	115, 118, 119, 120,	121, 122, 124, 125,	126, 127, 128, 130,	131, 132, 133, 134, 136, 137, 138, 139,	140, 142, 143, 144,	145, 146, 148, 149,	150, 151, 152, 154,	155, 156, 157, 158, 160, 161, 162, 163,	164 ], axis=0)

#nogo (77)
nogob5p2 = mne.pick_events(nogob5p2, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p2 = mne.merge_events(nogob5p2, [24], 77, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p2 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p2 = {'laststimleft':56,'laststimright':57, 'feedbackL':58, 'feedbackR':59, 'leftres':60, 'rightres':61, 'lastresleft':62, 'lastresright':63, 'prepleft':64, 'prepright':65, 'nogo': 66,}
event_dictB5_p2 = {'laststimleft':67,'laststimright':68, 'feedbackL':69, 'feedbackR':70, 'leftres':71, 'rightres':72, 'lastresleft':73, 'lastresright':74, 'prepleft':75, 'prepright':76, 'nogo': 77,}



#merging events togeher into one event list

finalB1_p2 = np.concatenate((laststimpositionleft_b1_p2, laststimpositionright_b1_p2, feedbackleft_b1_p2, feedbackright_b1_p2, leftresponse_b1_p2, rightresponse_b1_p2, lastresponseleft_b1_p2, lastresponseright_b1_p2, preparationleft_b1_p2, preparationright_b1_p2, events_p2, nogob1p2), axis=0)
finalB5_p2 = np.concatenate((laststimpositionleft_b5_p2, laststimpositionright_b5_p2, feedbackleft_b5_p2, feedbackright_b5_p2, leftresponse_b5_p2, rightresponse_b5_p2, lastresponseleft_b5_p2, lastresponseright_b5_p2, preparationleft_b5_p2, preparationright_b5_p2, events_p2, nogob5p2), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p2, event_id=event_dictB1_p2, 

                         sfreq=raw_p2.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p2, event_id=event_dictB5_p2, 

                         sfreq=raw5_p2.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend


#-----------------------------------------------------------------------------#
#                               PARTICIPANT 3
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_3_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part3_SeqL_ERD_6_B1.fif'
Part_3_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part3_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p3 = mne.io.read_raw_fif(Part_3_1, preload = True) 
 

#--------B5--------#
raw5_p3 = mne.io.read_raw_fif(Part_3_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p3, _ = mne.events_from_annotations(raw_p3, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p3, _ = mne.events_from_annotations(raw5_p3, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p3 = np.copy(events_p3)
laststimpositionright_b1_p3 = np.copy(events_p3)
feedbackleft_b1_p3 = np.copy(events_p3)
feedbackright_b1_p3 = np.copy(events_p3)
leftresponse_b1_p3 = np.copy(events_p3)
rightresponse_b1_p3 = np.copy(events_p3)
lastresponseleft_b1_p3 = np.copy(events_p3)
lastresponseright_b1_p3 = np.copy(events_p3)
preparationleft_b1_p3 = np.copy(events_p3)
preparationright_b1_p3 = np.copy(events_p3)
nogob1p3 = np.copy(events_p3)
#--------B5-------#
laststimpositionleft_b5_p3 = np.copy(eventsB5_p3)
laststimpositionright_b5_p3 = np.copy(eventsB5_p3)
feedbackleft_b5_p3 = np.copy(eventsB5_p3)
feedbackright_b5_p3 = np.copy(eventsB5_p3)
leftresponse_b5_p3 = np.copy(eventsB5_p3)
rightresponse_b5_p3 = np.copy(eventsB5_p3)
lastresponseleft_b5_p3 = np.copy(eventsB5_p3)
lastresponseright_b5_p3 = np.copy(eventsB5_p3)
preparationleft_b5_p3 = np.copy(eventsB5_p3)
preparationright_b5_p3 = np.copy(eventsB5_p3)
nogob5p3 = np.copy(eventsB5_p3)


#print to see if event times are correct

print(events_p3)

#--------B1-------#


#last  stimulus position left (78)
laststimpositionleft_b1_p3 = mne.pick_events(laststimpositionleft_b1_p3, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p3 = mne.merge_events(laststimpositionleft_b1_p3, [5, 6, 7, 8], 78, replace_events=True)
laststimpositionleft_b1_p3 = np.delete(laststimpositionleft_b1_p3, [11, 131, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#last  stimulus position right (79)
laststimpositionright_b1_p3 = mne.pick_events(laststimpositionright_b1_p3, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p3 = mne.merge_events(laststimpositionright_b1_p3, [9, 10, 11, 12], 79, replace_events=True)
laststimpositionright_b1_p3 = np.delete(laststimpositionright_b1_p3, [35, 131, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#feedback left (80)
feedbackleft_b1_p3 = mne.pick_events(feedbackleft_b1_p3, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p3 = mne.merge_events(feedbackleft_b1_p3, [25,26], 80, replace_events=True)
feedbackleft_b1_p3 = mne.event.shift_time_events(feedbackleft_b1_p3, 80, -1.000, 500)
feedbackleft_b1_p3 = np.delete(feedbackleft_b1_p3, [0, 1, 4, 6, 9, 15, 16, 17, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 34, 35, 37, 40, 42, 47], axis=0)

#feedback right (81)
feedbackright_b1_p3 = mne.pick_events(feedbackright_b1_p3, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p3 = mne.merge_events(feedbackright_b1_p3, [25,26], 81, replace_events=True)
feedbackright_b1_p3 = mne.event.shift_time_events(feedbackright_b1_p3, 81, -1.000, 500)
feedbackright_b1_p3 = np.delete(feedbackright_b1_p3, [2, 3, 5, 7, 8, 10, 11, 12, 13, 14, 18, 19, 24, 31, 32, 33, 36, 38, 39, 41, 43, 44, 45, 46], axis=0)

#left response (82) 
leftresponse_b1_p3 = mne.pick_events(leftresponse_b1_p3, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p3 = mne.merge_events(leftresponse_b1_p3, [14, 15, 16, 17], 82, replace_events=True)

#right response(83)
rightresponse_b1_p3 = mne.pick_events(rightresponse_b1_p3, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p3 = mne.merge_events(rightresponse_b1_p3, [18, 19, 20, 21], 83, replace_events=True)

#last response left(84)
lastresponseleft_b1_p3 = mne.pick_events(lastresponseleft_b1_p3, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p3 = mne.merge_events(lastresponseleft_b1_p3, [5, 6, 7, 8], 84, replace_events=True)

#last response right(85)
lastresponseright_b1_p3 = mne.pick_events(lastresponseright_b1_p3, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p3 = mne.merge_events(lastresponseright_b1_p3, [9, 10, 11, 12], 85, replace_events=True)


#preperation left (86) 
preparationleft_b1_p3 = mne.pick_events(preparationleft_b1_p3, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p3 = mne.merge_events(preparationleft_b1_p3, [5, 6, 7, 8], 86, replace_events=True)
preparationleft_b1_p3 = mne.event.shift_time_events(preparationleft_b1_p3, 86, 1.500, 500)
preparationleft_b1_p3 = np.delete(preparationleft_b1_p3, [11, 131, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#preperation right (87) 
preparationright_b1_p3 = mne.pick_events(preparationright_b1_p3, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p3 = mne.merge_events(preparationright_b1_p3, [9, 10, 11, 12], 87, replace_events=True)
preparationright_b1_p3 = mne.event.shift_time_events(preparationright_b1_p3, 87, 1.500, 500)
preparationright_b1_p3  = np.delete(preparationright_b1_p3, [35, 131, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#nogo (88)
nogob1p3 = mne.pick_events(nogob1p3, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p3 = mne.merge_events(nogob1p3, [24], 88, replace_events=True)
#--------B5-------#


#last  stimulus position left (89)
laststimpositionleft_b5_p3 = mne.pick_events(laststimpositionleft_b5_p3, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p3 = mne.merge_events(laststimpositionleft_b5_p3, [5, 6, 7, 8], 89, replace_events=True)
laststimpositionleft_b5_p3 = np.delete(laststimpositionleft_b5_p3, [5, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#last  stimulus position right (90)
laststimpositionright_b5_p3 = mne.pick_events(laststimpositionright_b5_p3, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p3 = mne.merge_events(laststimpositionright_b5_p3, [9, 10, 11, 12], 90, replace_events=True)
laststimpositionright_b5_p3 = np.delete(laststimpositionright_b5_p3, [35, 113, 161, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)

#feedback (91)
feedbackleft_b5_p3 = mne.pick_events(feedbackleft_b5_p3, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p3 = mne.merge_events(feedbackleft_b5_p3, [25,26], 91, replace_events=True)
feedbackleft_b5_p3 = mne.event.shift_time_events(feedbackleft_b5_p3, 91, -1.000, 500)
feedbackleft_b5_p3 = np.delete(feedbackleft_b5_p3, [0, 1, 2, 5, 6, 12, 16, 17, 18, 19, 20, 23, 27, 28, 29, 30, 31, 34, 36, 37, 38, 40, 41, 46], axis=0)


#feedback (92)
feedbackright_b5_p3 = mne.pick_events(feedbackright_b5_p3, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p3 = mne.merge_events(feedbackright_b5_p3, [25,26], 92, replace_events=True)
feedbackright_b5_p3 = mne.event.shift_time_events(feedbackright_b5_p3, 92, -1.000, 500)
feedbackright_b5_p3 = np.delete(feedbackright_b5_p3, [3, 4, 7, 8, 9, 10, 11, 13, 14, 15, 21, 22, 24, 25, 26, 32, 33, 35, 39, 42, 43, 44, 45, 47], axis=0)

#left response (93) 
leftresponse_b5_p3 = mne.pick_events(leftresponse_b5_p3, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p3 = mne.merge_events(leftresponse_b5_p3, [14, 15, 16, 17], 93, replace_events=True)

#right response(94)
rightresponse_b5_p3 = mne.pick_events(rightresponse_b5_p3, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p3 = mne.merge_events(rightresponse_b5_p3, [18, 19, 20, 21], 94, replace_events=True)

#last response left(95)
lastresponseleft_b5_p3 = mne.pick_events(lastresponseleft_b5_p3, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p3 = mne.merge_events(lastresponseleft_b5_p3, [5, 6, 7, 8], 95, replace_events=True)

#last response right(96)
lastresponseright_b5_p3 = mne.pick_events(lastresponseright_b5_p3, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p3 = mne.merge_events(lastresponseright_b5_p3, [9, 10, 11, 12], 96, replace_events=True)


#preperation left (97) 
preparationleft_b5_p3 = mne.pick_events(preparationleft_b5_p3, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p3 = mne.merge_events(preparationleft_b5_p3, [5, 6, 7, 8], 97, replace_events=True)
preparationleft_b5_p3 = mne.event.shift_time_events(preparationleft_b5_p3, 97, 1.500, 500)
preparationleft_b5_p3 = np.delete(preparationleft_b5_p3, [5, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#preperation right (98) 
preparationright_b5_p3 = mne.pick_events(preparationright_b5_p3, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p3 = mne.merge_events(preparationright_b5_p3, [9, 10, 11, 12], 98, replace_events=True)
preparationright_b5_p3 = mne.event.shift_time_events(preparationright_b5_p3, 98, 1.500, 500)
preparationright_b5_p3 = np.delete(preparationright_b5_p3, [35, 113, 161, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)

#nogo (99)
nogob5p3 = mne.pick_events(nogob5p3, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p3 = mne.merge_events(nogob5p3, [24], 99, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p3 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p3 = {'laststimleft':78,'laststimright':79, 'feedbackL':80, 'feedbackR':81, 'leftres':82, 'rightres':83, 'lastresleft':84, 'lastresright':85, 'prepleft':86, 'prepright':87, 'nogo': 88}
event_dictB5_p3 = {'laststimleft':89,'laststimright':90, 'feedbackL':91, 'feedbackR':92, 'leftres':93, 'rightres':94, 'lastresleft':95, 'lastresright':96, 'prepleft':97, 'prepright':98, 'nogo': 99}



#merging events togeher into one event list

finalB1_p3 = np.concatenate((laststimpositionleft_b1_p3, laststimpositionright_b1_p3, feedbackleft_b1_p3, feedbackright_b1_p3, leftresponse_b1_p3, rightresponse_b1_p3, lastresponseleft_b1_p3, lastresponseright_b1_p3, preparationleft_b1_p3, preparationright_b1_p3, events_p3, nogob1p3), axis=0)
finalB5_p3 = np.concatenate((laststimpositionleft_b5_p3, laststimpositionright_b5_p3, feedbackleft_b5_p3, feedbackright_b5_p3, leftresponse_b5_p3, rightresponse_b5_p3, lastresponseleft_b5_p3, lastresponseright_b5_p3, preparationleft_b5_p3, preparationright_b5_p3, events_p3, nogob5p3), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p3, event_id=event_dictB1_p3, 

                         sfreq=raw_p3.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p3, event_id=event_dictB5_p3, 

                         sfreq=raw5_p3.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend


#-----------------------------------------------------------------------------#
#                               PARTICIPANT 5
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_5_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part5_SeqL_ERD_6_B1.fif'
Part_5_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part5_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p5 = mne.io.read_raw_fif(Part_5_1, preload = True) 
 

#--------B5--------#
raw5_p5 = mne.io.read_raw_fif(Part_5_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p5, _ = mne.events_from_annotations(raw_p5, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p5, _ = mne.events_from_annotations(raw5_p5, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p5 = np.copy(events_p5)
laststimpositionright_b1_p5 = np.copy(events_p5)
feedbackleft_b1_p5 = np.copy(events_p5)
feedbackright_b1_p5 = np.copy(events_p5)
leftresponse_b1_p5 = np.copy(events_p5)
rightresponse_b1_p5 = np.copy(events_p5)
lastresponseleft_b1_p5 = np.copy(events_p5)
lastresponseright_b1_p5 = np.copy(events_p5)
preparationleft_b1_p5 = np.copy(events_p5)
preparationright_b1_p5 = np.copy(events_p5)
nogob1p5 = np.copy(events_p5)
#--------B5-------#
laststimpositionleft_b5_p5 = np.copy(eventsB5_p5)
laststimpositionright_b5_p5 = np.copy(eventsB5_p5)
feedbackleft_b5_p5 = np.copy(eventsB5_p5)
feedbackright_b5_p5 = np.copy(eventsB5_p5)
leftresponse_b5_p5 = np.copy(eventsB5_p5)
rightresponse_b5_p5 = np.copy(eventsB5_p5)
lastresponseleft_b5_p5 = np.copy(eventsB5_p5)
lastresponseright_b5_p5 = np.copy(eventsB5_p5)
preparationleft_b5_p5 = np.copy(eventsB5_p5)
preparationright_b5_p5 = np.copy(eventsB5_p5)
nogob5p5 = np.copy(eventsB5_p5)


#print to see if event times are correct

print(events_p5)

#--------B1-------#


#last  stimulus position left (100)
laststimpositionleft_b1_p5 = mne.pick_events(laststimpositionleft_b1_p5, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p5 = mne.merge_events(laststimpositionleft_b1_p5, [5, 6, 7, 8], 100, replace_events=True)
laststimpositionleft_b1_p5 = np.delete(laststimpositionleft_b1_p5, [23, 53, 113, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)

#last  stimulus position right (101)
laststimpositionright_b1_p5 = mne.pick_events(laststimpositionright_b1_p5, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p5 = mne.merge_events(laststimpositionright_b1_p5, [9, 10, 11, 12], 101, replace_events=True)
laststimpositionright_b1_p5 = np.delete(laststimpositionright_b1_p5, [89, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#feedback left (102)
feedbackleft_b1_p5 = mne.pick_events(feedbackleft_b1_p5, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p5 = mne.merge_events(feedbackleft_b1_p5, [25,26], 102, replace_events=True)
feedbackleft_b1_p5 = mne.event.shift_time_events(feedbackleft_b1_p5, 102, -1.000, 500)
feedbackleft_b1_p5 = np.delete(feedbackleft_b1_p5, [2, 9, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 25, 28, 31, 33, 34, 40, 41, 42, 43, 44, 45], axis=0)

#feedback right (103)
feedbackright_b1_p5 = mne.pick_events(feedbackright_b1_p5, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p5 = mne.merge_events(feedbackright_b1_p5, [25,26], 103, replace_events=True)
feedbackright_b1_p5 = mne.event.shift_time_events(feedbackright_b1_p5, 103, -1.000, 500)
feedbackright_b1_p5 = np.delete(feedbackright_b1_p5, [0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 15, 20, 26, 27, 29, 30, 32, 35, 36, 37, 38, 39], axis=0)

#left response (104) 
leftresponse_b1_p5 = mne.pick_events(leftresponse_b1_p5, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p5 = mne.merge_events(leftresponse_b1_p5, [14, 15, 16, 17], 104, replace_events=True)

#right response(105)
rightresponse_b1_p5 = mne.pick_events(rightresponse_b1_p5, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p5 = mne.merge_events(rightresponse_b1_p5, [18, 19, 20, 21], 105, replace_events=True)

#last response left(106)
lastresponseleft_b1_p5 = mne.pick_events(lastresponseleft_b1_p5, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p5 = mne.merge_events(lastresponseleft_b1_p5, [5, 6, 7, 8], 106, replace_events=True)

#last response right(107)
lastresponseright_b1_p5 = mne.pick_events(lastresponseright_b1_p5, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p5 = mne.merge_events(lastresponseright_b1_p5, [9, 10, 11, 12], 107, replace_events=True)


#preperation left (108) 
preparationleft_b1_p5 = mne.pick_events(preparationleft_b1_p5, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p5 = mne.merge_events(preparationleft_b1_p5, [5, 6, 7, 8], 108, replace_events=True)
preparationleft_b1_p5 = mne.event.shift_time_events(preparationleft_b1_p5, 108, 1.500, 500)
preparationleft_b1_p5 = np.delete(preparationleft_b1_p5, [23, 53, 113, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)

#preperation right (109) 
preparationright_b1_p5 = mne.pick_events(preparationright_b1_p5, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p5 = mne.merge_events(preparationright_b1_p5, [9, 10, 11, 12], 109, replace_events=True)
preparationright_b1_p5 = mne.event.shift_time_events(preparationright_b1_p5, 109, 1.500, 500)
preparationright_b1_p5 = np.delete(preparationright_b1_p5, [89, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)


#nogo (110)
nogob1p5 = mne.pick_events(nogob1p5, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p5 = mne.merge_events(nogob1p5, [24], 110, replace_events=True)


#--------B5-------#
#last  stimulus position left (111)
laststimpositionleft_b5_p5 = mne.pick_events(laststimpositionleft_b5_p5, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p5 = mne.merge_events(laststimpositionleft_b5_p5, [5, 6, 7, 8], 111, replace_events=True)
laststimpositionleft_b5_p5 = np.delete(laststimpositionleft_b5_p5, [5, 143, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#last  stimulus position right (112)
laststimpositionright_b5_p5 = mne.pick_events(laststimpositionright_b5_p5, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p5 = mne.merge_events(laststimpositionright_b5_p5, [9, 10, 11, 12], 112, replace_events=True)
laststimpositionright_b5_p5 = np.delete(laststimpositionright_b5_p5, [35, 89, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#feedback (113)
feedbackleft_b5_p5 = mne.pick_events(feedbackleft_b5_p5, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p5 = mne.merge_events(feedbackleft_b5_p5, [25,26], 113, replace_events=True)
feedbackleft_b5_p5 = mne.event.shift_time_events(feedbackleft_b5_p5, 113, -1.000, 500)
feedbackleft_b5_p5 = np.delete(feedbackleft_b5_p5, [0, 1, 2, 3, 5, 6, 7, 8, 11, 14, 16, 18, 25, 31, 33, 35, 36, 37, 39, 41, 42, 45, 46, 47], axis=0)

#feedback (114)
feedbackright_b5_p5 = mne.pick_events(feedbackright_b5_p5, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p5 = mne.merge_events(feedbackright_b5_p5, [25,26], 114, replace_events=True)
feedbackright_b5_p5 = mne.event.shift_time_events(feedbackright_b5_p5, 114, -1.000, 500)
feedbackright_b5_p5 = np.delete(feedbackright_b5_p5, [4, 9, 10, 12, 13, 15, 17, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 32, 34, 38, 40, 43, 44], axis=0)

#left response (115) 
leftresponse_b5_p5 = mne.pick_events(leftresponse_b5_p5, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p5 = mne.merge_events(leftresponse_b5_p5, [14, 15, 16, 17], 115, replace_events=True)

#right response(116)
rightresponse_b5_p5 = mne.pick_events(rightresponse_b5_p5, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p5 = mne.merge_events(rightresponse_b5_p5, [18, 19, 20, 21], 116, replace_events=True)

#last response left(117)
lastresponseleft_b5_p5 = mne.pick_events(lastresponseleft_b5_p5, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p5 = mne.merge_events(lastresponseleft_b5_p5, [5, 6, 7, 8], 117, replace_events=True)

#last response right(118)
lastresponseright_b5_p5 = mne.pick_events(lastresponseright_b5_p5, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p5 = mne.merge_events(lastresponseright_b5_p5, [9, 10, 11, 12], 118, replace_events=True)


#preperation left (119) 
preparationleft_b5_p5 = mne.pick_events(preparationleft_b5_p5, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p5 = mne.merge_events(preparationleft_b5_p5, [5, 6, 7, 8], 119, replace_events=True)
preparationleft_b5_p5 = mne.event.shift_time_events(preparationleft_b5_p5, 119, 1.500, 500)
preparationleft_b5_p5 = np.delete(preparationleft_b5_p5, [5, 143, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#preperation right (120) 
preparationright_b5_p5 = mne.pick_events(preparationright_b5_p5, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p5 = mne.merge_events(preparationright_b5_p5, [9, 10, 11, 12], 120, replace_events=True)
preparationright_b5_p5 = mne.event.shift_time_events(preparationright_b5_p5, 120, 1.500, 500)
preparationright_b5_p5 = np.delete(preparationright_b5_p5, [35, 89, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#nogo (121)
nogob5p5 = mne.pick_events(nogob5p5, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p5 = mne.merge_events(nogob5p5, [24], 121, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p5 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p5 = {'laststimleft':100,'laststimright':101, 'feedbackL':102, 'feedbackR':103, 'leftres':104, 'rightres':105, 'lastresleft':106, 'lastresright':107, 'prepleft':108, 'prepright':109, 'nogo': 110,}
event_dictB5_p5 = {'laststimleft':111,'laststimright':112, 'feedbackL':113, 'feedbackR':114, 'leftres':115, 'rightres':116, 'lastresleft':117, 'lastresright':118, 'prepleft':119, 'prepright':120, 'nogo': 121,}



#merging events togeher into one event list

finalB1_p5 = np.concatenate((laststimpositionleft_b1_p5, laststimpositionright_b1_p5, feedbackleft_b1_p5, feedbackright_b1_p5, leftresponse_b1_p5, rightresponse_b1_p5, lastresponseleft_b1_p5, lastresponseright_b1_p5, preparationleft_b1_p5, preparationright_b1_p5, events_p5, nogob1p5), axis=0)
finalB5_p5 = np.concatenate((laststimpositionleft_b5_p5, laststimpositionright_b5_p5, feedbackleft_b5_p5, feedbackright_b5_p5, leftresponse_b5_p5, rightresponse_b5_p5, lastresponseleft_b5_p5, lastresponseright_b5_p5, preparationleft_b5_p5, preparationright_b5_p5, events_p5, nogob5p5), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p5, event_id=event_dictB1_p5, 

                         sfreq=raw_p5.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p5, event_id=event_dictB5_p5, 

                         sfreq=raw5_p5.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend



#-----------------------------------------------------------------------------#
#                               PARTICIPANT 6
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_6_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part6_SeqL_ERD_6_B1.fif'
Part_6_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part6_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p6 = mne.io.read_raw_fif(Part_6_1, preload = True) 
 

#--------B5--------#
raw5_p6 = mne.io.read_raw_fif(Part_6_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p6, _ = mne.events_from_annotations(raw_p6, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p6, _ = mne.events_from_annotations(raw5_p6, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p6 = np.copy(events_p6)
laststimpositionright_b1_p6 = np.copy(events_p6)
feedbackleft_b1_p6 = np.copy(events_p6)
feedbackright_b1_p6 = np.copy(events_p6)
leftresponse_b1_p6 = np.copy(events_p6)
rightresponse_b1_p6 = np.copy(events_p6)
lastresponseleft_b1_p6 = np.copy(events_p6)
lastresponseright_b1_p6 = np.copy(events_p6)
preparationleft_b1_p6 = np.copy(events_p6)
preparationright_b1_p6 = np.copy(events_p6)
nogob1p6 = np.copy(events_p6)
#--------B5-------#
laststimpositionleft_b5_p6 = np.copy(eventsB5_p6)
laststimpositionright_b5_p6 = np.copy(eventsB5_p6)
feedbackleft_b5_p6 = np.copy(eventsB5_p6)
feedbackright_b5_p6 = np.copy(eventsB5_p6)
leftresponse_b5_p6 = np.copy(eventsB5_p6)
rightresponse_b5_p6 = np.copy(eventsB5_p6)
lastresponseleft_b5_p6 = np.copy(eventsB5_p6)
lastresponseright_b5_p6 = np.copy(eventsB5_p6)
preparationleft_b5_p6 = np.copy(eventsB5_p6)
preparationright_b5_p6 = np.copy(eventsB5_p6)
nogob5p6 = np.copy(eventsB5_p6)


#print to see if event times are correct

print(events_p6)

#--------B1-------#


#last  stimulus position left (122)
laststimpositionleft_b1_p6 = mne.pick_events(laststimpositionleft_b1_p6, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p6 = mne.merge_events(laststimpositionleft_b1_p6, [5, 6, 7, 8], 122, replace_events=True)
laststimpositionleft_b1_p6 = np.delete(laststimpositionleft_b1_p6, [41, 149, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#last  stimulus position right (123)
laststimpositionright_b1_p6 = mne.pick_events(laststimpositionright_b1_p6, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p6 = mne.merge_events(laststimpositionright_b1_p6, [9, 10, 11, 12], 123, replace_events=True)
laststimpositionright_b1_p6 = np.delete(laststimpositionright_b1_p6, [65, 125, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#feedback left (124)
feedbackleft_b1_p6 = mne.pick_events(feedbackleft_b1_p6, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p6 = mne.merge_events(feedbackleft_b1_p6, [25,26], 124, replace_events=True)
feedbackleft_b1_p6 = mne.event.shift_time_events(feedbackleft_b1_p6, 124, -1.000, 500)
feedbackleft_b1_p6 = np.delete(feedbackleft_b1_p6, [3, 4, 6, 7, 10, 12, 13, 14, 17, 18, 19, 21, 25, 27, 28, 29, 31, 36, 39, 40, 41, 42, 43, 44], axis=0)



#feedback right (125)
feedbackright_b1_p6 = mne.pick_events(feedbackright_b1_p6, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p6 = mne.merge_events(feedbackright_b1_p6, [25,26], 125, replace_events=True)
feedbackright_b1_p6 = mne.event.shift_time_events(feedbackright_b1_p6, 125, -1.000, 500)
feedbackright_b1_p6 = np.delete(feedbackright_b1_p6, [0, 1, 2, 5, 8, 9, 11, 15, 16, 20, 22, 23, 24, 26, 30, 32, 33, 34, 35, 37, 38, 45, 46, 47], axis=0)

#left response (126) 
leftresponse_b1_p6 = mne.pick_events(leftresponse_b1_p6, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p6 = mne.merge_events(leftresponse_b1_p6, [14, 15, 16, 17], 126, replace_events=True)

#right response(127)
rightresponse_b1_p6 = mne.pick_events(rightresponse_b1_p6, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p6 = mne.merge_events(rightresponse_b1_p6, [18, 19, 20, 21], 127, replace_events=True)

#last response left(128)
lastresponseleft_b1_p6 = mne.pick_events(lastresponseleft_b1_p6, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p6 = mne.merge_events(lastresponseleft_b1_p6, [5, 6, 7, 8], 128, replace_events=True)

#last response right(129)
lastresponseright_b1_p6 = mne.pick_events(lastresponseright_b1_p6, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p6 = mne.merge_events(lastresponseright_b1_p6, [9, 10, 11, 12], 129, replace_events=True)


#preperation left (130) 
preparationleft_b1_p6 = mne.pick_events(preparationleft_b1_p6, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p6 = mne.merge_events(preparationleft_b1_p6, [5, 6, 7, 8], 130, replace_events=True)
preparationleft_b1_p6 = mne.event.shift_time_events(preparationleft_b1_p6, 130, 1.500, 500)
preparationleft_b1_p6 = np.delete(preparationleft_b1_p6, [41, 149, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#preperation right (131) 
preparationright_b1_p6 = mne.pick_events(preparationright_b1_p6, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p6 = mne.merge_events(preparationright_b1_p6, [9, 10, 11, 12], 131, replace_events=True)
preparationright_b1_p6 = mne.event.shift_time_events(preparationright_b1_p6, 131, 1.500, 500)
preparationright_b1_p6 = np.delete(preparationright_b1_p6, [65, 125, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#nogo (132)
nogob1p6 = mne.pick_events(nogob1p6, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p6 = mne.merge_events(nogob1p6, [24], 132, replace_events=True)
#--------B5-------#


#last  stimulus position left (133)
laststimpositionleft_b5_p6 = mne.pick_events(laststimpositionleft_b5_p6, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p6 = mne.merge_events(laststimpositionleft_b5_p6, [5, 6, 7, 8], 133, replace_events=True)
laststimpositionleft_b5_p6 = np.delete(laststimpositionleft_b5_p6, [101, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#last  stimulus position right (134)
laststimpositionright_b5_p6 = mne.pick_events(laststimpositionright_b5_p6, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p6 = mne.merge_events(laststimpositionright_b5_p6, [9, 10, 11, 12], 134, replace_events=True)
laststimpositionright_b5_p6 = np.delete(laststimpositionright_b5_p6, [17, 29, 95, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)



#feedback (135)
feedbackleft_b5_p6 = mne.pick_events(feedbackleft_b5_p6, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p6 = mne.merge_events(feedbackleft_b5_p6, [25,26], 135, replace_events=True)
feedbackleft_b5_p6 = mne.event.shift_time_events(feedbackleft_b5_p6, 135, -1.000, 500)
feedbackleft_b5_p6 = np.delete(feedbackleft_b5_p6, [3, 7, 9, 10, 11, 13, 15, 16, 17, 20, 22, 23, 25, 27, 28, 30, 31, 32, 34, 35, 38, 39, 41, 45], axis=0)

#feedback (136)
feedbackright_b5_p6 = mne.pick_events(feedbackright_b5_p6, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p6 = mne.merge_events(feedbackright_b5_p6, [25,26], 136, replace_events=True)
feedbackright_b5_p6 = mne.event.shift_time_events(feedbackright_b5_p6, 136, -1.000, 500)
feedbackright_b5_p6 = np.delete(feedbackright_b5_p6, [0, 1, 2, 4, 5, 6, 8, 12, 14, 18, 19, 21, 24, 26, 29, 33, 36, 37, 40, 42, 43, 44, 46, 47], axis=0)


#left response (137) 
leftresponse_b5_p6 = mne.pick_events(leftresponse_b5_p6, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p6 = mne.merge_events(leftresponse_b5_p6, [14, 15, 16, 17], 137, replace_events=True)

#right response(138)
rightresponse_b5_p6 = mne.pick_events(rightresponse_b5_p6, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p6 = mne.merge_events(rightresponse_b5_p6, [18, 19, 20, 21], 138, replace_events=True)

#last response left(139)
lastresponseleft_b5_p6 = mne.pick_events(lastresponseleft_b5_p6, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p6 = mne.merge_events(lastresponseleft_b5_p6, [5, 6, 7, 8], 139, replace_events=True)

#last response right(140)
lastresponseright_b5_p6 = mne.pick_events(lastresponseright_b5_p6, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p6 = mne.merge_events(lastresponseright_b5_p6, [9, 10, 11, 12], 140, replace_events=True)


#preperation left (141) 
preparationleft_b5_p6 = mne.pick_events(preparationleft_b5_p6, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p6 = mne.merge_events(preparationleft_b5_p6, [5, 6, 7, 8], 141, replace_events=True)
preparationleft_b5_p6 = mne.event.shift_time_events(preparationleft_b5_p6, 141, 1.500, 500)
preparationleft_b5_p6 = np.delete(preparationleft_b5_p6, [101, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)



#preperation right (142) 
preparationright_b5_p6 = mne.pick_events(preparationright_b5_p6, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p6 = mne.merge_events(preparationright_b5_p6, [9, 10, 11, 12], 142, replace_events=True)
preparationright_b5_p6 = mne.event.shift_time_events(preparationright_b5_p6, 142, 1.500, 500)
preparationright_b5_p6 = np.delete(preparationright_b5_p6, [17, 29, 95, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)

#nogo (143)
nogob5p6 = mne.pick_events(nogob5p6, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p6 = mne.merge_events(nogob5p6, [24], 143, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p6 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p6 = {'laststimleft':122,'laststimright':123, 'feedbackL':124, 'feedbackR':125, 'leftres':126, 'rightres':127, 'lastresleft':128, 'lastresright':129, 'prepleft':130, 'prepright':131, 'nogo': 132}
event_dictB5_p6 = {'laststimleft':133,'laststimright':134, 'feedbackL':135, 'feedbackR':136, 'leftres':137, 'rightres':138, 'lastresleft':139, 'lastresright':140, 'prepleft':141, 'prepright':142, 'nogo': 143}



#merging events togeher into one event list

finalB1_p6 = np.concatenate((laststimpositionleft_b1_p6, laststimpositionright_b1_p6, feedbackleft_b1_p6, feedbackright_b1_p6, leftresponse_b1_p6, rightresponse_b1_p6, lastresponseleft_b1_p6, lastresponseright_b1_p6, preparationleft_b1_p6, preparationright_b1_p6, events_p6, nogob1p6), axis=0)
finalB5_p6 = np.concatenate((laststimpositionleft_b5_p6, laststimpositionright_b5_p6, feedbackleft_b5_p6, feedbackright_b5_p6, leftresponse_b5_p6, rightresponse_b5_p6, lastresponseleft_b5_p6, lastresponseright_b5_p6, preparationleft_b5_p6, preparationright_b5_p6, events_p6, nogob5p6), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p6, event_id=event_dictB1_p6, 

                         sfreq=raw_p6.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p6, event_id=event_dictB5_p6, 

                         sfreq=raw5_p6.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend


#-----------------------------------------------------------------------------#
#                               PARTICIPANT 7
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_7_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part7_SeqL_ERD_6_B1.fif'
Part_7_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part7_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p7 = mne.io.read_raw_fif(Part_7_1, preload = True) 
 

#--------B5--------#
raw5_p7 = mne.io.read_raw_fif(Part_7_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p7, _ = mne.events_from_annotations(raw_p7, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p7, _ = mne.events_from_annotations(raw5_p7, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p7 = np.copy(events_p7)
laststimpositionright_b1_p7 = np.copy(events_p7)
feedbackleft_b1_p7 = np.copy(events_p7)
feedbackright_b1_p7 = np.copy(events_p7)
leftresponse_b1_p7 = np.copy(events_p7)
rightresponse_b1_p7 = np.copy(events_p7)
lastresponseleft_b1_p7 = np.copy(events_p7)
lastresponseright_b1_p7 = np.copy(events_p7)
preparationleft_b1_p7 = np.copy(events_p7)
preparationright_b1_p7 = np.copy(events_p7)
nogob1p7 = np.copy(events_p7)
#--------B5-------#
laststimpositionleft_b5_p7 = np.copy(eventsB5_p7)
laststimpositionright_b5_p7 = np.copy(eventsB5_p7)
feedbackleft_b5_p7 = np.copy(eventsB5_p7)
feedbackright_b5_p7 = np.copy(eventsB5_p7)
leftresponse_b5_p7 = np.copy(eventsB5_p7)
rightresponse_b5_p7 = np.copy(eventsB5_p7)
lastresponseleft_b5_p7 = np.copy(eventsB5_p7)
lastresponseright_b5_p7 = np.copy(eventsB5_p7)
preparationleft_b5_p7 = np.copy(eventsB5_p7)
preparationright_b5_p7 = np.copy(eventsB5_p7)
nogob5p7 = np.copy(eventsB5_p7)


#print to see if event times are correct

print(events_p7)

#--------B1-------#


#last  stimulus position left (144)
laststimpositionleft_b1_p7 = mne.pick_events(laststimpositionleft_b1_p7, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p7 = mne.merge_events(laststimpositionleft_b1_p7, [5, 6, 7, 8], 144, replace_events=True)
laststimpositionleft_b1_p7 = np.delete(laststimpositionleft_b1_p7, [71, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#last  stimulus position right (145)
laststimpositionright_b1_p7 = mne.pick_events(laststimpositionright_b1_p7, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p7 = mne.merge_events(laststimpositionright_b1_p7, [9, 10, 11, 12], 145, replace_events=True)
laststimpositionright_b1_p7 = np.delete(laststimpositionright_b1_p7, [29, 95, 143, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)

#feedback left (146)
feedbackleft_b1_p7 = mne.pick_events(feedbackleft_b1_p7, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p7 = mne.merge_events(feedbackleft_b1_p7, [25,26], 146, replace_events=True)
feedbackleft_b1_p7 = mne.event.shift_time_events(feedbackleft_b1_p7, 146, -1.000, 500)
feedbackleft_b1_p7 = np.delete(feedbackleft_b1_p7, [1, 7, 8, 12, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 30, 31, 33, 34, 36, 37, 39, 40, 46, 47], axis=0)



#feedback right (147)
feedbackright_b1_p7 = mne.pick_events(feedbackright_b1_p7, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p7 = mne.merge_events(feedbackright_b1_p7, [25,26], 147, replace_events=True)
feedbackright_b1_p7 = mne.event.shift_time_events(feedbackright_b1_p7, 147, -1.000, 500)
feedbackright_b1_p7 = np.delete(feedbackright_b1_p7, [0, 2, 3, 4, 5, 6, 9, 10, 11, 13, 14, 17, 26, 27, 28, 29, 32, 35, 38, 41, 42, 43, 44, 45], axis=0)

#left response (148) 
leftresponse_b1_p7 = mne.pick_events(leftresponse_b1_p7, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p7 = mne.merge_events(leftresponse_b1_p7, [14, 15, 16, 17], 148, replace_events=True)

#right response(149)
rightresponse_b1_p7 = mne.pick_events(rightresponse_b1_p7, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p7 = mne.merge_events(rightresponse_b1_p7, [18, 19, 20, 21], 149, replace_events=True)

#last response left(150)
lastresponseleft_b1_p7 = mne.pick_events(lastresponseleft_b1_p7, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p7 = mne.merge_events(lastresponseleft_b1_p7, [5, 6, 7, 8], 150, replace_events=True)

#last response right(151)
lastresponseright_b1_p7 = mne.pick_events(lastresponseright_b1_p7, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p7 = mne.merge_events(lastresponseright_b1_p7, [9, 10, 11, 12], 151, replace_events=True)


#preperation left (152) 
preparationleft_b1_p7 = mne.pick_events(preparationleft_b1_p7, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p7 = mne.merge_events(preparationleft_b1_p7, [5, 6, 7, 8], 152, replace_events=True)
preparationleft_b1_p7 = mne.event.shift_time_events(preparationleft_b1_p7, 152, 1.500, 500)
preparationleft_b1_p7 = np.delete(preparationleft_b1_p7, [71, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#preperation right (153) 
preparationright_b1_p7 = mne.pick_events(preparationright_b1_p7, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p7 = mne.merge_events(preparationright_b1_p7, [9, 10, 11, 12], 153, replace_events=True)
preparationright_b1_p7 = mne.event.shift_time_events(preparationright_b1_p7, 153, 1.500, 500)
preparationright_b1_p7 = np.delete(preparationright_b1_p7, [29, 95, 143, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)

#nogo (154)
nogob1p7 = mne.pick_events(nogob1p7, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p7 = mne.merge_events(nogob1p7, [24], 154, replace_events=True)
#--------B5-------#


#last  stimulus position left (155)
laststimpositionleft_b5_p7 = mne.pick_events(laststimpositionleft_b5_p7, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p7 = mne.merge_events(laststimpositionleft_b5_p7, [5, 6, 7, 8], 155, replace_events=True)
laststimpositionleft_b5_p7 = np.delete(laststimpositionleft_b5_p7, [101, 125, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#last  stimulus position right (156)
laststimpositionright_b5_p7 = mne.pick_events(laststimpositionright_b5_p7, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p7 = mne.merge_events(laststimpositionright_b5_p7, [9, 10, 11, 12], 156, replace_events=True)
laststimpositionright_b5_p7 = np.delete(laststimpositionright_b5_p7, [5, 23, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#feedback (157)
feedbackleft_b5_p7 = mne.pick_events(feedbackleft_b5_p7, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p7 = mne.merge_events(feedbackleft_b5_p7, [25,26], 157, replace_events=True)
feedbackleft_b5_p7 = mne.event.shift_time_events(feedbackleft_b5_p7, 157, -1.000, 500)
feedbackleft_b5_p7 = np.delete(feedbackleft_b5_p7, [0, 1, 2, 3, 7, 8, 10, 13, 14, 18, 21, 22, 25, 26, 27, 29, 30, 31, 35, 39, 40, 42, 46, 47], axis=0)


#feedback (158)
feedbackright_b5_p7 = mne.pick_events(feedbackright_b5_p7, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p7 = mne.merge_events(feedbackright_b5_p7, [25,26], 158, replace_events=True)
feedbackright_b5_p7 = mne.event.shift_time_events(feedbackright_b5_p7, 158, -1.000, 500)
feedbackright_b5_p7 = np.delete(feedbackright_b5_p7, [4, 5, 6, 9, 11, 12, 15, 16, 17, 19, 20, 23, 24, 28, 32, 33, 34, 36, 37, 38, 41, 43, 44, 45], axis=0)

#left response (159) 
leftresponse_b5_p7 = mne.pick_events(leftresponse_b5_p7, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p7 = mne.merge_events(leftresponse_b5_p7, [14, 15, 16, 17], 159, replace_events=True)

#right response(160)
rightresponse_b5_p7 = mne.pick_events(rightresponse_b5_p7, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p7 = mne.merge_events(rightresponse_b5_p7, [18, 19, 20, 21], 160, replace_events=True)

#last response left(161)
lastresponseleft_b5_p7 = mne.pick_events(lastresponseleft_b5_p7, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p7 = mne.merge_events(lastresponseleft_b5_p7, [5, 6, 7, 8], 161, replace_events=True)

#last response right(162)
lastresponseright_b5_p7 = mne.pick_events(lastresponseright_b5_p7, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p7 = mne.merge_events(lastresponseright_b5_p7, [9, 10, 11, 12], 162, replace_events=True)


#preperation left (163) 
preparationleft_b5_p7 = mne.pick_events(preparationleft_b5_p7, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p7 = mne.merge_events(preparationleft_b5_p7, [5, 6, 7, 8], 163, replace_events=True)
preparationleft_b5_p7 = mne.event.shift_time_events(preparationleft_b5_p7, 163, 1.500, 500)
preparationleft_b5_p7 = np.delete(preparationleft_b5_p7, [101, 125, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#preperation right (164) 
preparationright_b5_p7 = mne.pick_events(preparationright_b5_p7, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p7 = mne.merge_events(preparationright_b5_p7, [9, 10, 11, 12], 164, replace_events=True)
preparationright_b5_p7 = mne.event.shift_time_events(preparationright_b5_p7, 164, 1.500, 500)
preparationright_b5_p7 = np.delete(preparationright_b5_p7, [5, 23, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#nogo (165)
nogob5p7 = mne.pick_events(nogob5p7, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p7 = mne.merge_events(nogob5p7, [24], 165, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p7 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p7 = {'laststimleft':144,'laststimright':145, 'feedbackL':146, 'feedbackR':147, 'leftres':148, 'rightres':149, 'lastresleft':150, 'lastresright':151, 'prepleft':152, 'prepright':153, 'nogo': 154,}
event_dictB5_p7 = {'laststimleft':155,'laststimright':156, 'feedbackL':157, 'feedbackR':158, 'leftres':159, 'rightres':160, 'lastresleft':161, 'lastresright':162, 'prepleft':163, 'prepright':164, 'nogo': 165,}



#merging events togeher into one event list

finalB1_p7 = np.concatenate((laststimpositionleft_b1_p7, laststimpositionright_b1_p7, feedbackleft_b1_p7, feedbackright_b1_p7, leftresponse_b1_p7, rightresponse_b1_p7, lastresponseleft_b1_p7, lastresponseright_b1_p7, preparationleft_b1_p7, preparationright_b1_p7, events_p7, nogob1p7), axis=0)
finalB5_p7 = np.concatenate((laststimpositionleft_b5_p7, laststimpositionright_b5_p7, feedbackleft_b5_p7, feedbackright_b5_p7, leftresponse_b5_p7, rightresponse_b5_p7, lastresponseleft_b5_p7, lastresponseright_b5_p7, preparationleft_b5_p7, preparationright_b5_p7, events_p7, nogob5p7), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p7, event_id=event_dictB1_p7, 

                         sfreq=raw_p7.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p7, event_id=event_dictB5_p7, 

                         sfreq=raw5_p7.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend




#-----------------------------------------------------------------------------#
#                               PARTICIPANT 8
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_8_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part8_SeqL_ERD_6_B1.fif'
Part_8_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part8_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p8 = mne.io.read_raw_fif(Part_8_1, preload = True) 
 

#--------B5--------#
raw5_p8 = mne.io.read_raw_fif(Part_8_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p8, _ = mne.events_from_annotations(raw_p8, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p8, _ = mne.events_from_annotations(raw5_p8, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p8 = np.copy(events_p8)
laststimpositionright_b1_p8 = np.copy(events_p8)
feedbackleft_b1_p8 = np.copy(events_p8)
feedbackright_b1_p8 = np.copy(events_p8)
leftresponse_b1_p8 = np.copy(events_p8)
rightresponse_b1_p8 = np.copy(events_p8)
lastresponseleft_b1_p8 = np.copy(events_p8)
lastresponseright_b1_p8 = np.copy(events_p8)
preparationleft_b1_p8 = np.copy(events_p8)
preparationright_b1_p8 = np.copy(events_p8)
nogob1p8 = np.copy(events_p8)
#--------B5-------#
laststimpositionleft_b5_p8 = np.copy(eventsB5_p8)
laststimpositionright_b5_p8 = np.copy(eventsB5_p8)
feedbackleft_b5_p8 = np.copy(eventsB5_p8)
feedbackright_b5_p8 = np.copy(eventsB5_p8)
leftresponse_b5_p8 = np.copy(eventsB5_p8)
rightresponse_b5_p8 = np.copy(eventsB5_p8)
lastresponseleft_b5_p8 = np.copy(eventsB5_p8)
lastresponseright_b5_p8 = np.copy(eventsB5_p8)
preparationleft_b5_p8 = np.copy(eventsB5_p8)
preparationright_b5_p8 = np.copy(eventsB5_p8)
nogob5p8 = np.copy(eventsB5_p8)


#print to see if event times are correct

print(events_p8)

#--------B1-------#


#last  stimulus position left (166)
laststimpositionleft_b1_p8 = mne.pick_events(laststimpositionleft_b1_p8, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p8 = mne.merge_events(laststimpositionleft_b1_p8, [5, 6, 7, 8], 166, replace_events=True)
laststimpositionleft_b1_p8 = np.delete(laststimpositionleft_b1_p8, [11, 71, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#last  stimulus position right (167)
laststimpositionright_b1_p8 = mne.pick_events(laststimpositionright_b1_p8, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p8 = mne.merge_events(laststimpositionright_b1_p8, [9, 10, 11, 12], 167, replace_events=True)
laststimpositionright_b1_p8 = np.delete(laststimpositionright_b1_p8, [101, 155, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#feedback left (168)
feedbackleft_b1_p8 = mne.pick_events(feedbackleft_b1_p8, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p8 = mne.merge_events(feedbackleft_b1_p8, [25,26], 168, replace_events=True)
feedbackleft_b1_p8 = mne.event.shift_time_events(feedbackleft_b1_p8, 168, -1.000, 500)
feedbackleft_b1_p8 = np.delete(feedbackleft_b1_p8, [1, 3, 4, 5, 7, 8, 10, 11, 14, 17, 18, 20, 24, 25, 26, 28, 29, 30, 32, 34, 35, 38, 39, 42], axis=0)


#feedback right (169)
feedbackright_b1_p8 = mne.pick_events(feedbackright_b1_p8, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p8 = mne.merge_events(feedbackright_b1_p8, [25,26], 169, replace_events=True)
feedbackright_b1_p8 = mne.event.shift_time_events(feedbackright_b1_p8, 169, -1.000, 500)
feedbackright_b1_p8 = np.delete(feedbackright_b1_p8, [0, 2, 6, 9, 12, 13, 15, 16, 19, 21, 22, 23, 27, 31, 33, 36, 37, 40, 41, 43, 44, 45, 46, 47], axis=0)

#left response (170) 
leftresponse_b1_p8 = mne.pick_events(leftresponse_b1_p8, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p8 = mne.merge_events(leftresponse_b1_p8, [14, 15, 16, 17], 170, replace_events=True)

#right response(171)
rightresponse_b1_p8 = mne.pick_events(rightresponse_b1_p8, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p8 = mne.merge_events(rightresponse_b1_p8, [18, 19, 20, 21], 171, replace_events=True)

#last response left(172)
lastresponseleft_b1_p8 = mne.pick_events(lastresponseleft_b1_p8, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p8 = mne.merge_events(lastresponseleft_b1_p8, [5, 6, 7, 8], 172, replace_events=True)

#last response right(173)
lastresponseright_b1_p8 = mne.pick_events(lastresponseright_b1_p8, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p8 = mne.merge_events(lastresponseright_b1_p8, [9, 10, 11, 12], 173, replace_events=True)


#preperation left (174) 
preparationleft_b1_p8 = mne.pick_events(preparationleft_b1_p8, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p8 = mne.merge_events(preparationleft_b1_p8, [5, 6, 7, 8], 174, replace_events=True)
preparationleft_b1_p8 = mne.event.shift_time_events(preparationleft_b1_p8, 174, 1.500, 500)
preparationleft_b1_p8 = np.delete(preparationleft_b1_p8, [11, 71, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#preperation right (175) 
preparationright_b1_p8 = mne.pick_events(preparationright_b1_p8, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p8 = mne.merge_events(preparationright_b1_p8, [9, 10, 11, 12], 175, replace_events=True)
preparationright_b1_p8 = mne.event.shift_time_events(preparationright_b1_p8, 175, 1.500, 500)
preparationright_b1_p8 = np.delete(preparationright_b1_p8, [101, 155, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#nogo (176)
nogob1p8 = mne.pick_events(nogob1p8, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p8 = mne.merge_events(nogob1p8, [24], 176, replace_events=True)
#--------B5-------#


#last  stimulus position left (177)
laststimpositionleft_b5_p8 = mne.pick_events(laststimpositionleft_b5_p8, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p8 = mne.merge_events(laststimpositionleft_b5_p8, [5, 6, 7, 8], 177, replace_events=True)
laststimpositionleft_b5_p8 = np.delete(laststimpositionleft_b5_p8, [41, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#last  stimulus position right (178)
laststimpositionright_b5_p8 = mne.pick_events(laststimpositionright_b5_p8, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p8 = mne.merge_events(laststimpositionright_b5_p8, [9, 10, 11, 12], 178, replace_events=True)
laststimpositionright_b5_p8 = np.delete(laststimpositionright_b5_p8, [47, 107, 125, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)

#feedback (179)
feedbackleft_b5_p8 = mne.pick_events(feedbackleft_b5_p8, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p8 = mne.merge_events(feedbackleft_b5_p8, [25,26], 179, replace_events=True)
feedbackleft_b5_p8 = mne.event.shift_time_events(feedbackleft_b5_p8, 179, -1.000, 500)
feedbackleft_b5_p8 = np.delete(feedbackleft_b5_p8, [0, 2, 7, 9, 11, 13, 14, 16, 18, 20, 22, 23, 26, 27, 29, 30, 31, 32, 34, 36, 40, 44, 45, 47], axis=0)


#feedback (180)
feedbackright_b5_p8 = mne.pick_events(feedbackright_b5_p8, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p8 = mne.merge_events(feedbackright_b5_p8, [25,26], 180, replace_events=True)
feedbackright_b5_p8 = mne.event.shift_time_events(feedbackright_b5_p8, 180, -1.000, 500)
feedbackright_b5_p8 = np.delete(feedbackright_b5_p8, [1, 3, 4, 5, 6, 8, 10, 12, 15, 17, 19, 21, 24, 25, 28, 33, 35, 37, 38, 39, 41, 42, 43, 46], axis=0)

#left response (181) 
leftresponse_b5_p8 = mne.pick_events(leftresponse_b5_p8, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p8 = mne.merge_events(leftresponse_b5_p8, [14, 15, 16, 17], 181, replace_events=True)

#right response(182)
rightresponse_b5_p8 = mne.pick_events(rightresponse_b5_p8, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p8 = mne.merge_events(rightresponse_b5_p8, [18, 19, 20, 21], 182, replace_events=True)

#last response left(183)
lastresponseleft_b5_p8 = mne.pick_events(lastresponseleft_b5_p8, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p8 = mne.merge_events(lastresponseleft_b5_p8, [5, 6, 7, 8], 183, replace_events=True)

#last response right(184)
lastresponseright_b5_p8 = mne.pick_events(lastresponseright_b5_p8, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p8 = mne.merge_events(lastresponseright_b5_p8, [9, 10, 11, 12], 184, replace_events=True)


#preperation left (185) 
preparationleft_b5_p8 = mne.pick_events(preparationleft_b5_p8, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p8 = mne.merge_events(preparationleft_b5_p8, [5, 6, 7, 8], 185, replace_events=True)
preparationleft_b5_p8 = mne.event.shift_time_events(preparationleft_b5_p8, 185, 1.500, 500)
preparationleft_b5_p8 = np.delete(preparationleft_b5_p8, [41, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#preperation right (186) 
preparationright_b5_p8 = mne.pick_events(preparationright_b5_p8, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p8 = mne.merge_events(preparationright_b5_p8, [9, 10, 11, 12], 186, replace_events=True)
preparationright_b5_p8 = mne.event.shift_time_events(preparationright_b5_p8, 186, 1.500, 500)
preparationright_b5_p8 = np.delete(preparationright_b5_p8, [47, 107, 125, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)

#nogo (187)
nogob5p8 = mne.pick_events(nogob5p8, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p8 = mne.merge_events(nogob5p8, [24], 187, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p8 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p8 = {'laststimleft':166,'laststimright':167, 'feedbackL':168, 'feedbackR':169, 'leftres':170, 'rightres':171, 'lastresleft':172, 'lastresright':173, 'prepleft':174, 'prepright':175, 'nogo': 176,}
event_dictB5_p8 = {'laststimleft':177,'laststimright':178, 'feedbackL':179, 'feedbackR':180, 'leftres':181, 'rightres':182, 'lastresleft':183, 'lastresright':184, 'prepleft':185, 'prepright':186, 'nogo': 187,}



#merging events togeher into one event list

finalB1_p8 = np.concatenate((laststimpositionleft_b1_p8, laststimpositionright_b1_p8, feedbackleft_b1_p8, feedbackright_b1_p8, leftresponse_b1_p8, rightresponse_b1_p8, lastresponseleft_b1_p8, lastresponseright_b1_p8, preparationleft_b1_p8, preparationright_b1_p8, events_p8, nogob1p8), axis=0)
finalB5_p8 = np.concatenate((laststimpositionleft_b5_p8, laststimpositionright_b5_p8, feedbackleft_b5_p8, feedbackright_b5_p8, leftresponse_b5_p8, rightresponse_b5_p8, lastresponseleft_b5_p8, lastresponseright_b5_p8, preparationleft_b5_p8, preparationright_b5_p8, events_p8, nogob5p8), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p8, event_id=event_dictB1_p8, 

                         sfreq=raw_p8.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p8, event_id=event_dictB5_p8, 

                         sfreq=raw5_p8.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend


#-----------------------------------------------------------------------------#
#                               PARTICIPANT 9
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_9_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part9_SeqL_ERD_6_B1.fif'
Part_9_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part9_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p9 = mne.io.read_raw_fif(Part_9_1, preload = True) 
 

#--------B5--------#
raw5_p9 = mne.io.read_raw_fif(Part_9_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p9, _ = mne.events_from_annotations(raw_p9, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p9, _ = mne.events_from_annotations(raw5_p9, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p9 = np.copy(events_p9)
laststimpositionright_b1_p9 = np.copy(events_p9)
feedbackleft_b1_p9 = np.copy(events_p9)
feedbackright_b1_p9 = np.copy(events_p9)
leftresponse_b1_p9 = np.copy(events_p9)
rightresponse_b1_p9 = np.copy(events_p9)
lastresponseleft_b1_p9 = np.copy(events_p9)
lastresponseright_b1_p9 = np.copy(events_p9)
preparationleft_b1_p9 = np.copy(events_p9)
preparationright_b1_p9 = np.copy(events_p9)
nogob1p9 = np.copy(events_p9)
#--------B5-------#
laststimpositionleft_b5_p9 = np.copy(eventsB5_p9)
laststimpositionright_b5_p9 = np.copy(eventsB5_p9)
feedbackleft_b5_p9 = np.copy(eventsB5_p9)
feedbackright_b5_p9 = np.copy(eventsB5_p9)
leftresponse_b5_p9 = np.copy(eventsB5_p9)
rightresponse_b5_p9 = np.copy(eventsB5_p9)
lastresponseleft_b5_p9 = np.copy(eventsB5_p9)
lastresponseright_b5_p9 = np.copy(eventsB5_p9)
preparationleft_b5_p9 = np.copy(eventsB5_p9)
preparationright_b5_p9 = np.copy(eventsB5_p9)
nogob5p9 = np.copy(eventsB5_p9)


#print to see if event times are correct

print(events_p9)

#--------B1-------#


#last  stimulus position left (188)
laststimpositionleft_b1_p9 = mne.pick_events(laststimpositionleft_b1_p9, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p9 = mne.merge_events(laststimpositionleft_b1_p9, [5, 6, 7, 8], 188, replace_events=True)
laststimpositionleft_b1_p9 = np.delete(laststimpositionleft_b1_p9, [137, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#last  stimulus position right (189)
laststimpositionright_b1_p9 = mne.pick_events(laststimpositionright_b1_p9, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p9 = mne.merge_events(laststimpositionright_b1_p9, [9, 10, 11, 12], 189, replace_events=True)
laststimpositionright_b1_p9 = np.delete(laststimpositionright_b1_p9, [17, 35, 155, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)


#feedback left (190)
feedbackleft_b1_p9 = mne.pick_events(feedbackleft_b1_p9, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p9 = mne.merge_events(feedbackleft_b1_p9, [25,26], 190, replace_events=True)
feedbackleft_b1_p9 = mne.event.shift_time_events(feedbackleft_b1_p9, 190, -1.000, 500)
feedbackleft_b1_p9 = np.delete(feedbackleft_b1_p9, [0, 2, 5, 6, 9, 10, 12, 13, 14, 16, 17, 22, 24, 25, 26, 27, 28, 29, 31, 32, 34, 35, 39, 45], axis=0)

#feedback right (191)
feedbackright_b1_p9 = mne.pick_events(feedbackright_b1_p9, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p9 = mne.merge_events(feedbackright_b1_p9, [25,26], 191, replace_events=True)
feedbackright_b1_p9 = mne.event.shift_time_events(feedbackright_b1_p9, 191, -1.000, 500)
feedbackright_b1_p9 = np.delete(feedbackright_b1_p9, [1, 3, 4, 7, 8, 11, 15, 18, 19, 20, 21, 23, 30, 33, 36, 37, 38, 40, 41, 42, 43, 44, 46, 47], axis=0)

#left response (192) 
leftresponse_b1_p9 = mne.pick_events(leftresponse_b1_p9, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p9 = mne.merge_events(leftresponse_b1_p9, [14, 15, 16, 17], 192, replace_events=True)

#right response(193)
rightresponse_b1_p9 = mne.pick_events(rightresponse_b1_p9, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p9 = mne.merge_events(rightresponse_b1_p9, [18, 19, 20, 21], 193, replace_events=True)

#last response left(194)
lastresponseleft_b1_p9 = mne.pick_events(lastresponseleft_b1_p9, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p9 = mne.merge_events(lastresponseleft_b1_p9, [5, 6, 7, 8], 194, replace_events=True)

#last response right(195)
lastresponseright_b1_p9 = mne.pick_events(lastresponseright_b1_p9, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p9 = mne.merge_events(lastresponseright_b1_p9, [9, 10, 11, 12], 195, replace_events=True)


#preperation left (196) 
preparationleft_b1_p9 = mne.pick_events(preparationleft_b1_p9, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p9 = mne.merge_events(preparationleft_b1_p9, [5, 6, 7, 8], 196, replace_events=True)
preparationleft_b1_p9 = mne.event.shift_time_events(preparationleft_b1_p9, 196, 1.500, 500)
preparationleft_b1_p9 = np.delete(preparationleft_b1_p9, [137, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#preperation right (197) 
preparationright_b1_p9 = mne.pick_events(preparationright_b1_p9, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p9 = mne.merge_events(preparationright_b1_p9, [9, 10, 11, 12], 197, replace_events=True)
preparationright_b1_p9 = mne.event.shift_time_events(preparationright_b1_p9, 197, 1.500, 500)
preparationright_b1_p9 = np.delete(preparationright_b1_p9, [17, 35, 155, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)

#nogo (198)
nogob1p9 = mne.pick_events(nogob1p9, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p9 = mne.merge_events(nogob1p9, [24], 198, replace_events=True)
#--------B5-------#


#last  stimulus position left (199)
laststimpositionleft_b5_p9 = mne.pick_events(laststimpositionleft_b5_p9, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p9 = mne.merge_events(laststimpositionleft_b5_p9, [5, 6, 7, 8], 199, replace_events=True)
laststimpositionleft_b5_p9 = np.delete(laststimpositionleft_b5_p9, [89, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#last  stimulus position right (200)
laststimpositionright_b5_p9 = mne.pick_events(laststimpositionright_b5_p9, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p9 = mne.merge_events(laststimpositionright_b5_p9, [9, 10, 11, 12], 200, replace_events=True)
laststimpositionright_b5_p9 = np.delete(laststimpositionright_b5_p9, [29, 47, 101, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)


#feedback (201)
feedbackleft_b5_p9 = mne.pick_events(feedbackleft_b5_p9, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p9 = mne.merge_events(feedbackleft_b5_p9, [25,26], 201, replace_events=True)
feedbackleft_b5_p9 = mne.event.shift_time_events(feedbackleft_b5_p9, 201, -1.000, 500)
feedbackleft_b5_p9 = np.delete(feedbackleft_b5_p9, [0, 1, 4, 5, 6, 7, 13, 14, 20, 21, 22, 23, 24, 25, 34, 36, 38, 39, 41, 43, 44, 45, 46, 47], axis=0)


#feedback (202)
feedbackright_b5_p9 = mne.pick_events(feedbackright_b5_p9, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p9 = mne.merge_events(feedbackright_b5_p9, [25,26], 202, replace_events=True)
feedbackright_b5_p9 = mne.event.shift_time_events(feedbackright_b5_p9, 202, -1.000, 500)
feedbackright_b5_p9 = np.delete(feedbackright_b5_p9, [2, 3, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 26, 27, 28, 29, 30, 31, 32, 33, 35, 37, 40, 42], axis=0)

#left response (203) 
leftresponse_b5_p9 = mne.pick_events(leftresponse_b5_p9, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p9 = mne.merge_events(leftresponse_b5_p9, [14, 15, 16, 17], 203, replace_events=True)

#right response(204)
rightresponse_b5_p9 = mne.pick_events(rightresponse_b5_p9, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p9 = mne.merge_events(rightresponse_b5_p9, [18, 19, 20, 21], 204, replace_events=True)

#last response left(205)
lastresponseleft_b5_p9 = mne.pick_events(lastresponseleft_b5_p9, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p9 = mne.merge_events(lastresponseleft_b5_p9, [5, 6, 7, 8], 205, replace_events=True)

#last response right(206)
lastresponseright_b5_p9 = mne.pick_events(lastresponseright_b5_p9, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p9 = mne.merge_events(lastresponseright_b5_p9, [9, 10, 11, 12], 206, replace_events=True)


#preperation left (207) 
preparationleft_b5_p9 = mne.pick_events(preparationleft_b5_p9, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p9 = mne.merge_events(preparationleft_b5_p9, [5, 6, 7, 8], 207, replace_events=True)
preparationleft_b5_p9 = mne.event.shift_time_events(preparationleft_b5_p9, 207, 1.500, 500)
preparationleft_b5_p9 = np.delete(preparationleft_b5_p9, [89, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#preperation right (208) 
preparationright_b5_p9 = mne.pick_events(preparationright_b5_p9, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p9 = mne.merge_events(preparationright_b5_p9, [9, 10, 11, 12], 208, replace_events=True)
preparationright_b5_p9 = mne.event.shift_time_events(preparationright_b5_p9, 208, 1.500, 500)
preparationright_b5_p9 = np.delete(preparationright_b5_p9, [29, 47, 101, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)

#nogo (209)
nogob5p9 = mne.pick_events(nogob5p9, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p9 = mne.merge_events(nogob5p9, [24], 209, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p9 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p9 = {'laststimleft':188,'laststimright':189, 'feedbackL':190, 'feedbackR':191, 'leftres':192, 'rightres':193, 'lastresleft':194, 'lastresright':195, 'prepleft':196, 'prepright':197, 'nogo': 198}
event_dictB5_p9 = {'laststimleft':199,'laststimright':200, 'feedbackL':201, 'feedbackR':202, 'leftres':203, 'rightres':204, 'lastresleft':205, 'lastresright':206, 'prepleft':207, 'prepright':208, 'nogo': 209}

finalB1_p9 = np.concatenate((laststimpositionleft_b1_p9, laststimpositionright_b1_p9, feedbackleft_b1_p9, feedbackright_b1_p9, leftresponse_b1_p9, rightresponse_b1_p9, lastresponseleft_b1_p9, lastresponseright_b1_p9, preparationleft_b1_p9, preparationright_b1_p9, events_p9, nogob1p9), axis=0)
finalB5_p9 = np.concatenate((laststimpositionleft_b5_p9, laststimpositionright_b5_p9, feedbackleft_b5_p9, feedbackright_b5_p9, leftresponse_b5_p9, rightresponse_b5_p9, lastresponseleft_b5_p9, lastresponseright_b5_p9, preparationleft_b5_p9, preparationright_b5_p9, events_p9, nogob5p9), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p9, event_id=event_dictB1_p9, 

                         sfreq=raw_p9.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p9, event_id=event_dictB5_p9, 

                         sfreq=raw5_p9.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend


#-----------------------------------------------------------------------------#
#                               PARTICIPANT 10
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_10_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part10_SeqL_ERD_6_B1.fif'
Part_10_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part10_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p10 = mne.io.read_raw_fif(Part_10_1, preload = True) 
 

#--------B5--------#
raw5_p10 = mne.io.read_raw_fif(Part_10_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p10, _ = mne.events_from_annotations(raw_p10, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p10, _ = mne.events_from_annotations(raw5_p10, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p10 = np.copy(events_p10)
laststimpositionright_b1_p10 = np.copy(events_p10)
feedbackleft_b1_p10 = np.copy(events_p10)
feedbackright_b1_p10 = np.copy(events_p10)
leftresponse_b1_p10 = np.copy(events_p10)
rightresponse_b1_p10 = np.copy(events_p10)
lastresponseleft_b1_p10 = np.copy(events_p10)
lastresponseright_b1_p10 = np.copy(events_p10)
preparationleft_b1_p10 = np.copy(events_p10)
preparationright_b1_p10 = np.copy(events_p10)
nogob1p10 = np.copy(events_p10)
#--------B5-------#
laststimpositionleft_b5_p10 = np.copy(eventsB5_p10)
laststimpositionright_b5_p10 = np.copy(eventsB5_p10)
feedbackleft_b5_p10 = np.copy(eventsB5_p10)
feedbackright_b5_p10 = np.copy(eventsB5_p10)
leftresponse_b5_p10 = np.copy(eventsB5_p10)
rightresponse_b5_p10 = np.copy(eventsB5_p10)
lastresponseleft_b5_p10 = np.copy(eventsB5_p10)
lastresponseright_b5_p10 = np.copy(eventsB5_p10)
preparationleft_b5_p10 = np.copy(eventsB5_p10)
preparationright_b5_p10 = np.copy(eventsB5_p10)
nogob5p10 = np.copy(eventsB5_p10)


#print to see if event times are correct

print(events_p10)

#--------B1-------#


#last  stimulus position left (210)
laststimpositionleft_b1_p10 = mne.pick_events(laststimpositionleft_b1_p10, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p10 = mne.merge_events(laststimpositionleft_b1_p10, [5, 6, 7, 8], 210, replace_events=True)
laststimpositionleft_b1_p10 = np.delete(laststimpositionleft_b1_p10, [119, 149, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#last  stimulus position right (211)
laststimpositionright_b1_p10 = mne.pick_events(laststimpositionright_b1_p10, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p10 = mne.merge_events(laststimpositionright_b1_p10, [9, 10, 11, 12], 211, replace_events=True)
laststimpositionright_b1_p10 = np.delete(laststimpositionright_b1_p10, [48, 85, 42, 79, 128, 153, 0,	1,	2,	3,	4, 6,	7,	8,	9,	10, 12,	13,	14,	15,	16, 18,	19,	20,	21,	22, 24,	25,	26,	27,	28, 30,	31,	32,	33,	34, 36,	37,	38,	39,	40, 43,	44,	45,	46,	47, 49,	50,	51,	52,	53,  55,	56,	57,	58,	59, 61,	62,	63,	64,	65, 67,	68,	69,	70,	71, 73,	74,	75,	76,	77, 80,	81,	82,	83,	84, 86,	87,	88,	89,	90, 92, 93,	94,	95,	96, 98,	99,	100,	101,	102, 104,	105,	106,	107,	108, 110,	111,	112,	113,	114, 116,	117,	118,	119,	120, 122,	123,	124,	125,	126, 129,	130,	131,	132,	133, 135,	136,	137,	138,	139, 141,	142,	143,	144,	145, 147,	148,	149,	150,	151, 154,	155,	156,	157,	158], axis=0)


#feedback left (212)
feedbackleft_b1_p10 = mne.pick_events(feedbackleft_b1_p10, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p10 = mne.merge_events(feedbackleft_b1_p10, [25,26], 212, replace_events=True)
feedbackleft_b1_p10 = mne.event.shift_time_events(feedbackleft_b1_p10, 212, -1.000, 500)
feedbackleft_b1_p10 = np.delete(feedbackleft_b1_p10, [1, 4, 5, 7, 8, 12, 16, 17, 20, 21, 22, 23, 24, 26, 28, 29, 32, 35, 36, 39, 40, 43, 45, 46], axis=0)

#feedback right (213)
feedbackright_b1_p10 = mne.pick_events(feedbackright_b1_p10, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p10 = mne.merge_events(feedbackright_b1_p10, [25,26], 213, replace_events=True)
feedbackright_b1_p10 = mne.event.shift_time_events(feedbackright_b1_p10, 213, -1.000, 500)
feedbackright_b1_p10 = np.delete(feedbackright_b1_p10, [0, 2, 3, 6, 9, 10, 11, 13, 14, 15, 18, 19, 25, 27, 30, 31, 33, 34, 37, 38, 41, 42, 44, 47], axis=0)

#left response (214) 
leftresponse_b1_p10 = mne.pick_events(leftresponse_b1_p10, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p10 = mne.merge_events(leftresponse_b1_p10, [14, 15, 16, 17], 214, replace_events=True)

#right response(215)
rightresponse_b1_p10 = mne.pick_events(rightresponse_b1_p10, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p10 = mne.merge_events(rightresponse_b1_p10, [18, 19, 20, 21], 215, replace_events=True)

#last response left(216)
lastresponseleft_b1_p10 = mne.pick_events(lastresponseleft_b1_p10, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p10 = mne.merge_events(lastresponseleft_b1_p10, [5, 6, 7, 8], 216, replace_events=True)

#last response right(217)
lastresponseright_b1_p10 = mne.pick_events(lastresponseright_b1_p10, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p10 = mne.merge_events(lastresponseright_b1_p10, [9, 10, 11, 12], 217, replace_events=True)


#preperation left (218) 
preparationleft_b1_p10 = mne.pick_events(preparationleft_b1_p10, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p10 = mne.merge_events(preparationleft_b1_p10, [5, 6, 7, 8], 218, replace_events=True)
preparationleft_b1_p10 = mne.event.shift_time_events(preparationleft_b1_p10, 218, 1.500, 500)
preparationleft_b1_p10 = np.delete(preparationleft_b1_p10, [119, 149, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#preperation right (219) 
preparationright_b1_p10 = mne.pick_events(preparationright_b1_p10, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p10 = mne.merge_events(preparationright_b1_p10, [9, 10, 11, 12], 219, replace_events=True)
preparationright_b1_p10 = mne.event.shift_time_events(preparationright_b1_p10, 219, 1.500, 500)
preparationright_b1_p10 = np.delete(preparationright_b1_p10, [48, 85, 42, 79, 128, 153, 0,	1,	2,	3,	4, 6,	7,	8,	9,	10, 12,	13,	14,	15,	16, 18,	19,	20,	21,	22, 24,	25,	26,	27,	28, 30,	31,	32,	33,	34, 36,	37,	38,	39,	40, 43,	44,	45,	46,	47, 49,	50,	51,	52,	53,  55,	56,	57,	58,	59, 61,	62,	63,	64,	65, 67,	68,	69,	70,	71, 73,	74,	75,	76,	77, 80,	81,	82,	83,	84, 86,	87,	88,	89,	90, 92, 93,	94,	95,	96, 98,	99,	100,	101,	102, 104,	105,	106,	107,	108, 110,	111,	112,	113,	114, 116,	117,	118,	119,	120, 122,	123,	124,	125,	126, 129,	130,	131,	132,	133, 135,	136,	137,	138,	139, 141,	142,	143,	144,	145, 147,	148,	149,	150,	151, 154,	155,	156,	157,	158], axis=0)

#nogo (220)
nogob1p10 = mne.pick_events(nogob1p10, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p10 = mne.merge_events(nogob1p10, [24], 220, replace_events=True)


#--------B5-------#
#last  stimulus position left (221)
laststimpositionleft_b5_p10 = mne.pick_events(laststimpositionleft_b5_p10, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p10 = mne.merge_events(laststimpositionleft_b5_p10, [5, 6, 7, 8], 221, replace_events=True)
laststimpositionleft_b5_p10 = np.delete(laststimpositionleft_b5_p10, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142], axis=0)

#last  stimulus position right (222)
laststimpositionright_b5_p10 = mne.pick_events(laststimpositionright_b5_p10, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p10 = mne.merge_events(laststimpositionright_b5_p10, [9, 10, 11, 12], 222, replace_events=True)
laststimpositionright_b5_p10 = np.delete(laststimpositionright_b5_p10, [36, 55, 128, 147, 30, 49, 122, 141, 0,	1,	2,	3,	4, 6,	7,	8,	9,	10, 12,	13,	14,	15,	16, 18,	19,	20,	21,	22, 24,	25,	26,	27,	28, 31,	32,	33,	34,	35, 37,	38,	39,	40,	41, 43,	44,	45,	46,	47, 50,	51,	52,	53,	54, 56,	57,	58,	59,	60, 62,	63,	64,	65,	66, 68,	69,	70,	71,	72, 74,	75,	76,	77,	78, 80,	81,	82,	83,	84, 86,	87,	88,	89,	90, 92,	93,	94,	95,	96, 98,	99,	100,	101,	102, 104,	105,	106,	107,	108, 110,	111,	112,	113,	114, 116,	117,	118,	119,	120, 123,	124,	125,	126,	127, 129,	130,	131,	132,	133, 135,	136,	137,	138,	139, 142,	143,	144,	145,	146, 148,	149,	150,	151,	152, 154,	155,	156,	157,	158, 160,	161,	162,	163,	164, 166,	167,	168,	169,	170], axis=0)


#feedback (223)
feedbackleft_b5_p10 = mne.pick_events(feedbackleft_b5_p10, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p10 = mne.merge_events(feedbackleft_b5_p10, [25,26], 223, replace_events=True)
feedbackleft_b5_p10 = mne.event.shift_time_events(feedbackleft_b5_p10, 223, -1.000, 500)
feedbackleft_b5_p10 = np.delete(feedbackleft_b5_p10, [0, 1, 4, 5, 7, 9, 10, 12, 13, 14, 15, 19, 25, 26, 28, 30, 31, 36, 37, 38, 42, 44, 45, 47], axis=0)

#feedback (224)
feedbackright_b5_p10 = mne.pick_events(feedbackright_b5_p10, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p10 = mne.merge_events(feedbackright_b5_p10, [25,26], 224, replace_events=True)
feedbackright_b5_p10 = mne.event.shift_time_events(feedbackright_b5_p10, 224, -1.000, 500)
feedbackright_b5_p10 = np.delete(feedbackright_b5_p10, [2, 3, 6, 8, 11, 16, 17, 18, 20, 21, 22, 23, 24, 27, 29, 32, 33, 34, 35, 39, 40, 41, 43, 46], axis=0)


#left response (225) 
leftresponse_b5_p10 = mne.pick_events(leftresponse_b5_p10, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p10 = mne.merge_events(leftresponse_b5_p10, [14, 15, 16, 17], 225, replace_events=True)

#right response(226)
rightresponse_b5_p10 = mne.pick_events(rightresponse_b5_p10, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p10 = mne.merge_events(rightresponse_b5_p10, [18, 19, 20, 21], 226, replace_events=True)

#last response left(227)
lastresponseleft_b5_p10 = mne.pick_events(lastresponseleft_b5_p10, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p10 = mne.merge_events(lastresponseleft_b5_p10, [5, 6, 7, 8], 227, replace_events=True)

#last response right(228)
lastresponseright_b5_p10 = mne.pick_events(lastresponseright_b5_p10, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p10 = mne.merge_events(lastresponseright_b5_p10, [9, 10, 11, 12], 228, replace_events=True)


#preperation left (229) 
preparationleft_b5_p10 = mne.pick_events(preparationleft_b5_p10, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p10 = mne.merge_events(preparationleft_b5_p10, [5, 6, 7, 8], 229, replace_events=True)
preparationleft_b5_p10 = mne.event.shift_time_events(preparationleft_b5_p10, 229, 1.500, 500)
preparationleft_b5_p10 = np.delete(preparationleft_b5_p10, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142], axis=0)

#preperation right (230) 
preparationright_b5_p10 = mne.pick_events(preparationright_b5_p10, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p10 = mne.merge_events(preparationright_b5_p10, [9, 10, 11, 12], 230, replace_events=True)
preparationright_b5_p10 = mne.event.shift_time_events(preparationright_b5_p10, 230, 1.500, 500)
preparationright_b5_p10 = np.delete(preparationright_b5_p10, [36, 55, 128, 147, 30, 49, 122, 141, 0,	1,	2,	3,	4, 6,	7,	8,	9,	10, 12,	13,	14,	15,	16, 18,	19,	20,	21,	22, 24,	25,	26,	27,	28, 31,	32,	33,	34,	35, 37,	38,	39,	40,	41, 43,	44,	45,	46,	47, 50,	51,	52,	53,	54, 56,	57,	58,	59,	60, 62,	63,	64,	65,	66, 68,	69,	70,	71,	72, 74,	75,	76,	77,	78, 80,	81,	82,	83,	84, 86,	87,	88,	89,	90, 92,	93,	94,	95,	96, 98,	99,	100,	101,	102, 104,	105,	106,	107,	108, 110,	111,	112,	113,	114, 116,	117,	118,	119,	120, 123,	124,	125,	126,	127, 129,	130,	131,	132,	133, 135,	136,	137,	138,	139, 142,	143,	144,	145,	146, 148,	149,	150,	151,	152, 154,	155,	156,	157,	158, 160,	161,	162,	163,	164, 166,	167,	168,	169,	170], axis=0)

#nogo (231)
nogob5p10 = mne.pick_events(nogob5p10, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p10 = mne.merge_events(nogob5p10, [24], 231, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p10 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p10 = {'laststimleft':210,'laststimright':211, 'feedbackL':212, 'feedbackR':213, 'leftres':214, 'rightres':215, 'lastresleft':216, 'lastresright':217, 'prepleft':218, 'prepright':219, 'nogo': 220,}
event_dictB5_p10 = {'laststimleft':221,'laststimright':222, 'feedbackL':223, 'feedbackR':224, 'leftres':225, 'rightres':226, 'lastresleft':227, 'lastresright':228, 'prepleft':229, 'prepright':230, 'nogo': 231,}



#merging events togeher into one event list

finalB1_p10 = np.concatenate((laststimpositionleft_b1_p10, laststimpositionright_b1_p10, feedbackleft_b1_p10, feedbackright_b1_p10, leftresponse_b1_p10, rightresponse_b1_p10, lastresponseleft_b1_p10, lastresponseright_b1_p10, preparationleft_b1_p10, preparationright_b1_p10, events_p10, nogob1p10), axis=0)
finalB5_p10 = np.concatenate((laststimpositionleft_b5_p10, laststimpositionright_b5_p10, feedbackleft_b5_p10, feedbackright_b5_p10, leftresponse_b5_p10, rightresponse_b5_p10, lastresponseleft_b5_p10, lastresponseright_b5_p10, preparationleft_b5_p10, preparationright_b5_p10, events_p10, nogob5p10), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p10, event_id=event_dictB1_p10, 

                         sfreq=raw_p10.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p10, event_id=event_dictB5_p10, 

                         sfreq=raw5_p10.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend


#merging events togeher into one event list

#-----------------------------------------------------------------------------#
#                               PARTICIPANT 11
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_11_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part11_SeqL_ERD_6_B1.fif'
Part_11_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part11_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p11 = mne.io.read_raw_fif(Part_11_1, preload = True) 
 

#--------B5--------#
raw5_p11 = mne.io.read_raw_fif(Part_11_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p11, _ = mne.events_from_annotations(raw_p11, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p11, _ = mne.events_from_annotations(raw5_p11, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p11 = np.copy(events_p11)
laststimpositionright_b1_p11 = np.copy(events_p11)
feedbackleft_b1_p11 = np.copy(events_p11)
feedbackright_b1_p11 = np.copy(events_p11)
leftresponse_b1_p11 = np.copy(events_p11)
rightresponse_b1_p11 = np.copy(events_p11)
lastresponseleft_b1_p11 = np.copy(events_p11)
lastresponseright_b1_p11 = np.copy(events_p11)
preparationleft_b1_p11 = np.copy(events_p11)
preparationright_b1_p11 = np.copy(events_p11)
nogob1p11 = np.copy(events_p11)
#--------B5-------#
laststimpositionleft_b5_p11 = np.copy(eventsB5_p11)
laststimpositionright_b5_p11 = np.copy(eventsB5_p11)
feedbackleft_b5_p11 = np.copy(eventsB5_p11)
feedbackright_b5_p11 = np.copy(eventsB5_p11)
leftresponse_b5_p11 = np.copy(eventsB5_p11)
rightresponse_b5_p11 = np.copy(eventsB5_p11)
lastresponseleft_b5_p11 = np.copy(eventsB5_p11)
lastresponseright_b5_p11 = np.copy(eventsB5_p11)
preparationleft_b5_p11 = np.copy(eventsB5_p11)
preparationright_b5_p11 = np.copy(eventsB5_p11)
nogob5p11 = np.copy(eventsB5_p11)


#print to see if event times are correct

print(events_p11)

#--------B1-------#


#last  stimulus position left (232)
laststimpositionleft_b1_p11 = mne.pick_events(laststimpositionleft_b1_p11, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p11 = mne.merge_events(laststimpositionleft_b1_p11, [5, 6, 7, 8], 232, replace_events=True)
laststimpositionleft_b1_p11 = np.delete(laststimpositionleft_b1_p11, [65, 77, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#last  stimulus position right (233)
laststimpositionright_b1_p11 = mne.pick_events(laststimpositionright_b1_p11, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p11 = mne.merge_events(laststimpositionright_b1_p11, [9, 10, 11, 12], 233, replace_events=True)
laststimpositionright_b1_p11 = np.delete(laststimpositionright_b1_p11, [125, 149, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#feedback left (234)
feedbackleft_b1_p11 = mne.pick_events(feedbackleft_b1_p11, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p11 = mne.merge_events(feedbackleft_b1_p11, [25,26], 234, replace_events=True)
feedbackleft_b1_p11 = mne.event.shift_time_events(feedbackleft_b1_p11, 234, -1.000, 500)
feedbackleft_b1_p11 = np.delete(feedbackleft_b1_p11, [0, 1, 9, 13, 14, 15, 16, 17, 18, 20, 22, 23, 24, 26, 27, 29, 30, 33, 34, 38, 39, 40, 41, 44], axis=0)


#feedback right (235)
feedbackright_b1_p11 = mne.pick_events(feedbackright_b1_p11, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p11 = mne.merge_events(feedbackright_b1_p11, [25,26], 235, replace_events=True)
feedbackright_b1_p11 = mne.event.shift_time_events(feedbackright_b1_p11, 235, -1.000, 500)
feedbackright_b1_p11 = np.delete(feedbackright_b1_p11, [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 19, 21, 25, 28, 31, 32, 35, 36, 37, 42, 43, 45, 46, 47], axis=0)


#left response (236) 
leftresponse_b1_p11 = mne.pick_events(leftresponse_b1_p11, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p11 = mne.merge_events(leftresponse_b1_p11, [14, 15, 16, 17], 236, replace_events=True)

#right response(237)
rightresponse_b1_p11 = mne.pick_events(rightresponse_b1_p11, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p11 = mne.merge_events(rightresponse_b1_p11, [18, 19, 20, 21], 237, replace_events=True)

#last response left(238)
lastresponseleft_b1_p11 = mne.pick_events(lastresponseleft_b1_p11, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p11 = mne.merge_events(lastresponseleft_b1_p11, [5, 6, 7, 8], 238, replace_events=True)

#last response right(239)
lastresponseright_b1_p11 = mne.pick_events(lastresponseright_b1_p11, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p11 = mne.merge_events(lastresponseright_b1_p11, [9, 10, 11, 12], 239, replace_events=True)


#preperation left (240) 
preparationleft_b1_p11 = mne.pick_events(preparationleft_b1_p11, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p11 = mne.merge_events(preparationleft_b1_p11, [5, 6, 7, 8], 240, replace_events=True)
preparationleft_b1_p11 = mne.event.shift_time_events(preparationleft_b1_p11, 240, 1.500, 500)
preparationleft_b1_p11 = np.delete(preparationleft_b1_p11, [65, 77, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#preperation right (241) 
preparationright_b1_p11 = mne.pick_events(preparationright_b1_p11, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p11 = mne.merge_events(preparationright_b1_p11, [9, 10, 11, 12], 241, replace_events=True)
preparationright_b1_p11 = mne.event.shift_time_events(preparationright_b1_p11, 241, 1.500, 500)
preparationright_b1_p11 = np.delete(preparationright_b1_p11, [125, 149, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#nogo (242)
nogob1p11 = mne.pick_events(nogob1p11, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p11 = mne.merge_events(nogob1p11, [24], 242, replace_events=True)
#--------B5-------#


#last  stimulus position left (243)
laststimpositionleft_b5_p11 = mne.pick_events(laststimpositionleft_b5_p11, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p11 = mne.merge_events(laststimpositionleft_b5_p11, [5, 6, 7, 8], 243, replace_events=True)
laststimpositionleft_b5_p11 = np.delete(laststimpositionleft_b5_p11, [119, 137, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#last  stimulus position right (244)
laststimpositionright_b5_p11 = mne.pick_events(laststimpositionright_b5_p11, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p11 = mne.merge_events(laststimpositionright_b5_p11, [9, 10, 11, 12], 244, replace_events=True)
laststimpositionright_b5_p11 = np.delete(laststimpositionright_b5_p11, [17, 41, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#feedback (245)
feedbackleft_b5_p11 = mne.pick_events(feedbackleft_b5_p11, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p11 = mne.merge_events(feedbackleft_b5_p11, [25,26], 245, replace_events=True)
feedbackleft_b5_p11 = mne.event.shift_time_events(feedbackleft_b5_p11, 245, -1.000, 500)
feedbackleft_b5_p11 = np.delete(feedbackleft_b5_p11, [0, 4, 9, 10, 11, 18, 20, 21, 22, 23, 27, 29, 30, 35, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47], axis=0)


#feedback (246)
feedbackright_b5_p11 = mne.pick_events(feedbackright_b5_p11, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p11 = mne.merge_events(feedbackright_b5_p11, [25,26], 246, replace_events=True)
feedbackright_b5_p11 = mne.event.shift_time_events(feedbackright_b5_p11, 246, -1.000, 500)
feedbackright_b5_p11 = np.delete(feedbackright_b5_p11, [1, 2, 3, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 19, 24, 25, 26, 28, 31, 32, 33, 34, 36, 42], axis=0)

#left response (247) 
leftresponse_b5_p11 = mne.pick_events(leftresponse_b5_p11, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p11 = mne.merge_events(leftresponse_b5_p11, [14, 15, 16, 17], 247, replace_events=True)

#right response(248)
rightresponse_b5_p11 = mne.pick_events(rightresponse_b5_p11, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p11 = mne.merge_events(rightresponse_b5_p11, [18, 19, 20, 21], 248, replace_events=True)

#last response left(249)
lastresponseleft_b5_p11 = mne.pick_events(lastresponseleft_b5_p11, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p11 = mne.merge_events(lastresponseleft_b5_p11, [5, 6, 7, 8], 249, replace_events=True)

#last response right(250)
lastresponseright_b5_p11 = mne.pick_events(lastresponseright_b5_p11, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p11 = mne.merge_events(lastresponseright_b5_p11, [9, 10, 11, 12], 250, replace_events=True)


#preperation left (251) 
preparationleft_b5_p11 = mne.pick_events(preparationleft_b5_p11, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p11 = mne.merge_events(preparationleft_b5_p11, [5, 6, 7, 8], 251, replace_events=True)
preparationleft_b5_p11 = mne.event.shift_time_events(preparationleft_b5_p11, 251, 1.500, 500)
preparationleft_b5_p11 = np.delete(preparationleft_b5_p11, [119, 137,0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#preperation right (252) 
preparationright_b5_p11 = mne.pick_events(preparationright_b5_p11, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p11 = mne.merge_events(preparationright_b5_p11, [9, 10, 11, 12], 252, replace_events=True)
preparationright_b5_p11 = mne.event.shift_time_events(preparationright_b5_p11, 252, 1.500, 500)
preparationright_b5_p11 = np.delete(preparationright_b5_p11, [17, 41, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#nogo (253)
nogob5p11 = mne.pick_events(nogob5p11, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p11 = mne.merge_events(nogob5p11, [24], 253, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p11 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p11 = {'laststimleft':232,'laststimright':233, 'feedbackL':234, 'feedbackR':235, 'leftres':236, 'rightres':237, 'lastresleft':238, 'lastresright':239, 'prepleft':240, 'prepright':241, 'nogo': 242}
event_dictB5_p11 = {'laststimleft':243,'laststimright':244, 'feedbackL':245, 'feedbackR':246, 'leftres':247, 'rightres':248, 'lastresleft':249, 'lastresright':250, 'prepleft':251, 'prepright':252, 'nogo': 253}



#merging events togeher into one event list

finalB1_p11 = np.concatenate((laststimpositionleft_b1_p11, laststimpositionright_b1_p11, feedbackleft_b1_p11, feedbackright_b1_p11, leftresponse_b1_p11, rightresponse_b1_p11, lastresponseleft_b1_p11, lastresponseright_b1_p11, preparationleft_b1_p11, preparationright_b1_p11, events_p11, nogob1p11), axis=0)
finalB5_p11 = np.concatenate((laststimpositionleft_b5_p11, laststimpositionright_b5_p11, feedbackleft_b5_p11, feedbackright_b5_p11, leftresponse_b5_p11, rightresponse_b5_p11, lastresponseleft_b5_p11, lastresponseright_b5_p11, preparationleft_b5_p11, preparationright_b5_p11, events_p11, nogob5p11), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p11, event_id=event_dictB1_p11, 

                         sfreq=raw_p11.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p11, event_id=event_dictB5_p11, 

                         sfreq=raw5_p11.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend

#-----------------------------------------------------------------------------#
#                               PARTICIPANT 12
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_12_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part12_SeqL_ERD_6_B1.fif'
Part_12_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part12_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p12 = mne.io.read_raw_fif(Part_12_1, preload = True) 
 

#--------B5--------#
raw5_p12 = mne.io.read_raw_fif(Part_12_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p12, _ = mne.events_from_annotations(raw_p12, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p12, _ = mne.events_from_annotations(raw5_p12, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p12 = np.copy(events_p12)
laststimpositionright_b1_p12 = np.copy(events_p12)
feedbackleft_b1_p12 = np.copy(events_p12)
feedbackright_b1_p12 = np.copy(events_p12)
leftresponse_b1_p12 = np.copy(events_p12)
rightresponse_b1_p12 = np.copy(events_p12)
lastresponseleft_b1_p12 = np.copy(events_p12)
lastresponseright_b1_p12 = np.copy(events_p12)
preparationleft_b1_p12 = np.copy(events_p12)
preparationright_b1_p12 = np.copy(events_p12)
nogob1p12 = np.copy(events_p12)
#--------B5-------#
laststimpositionleft_b5_p12 = np.copy(eventsB5_p12)
laststimpositionright_b5_p12 = np.copy(eventsB5_p12)
feedbackleft_b5_p12 = np.copy(eventsB5_p12)
feedbackright_b5_p12 = np.copy(eventsB5_p12)
leftresponse_b5_p12 = np.copy(eventsB5_p12)
rightresponse_b5_p12 = np.copy(eventsB5_p12)
lastresponseleft_b5_p12 = np.copy(eventsB5_p12)
lastresponseright_b5_p12 = np.copy(eventsB5_p12)
preparationleft_b5_p12 = np.copy(eventsB5_p12)
preparationright_b5_p12 = np.copy(eventsB5_p12)
nogob5p12 = np.copy(eventsB5_p12)


#print to see if event times are correct

print(events_p12)

#--------B1-------#


#last  stimulus position left (254)
laststimpositionleft_b1_p12 = mne.pick_events(laststimpositionleft_b1_p12, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p12 = mne.merge_events(laststimpositionleft_b1_p12, [5, 6, 7, 8], 254, replace_events=True)
laststimpositionleft_b1_p12 = np.delete(laststimpositionleft_b1_p12, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142], axis=0)


#last  stimulus position right (255)
laststimpositionright_b1_p12 = mne.pick_events(laststimpositionright_b1_p12, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p12 = mne.merge_events(laststimpositionright_b1_p12, [9, 10, 11, 12], 255, replace_events=True)
laststimpositionright_b1_p12 = np.delete(laststimpositionright_b1_p12, [11, 77, 125, 137, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166], axis=0)

#feedback left (256)
feedbackleft_b1_p12 = mne.pick_events(feedbackleft_b1_p12, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p12 = mne.merge_events(feedbackleft_b1_p12, [25,26], 256, replace_events=True)
feedbackleft_b1_p12 = mne.event.shift_time_events(feedbackleft_b1_p12, 256, -1.000, 500)
feedbackleft_b1_p12 = np.delete(feedbackleft_b1_p12, [1, 5, 7, 9, 12, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31, 32, 36, 38, 39, 40, 42], axis=0)


#feedback right (257)
feedbackright_b1_p12 = mne.pick_events(feedbackright_b1_p12, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p12 = mne.merge_events(feedbackright_b1_p12, [25,26], 257, replace_events=True)
feedbackright_b1_p12 = mne.event.shift_time_events(feedbackright_b1_p12, 257, -1.000, 500)
feedbackright_b1_p12 = np.delete(feedbackright_b1_p12, [0, 2, 3, 4, 6, 8, 10, 11, 13, 14, 18, 20, 24, 28, 33, 34, 35, 37, 41, 43, 44, 45, 46, 47], axis=0)


#left response (258) 
leftresponse_b1_p12 = mne.pick_events(leftresponse_b1_p12, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p12 = mne.merge_events(leftresponse_b1_p12, [14, 15, 16, 17], 258, replace_events=True)

#right response(259)
rightresponse_b1_p12 = mne.pick_events(rightresponse_b1_p12, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p12 = mne.merge_events(rightresponse_b1_p12, [18, 19, 20, 21], 259, replace_events=True)

#last response left(260)
lastresponseleft_b1_p12 = mne.pick_events(lastresponseleft_b1_p12, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p12 = mne.merge_events(lastresponseleft_b1_p12, [5, 6, 7, 8], 260, replace_events=True)

#last response right(261)
lastresponseright_b1_p12 = mne.pick_events(lastresponseright_b1_p12, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p12 = mne.merge_events(lastresponseright_b1_p12, [9, 10, 11, 12], 261, replace_events=True)


#preperation left (262) 
preparationleft_b1_p12 = mne.pick_events(preparationleft_b1_p12, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p12 = mne.merge_events(preparationleft_b1_p12, [5, 6, 7, 8], 262, replace_events=True)
preparationleft_b1_p12 = mne.event.shift_time_events(preparationleft_b1_p12, 262, 1.500, 500)
preparationleft_b1_p12 = np.delete(preparationleft_b1_p12, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142], axis=0)

#preperation right (263) 
preparationright_b1_p12 = mne.pick_events(preparationright_b1_p12, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p12 = mne.merge_events(preparationright_b1_p12, [9, 10, 11, 12], 263, replace_events=True)
preparationright_b1_p12 = mne.event.shift_time_events(preparationright_b1_p12, 263, 1.500, 500)
preparationright_b1_p12 = np.delete(preparationright_b1_p12, [11, 77, 125, 137, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166], axis=0)

#nogo (264)
nogob1p12 = mne.pick_events(nogob1p12, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p12 = mne.merge_events(nogob1p12, [24], 264, replace_events=True)
#--------B5-------#


#last  stimulus position left (265)
laststimpositionleft_b5_p12 = mne.pick_events(laststimpositionleft_b5_p12, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p12 = mne.merge_events(laststimpositionleft_b5_p12, [5, 6, 7, 8], 265, replace_events=True)
laststimpositionleft_b5_p12 = np.delete(laststimpositionleft_b5_p12, [53, 101, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#last  stimulus position right (266)
laststimpositionright_b5_p12 = mne.pick_events(laststimpositionright_b5_p12, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p12 = mne.merge_events(laststimpositionright_b5_p12, [9, 10, 11, 12], 266, replace_events=True)
laststimpositionright_b5_p12 = np.delete(laststimpositionright_b5_p12, [11, 155, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#feedback (267)
feedbackleft_b5_p12 = mne.pick_events(feedbackleft_b5_p12, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p12 = mne.merge_events(feedbackleft_b5_p12, [25,26], 267, replace_events=True)
feedbackleft_b5_p12 = mne.event.shift_time_events(feedbackleft_b5_p12, 267, -1.000, 500)
feedbackleft_b5_p12 = np.delete(feedbackleft_b5_p12, [4, 6, 9, 11, 13, 15, 17, 18, 20, 21, 22, 23, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 40, 41], axis=0)


#feedback (268)
feedbackright_b5_p12 = mne.pick_events(feedbackright_b5_p12, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p12 = mne.merge_events(feedbackright_b5_p12, [25,26], 268, replace_events=True)
feedbackright_b5_p12 = mne.event.shift_time_events(feedbackright_b5_p12, 268, -1.000, 500)
feedbackright_b5_p12 = np.delete(feedbackright_b5_p12, [0, 1, 2, 3, 5, 7, 8, 10, 12, 14, 16, 19, 24, 25, 31, 37, 38, 39, 42, 43, 44, 45, 46, 47], axis=0)

#left response (269) 
leftresponse_b5_p12 = mne.pick_events(leftresponse_b5_p12, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p12 = mne.merge_events(leftresponse_b5_p12, [14, 15, 16, 17], 269, replace_events=True)

#right response(270)
rightresponse_b5_p12 = mne.pick_events(rightresponse_b5_p12, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p12 = mne.merge_events(rightresponse_b5_p12, [18, 19, 20, 21], 270, replace_events=True)

#last response left(271)
lastresponseleft_b5_p12 = mne.pick_events(lastresponseleft_b5_p12, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p12 = mne.merge_events(lastresponseleft_b5_p12, [5, 6, 7, 8], 271, replace_events=True)

#last response right(272)
lastresponseright_b5_p12 = mne.pick_events(lastresponseright_b5_p12, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p12 = mne.merge_events(lastresponseright_b5_p12, [9, 10, 11, 12], 272, replace_events=True)


#preperation left (273) 
preparationleft_b5_p12 = mne.pick_events(preparationleft_b5_p12, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p12 = mne.merge_events(preparationleft_b5_p12, [5, 6, 7, 8], 273, replace_events=True)
preparationleft_b5_p12 = mne.event.shift_time_events(preparationleft_b5_p12, 273, 1.500, 500)
preparationleft_b5_p12 = np.delete(preparationleft_b5_p12, [53, 101, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#preperation right (274) 
preparationright_b5_p12 = mne.pick_events(preparationright_b5_p12, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p12 = mne.merge_events(preparationright_b5_p12, [9, 10, 11, 12], 274, replace_events=True)
preparationright_b5_p12 = mne.event.shift_time_events(preparationright_b5_p12, 274, 1.500, 500)
preparationright_b5_p12 = np.delete(preparationright_b5_p12, [11, 155, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#nogo (275)
nogob5p12 = mne.pick_events(nogob5p12, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p12 = mne.merge_events(nogob5p12, [24], 275, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p12 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p12 = {'laststimleft':254,'laststimright':255, 'feedbackL':256, 'feedbackR':257, 'leftres':258, 'rightres':259, 'lastresleft':260, 'lastresright':261, 'prepleft':262, 'prepright':263, 'nogo': 264,}
event_dictB5_p12 = {'laststimleft':265,'laststimright':266, 'feedbackL':267, 'feedbackR':268, 'leftres':269, 'rightres':270, 'lastresleft':271, 'lastresright':272, 'prepleft':273, 'prepright':274, 'nogo': 275,}



#merging events togeher into one event list

finalB1_p12 = np.concatenate((laststimpositionleft_b1_p12, laststimpositionright_b1_p12, feedbackleft_b1_p12, feedbackright_b1_p12, leftresponse_b1_p12, rightresponse_b1_p12, lastresponseleft_b1_p12, lastresponseright_b1_p12, preparationleft_b1_p12, preparationright_b1_p12, events_p12, nogob1p12), axis=0)
finalB5_p12 = np.concatenate((laststimpositionleft_b5_p12, laststimpositionright_b5_p12, feedbackleft_b5_p12, feedbackright_b5_p12, leftresponse_b5_p12, rightresponse_b5_p12, lastresponseleft_b5_p12, lastresponseright_b5_p12, preparationleft_b5_p12, preparationright_b5_p12, events_p12, nogob5p12), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p12, event_id=event_dictB1_p12, 

                         sfreq=raw_p12.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p12, event_id=event_dictB5_p12, 

                         sfreq=raw5_p12.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend


#-----------------------------------------------------------------------------#
#                               PARTICIPANT 14
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_14_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part14_SeqL_ERD_6_B1.fif'
Part_14_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part14_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p14 = mne.io.read_raw_fif(Part_14_1, preload = True) 
 

#--------B5--------#
raw5_p14 = mne.io.read_raw_fif(Part_14_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p14, _ = mne.events_from_annotations(raw_p14, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p14, _ = mne.events_from_annotations(raw5_p14, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p14 = np.copy(events_p14)
laststimpositionright_b1_p14 = np.copy(events_p14)
feedbackleft_b1_p14 = np.copy(events_p14)
feedbackright_b1_p14 = np.copy(events_p14)
leftresponse_b1_p14 = np.copy(events_p14)
rightresponse_b1_p14 = np.copy(events_p14)
lastresponseleft_b1_p14 = np.copy(events_p14)
lastresponseright_b1_p14 = np.copy(events_p14)
preparationleft_b1_p14 = np.copy(events_p14)
preparationright_b1_p14 = np.copy(events_p14)
nogob1p14 = np.copy(events_p14)
#--------B5-------#
laststimpositionleft_b5_p14 = np.copy(eventsB5_p14)
laststimpositionright_b5_p14 = np.copy(eventsB5_p14)
feedbackleft_b5_p14 = np.copy(eventsB5_p14)
feedbackright_b5_p14 = np.copy(eventsB5_p14)
leftresponse_b5_p14 = np.copy(eventsB5_p14)
rightresponse_b5_p14 = np.copy(eventsB5_p14)
lastresponseleft_b5_p14 = np.copy(eventsB5_p14)
lastresponseright_b5_p14 = np.copy(eventsB5_p14)
preparationleft_b5_p14 = np.copy(eventsB5_p14)
preparationright_b5_p14 = np.copy(eventsB5_p14)
nogob5p14 = np.copy(eventsB5_p14)


#print to see if event times are correct

print(events_p14)

#--------B1-------#


#last  stimulus position left (276)
laststimpositionleft_b1_p14 = mne.pick_events(laststimpositionleft_b1_p14, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p14 = mne.merge_events(laststimpositionleft_b1_p14, [5, 6, 7, 8], 276, replace_events=True)
laststimpositionleft_b1_p14 = np.delete(laststimpositionleft_b1_p14, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142], axis=0)


#last  stimulus position right (277)
laststimpositionright_b1_p14 = mne.pick_events(laststimpositionright_b1_p14, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p14 = mne.merge_events(laststimpositionright_b1_p14, [9, 10, 11, 12], 277, replace_events=True)
laststimpositionright_b1_p14 = np.delete(laststimpositionright_b1_p14, [29, 65, 143, 161, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166], axis=0)

#feedback left (278)
feedbackleft_b1_p14 = mne.pick_events(feedbackleft_b1_p14, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p14 = mne.merge_events(feedbackleft_b1_p14, [25,26], 278, replace_events=True)
feedbackleft_b1_p14 = mne.event.shift_time_events(feedbackleft_b1_p14, 278, -1.000, 500)
feedbackleft_b1_p14 = np.delete(feedbackleft_b1_p14, [0, 1, 2, 4, 6, 7, 8, 11, 13, 16, 18, 22, 30, 31, 32, 34, 37, 38, 39, 41, 42, 44, 45, 46], axis=0)


#feedback right (279)
feedbackright_b1_p14 = mne.pick_events(feedbackright_b1_p14, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p14 = mne.merge_events(feedbackright_b1_p14, [25,26], 279, replace_events=True)
feedbackright_b1_p14 = mne.event.shift_time_events(feedbackright_b1_p14, 279, -1.000, 500)
feedbackright_b1_p14 = np.delete(feedbackright_b1_p14, [3, 5, 9, 10, 12, 14, 15, 17, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 33, 35, 36, 40, 43, 47], axis=0)

#left response (280) 
leftresponse_b1_p14 = mne.pick_events(leftresponse_b1_p14, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p14 = mne.merge_events(leftresponse_b1_p14, [14, 15, 16, 17], 280, replace_events=True)

#right response(281)
rightresponse_b1_p14 = mne.pick_events(rightresponse_b1_p14, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p14 = mne.merge_events(rightresponse_b1_p14, [18, 19, 20, 21], 281, replace_events=True)

#last response left(282)
lastresponseleft_b1_p14 = mne.pick_events(lastresponseleft_b1_p14, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p14 = mne.merge_events(lastresponseleft_b1_p14, [5, 6, 7, 8], 282, replace_events=True)

#last response right(283)
lastresponseright_b1_p14 = mne.pick_events(lastresponseright_b1_p14, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p14 = mne.merge_events(lastresponseright_b1_p14, [9, 10, 11, 12], 283, replace_events=True)


#preperation left (284) 
preparationleft_b1_p14 = mne.pick_events(preparationleft_b1_p14, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p14 = mne.merge_events(preparationleft_b1_p14, [5, 6, 7, 8], 284, replace_events=True)
preparationleft_b1_p14 = mne.event.shift_time_events(preparationleft_b1_p14, 284, 1.500, 500)
preparationleft_b1_p14 = np.delete(preparationleft_b1_p14, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142], axis=0)

#preperation right (285) 
preparationright_b1_p14 = mne.pick_events(preparationright_b1_p14, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p14 = mne.merge_events(preparationright_b1_p14, [9, 10, 11, 12], 285, replace_events=True)
preparationright_b1_p14 = mne.event.shift_time_events(preparationright_b1_p14, 285, 1.500, 500)
preparationright_b1_p14 = np.delete(preparationright_b1_p14, [29, 65, 143, 161, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166], axis=0)

#nogo (286)
nogob1p14 = mne.pick_events(nogob1p14, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p14 = mne.merge_events(nogob1p14, [24], 286, replace_events=True)
#--------B5-------#


#last  stimulus position left (287)
laststimpositionleft_b5_p14 = mne.pick_events(laststimpositionleft_b5_p14, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p14 = mne.merge_events(laststimpositionleft_b5_p14, [5, 6, 7, 8], 287, replace_events=True)
laststimpositionleft_b5_p14 = np.delete(laststimpositionleft_b5_p14, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142], axis=0)


#last  stimulus position right (288)
laststimpositionright_b5_p14 = mne.pick_events(laststimpositionright_b5_p14, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p14 = mne.merge_events(laststimpositionright_b5_p14, [9, 10, 11, 12], 288, replace_events=True)
laststimpositionright_b5_p14 = np.delete(laststimpositionright_b5_p14, [5, 65, 131, 143, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166], axis=0)


#feedback (289)
feedbackleft_b5_p14 = mne.pick_events(feedbackleft_b5_p14, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p14 = mne.merge_events(feedbackleft_b5_p14, [25,26], 289, replace_events=True)
feedbackleft_b5_p14 = mne.event.shift_time_events(feedbackleft_b5_p14, 289, -1.000, 500)
feedbackleft_b5_p14 = np.delete(feedbackleft_b5_p14, [5, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28, 29, 31, 33, 34, 37, 40, 41, 42, 43, 44], axis=0)


#feedback (290)
feedbackright_b5_p14 = mne.pick_events(feedbackright_b5_p14, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p14 = mne.merge_events(feedbackright_b5_p14, [25,26], 290, replace_events=True)
feedbackright_b5_p14 = mne.event.shift_time_events(feedbackright_b5_p14, 290, -1.000, 500)
feedbackright_b5_p14 = np.delete(feedbackright_b5_p14, [0, 1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 25, 26, 27, 30, 32, 35, 36, 38, 39, 45, 46, 47], axis=0)

#left response (291) 
leftresponse_b5_p14 = mne.pick_events(leftresponse_b5_p14, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p14 = mne.merge_events(leftresponse_b5_p14, [14, 15, 16, 17], 291, replace_events=True)

#right response(292)
rightresponse_b5_p14 = mne.pick_events(rightresponse_b5_p14, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p14 = mne.merge_events(rightresponse_b5_p14, [18, 19, 20, 21], 292, replace_events=True)

#last response left(293)
lastresponseleft_b5_p14 = mne.pick_events(lastresponseleft_b5_p14, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p14 = mne.merge_events(lastresponseleft_b5_p14, [5, 6, 7, 8], 293, replace_events=True)

#last response right(294)
lastresponseright_b5_p14 = mne.pick_events(lastresponseright_b5_p14, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p14 = mne.merge_events(lastresponseright_b5_p14, [9, 10, 11, 12], 294, replace_events=True)


#preperation left (295) 
preparationleft_b5_p14 = mne.pick_events(preparationleft_b5_p14, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p14 = mne.merge_events(preparationleft_b5_p14, [5, 6, 7, 8], 295, replace_events=True)
preparationleft_b5_p14 = mne.event.shift_time_events(preparationleft_b5_p14, 295, 1.500, 500)
preparationleft_b5_p14 = np.delete(preparationleft_b5_p14, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142], axis=0)



#preperation right (296) 
preparationright_b5_p14 = mne.pick_events(preparationright_b5_p14, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p14 = mne.merge_events(preparationright_b5_p14, [9, 10, 11, 12], 296, replace_events=True)
preparationright_b5_p14 = mne.event.shift_time_events(preparationright_b5_p14, 296, 1.500, 500)
preparationright_b5_p14 = np.delete(preparationright_b5_p14, [5, 65, 131, 143, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166], axis=0)

#nogo (297)
nogob5p14 = mne.pick_events(nogob5p14, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p14 = mne.merge_events(nogob5p14, [24], 297, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p14 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p14 = {'laststimleft':276,'laststimright':277, 'feedbackL':278, 'feedbackR':279, 'leftres':280, 'rightres':281, 'lastresleft':282, 'lastresright':283, 'prepleft':284, 'prepright':285, 'nogo': 286}
event_dictB5_p14 = {'laststimleft':287,'laststimright':288, 'feedbackL':289, 'feedbackR':290, 'leftres':291, 'rightres':292, 'lastresleft':293, 'lastresright':294, 'prepleft':295, 'prepright':296, 'nogo': 297}



#merging events togeher into one event list

finalB1_p14 = np.concatenate((laststimpositionleft_b1_p14, laststimpositionright_b1_p14, feedbackleft_b1_p14, feedbackright_b1_p14, leftresponse_b1_p14, rightresponse_b1_p14, lastresponseleft_b1_p14, lastresponseright_b1_p14, preparationleft_b1_p14, preparationright_b1_p14, events_p14, nogob1p14), axis=0)
finalB5_p14 = np.concatenate((laststimpositionleft_b5_p14, laststimpositionright_b5_p14, feedbackleft_b5_p14, feedbackright_b5_p14, leftresponse_b5_p14, rightresponse_b5_p14, lastresponseleft_b5_p14, lastresponseright_b5_p14, preparationleft_b5_p14, preparationright_b5_p14, events_p14, nogob5p14), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p14, event_id=event_dictB1_p14, 

                         sfreq=raw_p14.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p14, event_id=event_dictB5_p14, 

                         sfreq=raw5_p14.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend


#-----------------------------------------------------------------------------#
#                               PARTICIPANT 15
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_15_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part15_SeqL_ERD_6_B1.fif'
Part_15_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part15_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p15 = mne.io.read_raw_fif(Part_15_1, preload = True) 
 

#--------B5--------#
raw5_p15 = mne.io.read_raw_fif(Part_15_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p15, _ = mne.events_from_annotations(raw_p15, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p15, _ = mne.events_from_annotations(raw5_p15, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p15 = np.copy(events_p15)
laststimpositionright_b1_p15 = np.copy(events_p15)
feedbackleft_b1_p15 = np.copy(events_p15)
feedbackright_b1_p15 = np.copy(events_p15)
leftresponse_b1_p15 = np.copy(events_p15)
rightresponse_b1_p15 = np.copy(events_p15)
lastresponseleft_b1_p15 = np.copy(events_p15)
lastresponseright_b1_p15 = np.copy(events_p15)
preparationleft_b1_p15 = np.copy(events_p15)
preparationright_b1_p15 = np.copy(events_p15)
nogob1p15 = np.copy(events_p15)
#--------B5-------#
laststimpositionleft_b5_p15 = np.copy(eventsB5_p15)
laststimpositionright_b5_p15 = np.copy(eventsB5_p15)
feedbackleft_b5_p15 = np.copy(eventsB5_p15)
feedbackright_b5_p15 = np.copy(eventsB5_p15)
leftresponse_b5_p15 = np.copy(eventsB5_p15)
rightresponse_b5_p15 = np.copy(eventsB5_p15)
lastresponseleft_b5_p15 = np.copy(eventsB5_p15)
lastresponseright_b5_p15 = np.copy(eventsB5_p15)
preparationleft_b5_p15 = np.copy(eventsB5_p15)
preparationright_b5_p15 = np.copy(eventsB5_p15)
nogob5p15 = np.copy(eventsB5_p15)


#print to see if event times are correct

print(events_p15)

#--------B1-------#


#last  stimulus position left (298)
laststimpositionleft_b1_p15 = mne.pick_events(laststimpositionleft_b1_p15, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p15 = mne.merge_events(laststimpositionleft_b1_p15, [5, 6, 7, 8], 298, replace_events=True)
laststimpositionleft_b1_p15 = np.delete(laststimpositionleft_b1_p15, [77, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)


#last  stimulus position right (299)
laststimpositionright_b1_p15 = mne.pick_events(laststimpositionright_b1_p15, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p15 = mne.merge_events(laststimpositionright_b1_p15, [9, 10, 11, 12], 299, replace_events=True)
laststimpositionright_b1_p15 = np.delete(laststimpositionright_b1_p15, [5, 11, 161, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)

#feedback left (300)
feedbackleft_b1_p15 = mne.pick_events(feedbackleft_b1_p15, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p15 = mne.merge_events(feedbackleft_b1_p15, [25,26], 300, replace_events=True)
feedbackleft_b1_p15 = mne.event.shift_time_events(feedbackleft_b1_p15, 300, -1.000, 500)
feedbackleft_b1_p15 = np.delete(feedbackleft_b1_p15, [5, 6, 10, 11, 12, 13, 16, 17, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 36, 37, 39], axis=0)


#feedback right (301)
feedbackright_b1_p15 = mne.pick_events(feedbackright_b1_p15, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p15 = mne.merge_events(feedbackright_b1_p15, [25,26], 301, replace_events=True)
feedbackright_b1_p15 = mne.event.shift_time_events(feedbackright_b1_p15, 301, -1.000, 500)
feedbackright_b1_p15 = np.delete(feedbackright_b1_p15, [0, 1, 2, 3, 4, 7, 8, 9, 14, 15, 19, 21, 30, 34, 35, 38, 40, 41, 42, 43, 44, 45, 46, 47], axis=0)

#left response (302) 
leftresponse_b1_p15 = mne.pick_events(leftresponse_b1_p15, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p15 = mne.merge_events(leftresponse_b1_p15, [14, 15, 16, 17], 302, replace_events=True)

#right response(303)
rightresponse_b1_p15 = mne.pick_events(rightresponse_b1_p15, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p15 = mne.merge_events(rightresponse_b1_p15, [18, 19, 20, 21], 303, replace_events=True)

#last response left(304)
lastresponseleft_b1_p15 = mne.pick_events(lastresponseleft_b1_p15, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p15 = mne.merge_events(lastresponseleft_b1_p15, [5, 6, 7, 8], 304, replace_events=True)

#last response right(305)
lastresponseright_b1_p15 = mne.pick_events(lastresponseright_b1_p15, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p15 = mne.merge_events(lastresponseright_b1_p15, [9, 10, 11, 12], 305, replace_events=True)


#preperation left (306) 
preparationleft_b1_p15 = mne.pick_events(preparationleft_b1_p15, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p15 = mne.merge_events(preparationleft_b1_p15, [5, 6, 7, 8], 306, replace_events=True)
preparationleft_b1_p15 = mne.event.shift_time_events(preparationleft_b1_p15, 306, 1.500, 500)
preparationleft_b1_p15 = np.delete(preparationleft_b1_p15, [77, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#preperation right (307) 
preparationright_b1_p15 = mne.pick_events(preparationright_b1_p15, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p15 = mne.merge_events(preparationright_b1_p15, [9, 10, 11, 12], 307, replace_events=True)
preparationright_b1_p15 = mne.event.shift_time_events(preparationright_b1_p15, 307, 1.500, 500)
preparationright_b1_p15 = np.delete(preparationright_b1_p15, [5, 11, 161, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)

#nogo (308)
nogob1p15 = mne.pick_events(nogob1p15, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p15 = mne.merge_events(nogob1p15, [24], 308, replace_events=True)
#--------B5-------#


#last  stimulus position left (309)
laststimpositionleft_b5_p15 = mne.pick_events(laststimpositionleft_b5_p15, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p15 = mne.merge_events(laststimpositionleft_b5_p15, [5, 6, 7, 8], 309, replace_events=True)
laststimpositionleft_b5_p15 = np.delete(laststimpositionleft_b5_p15, [11, 137, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#last  stimulus position right (310)
laststimpositionright_b5_p15 = mne.pick_events(laststimpositionright_b5_p15, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p15 = mne.merge_events(laststimpositionright_b5_p15, [9, 10, 11, 12], 310, replace_events=True)
laststimpositionright_b5_p15 = np.delete(laststimpositionright_b5_p15, [41, 155, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#feedback (311)
feedbackleft_b5_p15 = mne.pick_events(feedbackleft_b5_p15, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p15 = mne.merge_events(feedbackleft_b5_p15, [25,26], 311, replace_events=True)
feedbackleft_b5_p15 = mne.event.shift_time_events(feedbackleft_b5_p15, 311, -1.000, 500)
feedbackleft_b5_p15 = np.delete(feedbackleft_b5_p15, [0, 1, 2, 3, 5, 6, 9, 10, 14, 17, 18, 20, 24, 25, 28, 29, 31, 32, 35, 37, 40, 41, 43, 44], axis=0)


#feedback (312)
feedbackright_b5_p15 = mne.pick_events(feedbackright_b5_p15, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p15 = mne.merge_events(feedbackright_b5_p15, [25,26], 312, replace_events=True)
feedbackright_b5_p15 = mne.event.shift_time_events(feedbackright_b5_p15, 312, -1.000, 500)
feedbackright_b5_p15 = np.delete(feedbackright_b5_p15, [4, 7, 8, 11, 12, 13, 15, 16, 19, 21, 22, 23, 26, 27, 30, 33, 34, 36, 38, 39, 42, 45, 46, 47], axis=0)

#left response (313) 
leftresponse_b5_p15 = mne.pick_events(leftresponse_b5_p15, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p15 = mne.merge_events(leftresponse_b5_p15, [14, 15, 16, 17], 313, replace_events=True)

#right response(314)
rightresponse_b5_p15 = mne.pick_events(rightresponse_b5_p15, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p15 = mne.merge_events(rightresponse_b5_p15, [18, 19, 20, 21], 314, replace_events=True)

#last response left(315)
lastresponseleft_b5_p15 = mne.pick_events(lastresponseleft_b5_p15, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p15 = mne.merge_events(lastresponseleft_b5_p15, [5, 6, 7, 8], 315, replace_events=True)

#last response right(316)
lastresponseright_b5_p15 = mne.pick_events(lastresponseright_b5_p15, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p15 = mne.merge_events(lastresponseright_b5_p15, [9, 10, 11, 12], 316, replace_events=True)


#preperation left (317) 
preparationleft_b5_p15 = mne.pick_events(preparationleft_b5_p15, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p15 = mne.merge_events(preparationleft_b5_p15, [5, 6, 7, 8], 317, replace_events=True)
preparationleft_b5_p15 = mne.event.shift_time_events(preparationleft_b5_p15, 317, 1.500, 500)
preparationleft_b5_p15 = np.delete(preparationleft_b5_p15, [11, 137, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#preperation right (318) 
preparationright_b5_p15 = mne.pick_events(preparationright_b5_p15, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p15 = mne.merge_events(preparationright_b5_p15, [9, 10, 11, 12], 318, replace_events=True)
preparationright_b5_p15 = mne.event.shift_time_events(preparationright_b5_p15, 318, 1.500, 500)
preparationright_b5_p15 = np.delete(preparationright_b5_p15, [41, 155, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#nogo (319)
nogob5p15 = mne.pick_events(nogob5p15, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p15 = mne.merge_events(nogob5p15, [24], 319, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p15 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p15 = {'laststimleft':298,'laststimright':299, 'feedbackL':300, 'feedbackR':301, 'leftres':302, 'rightres':303, 'lastresleft':304, 'lastresright':305, 'prepleft':306, 'prepright':307, 'nogo': 308}
event_dictB5_p15 = {'laststimleft':309,'laststimright':310, 'feedbackL':311, 'feedbackR':312, 'leftres':313, 'rightres':314, 'lastresleft':315, 'lastresright':316, 'prepleft':317, 'prepright':318, 'nogo': 319}



#merging events togeher into one event list

finalB1_p15 = np.concatenate((laststimpositionleft_b1_p15, laststimpositionright_b1_p15, feedbackleft_b1_p15, feedbackright_b1_p15, leftresponse_b1_p15, rightresponse_b1_p15, lastresponseleft_b1_p15, lastresponseright_b1_p15, preparationleft_b1_p15, preparationright_b1_p15, events_p15, nogob1p15), axis=0)
finalB5_p15 = np.concatenate((laststimpositionleft_b5_p15, laststimpositionright_b5_p15, feedbackleft_b5_p15, feedbackright_b5_p15, leftresponse_b5_p15, rightresponse_b5_p15, lastresponseleft_b5_p15, lastresponseright_b5_p15, preparationleft_b5_p15, preparationright_b5_p15, events_p15, nogob5p15), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p15, event_id=event_dictB1_p15, 

                         sfreq=raw_p15.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p15, event_id=event_dictB5_p15, 

                         sfreq=raw5_p15.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend


#-----------------------------------------------------------------------------#
#                               PARTICIPANT 16
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_16_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part16_SeqL_ERD_6_B1.fif'
Part_16_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part16_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p16 = mne.io.read_raw_fif(Part_16_1, preload = True) 
 

#--------B5--------#
raw5_p16 = mne.io.read_raw_fif(Part_16_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p16, _ = mne.events_from_annotations(raw_p16, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p16, _ = mne.events_from_annotations(raw5_p16, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p16 = np.copy(events_p16)
laststimpositionright_b1_p16 = np.copy(events_p16)
feedbackleft_b1_p16 = np.copy(events_p16)
feedbackright_b1_p16 = np.copy(events_p16)
leftresponse_b1_p16 = np.copy(events_p16)
rightresponse_b1_p16 = np.copy(events_p16)
lastresponseleft_b1_p16 = np.copy(events_p16)
lastresponseright_b1_p16 = np.copy(events_p16)
preparationleft_b1_p16 = np.copy(events_p16)
preparationright_b1_p16 = np.copy(events_p16)
nogob1p16 = np.copy(events_p16)
#--------B5-------#
laststimpositionleft_b5_p16 = np.copy(eventsB5_p16)
laststimpositionright_b5_p16 = np.copy(eventsB5_p16)
feedbackleft_b5_p16 = np.copy(eventsB5_p16)
feedbackright_b5_p16 = np.copy(eventsB5_p16)
leftresponse_b5_p16 = np.copy(eventsB5_p16)
rightresponse_b5_p16 = np.copy(eventsB5_p16)
lastresponseleft_b5_p16 = np.copy(eventsB5_p16)
lastresponseright_b5_p16 = np.copy(eventsB5_p16)
preparationleft_b5_p16 = np.copy(eventsB5_p16)
preparationright_b5_p16 = np.copy(eventsB5_p16)
nogob5p16 = np.copy(eventsB5_p16)


#print to see if event times are correct

print(events_p16)

#--------B1-------#


#last  stimulus position left (320)
laststimpositionleft_b1_p16 = mne.pick_events(laststimpositionleft_b1_p16, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p16 = mne.merge_events(laststimpositionleft_b1_p16, [5, 6, 7, 8], 320, replace_events=True)
laststimpositionleft_b1_p16 = np.delete(laststimpositionleft_b1_p16, [35, 83, 155, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)


#last  stimulus position right (321)
laststimpositionright_b1_p16 = mne.pick_events(laststimpositionright_b1_p16, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p16 = mne.merge_events(laststimpositionright_b1_p16, [9, 10, 11, 12], 321, replace_events=True)
laststimpositionright_b1_p16 = np.delete(laststimpositionright_b1_p16, [83, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#feedback left (322)
feedbackleft_b1_p16 = mne.pick_events(feedbackleft_b1_p16, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p16 = mne.merge_events(feedbackleft_b1_p16, [25,26], 322, replace_events=True)
feedbackleft_b1_p16 = mne.event.shift_time_events(feedbackleft_b1_p16, 322, -1.000, 500)
feedbackleft_b1_p16 = np.delete(feedbackleft_b1_p16, [2, 6, 7, 9, 11, 12, 13, 14, 15, 16, 22, 23, 27, 29, 30, 34, 39, 40, 41, 42, 43, 44, 46, 47], axis=0)


#feedback right (323)
feedbackright_b1_p16 = mne.pick_events(feedbackright_b1_p16, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p16 = mne.merge_events(feedbackright_b1_p16, [25,26], 323, replace_events=True)
feedbackright_b1_p16 = mne.event.shift_time_events(feedbackright_b1_p16, 323, -1.000, 500)
feedbackright_b1_p16 = np.delete(feedbackright_b1_p16, [0, 1, 3, 4, 5, 8, 10, 17, 18, 19, 20, 21, 24, 25, 26, 28, 31, 32, 33, 35, 36, 37, 38, 45], axis=0)

#left response (324) 
leftresponse_b1_p16 = mne.pick_events(leftresponse_b1_p16, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p16 = mne.merge_events(leftresponse_b1_p16, [14, 15, 16, 17], 324, replace_events=True)

#right response(325)
rightresponse_b1_p16 = mne.pick_events(rightresponse_b1_p16, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p16 = mne.merge_events(rightresponse_b1_p16, [18, 19, 20, 21], 325, replace_events=True)

#last response left(326)
lastresponseleft_b1_p16 = mne.pick_events(lastresponseleft_b1_p16, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p16 = mne.merge_events(lastresponseleft_b1_p16, [5, 6, 7, 8], 326, replace_events=True)

#last response right(327)
lastresponseright_b1_p16 = mne.pick_events(lastresponseright_b1_p16, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p16 = mne.merge_events(lastresponseright_b1_p16, [9, 10, 11, 12], 327, replace_events=True)


#preperation left (328) 
preparationleft_b1_p16 = mne.pick_events(preparationleft_b1_p16, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p16 = mne.merge_events(preparationleft_b1_p16, [5, 6, 7, 8], 328, replace_events=True)
preparationleft_b1_p16 = mne.event.shift_time_events(preparationleft_b1_p16, 328, 1.500, 500)
preparationleft_b1_p16 = np.delete(preparationleft_b1_p16, [35, 83, 155, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)


#preperation right (329) 
preparationright_b1_p16 = mne.pick_events(preparationright_b1_p16, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p16 = mne.merge_events(preparationright_b1_p16, [9, 10, 11, 12], 329, replace_events=True)
preparationright_b1_p16 = mne.event.shift_time_events(preparationright_b1_p16, 329, 1.500, 500)
preparationright_b1_p16 = np.delete(preparationright_b1_p16, [83, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#nogo (110)
nogob1p16 = mne.pick_events(nogob1p16, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p16 = mne.merge_events(nogob1p16, [24], 330, replace_events=True)


#--------B5-------#
#last  stimulus position left (331)
laststimpositionleft_b5_p16 = mne.pick_events(laststimpositionleft_b5_p16, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p16 = mne.merge_events(laststimpositionleft_b5_p16, [5, 6, 7, 8], 331, replace_events=True)
laststimpositionleft_b5_p16 = np.delete(laststimpositionleft_b5_p16, [77, 101, 119, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)


#last  stimulus position right (332)
laststimpositionright_b5_p16 = mne.pick_events(laststimpositionright_b5_p16, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p16 = mne.merge_events(laststimpositionright_b5_p16, [9, 10, 11, 12], 332, replace_events=True)
laststimpositionright_b5_p16 = np.delete(laststimpositionright_b5_p16, [35, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)


#feedback (333)
feedbackleft_b5_p16 = mne.pick_events(feedbackleft_b5_p16, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p16 = mne.merge_events(feedbackleft_b5_p16, [25,26], 333, replace_events=True)
feedbackleft_b5_p16 = mne.event.shift_time_events(feedbackleft_b5_p16, 333, -1.000, 500)
feedbackleft_b5_p16 = np.delete(feedbackleft_b5_p16, [0, 1, 3, 9, 13, 17, 18, 19, 20, 21, 22, 23, 24, 26, 31, 32, 33, 34, 35, 36, 37, 39, 41, 42], axis=0)


#feedback (334)
feedbackright_b5_p16 = mne.pick_events(feedbackright_b5_p16, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p16 = mne.merge_events(feedbackright_b5_p16, [25,26], 334, replace_events=True)
feedbackright_b5_p16 = mne.event.shift_time_events(feedbackright_b5_p16, 334, -1.000, 500)
feedbackright_b5_p16 = np.delete(feedbackright_b5_p16, [2, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 25, 27, 28, 29, 30, 38, 40, 43, 44, 45, 46, 47], axis=0)


#left response (335) 
leftresponse_b5_p16 = mne.pick_events(leftresponse_b5_p16, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p16 = mne.merge_events(leftresponse_b5_p16, [14, 15, 16, 17], 335, replace_events=True)

#right response(336)
rightresponse_b5_p16 = mne.pick_events(rightresponse_b5_p16, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p16 = mne.merge_events(rightresponse_b5_p16, [18, 19, 20, 21], 336, replace_events=True)

#last response left(337)
lastresponseleft_b5_p16 = mne.pick_events(lastresponseleft_b5_p16, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p16 = mne.merge_events(lastresponseleft_b5_p16, [5, 6, 7, 8], 337, replace_events=True)

#last response right(338)
lastresponseright_b5_p16 = mne.pick_events(lastresponseright_b5_p16, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p16 = mne.merge_events(lastresponseright_b5_p16, [9, 10, 11, 12], 338, replace_events=True)


#preperation left (339) 
preparationleft_b5_p16 = mne.pick_events(preparationleft_b5_p16, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p16 = mne.merge_events(preparationleft_b5_p16, [5, 6, 7, 8], 339, replace_events=True)
preparationleft_b5_p16 = mne.event.shift_time_events(preparationleft_b5_p16, 339, 1.500, 500)
preparationleft_b5_p16 = np.delete(preparationleft_b5_p16, [77, 101, 119, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)


#preperation right (340) 
preparationright_b5_p16 = mne.pick_events(preparationright_b5_p16, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p16 = mne.merge_events(preparationright_b5_p16, [9, 10, 11, 12], 340, replace_events=True)
preparationright_b5_p16 = mne.event.shift_time_events(preparationright_b5_p16, 340, 1.500, 500)
preparationright_b5_p16  = np.delete(preparationright_b5_p16, [35, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#nogo (341)
nogob5p16 = mne.pick_events(nogob5p16, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p16 = mne.merge_events(nogob5p16, [24], 341, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p16 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p16 = {'laststimleft':320,'laststimright':321, 'feedbackL':322, 'feedbackR':323, 'leftres':324, 'rightres':325, 'lastresleft':326, 'lastresright':327, 'prepleft':328, 'prepright':329, 'nogo': 330,}
event_dictB5_p16 = {'laststimleft':331,'laststimright':332, 'feedbackL':333, 'feedbackR':334, 'leftres':335, 'rightres':336, 'lastresleft':337, 'lastresright':338, 'prepleft':339, 'prepright':340, 'nogo': 341,}



#merging events togeher into one event list

finalB1_p16 = np.concatenate((laststimpositionleft_b1_p16, laststimpositionright_b1_p16, feedbackleft_b1_p16, feedbackright_b1_p16, leftresponse_b1_p16, rightresponse_b1_p16, lastresponseleft_b1_p16, lastresponseright_b1_p16, preparationleft_b1_p16, preparationright_b1_p16, events_p16, nogob1p16), axis=0)
finalB5_p16 = np.concatenate((laststimpositionleft_b5_p16, laststimpositionright_b5_p16, feedbackleft_b5_p16, feedbackright_b5_p16, leftresponse_b5_p16, rightresponse_b5_p16, lastresponseleft_b5_p16, lastresponseright_b5_p16, preparationleft_b5_p16, preparationright_b5_p16, events_p16, nogob5p16), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p16, event_id=event_dictB1_p16, 

                         sfreq=raw_p16.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p16, event_id=event_dictB5_p16, 

                         sfreq=raw5_p16.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend



#-----------------------------------------------------------------------------#
#                               PARTICIPANT 17
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_17_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part17_SeqL_ERD_6_B1.fif'
Part_17_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part17_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p17 = mne.io.read_raw_fif(Part_17_1, preload = True) 
 

#--------B5--------#
raw5_p17 = mne.io.read_raw_fif(Part_17_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p17, _ = mne.events_from_annotations(raw_p17, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p17, _ = mne.events_from_annotations(raw5_p17, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p17 = np.copy(events_p17)
laststimpositionright_b1_p17 = np.copy(events_p17)
feedbackleft_b1_p17 = np.copy(events_p17)
feedbackright_b1_p17 = np.copy(events_p17)
leftresponse_b1_p17 = np.copy(events_p17)
rightresponse_b1_p17 = np.copy(events_p17)
lastresponseleft_b1_p17 = np.copy(events_p17)
lastresponseright_b1_p17 = np.copy(events_p17)
preparationleft_b1_p17 = np.copy(events_p17)
preparationright_b1_p17 = np.copy(events_p17)
nogob1p17 = np.copy(events_p17)
#--------B5-------#
laststimpositionleft_b5_p17 = np.copy(eventsB5_p17)
laststimpositionright_b5_p17 = np.copy(eventsB5_p17)
feedbackleft_b5_p17 = np.copy(eventsB5_p17)
feedbackright_b5_p17 = np.copy(eventsB5_p17)
leftresponse_b5_p17 = np.copy(eventsB5_p17)
rightresponse_b5_p17 = np.copy(eventsB5_p17)
lastresponseleft_b5_p17 = np.copy(eventsB5_p17)
lastresponseright_b5_p17 = np.copy(eventsB5_p17)
preparationleft_b5_p17 = np.copy(eventsB5_p17)
preparationright_b5_p17 = np.copy(eventsB5_p17)
nogob5p17 = np.copy(eventsB5_p17)


#print to see if event times are correct

print(events_p17)

#--------B1-------#


#last  stimulus position left (342)
laststimpositionleft_b1_p17 = mne.pick_events(laststimpositionleft_b1_p17, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p17 = mne.merge_events(laststimpositionleft_b1_p17, [5, 6, 7, 8], 342, replace_events=True)
laststimpositionleft_b1_p17 = np.delete(laststimpositionleft_b1_p17, [83, 89, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#last  stimulus position right (343)
laststimpositionright_b1_p17 = mne.pick_events(laststimpositionright_b1_p17, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p17 = mne.merge_events(laststimpositionright_b1_p17, [9, 10, 11, 12], 343, replace_events=True)
laststimpositionright_b1_p17 = np.delete(laststimpositionright_b1_p17, [29, 71, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#feedback left (344)
feedbackleft_b1_p17 = mne.pick_events(feedbackleft_b1_p17, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p17 = mne.merge_events(feedbackleft_b1_p17, [25,26], 344, replace_events=True)
feedbackleft_b1_p17 = mne.event.shift_time_events(feedbackleft_b1_p17, 344, -1.000, 500)
feedbackleft_b1_p17 = np.delete(feedbackleft_b1_p17, [0, 1, 2, 3, 7, 11, 13, 15, 16, 17, 18, 19, 25, 28, 32, 34, 36, 37, 39, 40, 41, 43, 44, 46], axis=0)

#feedback right (345)
feedbackright_b1_p17 = mne.pick_events(feedbackright_b1_p17, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p17 = mne.merge_events(feedbackright_b1_p17, [25,26], 345, replace_events=True)
feedbackright_b1_p17 = mne.event.shift_time_events(feedbackright_b1_p17, 345, -1.000, 500)
feedbackright_b1_p17 = np.delete(feedbackright_b1_p17, [4, 5, 6, 8, 9, 10, 12, 14, 20, 21, 22, 23, 24, 26, 27, 29, 30, 31, 33, 35, 38, 42, 45, 47], axis=0)

#left response (346) 
leftresponse_b1_p17 = mne.pick_events(leftresponse_b1_p17, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p17 = mne.merge_events(leftresponse_b1_p17, [14, 15, 16, 17], 346, replace_events=True)

#right response(347)
rightresponse_b1_p17 = mne.pick_events(rightresponse_b1_p17, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p17 = mne.merge_events(rightresponse_b1_p17, [18, 19, 20, 21], 347, replace_events=True)

#last response left(348)
lastresponseleft_b1_p17 = mne.pick_events(lastresponseleft_b1_p17, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p17 = mne.merge_events(lastresponseleft_b1_p17, [5, 6, 7, 8], 348, replace_events=True)

#last response right(349)
lastresponseright_b1_p17 = mne.pick_events(lastresponseright_b1_p17, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p17 = mne.merge_events(lastresponseright_b1_p17, [9, 10, 11, 12], 349, replace_events=True)


#preperation left (350) 
preparationleft_b1_p17 = mne.pick_events(preparationleft_b1_p17, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p17 = mne.merge_events(preparationleft_b1_p17, [5, 6, 7, 8], 350, replace_events=True)
preparationleft_b1_p17 = mne.event.shift_time_events(preparationleft_b1_p17, 350, 1.500, 500)
preparationleft_b1_p17 = np.delete(preparationleft_b1_p17, [83, 89, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)



#preperation right (351) 
preparationright_b1_p17 = mne.pick_events(preparationright_b1_p17, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p17 = mne.merge_events(preparationright_b1_p17, [9, 10, 11, 12], 351, replace_events=True)
preparationright_b1_p17 = mne.event.shift_time_events(preparationright_b1_p17, 351, 1.500, 500)
preparationright_b1_p17 = np.delete(preparationright_b1_p17, [29, 71, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#nogo (352)
nogob1p17 = mne.pick_events(nogob1p17, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p17 = mne.merge_events(nogob1p17, [24], 352, replace_events=True)
#--------B5-------#


#last  stimulus position left (353)
laststimpositionleft_b5_p17 = mne.pick_events(laststimpositionleft_b5_p17, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p17 = mne.merge_events(laststimpositionleft_b5_p17, [5, 6, 7, 8], 353, replace_events=True)
laststimpositionleft_b5_p17 = np.delete(laststimpositionleft_b5_p17, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142], axis=0)


#last  stimulus position right (354)
laststimpositionright_b5_p17 = mne.pick_events(laststimpositionright_b5_p17, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p17 = mne.merge_events(laststimpositionright_b5_p17, [9, 10, 11, 12], 354, replace_events=True)
laststimpositionright_b5_p17 = np.delete(laststimpositionright_b5_p17, [11, 29, 101, 119, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166], axis=0)


#feedback (355)
feedbackleft_b5_p17 = mne.pick_events(feedbackleft_b5_p17, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p17 = mne.merge_events(feedbackleft_b5_p17, [25,26], 355, replace_events=True)
feedbackleft_b5_p17 = mne.event.shift_time_events(feedbackleft_b5_p17, 355, -1.000, 500)
feedbackleft_b5_p17 = np.delete(feedbackleft_b5_p17, [2, 5, 7, 9, 12, 13, 15, 16, 19, 21, 22, 23, 24, 26, 30, 35, 36, 39, 41, 42, 44, 45, 46, 47], axis=0)

#feedback (356)
feedbackright_b5_p17 = mne.pick_events(feedbackright_b5_p17, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p17 = mne.merge_events(feedbackright_b5_p17, [25,26], 356, replace_events=True)
feedbackright_b5_p17 = mne.event.shift_time_events(feedbackright_b5_p17, 356, -1.000, 500)
feedbackright_b5_p17 = np.delete(feedbackright_b5_p17, [0, 1, 3, 4, 6, 8, 10, 11, 14, 17, 18, 20, 25, 27, 28, 29, 31, 32, 33, 34, 37, 38, 40, 43], axis=0)

#left response (357) 
leftresponse_b5_p17 = mne.pick_events(leftresponse_b5_p17, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p17 = mne.merge_events(leftresponse_b5_p17, [14, 15, 16, 17], 357, replace_events=True)

#right response(358)
rightresponse_b5_p17 = mne.pick_events(rightresponse_b5_p17, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p17 = mne.merge_events(rightresponse_b5_p17, [18, 19, 20, 21], 358, replace_events=True)


#last response left(359)
lastresponseleft_b5_p17 = mne.pick_events(lastresponseleft_b5_p17, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p17 = mne.merge_events(lastresponseleft_b5_p17, [5, 6, 7, 8], 359, replace_events=True)

#last response right(360)
lastresponseright_b5_p17 = mne.pick_events(lastresponseright_b5_p17, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p17 = mne.merge_events(lastresponseright_b5_p17, [9, 10, 11, 12], 360, replace_events=True)


#preperation left (361) 
preparationleft_b5_p17 = mne.pick_events(preparationleft_b5_p17, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p17 = mne.merge_events(preparationleft_b5_p17, [5, 6, 7, 8], 361, replace_events=True)
preparationleft_b5_p17 = mne.event.shift_time_events(preparationleft_b5_p17, 361, 1.500, 500)
preparationleft_b5_p17 = np.delete(preparationleft_b5_p17, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142], axis=0)



#preperation right (362) 
preparationright_b5_p17 = mne.pick_events(preparationright_b5_p17, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p17 = mne.merge_events(preparationright_b5_p17, [9, 10, 11, 12], 362, replace_events=True)
preparationright_b5_p17 = mne.event.shift_time_events(preparationright_b5_p17, 362, 1.500, 500)
preparationright_b5_p17= np.delete(preparationright_b5_p17, [11, 29, 101, 119, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166], axis=0)

#nogo (363)
nogob5p17 = mne.pick_events(nogob5p17, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p17 = mne.merge_events(nogob5p17, [24], 363, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p17 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p17 = {'laststimleft':342,'laststimright':343, 'feedbackL':344, 'feedbackR':345, 'leftres':346, 'rightres':347, 'lastresleft':348, 'lastresright':349, 'prepleft':350, 'prepright':351, 'nogo': 352}
event_dictB5_p17 = {'laststimleft':353,'laststimright':354, 'feedbackL':355, 'feedbackR':356, 'leftres':357, 'rightres':358, 'lastresleft':359, 'lastresright':360, 'prepleft':361, 'prepright':362, 'nogo': 363}



#merging events togeher into one event list

finalB1_p17 = np.concatenate((laststimpositionleft_b1_p17, laststimpositionright_b1_p17, feedbackleft_b1_p17, feedbackright_b1_p17, leftresponse_b1_p17, rightresponse_b1_p17, lastresponseleft_b1_p17, lastresponseright_b1_p17, preparationleft_b1_p17, preparationright_b1_p17, events_p17, nogob1p17), axis=0)
finalB5_p17 = np.concatenate((laststimpositionleft_b5_p17, laststimpositionright_b5_p17, feedbackleft_b5_p17, feedbackright_b5_p17, leftresponse_b5_p17, rightresponse_b5_p17, lastresponseleft_b5_p17, lastresponseright_b5_p17, preparationleft_b5_p17, preparationright_b5_p17, events_p17, nogob5p17), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p17, event_id=event_dictB1_p17, 

                         sfreq=raw_p17.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p17, event_id=event_dictB5_p17, 

                         sfreq=raw5_p17.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend


#-----------------------------------------------------------------------------#
#                               PARTICIPANT 18
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_18_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part18_SeqL_ERD_6_B1.fif'
Part_18_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part18_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p18 = mne.io.read_raw_fif(Part_18_1, preload = True) 
 

#--------B5--------#
raw5_p18 = mne.io.read_raw_fif(Part_18_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p18, _ = mne.events_from_annotations(raw_p18, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p18, _ = mne.events_from_annotations(raw5_p18, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p18 = np.copy(events_p18)
laststimpositionright_b1_p18 = np.copy(events_p18)
feedbackleft_b1_p18 = np.copy(events_p18)
feedbackright_b1_p18 = np.copy(events_p18)
leftresponse_b1_p18 = np.copy(events_p18)
rightresponse_b1_p18 = np.copy(events_p18)
lastresponseleft_b1_p18 = np.copy(events_p18)
lastresponseright_b1_p18 = np.copy(events_p18)
preparationleft_b1_p18 = np.copy(events_p18)
preparationright_b1_p18 = np.copy(events_p18)
nogob1p18 = np.copy(events_p18)
#--------B5-------#
laststimpositionleft_b5_p18 = np.copy(eventsB5_p18)
laststimpositionright_b5_p18 = np.copy(eventsB5_p18)
feedbackleft_b5_p18 = np.copy(eventsB5_p18)
feedbackright_b5_p18 = np.copy(eventsB5_p18)
leftresponse_b5_p18 = np.copy(eventsB5_p18)
rightresponse_b5_p18 = np.copy(eventsB5_p18)
lastresponseleft_b5_p18 = np.copy(eventsB5_p18)
lastresponseright_b5_p18 = np.copy(eventsB5_p18)
preparationleft_b5_p18 = np.copy(eventsB5_p18)
preparationright_b5_p18 = np.copy(eventsB5_p18)
nogob5p18 = np.copy(eventsB5_p18)


#print to see if event times are correct

print(events_p18)

#--------B1-------#


#last  stimulus position left (364)
laststimpositionleft_b1_p18 = mne.pick_events(laststimpositionleft_b1_p18, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p18 = mne.merge_events(laststimpositionleft_b1_p18, [5, 6, 7, 8], 364, replace_events=True)
laststimpositionleft_b1_p18 = np.delete (laststimpositionleft_b1_p18, [77, 155, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#last  stimulus position right (365)
laststimpositionright_b1_p18 = mne.pick_events(laststimpositionright_b1_p18, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p18 = mne.merge_events(laststimpositionright_b1_p18, [9, 10, 11, 12], 365, replace_events=True)
laststimpositionright_b1_p18 = np.delete(laststimpositionright_b1_p18, [36, 110, 30, 43, 104, 153, 0,	1,	2,	3,	4, 6,	7,	8,	9,	10, 12,	13,	14,	15,	16, 18,	19,	20,	21,	22, 24,	25,	26,	27,	28, 31,	32,	33,	34,	35, 37,	38,	39,	40,	41, 44,	45,	46,	47,	48, 50,	51,	52,	53,	54, 56,	57,	58,	59,	60, 62,	63,	64,	65,	66, 68,	69,	70,	71,	72, 74,	75,	76,	77,	78, 80,	81,	82,	83,	84, 86,	87,	88,	89,	90, 92,	93,	94,	95,	96, 98,	99,	100,	101,	102, 105,	106,	107,	108,	109, 111,	112,	113,	114,	115, 117,	118,	119,	120,	121, 123,	124,	125,	126,	127, 129,	130,	131,	132,	133, 135,	136,	137,	138,	139, 141,	142,	143,	144,	145, 147,	148,	149,	150,	151, 154,	155,	156,	157,	158], axis=0)



#feedback left (366)
feedbackleft_b1_p18 = mne.pick_events(feedbackleft_b1_p18, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p18 = mne.merge_events(feedbackleft_b1_p18, [25,26], 366, replace_events=True)
feedbackleft_b1_p18 = mne.event.shift_time_events(feedbackleft_b1_p18, 366, -1.000, 500)
feedbackleft_b1_p18 = np.delete(feedbackleft_b1_p18, [0, 6, 7, 8, 10, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 31, 34, 35, 41, 43, 44, 46, 47], axis=0)


#feedback right (367)
feedbackright_b1_p18 = mne.pick_events(feedbackright_b1_p18, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p18 = mne.merge_events(feedbackright_b1_p18, [25,26], 367, replace_events=True)
feedbackright_b1_p18 = mne.event.shift_time_events(feedbackright_b1_p18, 367, -1.000, 500)
feedbackright_b1_p18 = np.delete(feedbackright_b1_p18, [1, 2, 3, 4, 5, 9, 11, 12, 13, 14, 15, 17, 27, 29, 30, 32, 33, 36, 37, 38, 39, 40, 42, 45], axis=0)

#left response (368) 
leftresponse_b1_p18 = mne.pick_events(leftresponse_b1_p18, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p18 = mne.merge_events(leftresponse_b1_p18, [14, 15, 16, 17], 368, replace_events=True)

#right response(369)
rightresponse_b1_p18 = mne.pick_events(rightresponse_b1_p18, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p18 = mne.merge_events(rightresponse_b1_p18, [18, 19, 20, 21], 369, replace_events=True)

#last response left(370)
lastresponseleft_b1_p18 = mne.pick_events(lastresponseleft_b1_p18, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p18 = mne.merge_events(lastresponseleft_b1_p18, [5, 6, 7, 8], 370, replace_events=True)

#last response right(371)
lastresponseright_b1_p18 = mne.pick_events(lastresponseright_b1_p18, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p18 = mne.merge_events(lastresponseright_b1_p18, [9, 10, 11, 12], 371, replace_events=True)


#preperation left (372) 
preparationleft_b1_p18 = mne.pick_events(preparationleft_b1_p18, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p18 = mne.merge_events(preparationleft_b1_p18, [5, 6, 7, 8], 372, replace_events=True)
preparationleft_b1_p18 = mne.event.shift_time_events(preparationleft_b1_p18, 372, 1.500, 500)
preparationleft_b1_p18 = np.delete (preparationleft_b1_p18, [77, 155, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#preperation right (373) 
preparationright_b1_p18 = mne.pick_events(preparationright_b1_p18, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p18 = mne.merge_events(preparationright_b1_p18, [9, 10, 11, 12], 373, replace_events=True)
preparationright_b1_p18 = mne.event.shift_time_events(preparationright_b1_p18, 373, 1.500, 500)
preparationright_b1_p18 = np.delete(preparationright_b1_p18, [36, 110, 30, 43, 104, 153, 0,	1,	2,	3,	4, 6,	7,	8,	9,	10, 12,	13,	14,	15,	16, 18,	19,	20,	21,	22, 24,	25,	26,	27,	28, 31,	32,	33,	34,	35, 37,	38,	39,	40,	41, 44,	45,	46,	47,	48, 50,	51,	52,	53,	54, 56,	57,	58,	59,	60, 62,	63,	64,	65,	66, 68,	69,	70,	71,	72, 74,	75,	76,	77,	78, 80,	81,	82,	83,	84, 86,	87,	88,	89,	90, 92,	93,	94,	95,	96, 98,	99,	100,	101,	102, 105,	106,	107,	108,	109, 111,	112,	113,	114,	115, 117,	118,	119,	120,	121, 123,	124,	125,	126,	127, 129,	130,	131,	132,	133, 135,	136,	137,	138,	139, 141,	142,	143,	144,	145, 147,	148,	149,	150,	151, 154,	155,	156,	157,	158], axis=0)

#nogo (374)
nogob1p18 = mne.pick_events(nogob1p18, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p18 = mne.merge_events(nogob1p18, [24], 374, replace_events=True)
#--------B5-------#


#last  stimulus position left (375)
laststimpositionleft_b5_p18 = mne.pick_events(laststimpositionleft_b5_p18, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p18 = mne.merge_events(laststimpositionleft_b5_p18, [5, 6, 7, 8], 375, replace_events=True)
laststimpositionleft_b5_p18 = np.delete(laststimpositionleft_b5_p18, [23, 41, 89, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)

#last  stimulus position right (376)
laststimpositionright_b5_p18 = mne.pick_events(laststimpositionright_b5_p18, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p18 = mne.merge_events(laststimpositionright_b5_p18, [9, 10, 11, 12], 376, replace_events=True)
laststimpositionright_b5_p18 = np.delete(laststimpositionright_b5_p18, [117, 6, 37, 80, 111, 0,	1,	2,	3,	4, 7,	8,	9,	10,	11, 13,	14,	15,	16,	17, 19,	20,	21,	22,	23, 25,	26,	27,	28,	29, 31,	32,	33,	34,	35, 38,	39,	40,	41,	42, 44,	45,	46,	47,	48, 50,	51,	52,	53,	54, 56,	57,	58,	59,	60, 62,	63,	64,	65,	66, 68,	69,	70,	71,	72, 74,	75,	76,	77,	78, 81,	82,	83,	84,	85, 87,	88,	89,	90,	91, 93,	94,	95,	96,	97, 99,	100,	101,	102,	103, 105,	106,	107,	108,	109, 112,	113,	114,	115,	116, 118,	119,	120,	121,	122, 124,	125,	126,	127,	128, 130,	131,	132,	133,	134, 136,	137,	138,	139,	140, 142,	143,	144,	145,	146, 148,	149,	150,	151,	152], axis=0)



#feedback (377)
feedbackleft_b5_p18 = mne.pick_events(feedbackleft_b5_p18, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p18 = mne.merge_events(feedbackleft_b5_p18, [25,26], 377, replace_events=True)
feedbackleft_b5_p18 = mne.event.shift_time_events(feedbackleft_b5_p18, 377, -1.000, 500)
feedbackleft_b5_p18 = np.delete(feedbackleft_b5_p18, [0, 5, 6, 7, 8, 10, 11, 14, 15, 16, 17, 22, 24, 27, 28, 29, 30, 31, 40, 41, 43, 44, 45, 46], axis=0)


#feedback (378)
feedbackright_b5_p18 = mne.pick_events(feedbackright_b5_p18, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p18 = mne.merge_events(feedbackright_b5_p18, [25,26], 378, replace_events=True)
feedbackright_b5_p18 = mne.event.shift_time_events(feedbackright_b5_p18, 378, -1.000, 500)
feedbackright_b5_p18 = np.delete(feedbackright_b5_p18, [1, 2, 3, 4, 9, 12, 13, 18, 19, 20, 21, 23, 25, 26, 32, 33, 34, 35, 36, 37, 38, 39, 42, 47], axis=0)

#left response (379) 
leftresponse_b5_p18 = mne.pick_events(leftresponse_b5_p18, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p18 = mne.merge_events(leftresponse_b5_p18, [14, 15, 16, 17], 379, replace_events=True)

#right response(380)
rightresponse_b5_p18 = mne.pick_events(rightresponse_b5_p18, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p18 = mne.merge_events(rightresponse_b5_p18, [18, 19, 20, 21], 380, replace_events=True)

#last response left(381)
lastresponseleft_b5_p18 = mne.pick_events(lastresponseleft_b5_p18, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p18 = mne.merge_events(lastresponseleft_b5_p18, [5, 6, 7, 8], 381, replace_events=True)

#last response right(382)
lastresponseright_b5_p18 = mne.pick_events(lastresponseright_b5_p18, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p18 = mne.merge_events(lastresponseright_b5_p18, [9, 10, 11, 12], 382, replace_events=True)


#preperation left (383) 
preparationleft_b5_p18 = mne.pick_events(preparationleft_b5_p18, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p18 = mne.merge_events(preparationleft_b5_p18, [5, 6, 7, 8], 383, replace_events=True)
preparationleft_b5_p18 = mne.event.shift_time_events(preparationleft_b5_p18, 383, 1.500, 500)
preparationleft_b5_p18 = np.delete(preparationleft_b5_p18, [23, 41, 89, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)


#preperation right (384) 
preparationright_b5_p18 = mne.pick_events(preparationright_b5_p18, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p18 = mne.merge_events(preparationright_b5_p18, [9, 10, 11, 12], 384, replace_events=True)
preparationright_b5_p18 = mne.event.shift_time_events(preparationright_b5_p18, 384, 1.500, 500)
preparationright_b5_p18 = np.delete(preparationright_b5_p18, [117, 6, 37, 80, 111, 0,	1,	2,	3,	4, 7,	8,	9,	10,	11, 13,	14,	15,	16,	17, 19,	20,	21,	22,	23, 25,	26,	27,	28,	29, 31,	32,	33,	34,	35, 38,	39,	40,	41,	42, 44,	45,	46,	47,	48, 50,	51,	52,	53,	54, 56,	57,	58,	59,	60, 62,	63,	64,	65,	66, 68,	69,	70,	71,	72, 74,	75,	76,	77,	78, 81,	82,	83,	84,	85, 87,	88,	89,	90,	91, 93,	94,	95,	96,	97, 99,	100,	101,	102,	103, 105,	106,	107,	108,	109, 112,	113,	114,	115,	116, 118,	119,	120,	121,	122, 124,	125,	126,	127,	128, 130,	131,	132,	133,	134, 136,	137,	138,	139,	140, 142,	143,	144,	145,	146, 148,	149,	150,	151,	152], axis=0)

#nogo (385)
nogob5p18 = mne.pick_events(nogob5p18, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p18 = mne.merge_events(nogob5p18, [24], 385, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p18 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p18 = {'laststimleft':364,'laststimright':365, 'feedbackL':366, 'feedbackR':367, 'leftres':368, 'rightres':369, 'lastresleft':370, 'lastresright':371, 'prepleft':372, 'prepright':373, 'nogo': 374,}
event_dictB5_p18 = {'laststimleft':375,'laststimright':376, 'feedbackL':377, 'feedbackR':378, 'leftres':379, 'rightres':380, 'lastresleft':381, 'lastresright':382, 'prepleft':383, 'prepright':384, 'nogo': 385,}



#merging events togeher into one event list

finalB1_p18 = np.concatenate((laststimpositionleft_b1_p18, laststimpositionright_b1_p18, feedbackleft_b1_p18, feedbackright_b1_p18, leftresponse_b1_p18, rightresponse_b1_p18, lastresponseleft_b1_p18, lastresponseright_b1_p18, preparationleft_b1_p18, preparationright_b1_p18, events_p18, nogob1p18), axis=0)
finalB5_p18 = np.concatenate((laststimpositionleft_b5_p18, laststimpositionright_b5_p18, feedbackleft_b5_p18, feedbackright_b5_p18, leftresponse_b5_p18, rightresponse_b5_p18, lastresponseleft_b5_p18, lastresponseright_b5_p18, preparationleft_b5_p18, preparationright_b5_p18, events_p18, nogob5p18), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p18, event_id=event_dictB1_p18, 

                         sfreq=raw_p18.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p18, event_id=event_dictB5_p18, 

                         sfreq=raw5_p18.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend




#-----------------------------------------------------------------------------#
#                               PARTICIPANT 19
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_19_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part19_SeqL_ERD_6_B1.fif'
Part_19_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part19_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p19 = mne.io.read_raw_fif(Part_19_1, preload = True) 
 

#--------B5--------#
raw5_p19 = mne.io.read_raw_fif(Part_19_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p19, _ = mne.events_from_annotations(raw_p19, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p19, _ = mne.events_from_annotations(raw5_p19, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p19 = np.copy(events_p19)
laststimpositionright_b1_p19 = np.copy(events_p19)
feedbackleft_b1_p19 = np.copy(events_p19)
feedbackright_b1_p19 = np.copy(events_p19)
leftresponse_b1_p19 = np.copy(events_p19)
rightresponse_b1_p19 = np.copy(events_p19)
lastresponseleft_b1_p19 = np.copy(events_p19)
lastresponseright_b1_p19 = np.copy(events_p19)
preparationleft_b1_p19 = np.copy(events_p19)
preparationright_b1_p19 = np.copy(events_p19)
nogob1p19 = np.copy(events_p19)
#--------B5-------#
laststimpositionleft_b5_p19 = np.copy(eventsB5_p19)
laststimpositionright_b5_p19 = np.copy(eventsB5_p19)
feedbackleft_b5_p19 = np.copy(eventsB5_p19)
feedbackright_b5_p19 = np.copy(eventsB5_p19)
leftresponse_b5_p19 = np.copy(eventsB5_p19)
rightresponse_b5_p19 = np.copy(eventsB5_p19)
lastresponseleft_b5_p19 = np.copy(eventsB5_p19)
lastresponseright_b5_p19 = np.copy(eventsB5_p19)
preparationleft_b5_p19 = np.copy(eventsB5_p19)
preparationright_b5_p19 = np.copy(eventsB5_p19)
nogob5p19 = np.copy(eventsB5_p19)


#print to see if event times are correct

print(events_p19)

#--------B1-------#


#last  stimulus position left (386)
laststimpositionleft_b1_p19 = mne.pick_events(laststimpositionleft_b1_p19, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p19 = mne.merge_events(laststimpositionleft_b1_p19, [5, 6, 7, 8], 386, replace_events=True)
laststimpositionleft_b1_p19 = np.delete(laststimpositionleft_b1_p19, [107, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)


#last  stimulus position right (387)
laststimpositionright_b1_p19 = mne.pick_events(laststimpositionright_b1_p19, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p19 = mne.merge_events(laststimpositionright_b1_p19, [9, 10, 11, 12], 387, replace_events=True)
laststimpositionright_b1_p19 = np.delete(laststimpositionright_b1_p19, [11, 47, 119, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)


#feedback left (388)
feedbackleft_b1_p19 = mne.pick_events(feedbackleft_b1_p19, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p19 = mne.merge_events(feedbackleft_b1_p19, [25,26], 388, replace_events=True)
feedbackleft_b1_p19 = mne.event.shift_time_events(feedbackleft_b1_p19, 388, -1.000, 500)
feedbackleft_b1_p19 = np.delete(feedbackleft_b1_p19, [0, 1, 2, 5, 6, 8, 11, 12, 14, 17, 19, 21, 24, 25, 26, 29, 31, 37, 38, 39, 41, 42, 43, 47], axis=0)

#feedback right (389)
feedbackright_b1_p19 = mne.pick_events(feedbackright_b1_p19, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p19 = mne.merge_events(feedbackright_b1_p19, [25,26], 389, replace_events=True)
feedbackright_b1_p19 = mne.event.shift_time_events(feedbackright_b1_p19, 389, -1.000, 500)
feedbackright_b1_p19 = np.delete(feedbackright_b1_p19, [3, 4, 7, 9, 10, 13, 15, 16, 18, 20, 22, 23, 27, 28, 30, 32, 33, 34, 35, 36, 40, 44, 45, 46], axis=0)

#left response (390) 
leftresponse_b1_p19 = mne.pick_events(leftresponse_b1_p19, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p19 = mne.merge_events(leftresponse_b1_p19, [14, 15, 16, 17], 390, replace_events=True)

#right response(391)
rightresponse_b1_p19 = mne.pick_events(rightresponse_b1_p19, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p19 = mne.merge_events(rightresponse_b1_p19, [18, 19, 20, 21], 391, replace_events=True)

#last response left(392)
lastresponseleft_b1_p19 = mne.pick_events(lastresponseleft_b1_p19, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p19 = mne.merge_events(lastresponseleft_b1_p19, [5, 6, 7, 8], 392, replace_events=True)

#last response right(393)
lastresponseright_b1_p19 = mne.pick_events(lastresponseright_b1_p19, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p19 = mne.merge_events(lastresponseright_b1_p19, [9, 10, 11, 12], 393, replace_events=True)


#preperation left (394) 
preparationleft_b1_p19 = mne.pick_events(preparationleft_b1_p19, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p19 = mne.merge_events(preparationleft_b1_p19, [5, 6, 7, 8], 394, replace_events=True)
preparationleft_b1_p19 = mne.event.shift_time_events(preparationleft_b1_p19, 394, 1.500, 500)
preparationleft_b1_p19 = np.delete(preparationleft_b1_p19, [107, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)



#preperation right (395) 
preparationright_b1_p19 = mne.pick_events(preparationright_b1_p19, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p19 = mne.merge_events(preparationright_b1_p19, [9, 10, 11, 12], 395, replace_events=True)
preparationright_b1_p19 = mne.event.shift_time_events(preparationright_b1_p19, 395, 1.500, 500)
preparationright_b1_p19 = np.delete(preparationright_b1_p19, [11, 47, 119, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)

#nogo (396)
nogob1p19 = mne.pick_events(nogob1p19, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p19 = mne.merge_events(nogob1p19, [24], 396, replace_events=True)
#--------B5-------#


#last  stimulus position left (397)
laststimpositionleft_b5_p19 = mne.pick_events(laststimpositionleft_b5_p19, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p19 = mne.merge_events(laststimpositionleft_b5_p19, [5, 6, 7, 8], 397, replace_events=True)
laststimpositionleft_b5_p19 = np.delete(laststimpositionleft_b5_p19, [53, 131, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#last  stimulus position right (398)
laststimpositionright_b5_p19 = mne.pick_events(laststimpositionright_b5_p19, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p19 = mne.merge_events(laststimpositionright_b5_p19, [9, 10, 11, 12], 398, replace_events=True)
laststimpositionright_b5_p19 = np.delete(laststimpositionright_b5_p19, [71, 125, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#feedback (399)
feedbackleft_b5_p19 = mne.pick_events(feedbackleft_b5_p19, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p19 = mne.merge_events(feedbackleft_b5_p19, [25,26], 399, replace_events=True)
feedbackleft_b5_p19 = mne.event.shift_time_events(feedbackleft_b5_p19, 399, -1.000, 500)
feedbackleft_b5_p19 = np.delete(feedbackleft_b5_p19, [1, 3, 4, 5, 8, 9, 10, 11, 13, 17, 20, 23, 25, 26, 29, 32, 33, 37, 38, 43, 44, 45, 46, 47], axis=0)

#feedback (400)
feedbackright_b5_p19 = mne.pick_events(feedbackright_b5_p19, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p19 = mne.merge_events(feedbackright_b5_p19, [25,26], 400, replace_events=True)
feedbackright_b5_p19 = mne.event.shift_time_events(feedbackright_b5_p19, 400, -1.000, 500)
feedbackright_b5_p19 = np.delete(feedbackright_b5_p19, [0, 2, 6, 7, 12, 14, 15, 16, 18, 19, 21, 22, 24, 27, 28, 30, 31, 34, 35, 36, 39, 40, 41, 42], axis=0)

#left response (401) 
leftresponse_b5_p19 = mne.pick_events(leftresponse_b5_p19, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p19 = mne.merge_events(leftresponse_b5_p19, [14, 15, 16, 17], 401, replace_events=True)

#right response(402)
rightresponse_b5_p19 = mne.pick_events(rightresponse_b5_p19, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p19 = mne.merge_events(rightresponse_b5_p19, [18, 19, 20, 21], 402, replace_events=True)

#last response left(403)
lastresponseleft_b5_p19 = mne.pick_events(lastresponseleft_b5_p19, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p19 = mne.merge_events(lastresponseleft_b5_p19, [5, 6, 7, 8], 403, replace_events=True)

#last response right(404)
lastresponseright_b5_p19 = mne.pick_events(lastresponseright_b5_p19, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p19 = mne.merge_events(lastresponseright_b5_p19, [9, 10, 11, 12], 404, replace_events=True)


#preperation left (405) 
preparationleft_b5_p19 = mne.pick_events(preparationleft_b5_p19, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p19 = mne.merge_events(preparationleft_b5_p19, [5, 6, 7, 8], 405, replace_events=True)
preparationleft_b5_p19 = mne.event.shift_time_events(preparationleft_b5_p19, 405, 1.500, 500)
preparationleft_b5_p19 = np.delete(preparationleft_b5_p19, [53, 131, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#preperation right (406) 
preparationright_b5_p19 = mne.pick_events(preparationright_b5_p19, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p19 = mne.merge_events(preparationright_b5_p19, [9, 10, 11, 12], 406, replace_events=True)
preparationright_b5_p19 = mne.event.shift_time_events(preparationright_b5_p19, 406, 1.500, 500)
preparationright_b5_p19 = np.delete(preparationright_b5_p19, [71, 125, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#nogo (407)
nogob5p19 = mne.pick_events(nogob5p19, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p19 = mne.merge_events(nogob5p19, [24], 407, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p19 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p19 = {'laststimleft':386,'laststimright':387, 'feedbackL':388, 'feedbackR':389, 'leftres':390, 'rightres':391, 'lastresleft':392, 'lastresright':393, 'prepleft':394, 'prepright':395, 'nogo': 396,}
event_dictB5_p19 = {'laststimleft':397,'laststimright':398, 'feedbackL':399, 'feedbackR':400, 'leftres':401, 'rightres':402, 'lastresleft':403, 'lastresright':404, 'prepleft':405, 'prepright':406, 'nogo': 407,}



#merging events togeher into one event list

finalB1_p19 = np.concatenate((laststimpositionleft_b1_p19, laststimpositionright_b1_p19, feedbackleft_b1_p19, feedbackright_b1_p19, leftresponse_b1_p19, rightresponse_b1_p19, lastresponseleft_b1_p19, lastresponseright_b1_p19, preparationleft_b1_p19, preparationright_b1_p19, events_p19, nogob1p19), axis=0)
finalB5_p19 = np.concatenate((laststimpositionleft_b5_p19, laststimpositionright_b5_p19, feedbackleft_b5_p19, feedbackright_b5_p19, leftresponse_b5_p19, rightresponse_b5_p19, lastresponseleft_b5_p19, lastresponseright_b5_p19, preparationleft_b5_p19, preparationright_b5_p19, events_p19, nogob5p19), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p19, event_id=event_dictB1_p19, 

                         sfreq=raw_p19.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p19, event_id=event_dictB5_p19, 

                         sfreq=raw5_p19.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend


#-----------------------------------------------------------------------------#
#                               PARTICIPANT 22
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_22_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part22_SeqL_ERD_6_B1.fif'
Part_22_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part22_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p22 = mne.io.read_raw_fif(Part_22_1, preload = True) 
 

#--------B5--------#
raw5_p22 = mne.io.read_raw_fif(Part_22_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p22, _ = mne.events_from_annotations(raw_p22, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p22, _ = mne.events_from_annotations(raw5_p22, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p22 = np.copy(events_p22)
laststimpositionright_b1_p22 = np.copy(events_p22)
feedbackleft_b1_p22 = np.copy(events_p22)
feedbackright_b1_p22 = np.copy(events_p22)
leftresponse_b1_p22 = np.copy(events_p22)
rightresponse_b1_p22 = np.copy(events_p22)
lastresponseleft_b1_p22 = np.copy(events_p22)
lastresponseright_b1_p22 = np.copy(events_p22)
preparationleft_b1_p22 = np.copy(events_p22)
preparationright_b1_p22 = np.copy(events_p22)
nogob1p22 = np.copy(events_p22)
#--------B5-------#
laststimpositionleft_b5_p22 = np.copy(eventsB5_p22)
laststimpositionright_b5_p22 = np.copy(eventsB5_p22)
feedbackleft_b5_p22 = np.copy(eventsB5_p22)
feedbackright_b5_p22 = np.copy(eventsB5_p22)
leftresponse_b5_p22 = np.copy(eventsB5_p22)
rightresponse_b5_p22 = np.copy(eventsB5_p22)
lastresponseleft_b5_p22 = np.copy(eventsB5_p22)
lastresponseright_b5_p22 = np.copy(eventsB5_p22)
preparationleft_b5_p22 = np.copy(eventsB5_p22)
preparationright_b5_p22 = np.copy(eventsB5_p22)
nogob5p22 = np.copy(eventsB5_p22)


#print to see if event times are correct

print(events_p22)

#--------B1-------#


#last  stimulus position left (408)
laststimpositionleft_b1_p22 = mne.pick_events(laststimpositionleft_b1_p22, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p22 = mne.merge_events(laststimpositionleft_b1_p22, [5, 6, 7, 8], 408, replace_events=True)
laststimpositionleft_b1_p22 = np.delete(laststimpositionleft_b1_p22, [71, 107, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#last  stimulus position right (409)
laststimpositionright_b1_p22 = mne.pick_events(laststimpositionright_b1_p22, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p22 = mne.merge_events(laststimpositionright_b1_p22, [9, 10, 11, 12], 409, replace_events=True)
laststimpositionright_b1_p22 = np.delete(laststimpositionright_b1_p22, [41, 107, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#feedback left (410)
feedbackleft_b1_p22 = mne.pick_events(feedbackleft_b1_p22, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p22 = mne.merge_events(feedbackleft_b1_p22, [25,26], 410, replace_events=True)
feedbackleft_b1_p22 = mne.event.shift_time_events(feedbackleft_b1_p22, 410, -1.000, 500)
feedbackleft_b1_p22 = np.delete(feedbackleft_b1_p22, [0, 4, 5, 7, 8, 9, 11, 12, 16, 17, 20, 21, 28, 29, 30, 31, 34, 35, 36, 39, 40, 45, 46, 47], axis=0)

#feedback right (411)
feedbackright_b1_p22 = mne.pick_events(feedbackright_b1_p22, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p22 = mne.merge_events(feedbackright_b1_p22, [25,26], 411, replace_events=True)
feedbackright_b1_p22 = mne.event.shift_time_events(feedbackright_b1_p22, 411, -1.000, 500)
feedbackright_b1_p22 = np.delete(feedbackright_b1_p22, [1, 2, 3, 6, 10, 13, 14, 15, 18, 19, 22, 23, 24, 25, 26, 27, 32, 33, 37, 38, 41, 42, 43, 44], axis=0)

#left response (412) 
leftresponse_b1_p22 = mne.pick_events(leftresponse_b1_p22, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p22 = mne.merge_events(leftresponse_b1_p22, [14, 15, 16, 17], 412, replace_events=True)

#right response(413)
rightresponse_b1_p22 = mne.pick_events(rightresponse_b1_p22, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p22 = mne.merge_events(rightresponse_b1_p22, [18, 19, 20, 21], 413, replace_events=True)

#last response left(414)
lastresponseleft_b1_p22 = mne.pick_events(lastresponseleft_b1_p22, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p22 = mne.merge_events(lastresponseleft_b1_p22, [5, 6, 7, 8], 414, replace_events=True)

#last response right(415)
lastresponseright_b1_p22 = mne.pick_events(lastresponseright_b1_p22, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p22 = mne.merge_events(lastresponseright_b1_p22, [9, 10, 11, 12], 415, replace_events=True)


#preperation left (416) 
preparationleft_b1_p22 = mne.pick_events(preparationleft_b1_p22, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p22 = mne.merge_events(preparationleft_b1_p22, [5, 6, 7, 8], 416, replace_events=True)
preparationleft_b1_p22 = mne.event.shift_time_events(preparationleft_b1_p22, 416, 1.500, 500)
preparationleft_b1_p22 = np.delete(preparationleft_b1_p22, [71, 107, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)




#preperation right (417) 
preparationright_b1_p22 = mne.pick_events(preparationright_b1_p22, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p22 = mne.merge_events(preparationright_b1_p22, [9, 10, 11, 12], 417, replace_events=True)
preparationright_b1_p22 = mne.event.shift_time_events(preparationright_b1_p22, 417, 1.500, 500)
preparationright_b1_p22 = np.delete(preparationright_b1_p22, [41, 107, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#nogo (418)
nogob1p22 = mne.pick_events(nogob1p22, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p22 = mne.merge_events(nogob1p22, [24], 418, replace_events=True)
#--------B5-------#


#last  stimulus position left (419)
laststimpositionleft_b5_p22 = mne.pick_events(laststimpositionleft_b5_p22, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p22 = mne.merge_events(laststimpositionleft_b5_p22, [5, 6, 7, 8], 419, replace_events=True)
laststimpositionleft_b5_p22 = np.delete(laststimpositionleft_b5_p22, [23, 35, 113, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)


#last  stimulus position right (420)
laststimpositionright_b5_p22 = mne.pick_events(laststimpositionright_b5_p22, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p22 = mne.merge_events(laststimpositionright_b5_p22, [9, 10, 11, 12], 420, replace_events=True)
laststimpositionright_b5_p22 = np.delete(laststimpositionright_b5_p22, [125, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#feedback (421)
feedbackleft_b5_p22 = mne.pick_events(feedbackleft_b5_p22, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p22 = mne.merge_events(feedbackleft_b5_p22, [25,26], 421, replace_events=True)
feedbackleft_b5_p22 = mne.event.shift_time_events(feedbackleft_b5_p22, 421, -1.000, 500)
feedbackleft_b5_p22 = np.delete(feedbackleft_b5_p22, [0, 1, 4, 5, 6, 7, 8, 13, 16, 20, 21, 22, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 41, 44], axis=0)

#feedback (422)
feedbackright_b5_p22 = mne.pick_events(feedbackright_b5_p22, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p22 = mne.merge_events(feedbackright_b5_p22, [25,26], 422, replace_events=True)
feedbackright_b5_p22 = mne.event.shift_time_events(feedbackright_b5_p22, 422, -1.000, 500)
feedbackright_b5_p22 = np.delete(feedbackright_b5_p22, [2, 3, 9, 10, 11, 12, 14, 15, 17, 18, 19, 23, 24, 25, 26, 33, 38, 39, 40, 42, 43, 45, 46, 47], axis=0)


#left response (423) 
leftresponse_b5_p22 = mne.pick_events(leftresponse_b5_p22, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p22 = mne.merge_events(leftresponse_b5_p22, [14, 15, 16, 17], 423, replace_events=True)

#right response(424)
rightresponse_b5_p22 = mne.pick_events(rightresponse_b5_p22, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p22 = mne.merge_events(rightresponse_b5_p22, [18, 19, 20, 21], 424, replace_events=True)

#last response left(425)
lastresponseleft_b5_p22 = mne.pick_events(lastresponseleft_b5_p22, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p22 = mne.merge_events(lastresponseleft_b5_p22, [5, 6, 7, 8], 425, replace_events=True)

#last response right(426)
lastresponseright_b5_p22 = mne.pick_events(lastresponseright_b5_p22, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p22 = mne.merge_events(lastresponseright_b5_p22, [9, 10, 11, 12], 426, replace_events=True)


#preperation left (427) 
preparationleft_b5_p22 = mne.pick_events(preparationleft_b5_p22, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p22 = mne.merge_events(preparationleft_b5_p22, [5, 6, 7, 8], 427, replace_events=True)
preparationleft_b5_p22 = mne.event.shift_time_events(preparationleft_b5_p22, 427, 1.500, 500)
preparationleft_b5_p22 = np.delete(preparationleft_b5_p22, [23, 35, 113, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)


#preperation right (428) 
preparationright_b5_p22 = mne.pick_events(preparationright_b5_p22, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p22 = mne.merge_events(preparationright_b5_p22, [9, 10, 11, 12], 428, replace_events=True)
preparationright_b5_p22 = mne.event.shift_time_events(preparationright_b5_p22, 428, 1.500, 500)
preparationright_b5_p22 = np.delete(preparationright_b5_p22, [125, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#nogo (429)
nogob5p22 = mne.pick_events(nogob5p22, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p22 = mne.merge_events(nogob5p22, [24], 429, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p22 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p22 = {'laststimleft':408,'laststimright':409, 'feedbackL':410, 'feedbackR':411, 'leftres':412, 'rightres':413, 'lastresleft':414, 'lastresright':415, 'prepleft':416, 'prepright':417, 'nogo': 418}
event_dictB5_p22 = {'laststimleft':419,'laststimright':420, 'feedbackL':421, 'feedbackR':422, 'leftres':423, 'rightres':424, 'lastresleft':425, 'lastresright':426, 'prepleft':427, 'prepright':428, 'nogo': 429}

finalB1_p22 = np.concatenate((laststimpositionleft_b1_p22, laststimpositionright_b1_p22, feedbackleft_b1_p22, feedbackright_b1_p22, leftresponse_b1_p22, rightresponse_b1_p22, lastresponseleft_b1_p22, lastresponseright_b1_p22, preparationleft_b1_p22, preparationright_b1_p22, events_p22, nogob1p22), axis=0)
finalB5_p22 = np.concatenate((laststimpositionleft_b5_p22, laststimpositionright_b5_p22, feedbackleft_b5_p22, feedbackright_b5_p22, leftresponse_b5_p22, rightresponse_b5_p22, lastresponseleft_b5_p22, lastresponseright_b5_p22, preparationleft_b5_p22, preparationright_b5_p22, events_p22, nogob5p22), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p22, event_id=event_dictB1_p22, 

                         sfreq=raw_p22.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p22, event_id=event_dictB5_p22, 

                         sfreq=raw5_p22.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend


#-----------------------------------------------------------------------------#
#                               PARTICIPANT 23
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_23_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part23_SeqL_ERD_6_B1.fif'
Part_23_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part23_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p23 = mne.io.read_raw_fif(Part_23_1, preload = True) 
 

#--------B5--------#
raw5_p23 = mne.io.read_raw_fif(Part_23_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p23, _ = mne.events_from_annotations(raw_p23, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p23, _ = mne.events_from_annotations(raw5_p23, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p23 = np.copy(events_p23)
laststimpositionright_b1_p23 = np.copy(events_p23)
feedbackleft_b1_p23 = np.copy(events_p23)
feedbackright_b1_p23 = np.copy(events_p23)
leftresponse_b1_p23 = np.copy(events_p23)
rightresponse_b1_p23 = np.copy(events_p23)
lastresponseleft_b1_p23 = np.copy(events_p23)
lastresponseright_b1_p23 = np.copy(events_p23)
preparationleft_b1_p23 = np.copy(events_p23)
preparationright_b1_p23 = np.copy(events_p23)
nogob1p23 = np.copy(events_p23)
#--------B5-------#
laststimpositionleft_b5_p23 = np.copy(eventsB5_p23)
laststimpositionright_b5_p23 = np.copy(eventsB5_p23)
feedbackleft_b5_p23 = np.copy(eventsB5_p23)
feedbackright_b5_p23 = np.copy(eventsB5_p23)
leftresponse_b5_p23 = np.copy(eventsB5_p23)
rightresponse_b5_p23 = np.copy(eventsB5_p23)
lastresponseleft_b5_p23 = np.copy(eventsB5_p23)
lastresponseright_b5_p23 = np.copy(eventsB5_p23)
preparationleft_b5_p23 = np.copy(eventsB5_p23)
preparationright_b5_p23 = np.copy(eventsB5_p23)
nogob5p23 = np.copy(eventsB5_p23)


#print to see if event times are correct

print(events_p23)

#--------B1-------#


#last  stimulus position left (430)
laststimpositionleft_b1_p23 = mne.pick_events(laststimpositionleft_b1_p23, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p23 = mne.merge_events(laststimpositionleft_b1_p23, [5, 6, 7, 8], 430, replace_events=True)
laststimpositionleft_b1_p23 = np.delete(laststimpositionleft_b1_p23, [5, 83, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#last  stimulus position right (431)
laststimpositionright_b1_p23 = mne.pick_events(laststimpositionright_b1_p23, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p23 = mne.merge_events(laststimpositionright_b1_p23, [9, 10, 11, 12], 431, replace_events=True)
laststimpositionright_b1_p23 = np.delete(laststimpositionright_b1_p23, [77, 143, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#feedback left (432)
feedbackleft_b1_p23 = mne.pick_events(feedbackleft_b1_p23, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p23 = mne.merge_events(feedbackleft_b1_p23, [25,26], 432, replace_events=True)
feedbackleft_b1_p23 = mne.event.shift_time_events(feedbackleft_b1_p23, 432, -1.000, 500)
feedbackleft_b1_p23 = np.delete(feedbackleft_b1_p23, [0, 1, 4, 6, 9, 10, 12, 14, 15, 20, 21, 23, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47], axis=0)

#feedback right (433)
feedbackright_b1_p23 = mne.pick_events(feedbackright_b1_p23, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p23 = mne.merge_events(feedbackright_b1_p23, [25,26], 433, replace_events=True)
feedbackright_b1_p23 = mne.event.shift_time_events(feedbackright_b1_p23, 433, -1.000, 500)
feedbackright_b1_p23 = np.delete(feedbackright_b1_p23, [2, 3, 5, 7, 8, 11, 13, 16, 17, 18, 19, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 36, 37], axis=0)


#left response (434) 
leftresponse_b1_p23 = mne.pick_events(leftresponse_b1_p23, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p23 = mne.merge_events(leftresponse_b1_p23, [14, 15, 16, 17], 434, replace_events=True)

#right response(435)
rightresponse_b1_p23 = mne.pick_events(rightresponse_b1_p23, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p23 = mne.merge_events(rightresponse_b1_p23, [18, 19, 20, 21], 435, replace_events=True)

#last response left(436)
lastresponseleft_b1_p23 = mne.pick_events(lastresponseleft_b1_p23, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p23 = mne.merge_events(lastresponseleft_b1_p23, [5, 6, 7, 8], 436, replace_events=True)

#last response right(437)
lastresponseright_b1_p23 = mne.pick_events(lastresponseright_b1_p23, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p23 = mne.merge_events(lastresponseright_b1_p23, [9, 10, 11, 12], 437, replace_events=True)


#preperation left (438) 
preparationleft_b1_p23 = mne.pick_events(preparationleft_b1_p23, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p23 = mne.merge_events(preparationleft_b1_p23, [5, 6, 7, 8], 438, replace_events=True)
preparationleft_b1_p23 = mne.event.shift_time_events(preparationleft_b1_p23, 438, 1.500, 500)
preparationleft_b1_p23 = np.delete(preparationleft_b1_p23, [5, 83, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#preperation right (439) 
preparationright_b1_p23 = mne.pick_events(preparationright_b1_p23, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p23 = mne.merge_events(preparationright_b1_p23, [9, 10, 11, 12], 439, replace_events=True)
preparationright_b1_p23 = mne.event.shift_time_events(preparationright_b1_p23, 439, 1.500, 500)
preparationright_b1_p23 = np.delete(preparationright_b1_p23, [77, 143, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#nogo (440)
nogob1p23 = mne.pick_events(nogob1p23, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p23 = mne.merge_events(nogob1p23, [24], 440, replace_events=True)


#--------B5-------#
#last  stimulus position left (441)
laststimpositionleft_b5_p23 = mne.pick_events(laststimpositionleft_b5_p23, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p23 = mne.merge_events(laststimpositionleft_b5_p23, [5, 6, 7, 8], 441, replace_events=True)
laststimpositionleft_b5_p23 = np.delete(laststimpositionleft_b5_p23, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142], axis=0)


#last  stimulus position right (442)
laststimpositionright_b5_p23 = mne.pick_events(laststimpositionright_b5_p23, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p23 = mne.merge_events(laststimpositionright_b5_p23, [9, 10, 11, 12], 442, replace_events=True)
laststimpositionright_b5_p23 = np.delete(laststimpositionright_b5_p23, [41, 77, 149, 155, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166], axis=0)


#feedback (443)
feedbackleft_b5_p23 = mne.pick_events(feedbackleft_b5_p23, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p23 = mne.merge_events(feedbackleft_b5_p23, [25,26], 443, replace_events=True)
feedbackleft_b5_p23 = mne.event.shift_time_events(feedbackleft_b5_p23, 443, -1.000, 500)
feedbackleft_b5_p23 = np.delete(feedbackleft_b5_p23, [0, 2, 3, 5, 6, 10, 13, 14, 15, 16, 19, 22, 26, 28, 29, 30, 31, 33, 35, 36, 37, 38, 40, 43], axis=0)

#feedback (444)
feedbackright_b5_p23 = mne.pick_events(feedbackright_b5_p23, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p23 = mne.merge_events(feedbackright_b5_p23, [25,26], 444, replace_events=True)
feedbackright_b5_p23 = mne.event.shift_time_events(feedbackright_b5_p23, 444, -1.000, 500)
feedbackright_b5_p23 = np.delete(feedbackright_b5_p23, [1, 4, 7, 8, 9, 11, 12, 17, 18, 20, 21, 23, 24, 25, 27, 32, 34, 39, 41, 42, 44, 45, 46, 47], axis=0)

#left response (445) 
leftresponse_b5_p23 = mne.pick_events(leftresponse_b5_p23, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p23 = mne.merge_events(leftresponse_b5_p23, [14, 15, 16, 17], 445, replace_events=True)

#right response(446)
rightresponse_b5_p23 = mne.pick_events(rightresponse_b5_p23, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p23 = mne.merge_events(rightresponse_b5_p23, [18, 19, 20, 21], 446, replace_events=True)

#last response left(447)
lastresponseleft_b5_p23 = mne.pick_events(lastresponseleft_b5_p23, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p23 = mne.merge_events(lastresponseleft_b5_p23, [5, 6, 7, 8], 447, replace_events=True)

#last response right(448)
lastresponseright_b5_p23 = mne.pick_events(lastresponseright_b5_p23, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p23 = mne.merge_events(lastresponseright_b5_p23, [9, 10, 11, 12], 448, replace_events=True)


#preperation left (449) 
preparationleft_b5_p23 = mne.pick_events(preparationleft_b5_p23, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p23 = mne.merge_events(preparationleft_b5_p23, [5, 6, 7, 8], 449, replace_events=True)
preparationleft_b5_p23 = mne.event.shift_time_events(preparationleft_b5_p23, 449, 1.500, 500)
preparationleft_b5_p23 = np.delete(preparationleft_b5_p23, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142], axis=0)



#preperation right (450) 
preparationright_b5_p23 = mne.pick_events(preparationright_b5_p23, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p23 = mne.merge_events(preparationright_b5_p23, [9, 10, 11, 12], 450, replace_events=True)
preparationright_b5_p23 = mne.event.shift_time_events(preparationright_b5_p23, 450, 1.500, 500)
preparationright_b5_p23 = np.delete(preparationright_b5_p23, [41, 77, 149, 155, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166], axis=0)

#nogo (451)
nogob5p23 = mne.pick_events(nogob5p23, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p23 = mne.merge_events(nogob5p23, [24], 451, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p23 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p23 = {'laststimleft':430,'laststimright':431, 'feedbackL':432, 'feedbackR':433, 'leftres':434, 'rightres':435, 'lastresleft':436, 'lastresright':437, 'prepleft':438, 'prepright':439, 'nogo': 440,}
event_dictB5_p23 = {'laststimleft':441,'laststimright':442, 'feedbackL':443, 'feedbackR':444, 'leftres':445, 'rightres':446, 'lastresleft':447, 'lastresright':448, 'prepleft':449, 'prepright':450, 'nogo': 451,}



#merging events togeher into one event list

finalB1_p23 = np.concatenate((laststimpositionleft_b1_p23, laststimpositionright_b1_p23, feedbackleft_b1_p23, feedbackright_b1_p23, leftresponse_b1_p23, rightresponse_b1_p23, lastresponseleft_b1_p23, lastresponseright_b1_p23, preparationleft_b1_p23, preparationright_b1_p23, events_p23, nogob1p23), axis=0)
finalB5_p23 = np.concatenate((laststimpositionleft_b5_p23, laststimpositionright_b5_p23, feedbackleft_b5_p23, feedbackright_b5_p23, leftresponse_b5_p23, rightresponse_b5_p23, lastresponseleft_b5_p23, lastresponseright_b5_p23, preparationleft_b5_p23, preparationright_b5_p23, events_p23, nogob5p23), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p23, event_id=event_dictB1_p23, 

                         sfreq=raw_p23.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p23, event_id=event_dictB5_p23, 

                         sfreq=raw5_p23.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend


#merging events togeher into one event list

#-----------------------------------------------------------------------------#
#                               PARTICIPANT 24
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_24_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part24_SeqL_ERD_6_B1.fif'
Part_24_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part24_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p24 = mne.io.read_raw_fif(Part_24_1, preload = True) 
 

#--------B5--------#
raw5_p24 = mne.io.read_raw_fif(Part_24_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p24, _ = mne.events_from_annotations(raw_p24, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p24, _ = mne.events_from_annotations(raw5_p24, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p24 = np.copy(events_p24)
laststimpositionright_b1_p24 = np.copy(events_p24)
feedbackleft_b1_p24 = np.copy(events_p24)
feedbackright_b1_p24 = np.copy(events_p24)
leftresponse_b1_p24 = np.copy(events_p24)
rightresponse_b1_p24 = np.copy(events_p24)
lastresponseleft_b1_p24 = np.copy(events_p24)
lastresponseright_b1_p24 = np.copy(events_p24)
preparationleft_b1_p24 = np.copy(events_p24)
preparationright_b1_p24 = np.copy(events_p24)
nogob1p24 = np.copy(events_p24)
#--------B5-------#
laststimpositionleft_b5_p24 = np.copy(eventsB5_p24)
laststimpositionright_b5_p24 = np.copy(eventsB5_p24)
feedbackleft_b5_p24 = np.copy(eventsB5_p24)
feedbackright_b5_p24 = np.copy(eventsB5_p24)
leftresponse_b5_p24 = np.copy(eventsB5_p24)
rightresponse_b5_p24 = np.copy(eventsB5_p24)
lastresponseleft_b5_p24 = np.copy(eventsB5_p24)
lastresponseright_b5_p24 = np.copy(eventsB5_p24)
preparationleft_b5_p24 = np.copy(eventsB5_p24)
preparationright_b5_p24 = np.copy(eventsB5_p24)
nogob5p24 = np.copy(eventsB5_p24)


#print to see if event times are correct

print(events_p24)

#--------B1-------#


#last  stimulus position left (452)
laststimpositionleft_b1_p24 = mne.pick_events(laststimpositionleft_b1_p24, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p24 = mne.merge_events(laststimpositionleft_b1_p24, [5, 6, 7, 8], 452, replace_events=True)
laststimpositionleft_b1_p24 = np.delete(laststimpositionleft_b1_p24, [65, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)


#last  stimulus position right (453)
laststimpositionright_b1_p24 = mne.pick_events(laststimpositionright_b1_p24, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p24 = mne.merge_events(laststimpositionright_b1_p24, [9, 10, 11, 12], 453, replace_events=True)
laststimpositionright_b1_p24 = np.delete(laststimpositionright_b1_p24, [23, 113, 119, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)


#feedback left (454)
feedbackleft_b1_p24 = mne.pick_events(feedbackleft_b1_p24, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p24 = mne.merge_events(feedbackleft_b1_p24, [25,26], 454, replace_events=True)
feedbackleft_b1_p24 = mne.event.shift_time_events(feedbackleft_b1_p24, 454, -1.000, 500)
feedbackleft_b1_p24 = np.delete(feedbackleft_b1_p24, [0, 2, 4, 7, 8, 10, 11, 17, 18, 20, 22, 23, 24, 25, 26, 28, 29, 31, 32, 33, 36, 38, 40, 42], axis=0)


#feedback right (455)
feedbackright_b1_p24 = mne.pick_events(feedbackright_b1_p24, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p24 = mne.merge_events(feedbackright_b1_p24, [25,26], 455, replace_events=True)
feedbackright_b1_p24 = mne.event.shift_time_events(feedbackright_b1_p24, 455, -1.000, 500)
feedbackright_b1_p24 = np.delete(feedbackright_b1_p24, [1, 3, 5, 6, 9, 12, 13, 14, 15, 16, 19, 21, 27, 30, 34, 35, 37, 39, 41, 43, 44, 45, 46, 47], axis=0)

#left response (456) 
leftresponse_b1_p24 = mne.pick_events(leftresponse_b1_p24, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p24 = mne.merge_events(leftresponse_b1_p24, [14, 15, 16, 17], 456, replace_events=True)

#right response(457)
rightresponse_b1_p24 = mne.pick_events(rightresponse_b1_p24, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p24 = mne.merge_events(rightresponse_b1_p24, [18, 19, 20, 21], 457, replace_events=True)

#last response left(458)
lastresponseleft_b1_p24 = mne.pick_events(lastresponseleft_b1_p24, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p24 = mne.merge_events(lastresponseleft_b1_p24, [5, 6, 7, 8], 458, replace_events=True)

#last response right(459)
lastresponseright_b1_p24 = mne.pick_events(lastresponseright_b1_p24, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p24 = mne.merge_events(lastresponseright_b1_p24, [9, 10, 11, 12], 459, replace_events=True)


#preperation left (460) 
preparationleft_b1_p24 = mne.pick_events(preparationleft_b1_p24, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p24 = mne.merge_events(preparationleft_b1_p24, [5, 6, 7, 8], 460, replace_events=True)
preparationleft_b1_p24 = mne.event.shift_time_events(preparationleft_b1_p24, 460, 1.500, 500)
preparationleft_b1_p24 = np.delete(preparationleft_b1_p24, [65, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)



#preperation right (461) 
preparationright_b1_p24 = mne.pick_events(preparationright_b1_p24, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p24 = mne.merge_events(preparationright_b1_p24, [9, 10, 11, 12], 461, replace_events=True)
preparationright_b1_p24 = mne.event.shift_time_events(preparationright_b1_p24, 461, 1.500, 500)
preparationright_b1_p24 = np.delete(preparationright_b1_p24, [23, 113, 119, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)

#nogo (462)
nogob1p24 = mne.pick_events(nogob1p24, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p24 = mne.merge_events(nogob1p24, [24], 462, replace_events=True)
#--------B5-------#


#last  stimulus position left (463)
laststimpositionleft_b5_p24 = mne.pick_events(laststimpositionleft_b5_p24, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p24 = mne.merge_events(laststimpositionleft_b5_p24, [5, 6, 7, 8], 463, replace_events=True)
laststimpositionleft_b5_p24 = np.delete(laststimpositionleft_b5_p24, [5, 17, 119, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)

#last  stimulus position right (464)
laststimpositionright_b5_p24 = mne.pick_events(laststimpositionright_b5_p24, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p24 = mne.merge_events(laststimpositionright_b5_p24, [9, 10, 11, 12], 464, replace_events=True)
laststimpositionright_b5_p24 = np.delete(laststimpositionright_b5_p24, [77, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)


#feedback (465)
feedbackleft_b5_p24 = mne.pick_events(feedbackleft_b5_p24, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p24 = mne.merge_events(feedbackleft_b5_p24, [25,26], 465, replace_events=True)
feedbackleft_b5_p24 = mne.event.shift_time_events(feedbackleft_b5_p24, 465, -1.000, 500)
feedbackleft_b5_p24 = np.delete(feedbackleft_b5_p24, [0, 1, 2, 5, 6, 7, 8, 12, 14, 15, 18, 23, 25, 26, 27, 32, 33, 34, 36, 39, 41, 45, 46, 47], axis=0)


#feedback (466)
feedbackright_b5_p24 = mne.pick_events(feedbackright_b5_p24, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p24 = mne.merge_events(feedbackright_b5_p24, [25,26], 466, replace_events=True)
feedbackright_b5_p24 = mne.event.shift_time_events(feedbackright_b5_p24, 466, -1.000, 500)
feedbackright_b5_p24 = np.delete(feedbackright_b5_p24, [3, 4, 9, 10, 11, 13, 16, 17, 19, 20, 21, 22, 24, 28, 29, 30, 31, 35, 37, 38, 40, 42, 43, 44], axis=0)


#left response (467) 
leftresponse_b5_p24 = mne.pick_events(leftresponse_b5_p24, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p24 = mne.merge_events(leftresponse_b5_p24, [14, 15, 16, 17], 467, replace_events=True)

#right response(468)
rightresponse_b5_p24 = mne.pick_events(rightresponse_b5_p24, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p24 = mne.merge_events(rightresponse_b5_p24, [18, 19, 20, 21], 468, replace_events=True)

#last response left(469)
lastresponseleft_b5_p24 = mne.pick_events(lastresponseleft_b5_p24, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p24 = mne.merge_events(lastresponseleft_b5_p24, [5, 6, 7, 8], 469, replace_events=True)

#last response right(470)
lastresponseright_b5_p24 = mne.pick_events(lastresponseright_b5_p24, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p24 = mne.merge_events(lastresponseright_b5_p24, [9, 10, 11, 12], 470, replace_events=True)


#preperation left (471) 
preparationleft_b5_p24 = mne.pick_events(preparationleft_b5_p24, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p24 = mne.merge_events(preparationleft_b5_p24, [5, 6, 7, 8], 471, replace_events=True)
preparationleft_b5_p24 = mne.event.shift_time_events(preparationleft_b5_p24, 471, 1.500, 500)
preparationleft_b5_p24 = np.delete(preparationleft_b5_p24, [5, 17, 119, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)


#preperation right (472) 
preparationright_b5_p24 = mne.pick_events(preparationright_b5_p24, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p24 = mne.merge_events(preparationright_b5_p24, [9, 10, 11, 12], 472, replace_events=True)
preparationright_b5_p24 = mne.event.shift_time_events(preparationright_b5_p24, 472, 1.500, 500)
preparationright_b5_p24 = np.delete(preparationright_b5_p24, [77, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#nogo (473)
nogob5p24 = mne.pick_events(nogob5p24, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p24 = mne.merge_events(nogob5p24, [24], 473, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p24 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p24 = {'laststimleft':452,'laststimright':453, 'feedbackL':454, 'feedbackR':455, 'leftres':456, 'rightres':457, 'lastresleft':458, 'lastresright':459, 'prepleft':460, 'prepright':461, 'nogo': 462}
event_dictB5_p24 = {'laststimleft':463,'laststimright':464, 'feedbackL':465, 'feedbackR':466, 'leftres':467, 'rightres':468, 'lastresleft':469, 'lastresright':470, 'prepleft':471, 'prepright':472, 'nogo': 473}



#merging events togeher into one event list

finalB1_p24 = np.concatenate((laststimpositionleft_b1_p24, laststimpositionright_b1_p24, feedbackleft_b1_p24, feedbackright_b1_p24, leftresponse_b1_p24, rightresponse_b1_p24, lastresponseleft_b1_p24, lastresponseright_b1_p24, preparationleft_b1_p24, preparationright_b1_p24, events_p24, nogob1p24), axis=0)
finalB5_p24 = np.concatenate((laststimpositionleft_b5_p24, laststimpositionright_b5_p24, feedbackleft_b5_p24, feedbackright_b5_p24, leftresponse_b5_p24, rightresponse_b5_p24, lastresponseleft_b5_p24, lastresponseright_b5_p24, preparationleft_b5_p24, preparationright_b5_p24, events_p24, nogob5p24), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p24, event_id=event_dictB1_p24, 

                         sfreq=raw_p24.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p24, event_id=event_dictB5_p24, 

                         sfreq=raw5_p24.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend


#-----------------------------------------------------------------------------#
#                               PARTICIPANT 25
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_25_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part25_SeqL_ERD_6_B1.fif'
Part_25_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part25_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p25 = mne.io.read_raw_fif(Part_25_1, preload = True) 
 

#--------B5--------#
raw5_p25 = mne.io.read_raw_fif(Part_25_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p25, _ = mne.events_from_annotations(raw_p25, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p25, _ = mne.events_from_annotations(raw5_p25, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p25 = np.copy(events_p25)
laststimpositionright_b1_p25 = np.copy(events_p25)
feedbackleft_b1_p25 = np.copy(events_p25)
feedbackright_b1_p25 = np.copy(events_p25)
leftresponse_b1_p25 = np.copy(events_p25)
rightresponse_b1_p25 = np.copy(events_p25)
lastresponseleft_b1_p25 = np.copy(events_p25)
lastresponseright_b1_p25 = np.copy(events_p25)
preparationleft_b1_p25 = np.copy(events_p25)
preparationright_b1_p25 = np.copy(events_p25)
nogob1p25 = np.copy(events_p25)
#--------B5-------#
laststimpositionleft_b5_p25 = np.copy(eventsB5_p25)
laststimpositionright_b5_p25 = np.copy(eventsB5_p25)
feedbackleft_b5_p25 = np.copy(eventsB5_p25)
feedbackright_b5_p25 = np.copy(eventsB5_p25)
leftresponse_b5_p25 = np.copy(eventsB5_p25)
rightresponse_b5_p25 = np.copy(eventsB5_p25)
lastresponseleft_b5_p25 = np.copy(eventsB5_p25)
lastresponseright_b5_p25 = np.copy(eventsB5_p25)
preparationleft_b5_p25 = np.copy(eventsB5_p25)
preparationright_b5_p25 = np.copy(eventsB5_p25)
nogob5p25 = np.copy(eventsB5_p25)


#print to see if event times are correct

print(events_p25)

#--------B1-------#


#last  stimulus position left (474)
laststimpositionleft_b1_p25 = mne.pick_events(laststimpositionleft_b1_p25, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p25 = mne.merge_events(laststimpositionleft_b1_p25, [5, 6, 7, 8], 474, replace_events=True)
laststimpositionleft_b1_p25 = np.delete(laststimpositionleft_b1_p25, [125, 137, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#last  stimulus position right (475)
laststimpositionright_b1_p25 = mne.pick_events(laststimpositionright_b1_p25, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p25 = mne.merge_events(laststimpositionright_b1_p25, [9, 10, 11, 12], 475, replace_events=True)
laststimpositionright_b1_p25 = np.delete(laststimpositionright_b1_p25, [6, 85, 0, 79, 146, 147, 1,	2,	3,	4,	5, 7,	8,	9,	10,	11, 13,	14,	15,	16,	17, 19,	20,	21,	22,	23, 25,	26,	27,	28,	29, 31,	32,	33,	34,	35, 37,	38,	39,	40,	41, 43,	44,	45,	46,	47, 49,	50,	51,	52,	53, 55,	56,	57,	58,	59, 61,	62,	63,	64,	65, 67,	68,	69,	70,	71, 73,	74,	75,	76,	77, 80,	81,	82,	83,	84, 86,	87,	88,	89,	90, 92,	93,	94,	95,	96, 98,	99,	100, 101,	102, 104, 105, 106, 107, 108, 110, 111, 112, 113, 114, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 128, 129, 130, 131, 132, 134, 135, 136, 137, 138, 140, 141, 142, 143, 144, 148, 149, 150, 151, 152, 154, 155, 156, 157, 158], axis=0)

#feedback left (476)
feedbackleft_b1_p25 = mne.pick_events(feedbackleft_b1_p25, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p25 = mne.merge_events(feedbackleft_b1_p25, [25,26], 476, replace_events=True)
feedbackleft_b1_p25 = mne.event.shift_time_events(feedbackleft_b1_p25, 476, -1.000, 500)
feedbackleft_b1_p25 = np.delete(feedbackleft_b1_p25, [3, 4, 6, 7, 9, 11, 13, 14, 16, 18, 19, 20, 24, 26, 28, 29, 30, 31, 32, 34, 38, 40, 43, 44], axis=0)


#feedback right (477)
feedbackright_b1_p25 = mne.pick_events(feedbackright_b1_p25, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p25 = mne.merge_events(feedbackright_b1_p25, [25,26], 477, replace_events=True)
feedbackright_b1_p25 = mne.event.shift_time_events(feedbackright_b1_p25, 477, -1.000, 500)
feedbackright_b1_p25 = np.delete(feedbackright_b1_p25, [0, 1, 2, 5, 8, 10, 12, 15, 17, 21, 22, 23, 25, 27, 33, 35, 36, 37, 39, 41, 42, 45, 46, 47], axis=0)

#left response (478) 
leftresponse_b1_p25 = mne.pick_events(leftresponse_b1_p25, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p25 = mne.merge_events(leftresponse_b1_p25, [14, 15, 16, 17], 478, replace_events=True)

#right response(479)
rightresponse_b1_p25 = mne.pick_events(rightresponse_b1_p25, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p25 = mne.merge_events(rightresponse_b1_p25, [18, 19, 20, 21], 479, replace_events=True)

#last response left(480)
lastresponseleft_b1_p25 = mne.pick_events(lastresponseleft_b1_p25, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p25 = mne.merge_events(lastresponseleft_b1_p25, [5, 6, 7, 8], 480, replace_events=True)

#last response right(481)
lastresponseright_b1_p25 = mne.pick_events(lastresponseright_b1_p25, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p25 = mne.merge_events(lastresponseright_b1_p25, [9, 10, 11, 12], 481, replace_events=True)


#preperation left (482) 
preparationleft_b1_p25 = mne.pick_events(preparationleft_b1_p25, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p25 = mne.merge_events(preparationleft_b1_p25, [5, 6, 7, 8], 482, replace_events=True)
preparationleft_b1_p25 = mne.event.shift_time_events(preparationleft_b1_p25, 482, 1.500, 500)
preparationleft_b1_p25 = np.delete(preparationleft_b1_p25, [125, 137, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#preperation right (483) 
preparationright_b1_p25 = mne.pick_events(preparationright_b1_p25, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p25 = mne.merge_events(preparationright_b1_p25, [9, 10, 11, 12], 483, replace_events=True)
preparationright_b1_p25 = mne.event.shift_time_events(preparationright_b1_p25, 483, 1.500, 500)
preparationright_b1_p25  = np.delete(preparationright_b1_p25, [6, 85, 0, 79, 146, 147, 1,	2,	3,	4,	5, 7,	8,	9,	10,	11, 13,	14,	15,	16,	17, 19,	20,	21,	22,	23, 25,	26,	27,	28,	29, 31,	32,	33,	34,	35, 37,	38,	39,	40,	41, 43,	44,	45,	46,	47, 49,	50,	51,	52,	53, 55,	56,	57,	58,	59, 61,	62,	63,	64,	65, 67,	68,	69,	70,	71, 73,	74,	75,	76,	77, 80,	81,	82,	83,	84, 86,	87,	88,	89,	90, 92,	93,	94,	95,	96, 98,	99,	100, 101,	102, 104, 105, 106, 107, 108, 110, 111, 112, 113, 114, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 128, 129, 130, 131, 132, 134, 135, 136, 137, 138, 140, 141, 142, 143, 144, 148, 149, 150, 151, 152, 154, 155, 156, 157, 158], axis=0)

#nogo (484)
nogob1p25 = mne.pick_events(nogob1p25, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p25 = mne.merge_events(nogob1p25, [24], 484, replace_events=True)
#--------B5-------#


#last  stimulus position left (485)
laststimpositionleft_b5_p25 = mne.pick_events(laststimpositionleft_b5_p25, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p25 = mne.merge_events(laststimpositionleft_b5_p25, [5, 6, 7, 8], 485, replace_events=True)
laststimpositionleft_b5_p25 = np.delete(laststimpositionleft_b5_p25, [71, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#last  stimulus position right (486)
laststimpositionright_b5_p25 = mne.pick_events(laststimpositionright_b5_p25, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p25 = mne.merge_events(laststimpositionright_b5_p25, [9, 10, 11, 12], 486, replace_events=True)
laststimpositionright_b5_p25 = np.delete(laststimpositionright_b5_p25, [42, 104, 117, 36, 79, 98, 111, 0,	1,	2,	3,	4, 6,	7,	8,	9,	10, 12,	13,	14,	15,	16, 18,	19,	20,	21,	22, 24,	25,	26,	27,	28, 30,	31,	32,	33,	34, 37,	38,	39,	40,	41, 43,	44,	45,	46,	47, 49,	50,	51,	52,	53, 55,	56,	57,	58,	59, 61,	62,	63,	64,	65, 67,	68,	69,	70,	71, 73,	74,	75,	76,	77, 80,	81,	82,	83,	84, 86,	87,	88,	89,	90, 92,	93,	94,	95,	96, 99,	100,	101,	102,	103, 105,	106,	107,	108,	109,112,	113,	114,	115,	116, 118,	119,	120,	121,	122, 124,	125,	126,	127,	128, 130,	131,	132,	133,	134, 136,	137,	138,	139,	140, 142,	143,	144,	145,	146, 148,	149,	150,	151,	152, 154,	155,	156,	157,	158, 160,	161,	162,	163,	164], axis=0)



#feedback (487)
feedbackleft_b5_p25 = mne.pick_events(feedbackleft_b5_p25, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p25 = mne.merge_events(feedbackleft_b5_p25, [25,26], 487, replace_events=True)
feedbackleft_b5_p25 = mne.event.shift_time_events(feedbackleft_b5_p25, 487, -1.000, 500)
feedbackleft_b5_p25 = np.delete(feedbackleft_b5_p25, [0, 1, 2, 3, 4, 9, 14, 15, 18, 19, 20, 22, 27, 29, 30, 33, 34, 35, 37, 38, 42, 43, 44, 46], axis=0)


#feedback (488)
feedbackright_b5_p25 = mne.pick_events(feedbackright_b5_p25, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p25 = mne.merge_events(feedbackright_b5_p25, [25,26], 488, replace_events=True)
feedbackright_b5_p25 = mne.event.shift_time_events(feedbackright_b5_p25, 488, -1.000, 500)
feedbackright_b5_p25 = np.delete(feedbackright_b5_p25, [5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 21, 23, 24, 25, 26, 28, 31, 32, 36, 39, 40, 41, 45, 47], axis=0)


#left response (489) 
leftresponse_b5_p25 = mne.pick_events(leftresponse_b5_p25, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p25 = mne.merge_events(leftresponse_b5_p25, [14, 15, 16, 17], 489, replace_events=True)

#right response(490)
rightresponse_b5_p25 = mne.pick_events(rightresponse_b5_p25, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p25 = mne.merge_events(rightresponse_b5_p25, [18, 19, 20, 21], 490, replace_events=True)

#last response left(491)
lastresponseleft_b5_p25 = mne.pick_events(lastresponseleft_b5_p25, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p25 = mne.merge_events(lastresponseleft_b5_p25, [5, 6, 7, 8], 491, replace_events=True)

#last response right(492)
lastresponseright_b5_p25 = mne.pick_events(lastresponseright_b5_p25, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p25 = mne.merge_events(lastresponseright_b5_p25, [9, 10, 11, 12], 492, replace_events=True)


#preperation left (493) 
preparationleft_b5_p25 = mne.pick_events(preparationleft_b5_p25, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p25 = mne.merge_events(preparationleft_b5_p25, [5, 6, 7, 8], 493, replace_events=True)
preparationleft_b5_p25 = mne.event.shift_time_events(preparationleft_b5_p25, 493, 1.500, 500)
preparationleft_b5_p25 = np.delete(preparationleft_b5_p25, [71, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#preperation right (494) 
preparationright_b5_p25 = mne.pick_events(preparationright_b5_p25, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p25 = mne.merge_events(preparationright_b5_p25, [9, 10, 11, 12], 494, replace_events=True)
preparationright_b5_p25 = mne.event.shift_time_events(preparationright_b5_p25, 494, 1.500, 500)
preparationright_b5_p25 = np.delete(preparationright_b5_p25, [42, 104, 117, 36, 79, 98, 111, 0,	1,	2,	3,	4, 6,	7,	8,	9,	10, 12,	13,	14,	15,	16, 18,	19,	20,	21,	22, 24,	25,	26,	27,	28, 30,	31,	32,	33,	34, 37,	38,	39,	40,	41, 43,	44,	45,	46,	47, 49,	50,	51,	52,	53, 55,	56,	57,	58,	59, 61,	62,	63,	64,	65, 67,	68,	69,	70,	71, 73,	74,	75,	76,	77, 80,	81,	82,	83,	84, 86,	87,	88,	89,	90, 92,	93,	94,	95,	96, 99,	100,	101,	102,	103, 105,	106,	107,	108,	109,112,	113,	114,	115,	116, 118,	119,	120,	121,	122, 124,	125,	126,	127,	128, 130,	131,	132,	133,	134, 136,	137,	138,	139,	140, 142,	143,	144,	145,	146, 148,	149,	150,	151,	152, 154,	155,	156,	157,	158, 160,	161,	162,	163,	164], axis=0)

#nogo (495)
nogob5p25 = mne.pick_events(nogob5p25, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p25 = mne.merge_events(nogob5p25, [24], 495, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p25 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p25 = {'laststimleft':474,'laststimright':475, 'feedbackL':476, 'feedbackR':477, 'leftres':478, 'rightres':479, 'lastresleft':480, 'lastresright':481, 'prepleft':482, 'prepright':483, 'nogo': 484,}
event_dictB5_p25 = {'laststimleft':485,'laststimright':486, 'feedbackL':487, 'feedbackR':488, 'leftres':489, 'rightres':490, 'lastresleft':491, 'lastresright':492, 'prepleft':493, 'prepright':494, 'nogo': 495,}



#merging events togeher into one event list

finalB1_p25 = np.concatenate((laststimpositionleft_b1_p25, laststimpositionright_b1_p25, feedbackleft_b1_p25, feedbackright_b1_p25, leftresponse_b1_p25, rightresponse_b1_p25, lastresponseleft_b1_p25, lastresponseright_b1_p25, preparationleft_b1_p25, preparationright_b1_p25, events_p25, nogob1p25), axis=0)
finalB5_p25 = np.concatenate((laststimpositionleft_b5_p25, laststimpositionright_b5_p25, feedbackleft_b5_p25, feedbackright_b5_p25, leftresponse_b5_p25, rightresponse_b5_p25, lastresponseleft_b5_p25, lastresponseright_b5_p25, preparationleft_b5_p25, preparationright_b5_p25, events_p25, nogob5p25), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p25, event_id=event_dictB1_p25, 

                         sfreq=raw_p25.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p25, event_id=event_dictB5_p25, 

                         sfreq=raw5_p25.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 

#-----------------------------------------------------------------------------#
#                               PARTICIPANT 26
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_26_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part26_SeqL_ERD_6_B1.fif'
Part_26_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part26_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p26 = mne.io.read_raw_fif(Part_26_1, preload = True) 
 

#--------B5--------#
raw5_p26 = mne.io.read_raw_fif(Part_26_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p26, _ = mne.events_from_annotations(raw_p26, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p26, _ = mne.events_from_annotations(raw5_p26, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p26 = np.copy(events_p26)
laststimpositionright_b1_p26 = np.copy(events_p26)
feedbackleft_b1_p26 = np.copy(events_p26)
feedbackright_b1_p26 = np.copy(events_p26)
leftresponse_b1_p26 = np.copy(events_p26)
rightresponse_b1_p26 = np.copy(events_p26)
lastresponseleft_b1_p26 = np.copy(events_p26)
lastresponseright_b1_p26 = np.copy(events_p26)
preparationleft_b1_p26 = np.copy(events_p26)
preparationright_b1_p26 = np.copy(events_p26)
nogob1p26 = np.copy(events_p26)
#--------B5-------#
laststimpositionleft_b5_p26 = np.copy(eventsB5_p26)
laststimpositionright_b5_p26 = np.copy(eventsB5_p26)
feedbackleft_b5_p26 = np.copy(eventsB5_p26)
feedbackright_b5_p26 = np.copy(eventsB5_p26)
leftresponse_b5_p26 = np.copy(eventsB5_p26)
rightresponse_b5_p26 = np.copy(eventsB5_p26)
lastresponseleft_b5_p26 = np.copy(eventsB5_p26)
lastresponseright_b5_p26 = np.copy(eventsB5_p26)
preparationleft_b5_p26 = np.copy(eventsB5_p26)
preparationright_b5_p26 = np.copy(eventsB5_p26)
nogob5p26 = np.copy(eventsB5_p26)


#print to see if event times are correct

print(events_p26)

#--------B1-------#


#last  stimulus position left (496)
laststimpositionleft_b1_p26 = mne.pick_events(laststimpositionleft_b1_p26, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p26 = mne.merge_events(laststimpositionleft_b1_p26, [5, 6, 7, 8], 496, replace_events=True)
laststimpositionleft_b1_p26 = np.delete(laststimpositionleft_b1_p26, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142], axis=0)


#last  stimulus position right (497)
laststimpositionright_b1_p26 = mne.pick_events(laststimpositionright_b1_p26, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p26 = mne.merge_events(laststimpositionright_b1_p26, [9, 10, 11, 12], 497, replace_events=True)
laststimpositionright_b1_p26 = np.delete(laststimpositionright_b1_p26, [17, 65, 125, 161, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166], axis=0)


#feedback left (498)
feedbackleft_b1_p26 = mne.pick_events(feedbackleft_b1_p26, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p26 = mne.merge_events(feedbackleft_b1_p26, [25,26], 498, replace_events=True)
feedbackleft_b1_p26 = mne.event.shift_time_events(feedbackleft_b1_p26, 498, -1.000, 500)
feedbackleft_b1_p26 = np.delete(feedbackleft_b1_p26, [0, 1, 5, 12, 15, 16, 18, 19, 20, 21, 22, 23, 27, 28, 31, 32, 33, 35, 37, 38, 40, 41, 43, 45], axis=0)


#feedback right (499)
feedbackright_b1_p26 = mne.pick_events(feedbackright_b1_p26, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p26 = mne.merge_events(feedbackright_b1_p26, [25,26], 499, replace_events=True)
feedbackright_b1_p26 = mne.event.shift_time_events(feedbackright_b1_p26, 499, -1.000, 500)
feedbackright_b1_p26 = np.delete(feedbackright_b1_p26, [2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 17, 24, 25, 26, 29, 30, 34, 36, 39, 42, 44, 46, 47], axis=0)

#left response (500) 
leftresponse_b1_p26 = mne.pick_events(leftresponse_b1_p26, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p26 = mne.merge_events(leftresponse_b1_p26, [14, 15, 16, 17], 500, replace_events=True)

#right response(501)
rightresponse_b1_p26 = mne.pick_events(rightresponse_b1_p26, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p26 = mne.merge_events(rightresponse_b1_p26, [18, 19, 20, 21], 501, replace_events=True)

#last response left(502)
lastresponseleft_b1_p26 = mne.pick_events(lastresponseleft_b1_p26, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p26 = mne.merge_events(lastresponseleft_b1_p26, [5, 6, 7, 8], 502, replace_events=True)

#last response right(503)
lastresponseright_b1_p26 = mne.pick_events(lastresponseright_b1_p26, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p26 = mne.merge_events(lastresponseright_b1_p26, [9, 10, 11, 12], 503, replace_events=True)


#preperation left (504) 
preparationleft_b1_p26 = mne.pick_events(preparationleft_b1_p26, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p26 = mne.merge_events(preparationleft_b1_p26, [5, 6, 7, 8], 504, replace_events=True)
preparationleft_b1_p26 = mne.event.shift_time_events(preparationleft_b1_p26, 504, 1.500, 500)
preparationleft_b1_p26 = np.delete(preparationleft_b1_p26, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142], axis=0)



#preperation right (505) 
preparationright_b1_p26 = mne.pick_events(preparationright_b1_p26, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p26 = mne.merge_events(preparationright_b1_p26, [9, 10, 11, 12], 505, replace_events=True)
preparationright_b1_p26 = mne.event.shift_time_events(preparationright_b1_p26, 505, 1.500, 500)
preparationright_b1_p26 = np.delete(preparationright_b1_p26, [17, 65, 125, 161, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166], axis=0)

#nogo (506)
nogob1p26 = mne.pick_events(nogob1p26, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p26 = mne.merge_events(nogob1p26, [24], 506, replace_events=True)
#--------B5-------#


#last  stimulus position left (507)
laststimpositionleft_b5_p26 = mne.pick_events(laststimpositionleft_b5_p26, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p26 = mne.merge_events(laststimpositionleft_b5_p26, [5, 6, 7, 8], 507, replace_events=True)
laststimpositionleft_b5_p26 = np.delete(laststimpositionleft_b5_p26, [29, 101, 119, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)


#last  stimulus position right (508)
laststimpositionright_b5_p26 = mne.pick_events(laststimpositionright_b5_p26, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p26 = mne.merge_events(laststimpositionright_b5_p26, [9, 10, 11, 12], 508, replace_events=True)
laststimpositionright_b5_p26 = np.delete(laststimpositionright_b5_p26, [41, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)


#feedback (509)
feedbackleft_b5_p26 = mne.pick_events(feedbackleft_b5_p26, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p26 = mne.merge_events(feedbackleft_b5_p26, [25,26], 509, replace_events=True)
feedbackleft_b5_p26 = mne.event.shift_time_events(feedbackleft_b5_p26, 509, -1.000, 500)
feedbackleft_b5_p26 = np.delete(feedbackleft_b5_p26, [1, 2, 4, 5, 6, 7, 8, 11, 12, 17, 18, 19, 24, 26, 27, 29, 31, 32, 33, 34, 37, 38, 39, 40], axis=0)

#feedback (510)
feedbackright_b5_p26 = mne.pick_events(feedbackright_b5_p26, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p26 = mne.merge_events(feedbackright_b5_p26, [25,26], 510, replace_events=True)
feedbackright_b5_p26 = mne.event.shift_time_events(feedbackright_b5_p26, 510, -1.000, 500)
feedbackright_b5_p26 = np.delete(feedbackright_b5_p26, [0, 3, 9, 10, 13, 14, 15, 16, 20, 21, 22, 23, 25, 28, 30, 35, 36, 41, 42, 43, 44, 45, 46, 47], axis=0)

#left response (511) 
leftresponse_b5_p26 = mne.pick_events(leftresponse_b5_p26, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p26 = mne.merge_events(leftresponse_b5_p26, [14, 15, 16, 17], 511, replace_events=True)

#right response(512)
rightresponse_b5_p26 = mne.pick_events(rightresponse_b5_p26, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p26 = mne.merge_events(rightresponse_b5_p26, [18, 19, 20, 21], 512, replace_events=True)

#last response left(513)
lastresponseleft_b5_p26 = mne.pick_events(lastresponseleft_b5_p26, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p26 = mne.merge_events(lastresponseleft_b5_p26, [5, 6, 7, 8], 513, replace_events=True)

#last response right(514)
lastresponseright_b5_p26 = mne.pick_events(lastresponseright_b5_p26, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p26 = mne.merge_events(lastresponseright_b5_p26, [9, 10, 11, 12], 514, replace_events=True)


#preperation left (515) 
preparationleft_b5_p26 = mne.pick_events(preparationleft_b5_p26, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p26 = mne.merge_events(preparationleft_b5_p26, [5, 6, 7, 8], 515, replace_events=True)
preparationleft_b5_p26 = mne.event.shift_time_events(preparationleft_b5_p26, 515, 1.500, 500)
preparationleft_b5_p26 = np.delete(preparationleft_b5_p26, [29, 101, 119, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)



#preperation right (516) 
preparationright_b5_p26 = mne.pick_events(preparationright_b5_p26, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p26 = mne.merge_events(preparationright_b5_p26, [9, 10, 11, 12], 516, replace_events=True)
preparationright_b5_p26 = mne.event.shift_time_events(preparationright_b5_p26, 516, 1.500, 500)
preparationright_b5_p26 = np.delete(preparationright_b5_p26, [41, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#nogo (517)
nogob5p26 = mne.pick_events(nogob5p26, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p26 = mne.merge_events(nogob5p26, [24], 517, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p26 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p26 = {'laststimleft':496,'laststimright':497, 'feedbackL':498, 'feedbackR':499, 'leftres':500, 'rightres':501, 'lastresleft':502, 'lastresright':503, 'prepleft':504, 'prepright':505, 'nogo': 506,}
event_dictB5_p26 = {'laststimleft':507,'laststimright':508, 'feedbackL':509, 'feedbackR':510, 'leftres':511, 'rightres':512, 'lastresleft':513, 'lastresright':514, 'prepleft':515, 'prepright':516, 'nogo': 517,}



#merging events togeher into one event list

finalB1_p26 = np.concatenate((laststimpositionleft_b1_p26, laststimpositionright_b1_p26, feedbackleft_b1_p26, feedbackright_b1_p26, leftresponse_b1_p26, rightresponse_b1_p26, lastresponseleft_b1_p26, lastresponseright_b1_p26, preparationleft_b1_p26, preparationright_b1_p26, events_p26, nogob1p26), axis=0)
finalB5_p26 = np.concatenate((laststimpositionleft_b5_p26, laststimpositionright_b5_p26, feedbackleft_b5_p26, feedbackright_b5_p26, leftresponse_b5_p26, rightresponse_b5_p26, lastresponseleft_b5_p26, lastresponseright_b5_p26, preparationleft_b5_p26, preparationright_b5_p26, events_p26, nogob5p26), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p26, event_id=event_dictB1_p26, 

                         sfreq=raw_p26.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p26, event_id=event_dictB5_p26, 

                         sfreq=raw5_p26.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend


#-----------------------------------------------------------------------------#
#                               PARTICIPANT 27
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_27_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part27_SeqL_ERD_6_B1.fif'
Part_27_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part27_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p27 = mne.io.read_raw_fif(Part_27_1, preload = True) 
 

#--------B5--------#
raw5_p27 = mne.io.read_raw_fif(Part_27_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p27, _ = mne.events_from_annotations(raw_p27, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p27, _ = mne.events_from_annotations(raw5_p27, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p27 = np.copy(events_p27)
laststimpositionright_b1_p27 = np.copy(events_p27)
feedbackleft_b1_p27 = np.copy(events_p27)
feedbackright_b1_p27 = np.copy(events_p27)
leftresponse_b1_p27 = np.copy(events_p27)
rightresponse_b1_p27 = np.copy(events_p27)
lastresponseleft_b1_p27 = np.copy(events_p27)
lastresponseright_b1_p27 = np.copy(events_p27)
preparationleft_b1_p27 = np.copy(events_p27)
preparationright_b1_p27 = np.copy(events_p27)
nogob1p27 = np.copy(events_p27)
#--------B5-------#
laststimpositionleft_b5_p27 = np.copy(eventsB5_p27)
laststimpositionright_b5_p27 = np.copy(eventsB5_p27)
feedbackleft_b5_p27 = np.copy(eventsB5_p27)
feedbackright_b5_p27 = np.copy(eventsB5_p27)
leftresponse_b5_p27 = np.copy(eventsB5_p27)
rightresponse_b5_p27 = np.copy(eventsB5_p27)
lastresponseleft_b5_p27 = np.copy(eventsB5_p27)
lastresponseright_b5_p27 = np.copy(eventsB5_p27)
preparationleft_b5_p27 = np.copy(eventsB5_p27)
preparationright_b5_p27 = np.copy(eventsB5_p27)
nogob5p27 = np.copy(eventsB5_p27)


#print to see if event times are correct

print(events_p27)

#--------B1-------#


#last  stimulus position left (518)
laststimpositionleft_b1_p27 = mne.pick_events(laststimpositionleft_b1_p27, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p27 = mne.merge_events(laststimpositionleft_b1_p27, [5, 6, 7, 8], 518, replace_events=True)
laststimpositionleft_b1_p27 = np.delete(laststimpositionleft_b1_p27, [71, 89, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#last  stimulus position right (519)
laststimpositionright_b1_p27 = mne.pick_events(laststimpositionright_b1_p27, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p27 = mne.merge_events(laststimpositionright_b1_p27, [9, 10, 11, 12], 519, replace_events=True)
laststimpositionright_b1_p27 = np.delete(laststimpositionright_b1_p27, [5, 89, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#feedback left (520)
feedbackleft_b1_p27 = mne.pick_events(feedbackleft_b1_p27, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p27 = mne.merge_events(feedbackleft_b1_p27, [25,26], 520, replace_events=True)
feedbackleft_b1_p27 = mne.event.shift_time_events(feedbackleft_b1_p27, 520, -1.000, 500)
feedbackleft_b1_p27 = np.delete(feedbackleft_b1_p27, [1, 2, 3, 5, 8, 9, 10, 11, 13, 15, 16, 18, 27, 28, 33, 34, 35, 38, 39, 40, 43, 45, 46, 47], axis=0)


#feedback right (521)
feedbackright_b1_p27 = mne.pick_events(feedbackright_b1_p27, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p27 = mne.merge_events(feedbackright_b1_p27, [25,26], 521, replace_events=True)
feedbackright_b1_p27 = mne.event.shift_time_events(feedbackright_b1_p27, 521, -1.000, 500)
feedbackright_b1_p27 = np.delete(feedbackright_b1_p27, [0, 4, 6, 7, 12, 14, 17, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 31, 32, 36, 37, 41, 42, 44], axis=0)


#left response (522) 
leftresponse_b1_p27 = mne.pick_events(leftresponse_b1_p27, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p27 = mne.merge_events(leftresponse_b1_p27, [14, 15, 16, 17], 522, replace_events=True)

#right response(523)
rightresponse_b1_p27 = mne.pick_events(rightresponse_b1_p27, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p27 = mne.merge_events(rightresponse_b1_p27, [18, 19, 20, 21], 523, replace_events=True)

#last response left(524)
lastresponseleft_b1_p27 = mne.pick_events(lastresponseleft_b1_p27, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p27 = mne.merge_events(lastresponseleft_b1_p27, [5, 6, 7, 8], 524, replace_events=True)

#last response right(525)
lastresponseright_b1_p27 = mne.pick_events(lastresponseright_b1_p27, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p27 = mne.merge_events(lastresponseright_b1_p27, [9, 10, 11, 12], 525, replace_events=True)


#preperation left (526) 
preparationleft_b1_p27 = mne.pick_events(preparationleft_b1_p27, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p27 = mne.merge_events(preparationleft_b1_p27, [5, 6, 7, 8], 526, replace_events=True)
preparationleft_b1_p27 = mne.event.shift_time_events(preparationleft_b1_p27, 526, 1.500, 500)
preparationleft_b1_p27 = np.delete(preparationleft_b1_p27, [71, 89, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#preperation right (527) 
preparationright_b1_p27 = mne.pick_events(preparationright_b1_p27, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p27 = mne.merge_events(preparationright_b1_p27, [9, 10, 11, 12], 527, replace_events=True)
preparationright_b1_p27 = mne.event.shift_time_events(preparationright_b1_p27, 527, 1.500, 500)
preparationright_b1_p27 = np.delete(preparationright_b1_p27, [5, 89, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#nogo (528)
nogob1p27 = mne.pick_events(nogob1p27, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p27 = mne.merge_events(nogob1p27, [24], 528, replace_events=True)
#--------B5-------#


#last  stimulus position left (529)
laststimpositionleft_b5_p27 = mne.pick_events(laststimpositionleft_b5_p27, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p27 = mne.merge_events(laststimpositionleft_b5_p27, [5, 6, 7, 8], 529, replace_events=True)
laststimpositionleft_b5_p27 = np.delete(laststimpositionleft_b5_p27, [29, 59, 143, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)


#last  stimulus position right (530)
laststimpositionright_b5_p27 = mne.pick_events(laststimpositionright_b5_p27, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p27 = mne.merge_events(laststimpositionright_b5_p27, [9, 10, 11, 12], 530, replace_events=True)
laststimpositionright_b5_p27 = np.delete(laststimpositionright_b5_p27, [83, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)


#feedback (531)
feedbackleft_b5_p27 = mne.pick_events(feedbackleft_b5_p27, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p27 = mne.merge_events(feedbackleft_b5_p27, [25,26], 531, replace_events=True)
feedbackleft_b5_p27 = mne.event.shift_time_events(feedbackleft_b5_p27, 531, -1.000, 500)
feedbackleft_b5_p27 = np.delete(feedbackleft_b5_p27, [0, 2, 3, 8, 12, 13, 14, 15, 16, 17, 21, 23, 24, 25, 28, 29, 30, 31, 35, 37, 39, 40, 41, 46], axis=0)


#feedback (532)
feedbackright_b5_p27 = mne.pick_events(feedbackright_b5_p27, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p27 = mne.merge_events(feedbackright_b5_p27, [25,26], 532, replace_events=True)
feedbackright_b5_p27 = mne.event.shift_time_events(feedbackright_b5_p27, 532, -1.000, 500)
feedbackright_b5_p27 = np.delete(feedbackright_b5_p27, [1, 4, 5, 6, 7, 9, 10, 11, 18, 19, 20, 22, 26, 27, 32, 33, 34, 36, 38, 42, 43, 44, 45, 47], axis=0)


#left response (533) 
leftresponse_b5_p27 = mne.pick_events(leftresponse_b5_p27, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p27 = mne.merge_events(leftresponse_b5_p27, [14, 15, 16, 17], 533, replace_events=True)

#right response(534)
rightresponse_b5_p27 = mne.pick_events(rightresponse_b5_p27, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p27 = mne.merge_events(rightresponse_b5_p27, [18, 19, 20, 21], 534, replace_events=True)

#last response left(535)
lastresponseleft_b5_p27 = mne.pick_events(lastresponseleft_b5_p27, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p27 = mne.merge_events(lastresponseleft_b5_p27, [5, 6, 7, 8], 535, replace_events=True)

#last response right(536)
lastresponseright_b5_p27 = mne.pick_events(lastresponseright_b5_p27, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p27 = mne.merge_events(lastresponseright_b5_p27, [9, 10, 11, 12], 536, replace_events=True)


#preperation left (537) 
preparationleft_b5_p27 = mne.pick_events(preparationleft_b5_p27, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p27 = mne.merge_events(preparationleft_b5_p27, [5, 6, 7, 8], 537, replace_events=True)
preparationleft_b5_p27 = mne.event.shift_time_events(preparationleft_b5_p27, 537, 1.500, 500)
preparationleft_b5_p27 = np.delete(preparationleft_b5_p27, [29, 59, 143, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)

#preperation right (538) 
preparationright_b5_p27 = mne.pick_events(preparationright_b5_p27, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p27 = mne.merge_events(preparationright_b5_p27, [9, 10, 11, 12], 538, replace_events=True)
preparationright_b5_p27 = mne.event.shift_time_events(preparationright_b5_p27, 538, 1.500, 500)
preparationright_b5_p27 = np.delete(preparationright_b5_p27, [83, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#nogo (539)
nogob5p27 = mne.pick_events(nogob5p27, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p27 = mne.merge_events(nogob5p27, [24], 539, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p27 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p27 = {'laststimleft':518,'laststimright':519, 'feedbackL':520, 'feedbackR':521, 'leftres':522, 'rightres':523, 'lastresleft':524, 'lastresright':525, 'prepleft':526, 'prepright':527, 'nogo': 528}
event_dictB5_p27 = {'laststimleft':529,'laststimright':530, 'feedbackL':531, 'feedbackR':532, 'leftres':533, 'rightres':534, 'lastresleft':535, 'lastresright':536, 'prepleft':537, 'prepright':538, 'nogo': 539}



#merging events togeher into one event list

finalB1_p27 = np.concatenate((laststimpositionleft_b1_p27, laststimpositionright_b1_p27, feedbackleft_b1_p27, feedbackright_b1_p27, leftresponse_b1_p27, rightresponse_b1_p27, lastresponseleft_b1_p27, lastresponseright_b1_p27, preparationleft_b1_p27, preparationright_b1_p27, events_p27, nogob1p27), axis=0)
finalB5_p27 = np.concatenate((laststimpositionleft_b5_p27, laststimpositionright_b5_p27, feedbackleft_b5_p27, feedbackright_b5_p27, leftresponse_b5_p27, rightresponse_b5_p27, lastresponseleft_b5_p27, lastresponseright_b5_p27, preparationleft_b5_p27, preparationright_b5_p27, events_p27, nogob5p27), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p27, event_id=event_dictB1_p27, 

                         sfreq=raw_p27.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p27, event_id=event_dictB5_p27, 

                         sfreq=raw5_p27.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend


#-----------------------------------------------------------------------------#
#                               PARTICIPANT 28
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_28_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part28_SeqL_ERD_6_B1.fif'
Part_28_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part28_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p28 = mne.io.read_raw_fif(Part_28_1, preload = True) 
 

#--------B5--------#
raw5_p28 = mne.io.read_raw_fif(Part_28_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p28, _ = mne.events_from_annotations(raw_p28, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p28, _ = mne.events_from_annotations(raw5_p28, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p28 = np.copy(events_p28)
laststimpositionright_b1_p28 = np.copy(events_p28)
feedbackleft_b1_p28 = np.copy(events_p28)
feedbackright_b1_p28 = np.copy(events_p28)
leftresponse_b1_p28 = np.copy(events_p28)
rightresponse_b1_p28 = np.copy(events_p28)
lastresponseleft_b1_p28 = np.copy(events_p28)
lastresponseright_b1_p28 = np.copy(events_p28)
preparationleft_b1_p28 = np.copy(events_p28)
preparationright_b1_p28 = np.copy(events_p28)
nogob1p28 = np.copy(events_p28)
#--------B5-------#
laststimpositionleft_b5_p28 = np.copy(eventsB5_p28)
laststimpositionright_b5_p28 = np.copy(eventsB5_p28)
feedbackleft_b5_p28 = np.copy(eventsB5_p28)
feedbackright_b5_p28 = np.copy(eventsB5_p28)
leftresponse_b5_p28 = np.copy(eventsB5_p28)
rightresponse_b5_p28 = np.copy(eventsB5_p28)
lastresponseleft_b5_p28 = np.copy(eventsB5_p28)
lastresponseright_b5_p28 = np.copy(eventsB5_p28)
preparationleft_b5_p28 = np.copy(eventsB5_p28)
preparationright_b5_p28 = np.copy(eventsB5_p28)
nogob5p28 = np.copy(eventsB5_p28)


#print to see if event times are correct

print(events_p28)

#--------B1-------#


#last  stimulus position left (540)
laststimpositionleft_b1_p28 = mne.pick_events(laststimpositionleft_b1_p28, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p28 = mne.merge_events(laststimpositionleft_b1_p28, [5, 6, 7, 8], 540, replace_events=True)
laststimpositionleft_b1_p28 = np.delete(laststimpositionleft_b1_p28, [143, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)


#last  stimulus position right (541)
laststimpositionright_b1_p28 = mne.pick_events(laststimpositionright_b1_p28, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p28 = mne.merge_events(laststimpositionright_b1_p28, [9, 10, 11, 12], 541, replace_events=True)
laststimpositionright_b1_p28 = np.delete(laststimpositionright_b1_p28, [23, 59, 125, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)


#feedback left (542)
feedbackleft_b1_p28 = mne.pick_events(feedbackleft_b1_p28, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p28 = mne.merge_events(feedbackleft_b1_p28, [25,26], 542, replace_events=True)
feedbackleft_b1_p28 = mne.event.shift_time_events(feedbackleft_b1_p28, 542, -1.000, 500)
feedbackleft_b1_p28 = np.delete(feedbackleft_b1_p28, [0, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 25, 26, 27, 29, 31, 32, 34, 36, 37, 38, 40, 41], axis=0)


#feedback right (543)
feedbackright_b1_p28 = mne.pick_events(feedbackright_b1_p28, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p28 = mne.merge_events(feedbackright_b1_p28, [25,26], 543, replace_events=True)
feedbackright_b1_p28 = mne.event.shift_time_events(feedbackright_b1_p28, 543, -1.000, 500)
feedbackright_b1_p28 = np.delete(feedbackright_b1_p28, [1, 2, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 24, 28, 30, 33, 35, 39, 42, 43, 44, 45, 46, 47], axis=0)


#left response (544) 
leftresponse_b1_p28 = mne.pick_events(leftresponse_b1_p28, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p28 = mne.merge_events(leftresponse_b1_p28, [14, 15, 16, 17], 544, replace_events=True)

#right response(545)
rightresponse_b1_p28 = mne.pick_events(rightresponse_b1_p28, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p28 = mne.merge_events(rightresponse_b1_p28, [18, 19, 20, 21], 545, replace_events=True)

#last response left(546)
lastresponseleft_b1_p28 = mne.pick_events(lastresponseleft_b1_p28, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p28 = mne.merge_events(lastresponseleft_b1_p28, [5, 6, 7, 8], 546, replace_events=True)

#last response right(547)
lastresponseright_b1_p28 = mne.pick_events(lastresponseright_b1_p28, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p28 = mne.merge_events(lastresponseright_b1_p28, [9, 10, 11, 12], 547, replace_events=True)


#preperation left (548) 
preparationleft_b1_p28 = mne.pick_events(preparationleft_b1_p28, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p28 = mne.merge_events(preparationleft_b1_p28, [5, 6, 7, 8], 548, replace_events=True)
preparationleft_b1_p28 = mne.event.shift_time_events(preparationleft_b1_p28, 548, 1.500, 500)
preparationleft_b1_p28 = np.delete(preparationleft_b1_p28, [143, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)



#preperation right (549) 
preparationright_b1_p28 = mne.pick_events(preparationright_b1_p28, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p28 = mne.merge_events(preparationright_b1_p28, [9, 10, 11, 12], 549, replace_events=True)
preparationright_b1_p28 = mne.event.shift_time_events(preparationright_b1_p28, 549, 1.500, 500)
preparationright_b1_p28 = np.delete(preparationright_b1_p28, [23, 59, 125, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)

#nogo (550)
nogob1p28 = mne.pick_events(nogob1p28, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p28 = mne.merge_events(nogob1p28, [24], 550, replace_events=True)


#--------B5-------#
#last  stimulus position left (551)
laststimpositionleft_b5_p28 = mne.pick_events(laststimpositionleft_b5_p28, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p28 = mne.merge_events(laststimpositionleft_b5_p28, [5, 6, 7, 8], 551, replace_events=True)
laststimpositionleft_b5_p28 = np.delete(laststimpositionleft_b5_p28, [77, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)


#last  stimulus position right (552)
laststimpositionright_b5_p28 = mne.pick_events(laststimpositionright_b5_p28, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p28 = mne.merge_events(laststimpositionright_b5_p28, [9, 10, 11, 12], 552, replace_events=True)
laststimpositionright_b5_p28 = np.delete(laststimpositionright_b5_p28, [41, 65, 89, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)


#feedback (553)
feedbackleft_b5_p28 = mne.pick_events(feedbackleft_b5_p28, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p28 = mne.merge_events(feedbackleft_b5_p28, [25,26], 553, replace_events=True)
feedbackleft_b5_p28 = mne.event.shift_time_events(feedbackleft_b5_p28, 69, -1.000, 500)
feedbackleft_b5_p28 = np.delete(feedbackleft_b5_p28, [3, 4, 6, 9, 12, 13, 14, 16, 18, 19, 21, 22, 25, 26, 28, 30, 32, 36, 38, 39, 40, 44, 46, 47], axis=0)

#feedback (554)
feedbackright_b5_p28 = mne.pick_events(feedbackright_b5_p28, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p28 = mne.merge_events(feedbackright_b5_p28, [25,26], 554, replace_events=True)
feedbackright_b5_p28 = mne.event.shift_time_events(feedbackright_b5_p28, 554, -1.000, 500)
feedbackright_b5_p28 = np.delete(feedbackright_b5_p28, [0, 1, 2, 5, 7, 8, 10, 11, 15, 17, 20, 23, 24, 27, 29, 31, 33, 34, 35, 37, 41, 42, 43, 45], axis=0)

#left response (555) 
leftresponse_b5_p28 = mne.pick_events(leftresponse_b5_p28, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p28 = mne.merge_events(leftresponse_b5_p28, [14, 15, 16, 17], 555, replace_events=True)

#right response(556)
rightresponse_b5_p28 = mne.pick_events(rightresponse_b5_p28, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p28 = mne.merge_events(rightresponse_b5_p28, [18, 19, 20, 21], 556, replace_events=True)

#last response left(557)
lastresponseleft_b5_p28 = mne.pick_events(lastresponseleft_b5_p28, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p28 = mne.merge_events(lastresponseleft_b5_p28, [5, 6, 7, 8], 557, replace_events=True)

#last response right(558)
lastresponseright_b5_p28 = mne.pick_events(lastresponseright_b5_p28, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p28 = mne.merge_events(lastresponseright_b5_p28, [9, 10, 11, 12], 558, replace_events=True)


#preperation left (559) 
preparationleft_b5_p28 = mne.pick_events(preparationleft_b5_p28, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p28 = mne.merge_events(preparationleft_b5_p28, [5, 6, 7, 8], 559, replace_events=True)
preparationleft_b5_p28 = mne.event.shift_time_events(preparationleft_b5_p28, 559, 1.500, 500)
preparationleft_b5_p28 = np.delete(preparationleft_b5_p28, [77, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)


#preperation right (560) 
preparationright_b5_p28 = mne.pick_events(preparationright_b5_p28, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p28 = mne.merge_events(preparationright_b5_p28, [9, 10, 11, 12], 560, replace_events=True)
preparationright_b5_p28 = mne.event.shift_time_events(preparationright_b5_p28, 560, 1.500, 500)
preparationright_b5_p28 = np.delete(preparationright_b5_p28, [41, 65, 89, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)


#nogo (561)
nogob5p28 = mne.pick_events(nogob5p28, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p28 = mne.merge_events(nogob5p28, [24], 561, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p28 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p28 = {'laststimleft':540,'laststimright':541, 'feedbackL':542, 'feedbackR':543, 'leftres':544, 'rightres':545, 'lastresleft':546, 'lastresright':547, 'prepleft':548, 'prepright':549, 'nogo': 550,}
event_dictB5_p28 = {'laststimleft':551,'laststimright':552, 'feedbackL':553, 'feedbackR':554, 'leftres':555, 'rightres':556, 'lastresleft':557, 'lastresright':558, 'prepleft':559, 'prepright':560, 'nogo': 561,}



#merging events togeher into one event list

finalB1_p28 = np.concatenate((laststimpositionleft_b1_p28, laststimpositionright_b1_p28, feedbackleft_b1_p28, feedbackright_b1_p28, leftresponse_b1_p28, rightresponse_b1_p28, lastresponseleft_b1_p28, lastresponseright_b1_p28, preparationleft_b1_p28, preparationright_b1_p28, events_p28, nogob1p28), axis=0)
finalB5_p28 = np.concatenate((laststimpositionleft_b5_p28, laststimpositionright_b5_p28, feedbackleft_b5_p28, feedbackright_b5_p28, leftresponse_b5_p28, rightresponse_b5_p28, lastresponseleft_b5_p28, lastresponseright_b5_p28, preparationleft_b5_p28, preparationright_b5_p28, events_p28, nogob5p28), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p28, event_id=event_dictB1_p28, 

                         sfreq=raw_p28.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p28, event_id=event_dictB5_p28, 

                         sfreq=raw5_p28.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend



#-----------------------------------------------------------------------------#
#                               PARTICIPANT 29
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_29_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part29_SeqL_ERD_6_B1.fif'
Part_29_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part29_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p29 = mne.io.read_raw_fif(Part_29_1, preload = True) 
 

#--------B5--------#
raw5_p29 = mne.io.read_raw_fif(Part_29_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p29, _ = mne.events_from_annotations(raw_p29, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p29, _ = mne.events_from_annotations(raw5_p29, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p29 = np.copy(events_p29)
laststimpositionright_b1_p29 = np.copy(events_p29)
feedbackleft_b1_p29 = np.copy(events_p29)
feedbackright_b1_p29 = np.copy(events_p29)
leftresponse_b1_p29 = np.copy(events_p29)
rightresponse_b1_p29 = np.copy(events_p29)
lastresponseleft_b1_p29 = np.copy(events_p29)
lastresponseright_b1_p29 = np.copy(events_p29)
preparationleft_b1_p29 = np.copy(events_p29)
preparationright_b1_p29 = np.copy(events_p29)
nogob1p29 = np.copy(events_p29)
#--------B5-------#
laststimpositionleft_b5_p29 = np.copy(eventsB5_p29)
laststimpositionright_b5_p29 = np.copy(eventsB5_p29)
feedbackleft_b5_p29 = np.copy(eventsB5_p29)
feedbackright_b5_p29 = np.copy(eventsB5_p29)
leftresponse_b5_p29 = np.copy(eventsB5_p29)
rightresponse_b5_p29 = np.copy(eventsB5_p29)
lastresponseleft_b5_p29 = np.copy(eventsB5_p29)
lastresponseright_b5_p29 = np.copy(eventsB5_p29)
preparationleft_b5_p29 = np.copy(eventsB5_p29)
preparationright_b5_p29 = np.copy(eventsB5_p29)
nogob5p29 = np.copy(eventsB5_p29)


#print to see if event times are correct

print(events_p29)

#--------B1-------#


#last  stimulus position left (562)
laststimpositionleft_b1_p29 = mne.pick_events(laststimpositionleft_b1_p29, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p29 = mne.merge_events(laststimpositionleft_b1_p29, [5, 6, 7, 8], 562, replace_events=True)
laststimpositionleft_b1_p29 = np.delete(laststimpositionleft_b1_p29, [11, 143, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#last  stimulus position right (563)
laststimpositionright_b1_p29 = mne.pick_events(laststimpositionright_b1_p29, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p29 = mne.merge_events(laststimpositionright_b1_p29, [9, 10, 11, 12], 563, replace_events=True)
laststimpositionright_b1_p29 = np.delete(laststimpositionright_b1_p29, [59, 83, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#feedback left (564)
feedbackleft_b1_p29 = mne.pick_events(feedbackleft_b1_p29, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p29 = mne.merge_events(feedbackleft_b1_p29, [25,26], 564, replace_events=True)
feedbackleft_b1_p29 = mne.event.shift_time_events(feedbackleft_b1_p29, 564, -1.000, 500)
feedbackleft_b1_p29 = np.delete(feedbackleft_b1_p29, [1, 2, 5, 6, 11, 12, 14, 16, 17, 21, 22, 23, 26, 28, 31, 36, 37, 38, 39, 42, 43, 44, 45, 46], axis=0)


#feedback right (565)
feedbackright_b1_p29 = mne.pick_events(feedbackright_b1_p29, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p29 = mne.merge_events(feedbackright_b1_p29, [25,26], 565, replace_events=True)
feedbackright_b1_p29 = mne.event.shift_time_events(feedbackright_b1_p29, 565, -1.000, 500)
feedbackright_b1_p29 = np.delete(feedbackright_b1_p29, [0, 3, 4, 7, 8, 9, 10, 13, 15, 18, 19, 20, 24, 25, 27, 29, 30, 32, 33, 34, 35, 40, 41, 47], axis=0)

#left response (566) 
leftresponse_b1_p29 = mne.pick_events(leftresponse_b1_p29, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p29 = mne.merge_events(leftresponse_b1_p29, [14, 15, 16, 17], 566, replace_events=True)

#right response(567)
rightresponse_b1_p29 = mne.pick_events(rightresponse_b1_p29, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p29 = mne.merge_events(rightresponse_b1_p29, [18, 19, 20, 21], 567, replace_events=True)

#last response left(568)
lastresponseleft_b1_p29 = mne.pick_events(lastresponseleft_b1_p29, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p29 = mne.merge_events(lastresponseleft_b1_p29, [5, 6, 7, 8], 568, replace_events=True)

#last response right(569)
lastresponseright_b1_p29 = mne.pick_events(lastresponseright_b1_p29, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p29 = mne.merge_events(lastresponseright_b1_p29, [9, 10, 11, 12], 569, replace_events=True)


#preperation left (570) 
preparationleft_b1_p29 = mne.pick_events(preparationleft_b1_p29, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p29 = mne.merge_events(preparationleft_b1_p29, [5, 6, 7, 8], 570, replace_events=True)
preparationleft_b1_p29 = mne.event.shift_time_events(preparationleft_b1_p29, 570, 1.500, 500)
preparationleft_b1_p29 = np.delete(preparationleft_b1_p29, [11, 143, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#preperation right (571) 
preparationright_b1_p29 = mne.pick_events(preparationright_b1_p29, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p29 = mne.merge_events(preparationright_b1_p29, [9, 10, 11, 12], 571, replace_events=True)
preparationright_b1_p29 = mne.event.shift_time_events(preparationright_b1_p29, 571, 1.500, 500)
preparationright_b1_p29 = np.delete(preparationright_b1_p29, [59, 83, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#nogo (572)
nogob1p29 = mne.pick_events(nogob1p29, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p29 = mne.merge_events(nogob1p29, [24], 572, replace_events=True)
#--------B5-------#


#last  stimulus position left (573)
laststimpositionleft_b5_p29 = mne.pick_events(laststimpositionleft_b5_p29, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p29 = mne.merge_events(laststimpositionleft_b5_p29, [5, 6, 7, 8], 573, replace_events=True)
laststimpositionleft_b5_p29 = np.delete(laststimpositionleft_b5_p29, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142], axis=0)


#last  stimulus position right (574)
laststimpositionright_b5_p29 = mne.pick_events(laststimpositionright_b5_p29, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p29 = mne.merge_events(laststimpositionright_b5_p29, [9, 10, 11, 12], 574, replace_events=True)
laststimpositionright_b5_p29 = np.delete(laststimpositionright_b5_p29, [59, 83, 107, 125, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166], axis=0)


#feedback (575)
feedbackleft_b5_p29 = mne.pick_events(feedbackleft_b5_p29, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p29 = mne.merge_events(feedbackleft_b5_p29, [25,26], 575, replace_events=True)
feedbackleft_b5_p29 = mne.event.shift_time_events(feedbackleft_b5_p29, 575, -1.000, 500)
feedbackleft_b5_p29 = np.delete(feedbackleft_b5_p29, [0, 1, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 24, 26, 27, 30, 31, 33, 34, 35, 38, 39, 44, 45], axis=0)

#feedback (576)
feedbackright_b5_p29 = mne.pick_events(feedbackright_b5_p29, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p29 = mne.merge_events(feedbackright_b5_p29, [25,26], 576, replace_events=True)
feedbackright_b5_p29 = mne.event.shift_time_events(feedbackright_b5_p29, 576, -1.000, 500)
feedbackright_b5_p29 = np.delete(feedbackright_b5_p29, [2, 8, 9, 14, 15, 17, 18, 19, 20, 21, 22, 23, 25, 28, 29, 32, 36, 37, 40, 41, 42, 43, 46, 47], axis=0)


#left response (577) 
leftresponse_b5_p29 = mne.pick_events(leftresponse_b5_p29, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p29 = mne.merge_events(leftresponse_b5_p29, [14, 15, 16, 17], 577, replace_events=True)

#right response(578)
rightresponse_b5_p29 = mne.pick_events(rightresponse_b5_p29, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p29 = mne.merge_events(rightresponse_b5_p29, [18, 19, 20, 21], 578, replace_events=True)

#last response left(579)
lastresponseleft_b5_p29 = mne.pick_events(lastresponseleft_b5_p29, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p29 = mne.merge_events(lastresponseleft_b5_p29, [5, 6, 7, 8], 579, replace_events=True)

#last response right(580)
lastresponseright_b5_p29 = mne.pick_events(lastresponseright_b5_p29, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p29 = mne.merge_events(lastresponseright_b5_p29, [9, 10, 11, 12], 580, replace_events=True)


#preperation left (581) 
preparationleft_b5_p29 = mne.pick_events(preparationleft_b5_p29, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p29 = mne.merge_events(preparationleft_b5_p29, [5, 6, 7, 8], 581, replace_events=True)
preparationleft_b5_p29 = mne.event.shift_time_events(preparationleft_b5_p29, 581, 1.500, 500)
preparationleft_b5_p29 = np.delete(preparationleft_b5_p29, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142], axis=0)



#preperation right (582) 
preparationright_b5_p29 = mne.pick_events(preparationright_b5_p29, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p29 = mne.merge_events(preparationright_b5_p29, [9, 10, 11, 12], 582, replace_events=True)
preparationright_b5_p29 = mne.event.shift_time_events(preparationright_b5_p29, 582, 1.500, 500)
preparationright_b5_p29 = np.delete(preparationright_b5_p29, [59, 83, 107, 125, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166], axis=0)


#nogo (583)
nogob5p29 = mne.pick_events(nogob5p29, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p29 = mne.merge_events(nogob5p29, [24], 583, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p29 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p29 = {'laststimleft':562,'laststimright':563, 'feedbackL':564, 'feedbackR':565, 'leftres':566, 'rightres':567, 'lastresleft':568, 'lastresright':569, 'prepleft':570, 'prepright':571, 'nogo': 572}
event_dictB5_p29 = {'laststimleft':573,'laststimright':574, 'feedbackL':575, 'feedbackR':576, 'leftres':577, 'rightres':578, 'lastresleft':579, 'lastresright':580, 'prepleft':581, 'prepright':582, 'nogo': 583}



#merging events togeher into one event list

finalB1_p29 = np.concatenate((laststimpositionleft_b1_p29, laststimpositionright_b1_p29, feedbackleft_b1_p29, feedbackright_b1_p29, leftresponse_b1_p29, rightresponse_b1_p29, lastresponseleft_b1_p29, lastresponseright_b1_p29, preparationleft_b1_p29, preparationright_b1_p29, events_p29, nogob1p29), axis=0)
finalB5_p29 = np.concatenate((laststimpositionleft_b5_p29, laststimpositionright_b5_p29, feedbackleft_b5_p29, feedbackright_b5_p29, leftresponse_b5_p29, rightresponse_b5_p29, lastresponseleft_b5_p29, lastresponseright_b5_p29, preparationleft_b5_p29, preparationright_b5_p29, events_p29, nogob5p29), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p29, event_id=event_dictB1_p29, 

                         sfreq=raw_p29.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p29, event_id=event_dictB5_p29, 

                         sfreq=raw5_p29.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend


#-----------------------------------------------------------------------------#
#                               PARTICIPANT 30
#-----------------------------------------------------------------------------#

#give file a name and save it#

Part_30_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part30_SeqL_ERD_6_B1.fif'
Part_30_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part30_SeqL_ERD_6_B5.fif'

#read file from folder

#--------B1--------#
raw_p30 = mne.io.read_raw_fif(Part_30_1, preload = True) 
 

#--------B5--------#
raw5_p30 = mne.io.read_raw_fif(Part_30_5, preload = True) 

#-----------------------------------------------------------------------------#
#                                 EVENTS
#-----------------------------------------------------------------------------#  

#get and save stimuli times --> make an event  
#--------B1-------#
events_p30, _ = mne.events_from_annotations(raw_p30, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                             
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29}) 


#--------B5-------# 
eventsB5_p30, _ = mne.events_from_annotations(raw5_p30, event_id={'Stimulus/S  1': 1,'Stimulus/S  2': 2,'Stimulus/S  3': 3,'Stimulus/S  4': 4,'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
                                                       'Stimulus/S  7': 7,'Stimulus/S  8': 8, 
                                                       
                                                       'Stimulus/S  9': 9, 'Stimulus/S 10': 10, 

                                                       'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 

                                                       'Stimulus/S 14': 14, 
                                                       
                                                       'Stimulus/S 15': 15, 'Stimulus/S 16': 16,
                                                       
                                                       'Stimulus/S 17': 17, 'Stimulus/S 18': 18,
                                                         
                                                       'Stimulus/S 19': 19, 'Stimulus/S 20': 20,
                                                           
                                                       'Stimulus/S 21': 21,
                                                       
                                                       'Stimulus/S 22': 22,
                                                       
                                                       'Stimulus/S 24': 24,
                                                               
                                                       'Stimulus/S 25': 25, 'Stimulus/S 26': 26,
                                                                 
                                                       'Stimulus/S 27': 27, 'Stimulus/S 29': 29})                                        


#creating new events based on copy

#--------B1-------#
laststimpositionleft_b1_p30 = np.copy(events_p30)
laststimpositionright_b1_p30 = np.copy(events_p30)
feedbackleft_b1_p30 = np.copy(events_p30)
feedbackright_b1_p30 = np.copy(events_p30)
leftresponse_b1_p30 = np.copy(events_p30)
rightresponse_b1_p30 = np.copy(events_p30)
lastresponseleft_b1_p30 = np.copy(events_p30)
lastresponseright_b1_p30 = np.copy(events_p30)
preparationleft_b1_p30 = np.copy(events_p30)
preparationright_b1_p30 = np.copy(events_p30)
nogob1p30 = np.copy(events_p30)
#--------B5-------#
laststimpositionleft_b5_p30 = np.copy(eventsB5_p30)
laststimpositionright_b5_p30 = np.copy(eventsB5_p30)
feedbackleft_b5_p30 = np.copy(eventsB5_p30)
feedbackright_b5_p30 = np.copy(eventsB5_p30)
leftresponse_b5_p30 = np.copy(eventsB5_p30)
rightresponse_b5_p30 = np.copy(eventsB5_p30)
lastresponseleft_b5_p30 = np.copy(eventsB5_p30)
lastresponseright_b5_p30 = np.copy(eventsB5_p30)
preparationleft_b5_p30 = np.copy(eventsB5_p30)
preparationright_b5_p30 = np.copy(eventsB5_p30)
nogob5p30 = np.copy(eventsB5_p30)


#print to see if event times are correct

print(events_p30)

#--------B1-------#


#last  stimulus position left (584)
laststimpositionleft_b1_p30 = mne.pick_events(laststimpositionleft_b1_p30, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b1_p30 = mne.merge_events(laststimpositionleft_b1_p30, [5, 6, 7, 8], 584, replace_events=True)
laststimpositionleft_b1_p30 = np.delete(laststimpositionleft_b1_p30, [17, 89, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#last  stimulus position right (585)
laststimpositionright_b1_p30 = mne.pick_events(laststimpositionright_b1_p30, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b1_p30 = mne.merge_events(laststimpositionright_b1_p30, [9, 10, 11, 12], 585, replace_events=True)
laststimpositionright_b1_p30 = np.delete(laststimpositionright_b1_p30, [41, 101, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#feedback left (586)
feedbackleft_b1_p30 = mne.pick_events(feedbackleft_b1_p30, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b1_p30 = mne.merge_events(feedbackleft_b1_p30, [25,26], 586, replace_events=True)
feedbackleft_b1_p30 = mne.event.shift_time_events(feedbackleft_b1_p30, 586, -1.000, 500)
feedbackleft_b1_p30 = np.delete(feedbackleft_b1_p30, [0, 2, 4, 5, 7, 8, 14, 18, 19, 20, 21, 23, 25, 26, 27, 32, 34, 37, 38, 39, 44, 45, 46, 47], axis=0)


#feedback right (587)
feedbackright_b1_p30 = mne.pick_events(feedbackright_b1_p30, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b1_p30 = mne.merge_events(feedbackright_b1_p30, [25,26], 587, replace_events=True)
feedbackright_b1_p30 = mne.event.shift_time_events(feedbackright_b1_p30, 587, -1.000, 500)
feedbackright_b1_p30 = np.delete(feedbackright_b1_p30, [1, 3, 6, 9, 10, 11, 12, 13, 15, 16, 17, 22, 24, 28, 29, 30, 31, 33, 35, 36, 40, 41, 42, 43], axis=0)

#left response (588) 
leftresponse_b1_p30 = mne.pick_events(leftresponse_b1_p30, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b1_p30 = mne.merge_events(leftresponse_b1_p30, [14, 15, 16, 17], 588, replace_events=True)

#right response(589)
rightresponse_b1_p30 = mne.pick_events(rightresponse_b1_p30, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b1_p30 = mne.merge_events(rightresponse_b1_p30, [18, 19, 20, 21], 589, replace_events=True)

#last response left(590)
lastresponseleft_b1_p30 = mne.pick_events(lastresponseleft_b1_p30, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b1_p30 = mne.merge_events(lastresponseleft_b1_p30, [5, 6, 7, 8], 590, replace_events=True)

#last response right(591)
lastresponseright_b1_p30 = mne.pick_events(lastresponseright_b1_p30, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b1_p30 = mne.merge_events(lastresponseright_b1_p30, [9, 10, 11, 12], 591, replace_events=True)


#preperation left (592) 
preparationleft_b1_p30 = mne.pick_events(preparationleft_b1_p30, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b1_p30 = mne.merge_events(preparationleft_b1_p30, [5, 6, 7, 8], 592, replace_events=True)
preparationleft_b1_p30 = mne.event.shift_time_events(preparationleft_b1_p30, 592, 1.500, 500)
preparationleft_b1_p30 = np.delete(preparationleft_b1_p30, [17, 89, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)


#preperation right (593) 
preparationright_b1_p30 = mne.pick_events(preparationright_b1_p30, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b1_p30 = mne.merge_events(preparationright_b1_p30, [9, 10, 11, 12], 593, replace_events=True)
preparationright_b1_p30 = mne.event.shift_time_events(preparationright_b1_p30, 593, 1.500, 500)
preparationright_b1_p30 = np.delete(preparationright_b1_p30, [41, 101, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154], axis=0)

#nogo (594)
nogob1p30 = mne.pick_events(nogob1p30, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob1p30 = mne.merge_events(nogob1p30, [24], 594, replace_events=True)
#--------B5-------#


#last  stimulus position left (595)
laststimpositionleft_b5_p30 = mne.pick_events(laststimpositionleft_b5_p30, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionleft_b5_p30 = mne.merge_events(laststimpositionleft_b5_p30, [5, 6, 7, 8], 595, replace_events=True)
laststimpositionleft_b5_p30 = np.delete(laststimpositionleft_b5_p30, [23, 125, 155, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)

#last  stimulus position right (596)
laststimpositionright_b5_p30 = mne.pick_events(laststimpositionright_b5_p30, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
laststimpositionright_b5_p30 = mne.merge_events(laststimpositionright_b5_p30, [9, 10, 11, 12], 596, replace_events=True)
laststimpositionright_b5_p30 = np.delete(laststimpositionright_b5_p30, [41, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)


#feedback (597)
feedbackleft_b5_p30 = mne.pick_events(feedbackleft_b5_p30, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackleft_b5_p30 = mne.merge_events(feedbackleft_b5_p30, [25,26], 597, replace_events=True)
feedbackleft_b5_p30 = mne.event.shift_time_events(feedbackleft_b5_p30, 597, -1.000, 500)
feedbackleft_b5_p30 = np.delete(feedbackleft_b5_p30, [3, 4, 6, 10, 12, 13, 16, 18, 20, 21, 22, 23, 27, 32, 33, 34, 35, 36, 39, 42, 43, 44, 45, 46], axis=0)


#feedback (598)
feedbackright_b5_p30 = mne.pick_events(feedbackright_b5_p30, include=[25,26], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29], step=False)
feedbackright_b5_p30 = mne.merge_events(feedbackright_b5_p30, [25,26], 598, replace_events=True)
feedbackright_b5_p30 = mne.event.shift_time_events(feedbackright_b5_p30, 598, -1.000, 500)
feedbackright_b5_p30 = np.delete(feedbackright_b5_p30, [0, 1, 2, 5, 7, 8, 9, 11, 14, 15, 17, 19, 24, 25, 26, 28, 29, 30, 31, 37, 38, 40, 41, 47], axis=0)


#left response (599) 
leftresponse_b5_p30 = mne.pick_events(leftresponse_b5_p30, include=[14, 15, 16, 17], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
leftresponse_b5_p30 = mne.merge_events(leftresponse_b5_p30, [14, 15, 16, 17], 599, replace_events=True)

#right response(600)
rightresponse_b5_p30 = mne.pick_events(rightresponse_b5_p30, include=[18, 19, 20, 21], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 24, 25, 26, 27, 29], step=False)
rightresponse_b5_p30 = mne.merge_events(rightresponse_b5_p30, [18, 19, 20, 21], 600, replace_events=True)

#last response left(601)
lastresponseleft_b5_p30 = mne.pick_events(lastresponseleft_b5_p30, [5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29] , step=False)
lastresponseleft_b5_p30 = mne.merge_events(lastresponseleft_b5_p30, [5, 6, 7, 8], 601, replace_events=True)

#last response right(602)
lastresponseright_b5_p30 = mne.pick_events(lastresponseright_b5_p30, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
lastresponseright_b5_p30 = mne.merge_events(lastresponseright_b5_p30, [9, 10, 11, 12], 602, replace_events=True)


#preperation left (603) 
preparationleft_b5_p30 = mne.pick_events(preparationleft_b5_p30, include=[5, 6, 7, 8], exclude=[1, 2, 3, 4, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationleft_b5_p30 = mne.merge_events(preparationleft_b5_p30, [5, 6, 7, 8], 603, replace_events=True)
preparationleft_b5_p30 = mne.event.shift_time_events(preparationleft_b5_p30, 603, 1.500, 500)
preparationleft_b5_p30 = np.delete(preparationleft_b5_p30, [23, 125, 155, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151,	152, 153, 154, 156, 157, 158, 159, 160], axis=0)


#preperation right (604) 
preparationright_b5_p30 = mne.pick_events(preparationright_b5_p30, include=[9, 10, 11, 12], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29], step=False)
preparationright_b5_p30 = mne.merge_events(preparationright_b5_p30, [9, 10, 11, 12], 604, replace_events=True)
preparationright_b5_p30 = mne.event.shift_time_events(preparationright_b5_p30, 604, 1.500, 500)
preparationright_b5_p30 = np.delete(preparationright_b5_p30, [41, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108,	109, 110, 111, 112, 114, 115, 116, 117,	118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148], axis=0)

#nogo (605)
nogob5p30 = mne.pick_events(nogob5p30, include=[24], exclude=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29], step=False)
nogob5p30 = mne.merge_events(nogob5p30, [24], 605, replace_events=True)

#set event dictonairy with new events

#(making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the eXcel file and put in late)

event_dict_p30 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '24': 24, '25': 25, '26': 26, '29': 29} 
event_dictB1_p30 = {'laststimleft':584,'laststimright':585, 'feedbackL':586, 'feedbackR':587, 'leftres':588, 'rightres':589, 'lastresleft':590, 'lastresright':591, 'prepleft':592, 'prepright':593, 'nogo': 594,}
event_dictB5_p30 = {'laststimleft':595,'laststimright':596, 'feedbackL':597, 'feedbackR':598, 'leftres':599, 'rightres':600, 'lastresleft':601, 'lastresright':602, 'prepleft':603, 'prepright':604, 'nogo': 605,}



#merging events togeher into one event list

finalB1_p30 = np.concatenate((laststimpositionleft_b1_p30, laststimpositionright_b1_p30, feedbackleft_b1_p30, feedbackright_b1_p30, leftresponse_b1_p30, rightresponse_b1_p30, lastresponseleft_b1_p30, lastresponseright_b1_p30, preparationleft_b1_p30, preparationright_b1_p30, events_p30, nogob1p30), axis=0)
finalB5_p30 = np.concatenate((laststimpositionleft_b5_p30, laststimpositionright_b5_p30, feedbackleft_b5_p30, feedbackright_b5_p30, leftresponse_b5_p30, rightresponse_b5_p30, lastresponseleft_b5_p30, lastresponseright_b5_p30, preparationleft_b5_p30, preparationright_b5_p30, events_p30, nogob5p30), axis=0)


#finalepoch = np.concatenate((new_events2, new_events7, new_events16), axis=0)
#finalepoch2 = np.concatenate((new_events9, new_events15, new_events17), axis=0)
#finalepoch3 = np.concatenate((new_events16, new_events18, new_events6), axis=0)
#finalepoch4 = np.concatenate((new_events17, new_events19, new_events14), axis=0)

#checking 



#create plot showing at what times selected stimuli are
fig = mne.viz.plot_events(finalB1_p30, event_id=event_dictB1_p30, 

                         sfreq=raw_p30.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend 


fig = mne.viz.plot_events(finalB5_p30, event_id=event_dictB5_p30, 

                         sfreq=raw5_p30.info['sfreq']) 
fig.subplots_adjust(right=0.6) #to make room for legend(description)<- smaller number bigger legend


#

#-----------------------------------------------------------------------------#
#                                  EPOCHS
#-----------------------------------------------------------------------------#

reject_criteria = dict(eeg=150e-6) #100uV 

flat_criteria = dict(eeg=5e-6)#1uV 

tmin, tmax = -6.5, 3 #for preparation period
tmin3, tmax3 = -4, 2.5  #for feedback period

want_chs = ['C3','C4', 'FC3', 'FC4'] #wanted channels
picks = mne.pick_channels(raw_p2.info["ch_names"], want_chs)

want_chs2 = ['Fp1', 'Fp2', 'F7', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', 'CPz', 'CP4', 'Pz', 'PO7', 'Oz', 'PO8'] #wanted channels
picks2 = mne.pick_channels(raw_p2.info["ch_names"], want_chs2)

#(NOT CORRECT YET)

#_p1_#
#-------------------PREPARATION------------------#

#--------B1-------#
epochsprep1left_p1 = mne.Epochs(raw_p1, preparationleft_b1_p1, event_id=42, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p1 = mne.Epochs(raw_p1, preparationright_b1_p1, event_id=43, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p1 = mne.Epochs(raw5_p1, preparationleft_b5_p1, event_id=53, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p1 = mne.Epochs(raw5_p1, preparationright_b5_p1, event_id=54, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p1 = mne.Epochs(raw_p1, preparationleft_b1_p1, event_id=42, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p1 = mne.Epochs(raw_p1, preparationright_b1_p1, event_id=43, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p1 = mne.Epochs(raw5_p1, preparationleft_b5_p1, event_id=53, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p1 = mne.Epochs(raw5_p1, preparationright_b5_p1, event_id=54, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p1 = mne.Epochs(raw_p1, feedbackleft_b1_p1, event_id=36, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p1 = mne.Epochs(raw_p1, feedbackright_b1_p1, event_id=37, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p1 = mne.Epochs(raw5_p1, feedbackleft_b5_p1, event_id=47, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p1 = mne.Epochs(raw5_p1, feedbackright_b5_p1, event_id=48, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p1 = mne.Epochs(raw_p1, feedbackleft_b1_p1, event_id=36, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p1 = mne.Epochs(raw_p1, feedbackright_b1_p1, event_id=37, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p1 = mne.Epochs(raw5_p1, feedbackleft_b5_p1, event_id=47, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p1 = mne.Epochs(raw5_p1, feedbackright_b5_p1, event_id=48, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#_p2_#
#-------------------PREPARATION------------------#

#--------B1-------#
epochsprep1left_p2 = mne.Epochs(raw_p2, preparationleft_b1_p2, event_id=64, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p2 = mne.Epochs(raw_p2, preparationright_b1_p2, event_id=65, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p2 = mne.Epochs(raw5_p2, preparationleft_b5_p2, event_id=75, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p2 = mne.Epochs(raw5_p2, preparationright_b5_p2, event_id=76, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p2 = mne.Epochs(raw_p2, preparationleft_b1_p2, event_id=64, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p2 = mne.Epochs(raw_p2, preparationright_b1_p2, event_id=65, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p2 = mne.Epochs(raw5_p2, preparationleft_b5_p2, event_id=75, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p2 = mne.Epochs(raw5_p2, preparationright_b5_p2, event_id=76, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p2 = mne.Epochs(raw_p2, feedbackleft_b1_p2, event_id=58, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p2 = mne.Epochs(raw_p2, feedbackright_b1_p2, event_id=59, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p2 = mne.Epochs(raw5_p2, feedbackleft_b5_p2, event_id=69, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p2 = mne.Epochs(raw5_p2, feedbackright_b5_p2, event_id=70, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p2 = mne.Epochs(raw_p2, feedbackleft_b1_p2, event_id=58, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p2 = mne.Epochs(raw_p2, feedbackright_b1_p2, event_id=59, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p2 = mne.Epochs(raw5_p2, feedbackleft_b5_p2, event_id=69, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p2 = mne.Epochs(raw5_p2, feedbackright_b5_p2, event_id=70, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#_p3_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p3 = mne.Epochs(raw_p3, preparationleft_b1_p3, event_id=86, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p3 = mne.Epochs(raw_p3, preparationright_b1_p3, event_id=87, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p3 = mne.Epochs(raw5_p3, preparationleft_b5_p3, event_id=97, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p3 = mne.Epochs(raw5_p3, preparationright_b5_p3, event_id=98, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p3 = mne.Epochs(raw_p3, preparationleft_b1_p3, event_id=86, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p3 = mne.Epochs(raw_p3, preparationright_b1_p3, event_id=87, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p3 = mne.Epochs(raw5_p3, preparationleft_b5_p3, event_id=97, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p3 = mne.Epochs(raw5_p3, preparationright_b5_p3, event_id=98, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p3 = mne.Epochs(raw_p3, feedbackleft_b1_p3, event_id=80, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p3 = mne.Epochs(raw_p3, feedbackright_b1_p3, event_id=81, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p3 = mne.Epochs(raw5_p3, feedbackleft_b5_p3, event_id=91, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p3 = mne.Epochs(raw5_p3, feedbackright_b5_p3, event_id=92, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p3 = mne.Epochs(raw_p3, feedbackleft_b1_p3, event_id=80, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p3 = mne.Epochs(raw_p3, feedbackright_b1_p3, event_id=81, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p3 = mne.Epochs(raw5_p3, feedbackleft_b5_p3, event_id=91, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p3 = mne.Epochs(raw5_p3, feedbackright_b5_p3, event_id=92, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p5_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p5 = mne.Epochs(raw_p5, preparationleft_b1_p5, event_id=108, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p5 = mne.Epochs(raw_p5, preparationright_b1_p5, event_id=109, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p5 = mne.Epochs(raw5_p5, preparationleft_b5_p5, event_id=119, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p5 = mne.Epochs(raw5_p5, preparationright_b5_p5, event_id=120, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p5 = mne.Epochs(raw_p5, preparationleft_b1_p5, event_id=108, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p5 = mne.Epochs(raw_p5, preparationright_b1_p5, event_id=109, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p5 = mne.Epochs(raw5_p5, preparationleft_b5_p5, event_id=119, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p5 = mne.Epochs(raw5_p5, preparationright_b5_p5, event_id=120, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p5 = mne.Epochs(raw_p5, feedbackleft_b1_p5, event_id=102, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p5 = mne.Epochs(raw_p5, feedbackright_b1_p5, event_id=103, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p5 = mne.Epochs(raw5_p5, feedbackleft_b5_p5, event_id=113, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p5 = mne.Epochs(raw5_p5, feedbackright_b5_p5, event_id=114, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p5 = mne.Epochs(raw_p5, feedbackleft_b1_p5, event_id=102, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p5 = mne.Epochs(raw_p5, feedbackright_b1_p5, event_id=103, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p5 = mne.Epochs(raw5_p5, feedbackleft_b5_p5, event_id=113, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p5 = mne.Epochs(raw5_p5, feedbackright_b5_p5, event_id=114, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p6_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p6 = mne.Epochs(raw_p6, preparationleft_b1_p6, event_id=130, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p6 = mne.Epochs(raw_p6, preparationright_b1_p6, event_id=131, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p6 = mne.Epochs(raw5_p6, preparationleft_b5_p6, event_id=141, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p6 = mne.Epochs(raw5_p6, preparationright_b5_p6, event_id=142, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p6 = mne.Epochs(raw_p6, preparationleft_b1_p6, event_id=130, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p6 = mne.Epochs(raw_p6, preparationright_b1_p6, event_id=131, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p6 = mne.Epochs(raw5_p6, preparationleft_b5_p6, event_id=141, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p6 = mne.Epochs(raw5_p6, preparationright_b5_p6, event_id=142, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p6 = mne.Epochs(raw_p6, feedbackleft_b1_p6, event_id=124, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p6 = mne.Epochs(raw_p6, feedbackright_b1_p6, event_id=125, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p6 = mne.Epochs(raw5_p6, feedbackleft_b5_p6, event_id=135, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p6 = mne.Epochs(raw5_p6, feedbackright_b5_p6, event_id=136, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p6 = mne.Epochs(raw_p6, feedbackleft_b1_p6, event_id=124, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p6 = mne.Epochs(raw_p6, feedbackright_b1_p6, event_id=125, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p6 = mne.Epochs(raw5_p6, feedbackleft_b5_p6, event_id=135, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p6 = mne.Epochs(raw5_p6, feedbackright_b5_p6, event_id=136, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p7_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p7 = mne.Epochs(raw_p7, preparationleft_b1_p7, event_id=152, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p7 = mne.Epochs(raw_p7, preparationright_b1_p7, event_id=153, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p7 = mne.Epochs(raw5_p7, preparationleft_b5_p7, event_id=163, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p7 = mne.Epochs(raw5_p7, preparationright_b5_p7, event_id=164, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p7 = mne.Epochs(raw_p7, preparationleft_b1_p7, event_id=152, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p7 = mne.Epochs(raw_p7, preparationright_b1_p7, event_id=153, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p7 = mne.Epochs(raw5_p7, preparationleft_b5_p7, event_id=163, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p7 = mne.Epochs(raw5_p7, preparationright_b5_p7, event_id=164, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p7 = mne.Epochs(raw_p7, feedbackleft_b1_p7, event_id=146, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p7 = mne.Epochs(raw_p7, feedbackright_b1_p7, event_id=147, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p7 = mne.Epochs(raw5_p7, feedbackleft_b5_p7, event_id=157, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p7 = mne.Epochs(raw5_p7, feedbackright_b5_p7, event_id=158, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p7 = mne.Epochs(raw_p7, feedbackleft_b1_p7, event_id=146, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p7 = mne.Epochs(raw_p7, feedbackright_b1_p7, event_id=147, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p7 = mne.Epochs(raw5_p7, feedbackleft_b5_p7, event_id=157, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p7 = mne.Epochs(raw5_p7, feedbackright_b5_p7, event_id=158, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p8_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p8 = mne.Epochs(raw_p8, preparationleft_b1_p8, event_id=174, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p8 = mne.Epochs(raw_p8, preparationright_b1_p8, event_id=175, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p8 = mne.Epochs(raw5_p8, preparationleft_b5_p8, event_id=185, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p8 = mne.Epochs(raw5_p8, preparationright_b5_p8, event_id=186, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p8 = mne.Epochs(raw_p8, preparationleft_b1_p8, event_id=174, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p8 = mne.Epochs(raw_p8, preparationright_b1_p8, event_id=175, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p8 = mne.Epochs(raw5_p8, preparationleft_b5_p8, event_id=185, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p8 = mne.Epochs(raw5_p8, preparationright_b5_p8, event_id=186, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p8 = mne.Epochs(raw_p8, feedbackleft_b1_p8, event_id=168, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p8 = mne.Epochs(raw_p8, feedbackright_b1_p8, event_id=169, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p8 = mne.Epochs(raw5_p8, feedbackleft_b5_p8, event_id=179, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p8 = mne.Epochs(raw5_p8, feedbackright_b5_p8, event_id=180, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p8 = mne.Epochs(raw_p8, feedbackleft_b1_p8, event_id=168, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p8 = mne.Epochs(raw_p8, feedbackright_b1_p8, event_id=169, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p8 = mne.Epochs(raw5_p8, feedbackleft_b5_p8, event_id=179, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p8 = mne.Epochs(raw5_p8, feedbackright_b5_p8, event_id=180, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p9_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p9 = mne.Epochs(raw_p9, preparationleft_b1_p9, event_id=196, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p9 = mne.Epochs(raw_p9, preparationright_b1_p9, event_id=197, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p9 = mne.Epochs(raw5_p9, preparationleft_b5_p9, event_id=207, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p9 = mne.Epochs(raw5_p9, preparationright_b5_p9, event_id=208, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p9 = mne.Epochs(raw_p9, preparationleft_b1_p9, event_id=196, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p9 = mne.Epochs(raw_p9, preparationright_b1_p9, event_id=197, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p9 = mne.Epochs(raw5_p9, preparationleft_b5_p9, event_id=207, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p9 = mne.Epochs(raw5_p9, preparationright_b5_p9, event_id=208, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p9 = mne.Epochs(raw_p9, feedbackleft_b1_p9, event_id=190, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p9 = mne.Epochs(raw_p9, feedbackright_b1_p9, event_id=191, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p9 = mne.Epochs(raw5_p9, feedbackleft_b5_p9, event_id=201, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p9 = mne.Epochs(raw5_p9, feedbackright_b5_p9, event_id=202, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p9 = mne.Epochs(raw_p9, feedbackleft_b1_p9, event_id=190, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p9 = mne.Epochs(raw_p9, feedbackright_b1_p9, event_id=191, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p9 = mne.Epochs(raw5_p9, feedbackleft_b5_p9, event_id=201, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p9 = mne.Epochs(raw5_p9, feedbackright_b5_p9, event_id=202, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p10_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p10 = mne.Epochs(raw_p10, preparationleft_b1_p10, event_id=218, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p10 = mne.Epochs(raw_p10, preparationright_b1_p10, event_id=219, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p10 = mne.Epochs(raw5_p10, preparationleft_b5_p10, event_id=229, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p10 = mne.Epochs(raw5_p10, preparationright_b5_p10, event_id=230, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p10 = mne.Epochs(raw_p10, preparationleft_b1_p10, event_id=218, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p10 = mne.Epochs(raw_p10, preparationright_b1_p10, event_id=219, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p10 = mne.Epochs(raw5_p10, preparationleft_b5_p10, event_id=229, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p10 = mne.Epochs(raw5_p10, preparationright_b5_p10, event_id=230, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p10 = mne.Epochs(raw_p10, feedbackleft_b1_p10, event_id=212, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p10 = mne.Epochs(raw_p10, feedbackright_b1_p10, event_id=213, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p10 = mne.Epochs(raw5_p10, feedbackleft_b5_p10, event_id=223, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p10 = mne.Epochs(raw5_p10, feedbackright_b5_p10, event_id=224, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p10 = mne.Epochs(raw_p10, feedbackleft_b1_p10, event_id=212, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p10 = mne.Epochs(raw_p10, feedbackright_b1_p10, event_id=213, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p10 = mne.Epochs(raw5_p10, feedbackleft_b5_p10, event_id=223, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p10 = mne.Epochs(raw5_p10, feedbackright_b5_p10, event_id=224, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p11_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p11 = mne.Epochs(raw_p11, preparationleft_b1_p11, event_id=240, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p11 = mne.Epochs(raw_p11, preparationright_b1_p11, event_id=241, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p11 = mne.Epochs(raw5_p11, preparationleft_b5_p11, event_id=251, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p11 = mne.Epochs(raw5_p11, preparationright_b5_p11, event_id=252, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p11 = mne.Epochs(raw_p11, preparationleft_b1_p11, event_id=240, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p11 = mne.Epochs(raw_p11, preparationright_b1_p11, event_id=241, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p11 = mne.Epochs(raw5_p11, preparationleft_b5_p11, event_id=251, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p11 = mne.Epochs(raw5_p11, preparationright_b5_p11, event_id=252, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p11 = mne.Epochs(raw_p11, feedbackleft_b1_p11, event_id=234, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p11 = mne.Epochs(raw_p11, feedbackright_b1_p11, event_id=235, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p11 = mne.Epochs(raw5_p11, feedbackleft_b5_p11, event_id=245, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p11 = mne.Epochs(raw5_p11, feedbackright_b5_p11, event_id=246, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p11 = mne.Epochs(raw_p11, feedbackleft_b1_p11, event_id=234, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p11 = mne.Epochs(raw_p11, feedbackright_b1_p11, event_id=235, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p11 = mne.Epochs(raw5_p11, feedbackleft_b5_p11, event_id=245, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p11 = mne.Epochs(raw5_p11, feedbackright_b5_p11, event_id=246, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p12_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p12 = mne.Epochs(raw_p12, preparationleft_b1_p12, event_id=262, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p12 = mne.Epochs(raw_p12, preparationright_b1_p12, event_id=263, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p12 = mne.Epochs(raw5_p12, preparationleft_b5_p12, event_id=273, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p12 = mne.Epochs(raw5_p12, preparationright_b5_p12, event_id=274, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p12 = mne.Epochs(raw_p12, preparationleft_b1_p12, event_id=262, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p12 = mne.Epochs(raw_p12, preparationright_b1_p12, event_id=263, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p12 = mne.Epochs(raw5_p12, preparationleft_b5_p12, event_id=273, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p12 = mne.Epochs(raw5_p12, preparationright_b5_p12, event_id=274, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p12 = mne.Epochs(raw_p12, feedbackleft_b1_p12, event_id=256, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p12 = mne.Epochs(raw_p12, feedbackright_b1_p12, event_id=257, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p12 = mne.Epochs(raw5_p12, feedbackleft_b5_p12, event_id=267, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p12 = mne.Epochs(raw5_p12, feedbackright_b5_p12, event_id=268, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p12 = mne.Epochs(raw_p12, feedbackleft_b1_p12, event_id=256, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p12 = mne.Epochs(raw_p12, feedbackright_b1_p12, event_id=257, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p12 = mne.Epochs(raw5_p12, feedbackleft_b5_p12, event_id=267, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p12 = mne.Epochs(raw5_p12, feedbackright_b5_p12, event_id=268, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p14_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p14 = mne.Epochs(raw_p14, preparationleft_b1_p14, event_id=284, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p14 = mne.Epochs(raw_p14, preparationright_b1_p14, event_id=285, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p14 = mne.Epochs(raw5_p14, preparationleft_b5_p14, event_id=295, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p14 = mne.Epochs(raw5_p14, preparationright_b5_p14, event_id=296, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p14 = mne.Epochs(raw_p14, preparationleft_b1_p14, event_id=284, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p14 = mne.Epochs(raw_p14, preparationright_b1_p14, event_id=285, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p14 = mne.Epochs(raw5_p14, preparationleft_b5_p14, event_id=295, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p14 = mne.Epochs(raw5_p14, preparationright_b5_p14, event_id=296, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p14 = mne.Epochs(raw_p14, feedbackleft_b1_p14, event_id=278, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p14 = mne.Epochs(raw_p14, feedbackright_b1_p14, event_id=279, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p14 = mne.Epochs(raw5_p14, feedbackleft_b5_p14, event_id=289, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p14 = mne.Epochs(raw5_p14, feedbackright_b5_p14, event_id=290, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p14 = mne.Epochs(raw_p14, feedbackleft_b1_p14, event_id=278, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p14 = mne.Epochs(raw_p14, feedbackright_b1_p14, event_id=279, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p14 = mne.Epochs(raw5_p14, feedbackleft_b5_p14, event_id=289, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p14 = mne.Epochs(raw5_p14, feedbackright_b5_p14, event_id=290, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p15_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p15 = mne.Epochs(raw_p15, preparationleft_b1_p15, event_id=306, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p15 = mne.Epochs(raw_p15, preparationright_b1_p15, event_id=307, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p15 = mne.Epochs(raw5_p15, preparationleft_b5_p15, event_id=317, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p15 = mne.Epochs(raw5_p15, preparationright_b5_p15, event_id=318, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p15 = mne.Epochs(raw_p15, preparationleft_b1_p15, event_id=306, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p15 = mne.Epochs(raw_p15, preparationright_b1_p15, event_id=307, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p15 = mne.Epochs(raw5_p15, preparationleft_b5_p15, event_id=317, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p15 = mne.Epochs(raw5_p15, preparationright_b5_p15, event_id=318, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p15 = mne.Epochs(raw_p15, feedbackleft_b1_p15, event_id=300, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p15 = mne.Epochs(raw_p15, feedbackright_b1_p15, event_id=301, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p15 = mne.Epochs(raw5_p15, feedbackleft_b5_p15, event_id=311, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p15 = mne.Epochs(raw5_p15, feedbackright_b5_p15, event_id=312, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p15 = mne.Epochs(raw_p15, feedbackleft_b1_p15, event_id=300, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p15 = mne.Epochs(raw_p15, feedbackright_b1_p15, event_id=301, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p15 = mne.Epochs(raw5_p15, feedbackleft_b5_p15, event_id=311, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p15 = mne.Epochs(raw5_p15, feedbackright_b5_p15, event_id=312, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p16_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p16 = mne.Epochs(raw_p16, preparationleft_b1_p16, event_id=328, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p16 = mne.Epochs(raw_p16, preparationright_b1_p16, event_id=329, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p16 = mne.Epochs(raw5_p16, preparationleft_b5_p16, event_id=339, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p16 = mne.Epochs(raw5_p16, preparationright_b5_p16, event_id=340, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p16 = mne.Epochs(raw_p16, preparationleft_b1_p16, event_id=328, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p16 = mne.Epochs(raw_p16, preparationright_b1_p16, event_id=329, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p16 = mne.Epochs(raw5_p16, preparationleft_b5_p16, event_id=339, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p16 = mne.Epochs(raw5_p16, preparationright_b5_p16, event_id=340, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p16 = mne.Epochs(raw_p16, feedbackleft_b1_p16, event_id=322, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p16 = mne.Epochs(raw_p16, feedbackright_b1_p16, event_id=323, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p16 = mne.Epochs(raw5_p16, feedbackleft_b5_p16, event_id=333, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p16 = mne.Epochs(raw5_p16, feedbackright_b5_p16, event_id=334, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p16 = mne.Epochs(raw_p16, feedbackleft_b1_p16, event_id=322, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p16 = mne.Epochs(raw_p16, feedbackright_b1_p16, event_id=323, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p16 = mne.Epochs(raw5_p16, feedbackleft_b5_p16, event_id=333, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p16 = mne.Epochs(raw5_p16, feedbackright_b5_p16, event_id=334, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p17_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p17 = mne.Epochs(raw_p17, preparationleft_b1_p17, event_id=350, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p17 = mne.Epochs(raw_p17, preparationright_b1_p17, event_id=351, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p17 = mne.Epochs(raw5_p17, preparationleft_b5_p17, event_id=361, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p17 = mne.Epochs(raw5_p17, preparationright_b5_p17, event_id=362, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p17 = mne.Epochs(raw_p17, preparationleft_b1_p17, event_id=350, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p17 = mne.Epochs(raw_p17, preparationright_b1_p17, event_id=351, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p17 = mne.Epochs(raw5_p17, preparationleft_b5_p17, event_id=361, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p17 = mne.Epochs(raw5_p17, preparationright_b5_p17, event_id=362, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p17 = mne.Epochs(raw_p17, feedbackleft_b1_p17, event_id=344, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p17 = mne.Epochs(raw_p17, feedbackright_b1_p17, event_id=345, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p17 = mne.Epochs(raw5_p17, feedbackleft_b5_p17, event_id=355, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p17 = mne.Epochs(raw5_p17, feedbackright_b5_p17, event_id=356, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p17 = mne.Epochs(raw_p17, feedbackleft_b1_p17, event_id=344, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p17 = mne.Epochs(raw_p17, feedbackright_b1_p17, event_id=345, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p17 = mne.Epochs(raw5_p17, feedbackleft_b5_p17, event_id=355, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p17 = mne.Epochs(raw5_p17, feedbackright_b5_p17, event_id=356, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p18_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p18 = mne.Epochs(raw_p18, preparationleft_b1_p18, event_id=372, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p18 = mne.Epochs(raw_p18, preparationright_b1_p18, event_id=373, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p18 = mne.Epochs(raw5_p18, preparationleft_b5_p18, event_id=383, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p18 = mne.Epochs(raw5_p18, preparationright_b5_p18, event_id=384, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p18 = mne.Epochs(raw_p18, preparationleft_b1_p18, event_id=372, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p18 = mne.Epochs(raw_p18, preparationright_b1_p18, event_id=373, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p18 = mne.Epochs(raw5_p18, preparationleft_b5_p18, event_id=383, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p18 = mne.Epochs(raw5_p18, preparationright_b5_p18, event_id=384, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p18 = mne.Epochs(raw_p18, feedbackleft_b1_p18, event_id=366, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p18 = mne.Epochs(raw_p18, feedbackright_b1_p18, event_id=367, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p18 = mne.Epochs(raw5_p18, feedbackleft_b5_p18, event_id=377, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p18 = mne.Epochs(raw5_p18, feedbackright_b5_p18, event_id=378, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p18 = mne.Epochs(raw_p18, feedbackleft_b1_p18, event_id=366, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p18 = mne.Epochs(raw_p18, feedbackright_b1_p18, event_id=367, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p18 = mne.Epochs(raw5_p18, feedbackleft_b5_p18, event_id=377, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p18 = mne.Epochs(raw5_p18, feedbackright_b5_p18, event_id=378, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p19_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p19 = mne.Epochs(raw_p19, preparationleft_b1_p19, event_id=394, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p19 = mne.Epochs(raw_p19, preparationright_b1_p19, event_id=395, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p19 = mne.Epochs(raw5_p19, preparationleft_b5_p19, event_id=405, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p19 = mne.Epochs(raw5_p19, preparationright_b5_p19, event_id=406, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p19 = mne.Epochs(raw_p19, preparationleft_b1_p19, event_id=394, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p19 = mne.Epochs(raw_p19, preparationright_b1_p19, event_id=395, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p19 = mne.Epochs(raw5_p19, preparationleft_b5_p19, event_id=405, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p19 = mne.Epochs(raw5_p19, preparationright_b5_p19, event_id=406, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p19 = mne.Epochs(raw_p19, feedbackleft_b1_p19, event_id=388, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p19 = mne.Epochs(raw_p19, feedbackright_b1_p19, event_id=389, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p19 = mne.Epochs(raw5_p19, feedbackleft_b5_p19, event_id=399, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p19 = mne.Epochs(raw5_p19, feedbackright_b5_p19, event_id=400, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p19 = mne.Epochs(raw_p19, feedbackleft_b1_p19, event_id=388, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p19 = mne.Epochs(raw_p19, feedbackright_b1_p19, event_id=389, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p19 = mne.Epochs(raw5_p19, feedbackleft_b5_p19, event_id=399, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p19 = mne.Epochs(raw5_p19, feedbackright_b5_p19, event_id=400, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p22_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p22 = mne.Epochs(raw_p22, preparationleft_b1_p22, event_id=416, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p22 = mne.Epochs(raw_p22, preparationright_b1_p22, event_id=417, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p22 = mne.Epochs(raw5_p22, preparationleft_b5_p22, event_id=427, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p22 = mne.Epochs(raw5_p22, preparationright_b5_p22, event_id=428, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p22 = mne.Epochs(raw_p22, preparationleft_b1_p22, event_id=416, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p22 = mne.Epochs(raw_p22, preparationright_b1_p22, event_id=417, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p22 = mne.Epochs(raw5_p22, preparationleft_b5_p22, event_id=427, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p22 = mne.Epochs(raw5_p22, preparationright_b5_p22, event_id=428, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p22 = mne.Epochs(raw_p22, feedbackleft_b1_p22, event_id=410, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p22 = mne.Epochs(raw_p22, feedbackright_b1_p22, event_id=411, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p22 = mne.Epochs(raw5_p22, feedbackleft_b5_p22, event_id=421, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p22 = mne.Epochs(raw5_p22, feedbackright_b5_p22, event_id=422, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p22 = mne.Epochs(raw_p22, feedbackleft_b1_p22, event_id=410, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p22 = mne.Epochs(raw_p22, feedbackright_b1_p22, event_id=411, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p22 = mne.Epochs(raw5_p22, feedbackleft_b5_p22, event_id=421, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p22 = mne.Epochs(raw5_p22, feedbackright_b5_p22, event_id=422, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p23_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p23 = mne.Epochs(raw_p23, preparationleft_b1_p23, event_id=438, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p23 = mne.Epochs(raw_p23, preparationright_b1_p23, event_id=439, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p23 = mne.Epochs(raw5_p23, preparationleft_b5_p23, event_id=449, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p23 = mne.Epochs(raw5_p23, preparationright_b5_p23, event_id=450, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p23 = mne.Epochs(raw_p23, preparationleft_b1_p23, event_id=438, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p23 = mne.Epochs(raw_p23, preparationright_b1_p23, event_id=439, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p23 = mne.Epochs(raw5_p23, preparationleft_b5_p23, event_id=449, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p23 = mne.Epochs(raw5_p23, preparationright_b5_p23, event_id=450, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p23 = mne.Epochs(raw_p23, feedbackleft_b1_p23, event_id=432, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p23 = mne.Epochs(raw_p23, feedbackright_b1_p23, event_id=433, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p23 = mne.Epochs(raw5_p23, feedbackleft_b5_p23, event_id=443, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p23 = mne.Epochs(raw5_p23, feedbackright_b5_p23, event_id=444, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p23 = mne.Epochs(raw_p23, feedbackleft_b1_p23, event_id=432, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p23 = mne.Epochs(raw_p23, feedbackright_b1_p23, event_id=433, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p23 = mne.Epochs(raw5_p23, feedbackleft_b5_p23, event_id=443, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p23 = mne.Epochs(raw5_p23, feedbackright_b5_p23, event_id=444, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p24_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p24 = mne.Epochs(raw_p24, preparationleft_b1_p24, event_id=460, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p24 = mne.Epochs(raw_p24, preparationright_b1_p24, event_id=461, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p24 = mne.Epochs(raw5_p24, preparationleft_b5_p24, event_id=471, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p24 = mne.Epochs(raw5_p24, preparationright_b5_p24, event_id=472, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p24 = mne.Epochs(raw_p24, preparationleft_b1_p24, event_id=460, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p24 = mne.Epochs(raw_p24, preparationright_b1_p24, event_id=461, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p24 = mne.Epochs(raw5_p24, preparationleft_b5_p24, event_id=471, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p24 = mne.Epochs(raw5_p24, preparationright_b5_p24, event_id=472, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p24 = mne.Epochs(raw_p24, feedbackleft_b1_p24, event_id=454, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p24 = mne.Epochs(raw_p24, feedbackright_b1_p24, event_id=455, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p24 = mne.Epochs(raw5_p24, feedbackleft_b5_p24, event_id=465, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p24 = mne.Epochs(raw5_p24, feedbackright_b5_p24, event_id=466, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p24 = mne.Epochs(raw_p24, feedbackleft_b1_p24, event_id=454, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p24 = mne.Epochs(raw_p24, feedbackright_b1_p24, event_id=455, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p24 = mne.Epochs(raw5_p24, feedbackleft_b5_p24, event_id=465, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p24 = mne.Epochs(raw5_p24, feedbackright_b5_p24, event_id=466, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p25_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p25 = mne.Epochs(raw_p25, preparationleft_b1_p25, event_id=482, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p25 = mne.Epochs(raw_p25, preparationright_b1_p25, event_id=483, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p25 = mne.Epochs(raw5_p25, preparationleft_b5_p25, event_id=493, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p25 = mne.Epochs(raw5_p25, preparationright_b5_p25, event_id=494, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p25 = mne.Epochs(raw_p25, preparationleft_b1_p25, event_id=482, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p25 = mne.Epochs(raw_p25, preparationright_b1_p25, event_id=483, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p25 = mne.Epochs(raw5_p25, preparationleft_b5_p25, event_id=493, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p25 = mne.Epochs(raw5_p25, preparationright_b5_p25, event_id=494, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p25 = mne.Epochs(raw_p25, feedbackleft_b1_p25, event_id=476, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p25 = mne.Epochs(raw_p25, feedbackright_b1_p25, event_id=477, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p25 = mne.Epochs(raw5_p25, feedbackleft_b5_p25, event_id=487, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p25 = mne.Epochs(raw5_p25, feedbackright_b5_p25, event_id=488, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p25 = mne.Epochs(raw_p25, feedbackleft_b1_p25, event_id=476, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p25 = mne.Epochs(raw_p25, feedbackright_b1_p25, event_id=477, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p25 = mne.Epochs(raw5_p25, feedbackleft_b5_p25, event_id=487, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p25 = mne.Epochs(raw5_p25, feedbackright_b5_p25, event_id=488, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p26_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p26 = mne.Epochs(raw_p26, preparationleft_b1_p26, event_id=504, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p26 = mne.Epochs(raw_p26, preparationright_b1_p26, event_id=505, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p26 = mne.Epochs(raw5_p26, preparationleft_b5_p26, event_id=515, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p26 = mne.Epochs(raw5_p26, preparationright_b5_p26, event_id=516, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p26 = mne.Epochs(raw_p26, preparationleft_b1_p26, event_id=504, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p26 = mne.Epochs(raw_p26, preparationright_b1_p26, event_id=505, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p26 = mne.Epochs(raw5_p26, preparationleft_b5_p26, event_id=515, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p26 = mne.Epochs(raw5_p26, preparationright_b5_p26, event_id=516, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p26 = mne.Epochs(raw_p26, feedbackleft_b1_p26, event_id=498, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p26 = mne.Epochs(raw_p26, feedbackright_b1_p26, event_id=499, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p26 = mne.Epochs(raw5_p26, feedbackleft_b5_p26, event_id=509, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p26 = mne.Epochs(raw5_p26, feedbackright_b5_p26, event_id=510, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p26 = mne.Epochs(raw_p26, feedbackleft_b1_p26, event_id=498, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p26 = mne.Epochs(raw_p26, feedbackright_b1_p26, event_id=499, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p26 = mne.Epochs(raw5_p26, feedbackleft_b5_p26, event_id=509, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p26 = mne.Epochs(raw5_p26, feedbackright_b5_p26, event_id=510, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p27_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p27 = mne.Epochs(raw_p27, preparationleft_b1_p27, event_id=526, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p27 = mne.Epochs(raw_p27, preparationright_b1_p27, event_id=527, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p27 = mne.Epochs(raw5_p27, preparationleft_b5_p27, event_id=537, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p27 = mne.Epochs(raw5_p27, preparationright_b5_p27, event_id=538, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p27 = mne.Epochs(raw_p27, preparationleft_b1_p27, event_id=526, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p27 = mne.Epochs(raw_p27, preparationright_b1_p27, event_id=527, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p27 = mne.Epochs(raw5_p27, preparationleft_b5_p27, event_id=537, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p27 = mne.Epochs(raw5_p27, preparationright_b5_p27, event_id=538, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p27 = mne.Epochs(raw_p27, feedbackleft_b1_p27, event_id=520, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p27 = mne.Epochs(raw_p27, feedbackright_b1_p27, event_id=521, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p27 = mne.Epochs(raw5_p27, feedbackleft_b5_p27, event_id=531, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p27 = mne.Epochs(raw5_p27, feedbackright_b5_p27, event_id=532, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p27 = mne.Epochs(raw_p27, feedbackleft_b1_p27, event_id=520, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p27 = mne.Epochs(raw_p27, feedbackright_b1_p27, event_id=521, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p27 = mne.Epochs(raw5_p27, feedbackleft_b5_p27, event_id=531, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p27 = mne.Epochs(raw5_p27, feedbackright_b5_p27, event_id=532, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p28_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p28 = mne.Epochs(raw_p28, preparationleft_b1_p28, event_id=548, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p28 = mne.Epochs(raw_p28, preparationright_b1_p28, event_id=549, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p28 = mne.Epochs(raw5_p28, preparationleft_b5_p28, event_id=559, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p28 = mne.Epochs(raw5_p28, preparationright_b5_p28, event_id=560, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p28 = mne.Epochs(raw_p28, preparationleft_b1_p28, event_id=548, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p28 = mne.Epochs(raw_p28, preparationright_b1_p28, event_id=549, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p28 = mne.Epochs(raw5_p28, preparationleft_b5_p28, event_id=559, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p28 = mne.Epochs(raw5_p28, preparationright_b5_p28, event_id=560, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p28 = mne.Epochs(raw_p28, feedbackleft_b1_p28, event_id=542, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p28 = mne.Epochs(raw_p28, feedbackright_b1_p28, event_id=543, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p28 = mne.Epochs(raw5_p28, feedbackleft_b5_p28, event_id=553, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p28 = mne.Epochs(raw5_p28, feedbackright_b5_p28, event_id=554, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p28 = mne.Epochs(raw_p28, feedbackleft_b1_p28, event_id=542, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p28 = mne.Epochs(raw_p28, feedbackright_b1_p28, event_id=543, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p28 = mne.Epochs(raw5_p28, feedbackleft_b5_p28, event_id=553, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p28 = mne.Epochs(raw5_p28, feedbackright_b5_p28, event_id=554, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p29_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p29 = mne.Epochs(raw_p29, preparationleft_b1_p29, event_id=570, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p29 = mne.Epochs(raw_p29, preparationright_b1_p29, event_id=571, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p29 = mne.Epochs(raw5_p29, preparationleft_b5_p29, event_id=581, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p29 = mne.Epochs(raw5_p29, preparationright_b5_p29, event_id=582, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p29 = mne.Epochs(raw_p29, preparationleft_b1_p29, event_id=570, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p29 = mne.Epochs(raw_p29, preparationright_b1_p29, event_id=571, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p29 = mne.Epochs(raw5_p29, preparationleft_b5_p29, event_id=581, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p29 = mne.Epochs(raw5_p29, preparationright_b5_p29, event_id=582, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p29 = mne.Epochs(raw_p29, feedbackleft_b1_p29, event_id=564, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p29 = mne.Epochs(raw_p29, feedbackright_b1_p29, event_id=565, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p29 = mne.Epochs(raw5_p29, feedbackleft_b5_p29, event_id=575, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p29 = mne.Epochs(raw5_p29, feedbackright_b5_p29, event_id=576, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p29 = mne.Epochs(raw_p29, feedbackleft_b1_p29, event_id=564, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p29 = mne.Epochs(raw_p29, feedbackright_b1_p29, event_id=565, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p29 = mne.Epochs(raw5_p29, feedbackleft_b5_p29, event_id=575, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p29 = mne.Epochs(raw5_p29, feedbackright_b5_p29, event_id=576, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)
#_p30_#
#-------------------PREPARATION------------------#
#--------B1-------#
epochsprep1left_p30 = mne.Epochs(raw_p30, preparationleft_b1_p30, event_id=592, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1right_p30 = mne.Epochs(raw_p30, preparationright_b1_p30, event_id=593, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5left_p30 = mne.Epochs(raw5_p30, preparationleft_b5_p30, event_id=603, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5right_p30 = mne.Epochs(raw5_p30, preparationright_b5_p30, event_id=604, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-6.5, -5.5), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------PREPARATION ALL CHANNELS------------------#
#--------B1-------#
epochsprep1leftall_p30 = mne.Epochs(raw_p30, preparationleft_b1_p30, event_id=592, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep1rightall_p30 = mne.Epochs(raw_p30, preparationright_b1_p30, event_id=593, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsprep5leftall_p30 = mne.Epochs(raw5_p30, preparationleft_b5_p30, event_id=603, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsprep5rightall_p30 = mne.Epochs(raw5_p30, preparationright_b5_p30, event_id=604, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-6.5, -5.5), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)


#-------------------FEEDBACK PERIOD------------------#
#--------B1-------#
epochsfeedback1left_p30 = mne.Epochs(raw_p30, feedbackleft_b1_p30, event_id=586, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1right_p30 = mne.Epochs(raw_p30, feedbackright_b1_p30, event_id=587, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5left_p30 = mne.Epochs(raw5_p30, feedbackleft_b5_p30, event_id=597, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5right_p30 = mne.Epochs(raw5_p30, feedbackright_b5_p30, event_id=598, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=reject_criteria, flat=flat_criteria, baseline=(-4, -3), picks=picks, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#-------------------FEEDBACK PERIOD ALL CHANNELS------------------#
#--------B1-------#
epochsfeedback1leftall_p30 = mne.Epochs(raw_p30, feedbackleft_b1_p30, event_id=586, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback1rightall_p30 = mne.Epochs(raw_p30, feedbackright_b1_p30, event_id=587, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

#--------B5-------#
epochsfeedback5leftall_p30 = mne.Epochs(raw5_p30, feedbackleft_b5_p30, event_id=597, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)

epochsfeedback5rightall_p30 = mne.Epochs(raw5_p30, feedbackright_b5_p30, event_id=598, 

                    tmin=tmin3, tmax=tmax3, reject_tmax=0, 

                    reject=None, flat=None, baseline=(-4, -3), picks=picks2, detrend=1, reject_by_annotation=True, preload=True, event_repeated=None) #detrending is set here)



#concatenating epochs for erds and morlet
#all channels
#prep
concatpreparationallb1left = mne.epochs.concatenate_epochs([epochsprep1leftall_p14, epochsprep1leftall_p1, epochsprep1leftall_p2, epochsprep1leftall_p3, epochsprep1leftall_p5, epochsprep1leftall_p6, epochsprep1leftall_p7, epochsprep1leftall_p8, epochsprep1leftall_p9, epochsprep1leftall_p10, epochsprep1leftall_p11, epochsprep1leftall_p12, epochsprep1leftall_p15, epochsprep1leftall_p16, epochsprep1leftall_p17, epochsprep1leftall_p18, epochsprep1leftall_p19, epochsprep1leftall_p22, epochsprep1leftall_p23, epochsprep1leftall_p24, epochsprep1leftall_p25, epochsprep1leftall_p26, epochsprep1leftall_p27, epochsprep1leftall_p28, epochsprep1leftall_p29, epochsprep1leftall_p30])
concatpreparationallb1right = mne.epochs.concatenate_epochs([epochsprep1rightall_p14, epochsprep1rightall_p1, epochsprep1rightall_p2, epochsprep1rightall_p3, epochsprep1rightall_p5, epochsprep1rightall_p6, epochsprep1rightall_p7, epochsprep1rightall_p8, epochsprep1rightall_p9, epochsprep1rightall_p10, epochsprep1rightall_p11, epochsprep1rightall_p12, epochsprep1rightall_p15, epochsprep1rightall_p16, epochsprep1rightall_p17, epochsprep1rightall_p18, epochsprep1rightall_p19, epochsprep1rightall_p22, epochsprep1rightall_p23, epochsprep1rightall_p24, epochsprep1rightall_p25, epochsprep1rightall_p26, epochsprep1rightall_p27, epochsprep1rightall_p28, epochsprep1rightall_p29, epochsprep1rightall_p30])

concatpreparationallb5left = mne.epochs.concatenate_epochs([epochsprep5leftall_p14, epochsprep5leftall_p1, epochsprep5leftall_p2, epochsprep5leftall_p3, epochsprep5leftall_p5, epochsprep5leftall_p6, epochsprep5leftall_p7, epochsprep5leftall_p8, epochsprep5leftall_p9, epochsprep5leftall_p10, epochsprep5leftall_p11, epochsprep5leftall_p12, epochsprep5leftall_p15, epochsprep5leftall_p16, epochsprep5leftall_p17, epochsprep5leftall_p18, epochsprep5leftall_p19, epochsprep5leftall_p22, epochsprep5leftall_p23, epochsprep5leftall_p24, epochsprep5leftall_p25, epochsprep5leftall_p26, epochsprep5leftall_p27, epochsprep5leftall_p28, epochsprep5leftall_p29, epochsprep5leftall_p30])
concatpreparationallb5right = mne.epochs.concatenate_epochs([epochsprep5rightall_p14, epochsprep5rightall_p1, epochsprep5rightall_p2, epochsprep5rightall_p3, epochsprep5rightall_p5, epochsprep5rightall_p6, epochsprep5rightall_p7, epochsprep5rightall_p8, epochsprep5rightall_p9, epochsprep5rightall_p10, epochsprep5rightall_p11, epochsprep5rightall_p12, epochsprep5rightall_p15, epochsprep5rightall_p16, epochsprep5rightall_p17, epochsprep5rightall_p18, epochsprep5rightall_p19, epochsprep5rightall_p22, epochsprep5rightall_p23, epochsprep5rightall_p24, epochsprep5rightall_p25, epochsprep5rightall_p26, epochsprep5rightall_p27, epochsprep5rightall_p28, epochsprep5rightall_p29, epochsprep5rightall_p30])

#feedback
concatfeedbackallb1left = mne.epochs.concatenate_epochs([epochsfeedback1leftall_p14, epochsfeedback1leftall_p1, epochsfeedback1leftall_p2, epochsfeedback1leftall_p3, epochsfeedback1leftall_p5, epochsfeedback1leftall_p6, epochsfeedback1leftall_p7, epochsfeedback1leftall_p8, epochsfeedback1leftall_p9, epochsfeedback1leftall_p10, epochsfeedback1leftall_p11, epochsfeedback1leftall_p12, epochsfeedback1leftall_p15, epochsfeedback1leftall_p16, epochsfeedback1leftall_p17, epochsfeedback1leftall_p18, epochsfeedback1leftall_p19, epochsfeedback1leftall_p22, epochsfeedback1leftall_p23, epochsfeedback1leftall_p24, epochsfeedback1leftall_p25, epochsfeedback1leftall_p26, epochsfeedback1leftall_p27, epochsfeedback1leftall_p28, epochsfeedback1leftall_p29, epochsfeedback1leftall_p30])
concatfeedbackallb1right = mne.epochs.concatenate_epochs([epochsfeedback1rightall_p14, epochsfeedback1rightall_p1, epochsfeedback1rightall_p2, epochsfeedback1rightall_p3, epochsfeedback1rightall_p5, epochsfeedback1rightall_p6, epochsfeedback1rightall_p7, epochsfeedback1rightall_p8, epochsfeedback1rightall_p9, epochsfeedback1rightall_p10, epochsfeedback1rightall_p11, epochsfeedback1rightall_p12, epochsfeedback1rightall_p15, epochsfeedback1rightall_p16, epochsfeedback1rightall_p17, epochsfeedback1rightall_p18, epochsfeedback1rightall_p19, epochsfeedback1rightall_p22, epochsfeedback1rightall_p23, epochsfeedback1rightall_p24, epochsfeedback1rightall_p25, epochsfeedback1rightall_p26, epochsfeedback1rightall_p27, epochsfeedback1rightall_p28, epochsfeedback1rightall_p29, epochsfeedback1rightall_p30])

concatfeedbackallb5left = mne.epochs.concatenate_epochs([epochsfeedback5leftall_p14, epochsfeedback5leftall_p1, epochsfeedback5leftall_p2, epochsfeedback5leftall_p3, epochsfeedback5leftall_p5, epochsfeedback5leftall_p6, epochsfeedback5leftall_p7, epochsfeedback5leftall_p8, epochsfeedback5leftall_p9, epochsfeedback5leftall_p10, epochsfeedback5leftall_p11, epochsfeedback5leftall_p12, epochsfeedback5leftall_p15, epochsfeedback5leftall_p16, epochsfeedback5leftall_p17, epochsfeedback5leftall_p18, epochsfeedback5leftall_p19, epochsfeedback5leftall_p22, epochsfeedback5leftall_p23, epochsfeedback5leftall_p24, epochsfeedback5leftall_p25, epochsfeedback5leftall_p26, epochsfeedback5leftall_p27, epochsfeedback5leftall_p28, epochsfeedback5leftall_p29, epochsfeedback5leftall_p30])
concatfeedbackallb5right= mne.epochs.concatenate_epochs([epochsfeedback5rightall_p14, epochsfeedback5rightall_p1, epochsfeedback5rightall_p2, epochsfeedback5rightall_p3, epochsfeedback5rightall_p5, epochsfeedback5rightall_p6, epochsfeedback5rightall_p7, epochsfeedback5rightall_p8, epochsfeedback5rightall_p9, epochsfeedback5rightall_p10, epochsfeedback5rightall_p11, epochsfeedback5rightall_p12, epochsfeedback5rightall_p15, epochsfeedback5rightall_p16, epochsfeedback5rightall_p17, epochsfeedback5rightall_p18, epochsfeedback5rightall_p19, epochsfeedback5rightall_p22, epochsfeedback5rightall_p23, epochsfeedback5rightall_p24, epochsfeedback5rightall_p25, epochsfeedback5rightall_p26, epochsfeedback5rightall_p27, epochsfeedback5rightall_p28, epochsfeedback5rightall_p29, epochsfeedback5rightall_p30])


#C4, C3, Fc3, Fc4
#prep
concatpreparationb1left = mne.epochs.concatenate_epochs([epochsprep1left_p14, epochsprep1left_p1, epochsprep1left_p2, epochsprep1left_p3, epochsprep1left_p5, epochsprep1left_p6, epochsprep1left_p7, epochsprep1left_p8, epochsprep1left_p9, epochsprep1left_p10, epochsprep1left_p11, epochsprep1left_p12, epochsprep1left_p15, epochsprep1left_p16, epochsprep1left_p17, epochsprep1left_p18, epochsprep1left_p19, epochsprep1left_p22, epochsprep1left_p23, epochsprep1left_p24, epochsprep1left_p25, epochsprep1left_p26, epochsprep1left_p27, epochsprep1left_p28, epochsprep1left_p29, epochsprep1left_p30])
concatpreparationb1right = mne.epochs.concatenate_epochs([epochsprep1right_p14, epochsprep1right_p1, epochsprep1right_p2, epochsprep1right_p3, epochsprep1right_p5, epochsprep1right_p6, epochsprep1right_p7, epochsprep1right_p8, epochsprep1right_p9, epochsprep1right_p10, epochsprep1right_p11, epochsprep1right_p12, epochsprep1right_p15, epochsprep1right_p16, epochsprep1right_p17, epochsprep1right_p18, epochsprep1right_p19, epochsprep1right_p22, epochsprep1right_p23, epochsprep1right_p24, epochsprep1right_p25, epochsprep1right_p26, epochsprep1right_p27, epochsprep1right_p28, epochsprep1right_p29, epochsprep1right_p30])

concatpreparationb5left = mne.epochs.concatenate_epochs([epochsprep5left_p14, epochsprep5left_p1, epochsprep5left_p2, epochsprep5left_p3, epochsprep5left_p5, epochsprep5left_p6, epochsprep5left_p7, epochsprep5left_p8, epochsprep5left_p9, epochsprep5left_p10, epochsprep5left_p11, epochsprep5left_p12, epochsprep5left_p15, epochsprep5left_p16, epochsprep5left_p17, epochsprep5left_p18, epochsprep5left_p19, epochsprep5left_p22, epochsprep5left_p23, epochsprep5left_p24, epochsprep5left_p25, epochsprep5left_p26, epochsprep5left_p27, epochsprep5left_p28, epochsprep5left_p29, epochsprep5left_p30])
concatpreparationb5right = mne.epochs.concatenate_epochs([epochsprep5right_p14, epochsprep5right_p1, epochsprep5right_p2, epochsprep5right_p3, epochsprep5right_p5, epochsprep5right_p6, epochsprep5right_p7, epochsprep5right_p8, epochsprep5right_p9, epochsprep5right_p10, epochsprep5right_p11, epochsprep5right_p12, epochsprep5right_p15, epochsprep5right_p16, epochsprep5right_p17, epochsprep5right_p18, epochsprep5right_p19, epochsprep5right_p22, epochsprep5right_p23, epochsprep5right_p24, epochsprep5right_p25, epochsprep5right_p26, epochsprep5right_p27, epochsprep5right_p28, epochsprep5right_p29, epochsprep5right_p30])


#feedback
concatfeedbackb1left = mne.epochs.concatenate_epochs([epochsfeedback1left_p14, epochsfeedback1left_p1, epochsfeedback1left_p2, epochsfeedback1left_p3, epochsfeedback1left_p5, epochsfeedback1left_p6, epochsfeedback1left_p7, epochsfeedback1left_p8, epochsfeedback1left_p9, epochsfeedback1left_p10, epochsfeedback1left_p11, epochsfeedback1left_p12, epochsfeedback1left_p15, epochsfeedback1left_p16, epochsfeedback1left_p17, epochsfeedback1left_p18, epochsfeedback1left_p19, epochsfeedback1left_p22, epochsfeedback1left_p23, epochsfeedback1left_p24, epochsfeedback1left_p25, epochsfeedback1left_p26, epochsfeedback1left_p27, epochsfeedback1left_p28, epochsfeedback1left_p29, epochsfeedback1left_p30])
concatfeedbackb1right = mne.epochs.concatenate_epochs([epochsfeedback1right_p14, epochsfeedback1right_p1, epochsfeedback1right_p2, epochsfeedback1right_p3, epochsfeedback1right_p5, epochsfeedback1right_p6, epochsfeedback1right_p7, epochsfeedback1right_p8, epochsfeedback1right_p9, epochsfeedback1right_p10, epochsfeedback1right_p11, epochsfeedback1right_p12, epochsfeedback1right_p15, epochsfeedback1right_p16, epochsfeedback1right_p17, epochsfeedback1right_p18, epochsfeedback1right_p19, epochsfeedback1right_p22, epochsfeedback1right_p23, epochsfeedback1right_p24, epochsfeedback1right_p25, epochsfeedback1right_p26, epochsfeedback1right_p27, epochsfeedback1right_p28, epochsfeedback1right_p29, epochsfeedback1right_p30])

concatfeedbackb5left = mne.epochs.concatenate_epochs([epochsfeedback5left_p14, epochsfeedback5left_p1, epochsfeedback5left_p2, epochsfeedback5left_p3, epochsfeedback5left_p5, epochsfeedback5left_p6, epochsfeedback5left_p7, epochsfeedback5left_p8, epochsfeedback5left_p9, epochsfeedback5left_p10, epochsfeedback5left_p11, epochsfeedback5left_p12, epochsfeedback5left_p15, epochsfeedback5left_p16, epochsfeedback5left_p17, epochsfeedback5left_p18, epochsfeedback5left_p19, epochsfeedback5left_p22, epochsfeedback5left_p23, epochsfeedback5left_p24, epochsfeedback5left_p25, epochsfeedback5left_p26, epochsfeedback5left_p27, epochsfeedback5left_p28, epochsfeedback5left_p29, epochsfeedback5left_p30])
concatfeedbackb5right= mne.epochs.concatenate_epochs([epochsfeedback5right_p14, epochsfeedback5right_p1, epochsfeedback5right_p2, epochsfeedback5right_p3, epochsfeedback5right_p5, epochsfeedback5right_p6, epochsfeedback5right_p7, epochsfeedback5right_p8, epochsfeedback5right_p9, epochsfeedback5right_p10, epochsfeedback5right_p11, epochsfeedback5right_p12, epochsfeedback5right_p15, epochsfeedback5right_p16, epochsfeedback5right_p17, epochsfeedback5right_p18, epochsfeedback5right_p19, epochsfeedback5right_p22, epochsfeedback5right_p23, epochsfeedback5right_p24, epochsfeedback5right_p25, epochsfeedback5right_p26, epochsfeedback5right_p27, epochsfeedback5right_p28, epochsfeedback5right_p29, epochsfeedback5right_p30])



#-----------------------------------------------------------------------------#
#                               MORLET WAVELETS
#-----------------------------------------------------------------------------#                          
                            

###determining the frequencies###
freqs = np.arange(12, 29, 1)      # full range
freqsbeta1 = np.arange(12, 16, 1)  # beta1 (12-17Hz)
freqsbeta2 = np.arange(17, 20, 1)  # beta2 (17-23Hz)
freqsbeta3 = np.arange(21, 29, 1)  # beta3 (23-29Hz)

#different number of cycle per frequency
n_cycles = 3 
n_cycles1 = freqsbeta1 / 2.
n_cycles2 = freqsbeta2 / 2.
n_cycles3 = freqsbeta3 / 2.

baseline = [-6.5, -5.5]  # baseline interval (in s)
baseline2 = [-7, -6]
baseline3 = [-4, -3]



#-----------------------------------------------------------------------------#
#                        MORLET PER PARTICIPANT
#-----------------------------------------------------------------------------#

#------------#
#PARTICIPANT 1
#------------#

#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p1, itc1 = tfr_morlet(epochsprep1left_p1, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p1, itc2 = tfr_morlet(epochsprep1right_p1, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p1, itc3 = tfr_morlet(epochsprep5left_p1, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p1, itc4 = tfr_morlet(epochsprep5right_p1, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   

#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p1, itc9 = tfr_morlet(epochsfeedback1left_p1, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p1, itc10 = tfr_morlet(epochsfeedback1right_p1, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p1, itc11 = tfr_morlet(epochsfeedback5left_p1, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p1, itc12 = tfr_morlet(epochsfeedback5right_p1, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#EXPORTING TO DATAFRAME

#------------#
#PARTICIPANT 2
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p2, itc13 = tfr_morlet(epochsprep1left_p2, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p2, itc14 = tfr_morlet(epochsprep1right_p2, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p2, itc15 = tfr_morlet(epochsprep5left_p2, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p2, itc16 = tfr_morlet(epochsprep5right_p2, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p2, itc21 = tfr_morlet(epochsfeedback1left_p2, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p2, itc22 = tfr_morlet(epochsfeedback1right_p2, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p2, itc23 = tfr_morlet(epochsfeedback5left_p2, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p2, itc24 = tfr_morlet(epochsfeedback5right_p2, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#------------#
#PARTICIPANT 3
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p3, itc25 = tfr_morlet(epochsprep1left_p3, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p3, itc26 = tfr_morlet(epochsprep1right_p3, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p3, itc27 = tfr_morlet(epochsprep5left_p3, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p3, itc28 = tfr_morlet(epochsprep5right_p3, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p3, itc33 = tfr_morlet(epochsfeedback1left_p3, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p3, itc34 = tfr_morlet(epochsfeedback1right_p3, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p3, itc35 = tfr_morlet(epochsfeedback5left_p3, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p3, itc36 = tfr_morlet(epochsfeedback5right_p3, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#------------#
#PARTICIPANT 5
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p5, itc49 = tfr_morlet(epochsprep1left_p5, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p5, itc50 = tfr_morlet(epochsprep1right_p5, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p5, itc51 = tfr_morlet(epochsprep5left_p5, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p5, itc52 = tfr_morlet(epochsprep5right_p5, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p5, itc57 = tfr_morlet(epochsfeedback1left_p5, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p5, itc58 = tfr_morlet(epochsfeedback1right_p5, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p5, itc59 = tfr_morlet(epochsfeedback5left_p5, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p5, itc60 = tfr_morlet(epochsfeedback5right_p5, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#------------#
#PARTICIPANT 6
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p6, itc61 = tfr_morlet(epochsprep1left_p6, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p6, itc62 = tfr_morlet(epochsprep1right_p6, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p6, itc63 = tfr_morlet(epochsprep5left_p6, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p6, itc64 = tfr_morlet(epochsprep5right_p6, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p6, itc69 = tfr_morlet(epochsfeedback1left_p6, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p6, itc70 = tfr_morlet(epochsfeedback1right_p6, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p6, itc71 = tfr_morlet(epochsfeedback5left_p6, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p6, itc72 = tfr_morlet(epochsfeedback5right_p6, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#------------#
#PARTICIPANT 7
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p7, itc73 = tfr_morlet(epochsprep1left_p7, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p7, itc74 = tfr_morlet(epochsprep1right_p7, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p7, itc75 = tfr_morlet(epochsprep5left_p7, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p7, itc76 = tfr_morlet(epochsprep5right_p7, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p7, itc81 = tfr_morlet(epochsfeedback1left_p7, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p7, itc82 = tfr_morlet(epochsfeedback1right_p7, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p7, itc83 = tfr_morlet(epochsfeedback5left_p7, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p7, itc84 = tfr_morlet(epochsfeedback5right_p7, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
#------------#
#PARTICIPANT 8
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p8, itc85 = tfr_morlet(epochsprep1left_p8, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p8, itc86 = tfr_morlet(epochsprep1right_p8, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p8, itc87 = tfr_morlet(epochsprep5left_p8, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p8, itc88 = tfr_morlet(epochsprep5right_p8, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p8, itc93 = tfr_morlet(epochsfeedback1left_p8, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p8, itc94 = tfr_morlet(epochsfeedback1right_p8, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p8, itc95 = tfr_morlet(epochsfeedback5left_p8, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p8, itc96 = tfr_morlet(epochsfeedback5right_p8, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#------------#
#PARTICIPANT 9
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p9, itc97 = tfr_morlet(epochsprep1left_p9, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p9, itc98 = tfr_morlet(epochsprep1right_p9, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p9, itc99 = tfr_morlet(epochsprep5left_p9, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p9, itc100 = tfr_morlet(epochsprep5right_p9, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p9, itc105 = tfr_morlet(epochsfeedback1left_p9, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p9, itc106 = tfr_morlet(epochsfeedback1right_p9, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p9, itc107 = tfr_morlet(epochsfeedback5left_p9, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p9, itc108 = tfr_morlet(epochsfeedback5right_p9, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#------------#
#PARTICIPANT 10
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p10, itc109 = tfr_morlet(epochsprep1left_p10, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p10, itc110 = tfr_morlet(epochsprep1right_p10, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p10, itc111 = tfr_morlet(epochsprep5left_p10, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p10, itc112 = tfr_morlet(epochsprep5right_p10, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p10, itc117 = tfr_morlet(epochsfeedback1left_p10, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p10, itc118 = tfr_morlet(epochsfeedback1right_p10, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p10, itc119 = tfr_morlet(epochsfeedback5left_p10, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p10, itc120 = tfr_morlet(epochsfeedback5right_p10, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#------------#
#PARTICIPANT 11
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p11, itc121 = tfr_morlet(epochsprep1left_p11, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p11, itc122 = tfr_morlet(epochsprep1right_p11, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p11, itc123 = tfr_morlet(epochsprep5left_p11, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p11, itc124 = tfr_morlet(epochsprep5right_p11, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p11, itc129 = tfr_morlet(epochsfeedback1left_p11, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p11, itc130 = tfr_morlet(epochsfeedback1right_p11, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p11, itc131 = tfr_morlet(epochsfeedback5left_p11, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p11, itc132 = tfr_morlet(epochsfeedback5right_p11, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#------------#
#PARTICIPANT 12
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p12, itc133 = tfr_morlet(epochsprep1left_p12, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p12, itc134 = tfr_morlet(epochsprep1right_p12, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p12, itc135 = tfr_morlet(epochsprep5left_p12, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p12, itc136 = tfr_morlet(epochsprep5right_p12, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p12, itc141 = tfr_morlet(epochsfeedback1left_p12, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p12, itc142 = tfr_morlet(epochsfeedback1right_p12, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p12, itc143 = tfr_morlet(epochsfeedback5left_p12, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p12, itc144 = tfr_morlet(epochsfeedback5right_p12, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#------------#
#PARTICIPANT 14
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p14, itc145 = tfr_morlet(epochsprep1left_p14, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p14, itc146 = tfr_morlet(epochsprep1right_p14, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p14, itc147 = tfr_morlet(epochsprep5left_p14, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p14, itc148 = tfr_morlet(epochsprep5right_p14, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p14, itc153 = tfr_morlet(epochsfeedback1left_p14, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p14, itc154 = tfr_morlet(epochsfeedback1right_p14, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p14, itc155 = tfr_morlet(epochsfeedback5left_p14, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p14, itc156 = tfr_morlet(epochsfeedback5right_p14, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
#------------#
#PARTICIPANT 15
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p15, itc157 = tfr_morlet(epochsprep1left_p15, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p15, itc158 = tfr_morlet(epochsprep1right_p15, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p15, itc159 = tfr_morlet(epochsprep5left_p15, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p15, itc160 = tfr_morlet(epochsprep5right_p15, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p15, itc165 = tfr_morlet(epochsfeedback1left_p15, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p15, itc166 = tfr_morlet(epochsfeedback1right_p15, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p15, itc167 = tfr_morlet(epochsfeedback5left_p15, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p15, itc168 = tfr_morlet(epochsfeedback5right_p15, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
#------------#
#PARTICIPANT 16
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p16, itc169 = tfr_morlet(epochsprep1left_p16, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p16, itc170 = tfr_morlet(epochsprep1right_p16, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p16, itc171 = tfr_morlet(epochsprep5left_p16, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p16, itc172 = tfr_morlet(epochsprep5right_p16, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p16, itc177 = tfr_morlet(epochsfeedback1left_p16, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p16, itc178 = tfr_morlet(epochsfeedback1right_p16, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p16, itc179 = tfr_morlet(epochsfeedback5left_p16, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p16, itc180 = tfr_morlet(epochsfeedback5right_p16, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#------------#
#PARTICIPANT 17
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p17, itc181 = tfr_morlet(epochsprep1left_p17, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p17, itc182 = tfr_morlet(epochsprep1right_p17, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p17, itc183 = tfr_morlet(epochsprep5left_p17, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p17, itc184 = tfr_morlet(epochsprep5right_p17, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p17, itc189 = tfr_morlet(epochsfeedback1left_p17, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p17, itc190 = tfr_morlet(epochsfeedback1right_p17, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p17, itc191 = tfr_morlet(epochsfeedback5left_p17, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p17, itc192 = tfr_morlet(epochsfeedback5right_p17, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#------------#
#PARTICIPANT 18
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p18, itc193 = tfr_morlet(epochsprep1left_p18, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p18, itc194 = tfr_morlet(epochsprep1right_p18, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p18, itc195 = tfr_morlet(epochsprep5left_p18, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p18, itc196 = tfr_morlet(epochsprep5right_p18, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p18, itc201 = tfr_morlet(epochsfeedback1left_p18, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p18, itc202 = tfr_morlet(epochsfeedback1right_p18, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p18, itc203 = tfr_morlet(epochsfeedback5left_p18, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p18, itc204 = tfr_morlet(epochsfeedback5right_p18, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#------------#
#PARTICIPANT 19
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p19, itc205 = tfr_morlet(epochsprep1left_p19, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p19, itc206 = tfr_morlet(epochsprep1right_p19, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p19, itc207 = tfr_morlet(epochsprep5left_p19, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p19, itc208 = tfr_morlet(epochsprep5right_p19, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p19, itc213 = tfr_morlet(epochsfeedback1left_p19, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p19, itc214 = tfr_morlet(epochsfeedback1right_p19, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p19, itc215 = tfr_morlet(epochsfeedback5left_p19, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p19, itc216 = tfr_morlet(epochsfeedback5right_p19, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#------------#
#PARTICIPANT 22
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p22, itc217 = tfr_morlet(epochsprep1left_p22, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p22, itc218 = tfr_morlet(epochsprep1right_p22, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p22, itc219 = tfr_morlet(epochsprep5left_p22, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p22, itc220 = tfr_morlet(epochsprep5right_p22, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p22, itc225 = tfr_morlet(epochsfeedback1left_p22, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p22, itc226 = tfr_morlet(epochsfeedback1right_p22, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p22, itc227 = tfr_morlet(epochsfeedback5left_p22, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p22, itc228 = tfr_morlet(epochsfeedback5right_p22, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#------------#
#PARTICIPANT 23
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p23, itc229 = tfr_morlet(epochsprep1left_p23, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p23, itc230 = tfr_morlet(epochsprep1right_p23, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p23, itc231 = tfr_morlet(epochsprep5left_p23, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p23, it232 = tfr_morlet(epochsprep5right_p23, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p23, itc237 = tfr_morlet(epochsfeedback1left_p23, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p23, itc238 = tfr_morlet(epochsfeedback1right_p23, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p23, itc239 = tfr_morlet(epochsfeedback5left_p23, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p23, itc240 = tfr_morlet(epochsfeedback5right_p23, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#------------#
#PARTICIPANT 24
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p24, itc241 = tfr_morlet(epochsprep1left_p24, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p24, itc242 = tfr_morlet(epochsprep1right_p24, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p24, itc243 = tfr_morlet(epochsprep5left_p24, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p24, itc244 = tfr_morlet(epochsprep5right_p24, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p24, itc249 = tfr_morlet(epochsfeedback1left_p24, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p24, itc250 = tfr_morlet(epochsfeedback1right_p24, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p24, itc251 = tfr_morlet(epochsfeedback5left_p24, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p24, itc252 = tfr_morlet(epochsfeedback5right_p24, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#------------#
#PARTICIPANT 25
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p25, itc253 = tfr_morlet(epochsprep1left_p25, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p25, itc254 = tfr_morlet(epochsprep1right_p25, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p25, itc255 = tfr_morlet(epochsprep5left_p25, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p25, itc256 = tfr_morlet(epochsprep5right_p25, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   



#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p25, itc261 = tfr_morlet(epochsfeedback1left_p25, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p25, itc262 = tfr_morlet(epochsfeedback1right_p25, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p25, itc263 = tfr_morlet(epochsfeedback5left_p25, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p25, itc264 = tfr_morlet(epochsfeedback5right_p25, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#------------#
#PARTICIPANT 26
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p26, itc265 = tfr_morlet(epochsprep1left_p26, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p26, itc266 = tfr_morlet(epochsprep1right_p26, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p26, itc263 = tfr_morlet(epochsprep5left_p26, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p26, itc264 = tfr_morlet(epochsprep5right_p26, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p26, itc269 = tfr_morlet(epochsfeedback1left_p26, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p26, itc270 = tfr_morlet(epochsfeedback1right_p26, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p26, itc271 = tfr_morlet(epochsfeedback5left_p26, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p26, itc272 = tfr_morlet(epochsfeedback5right_p26, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#------------#
#PARTICIPANT 27
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p27, itc273 = tfr_morlet(epochsprep1left_p27, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p27, itc274 = tfr_morlet(epochsprep1right_p27, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p27, itc275 = tfr_morlet(epochsprep5left_p27, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p27, itc276 = tfr_morlet(epochsprep5right_p27, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p27, itc281 = tfr_morlet(epochsfeedback1left_p27, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p27, itc282 = tfr_morlet(epochsfeedback1right_p27, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p27, itc283 = tfr_morlet(epochsfeedback5left_p27, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p27, itc284 = tfr_morlet(epochsfeedback5right_p27, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
#------------#
#PARTICIPANT 28
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p28, itc285 = tfr_morlet(epochsprep1left_p28, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p28, itc286 = tfr_morlet(epochsprep1right_p28, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p28, itc287 = tfr_morlet(epochsprep5left_p28, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p28, itc288 = tfr_morlet(epochsprep5right_p28, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   

#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p28, itc293 = tfr_morlet(epochsfeedback1left_p28, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p28, itc294 = tfr_morlet(epochsfeedback1right_p28, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p28, itc295 = tfr_morlet(epochsfeedback5left_p28, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p28, itc296 = tfr_morlet(epochsfeedback5right_p28, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#------------#
#PARTICIPANT 29
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p29, itc297 = tfr_morlet(epochsprep1left_p29, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p29, itc298 = tfr_morlet(epochsprep1right_p29, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p29, itc299 = tfr_morlet(epochsprep5left_p29, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p29, itc300 = tfr_morlet(epochsprep5right_p29, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p29, itc305 = tfr_morlet(epochsfeedback1left_p29, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p29, itc306 = tfr_morlet(epochsfeedback1right_p29, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p29, itc307 = tfr_morlet(epochsfeedback5left_p29, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p29, itc308 = tfr_morlet(epochsfeedback5right_p29, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#------------#
#PARTICIPANT 30
#------------#
#-------------------PREPARATION----------------#

#----B1----#
powerprep1allbandsleft_p30, itc309 = tfr_morlet(epochsprep1left_p30, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright_p30, itc310 = tfr_morlet(epochsprep1right_p30, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#----B5----#
powerprep5allbandsleft_p30, itc311 = tfr_morlet(epochsprep5left_p30, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright_p30, itc312 = tfr_morlet(epochsprep5right_p30, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
                   


#-------------------FEEDBACK------------------#

#----B1----#
powerfeedback1allbandsleft_p30, itc317 = tfr_morlet(epochsfeedback1left_p30, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright_p30, itc318 = tfr_morlet(epochsfeedback1right_p30, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allbandsleft_p30, itc319 = tfr_morlet(epochsfeedback5left_p30, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allbandsright_p30, itc320 = tfr_morlet(epochsfeedback5right_p30, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)


#------------#
#PARTICIPANT 1
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p1 = powerprep1allbandsleft_p1.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p1['Block'] = 'B1'
dfpowerprep1allbandsleft_p1['Hand'] = 'left'
dfpowerprep1allbandsleft_p1['Participant'] = '1'
print(dfpowerprep1allbandsleft_p1)

dfpowerprep1allbandsright_p1 = powerprep1allbandsright_p1.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p1['Block'] = 'B1'
dfpowerprep1allbandsright_p1['Hand'] = 'right'
dfpowerprep1allbandsright_p1['Participant'] = '1'
print(dfpowerprep1allbandsright_p1)

#b5
dfpowerprep5allbandsleft_p1 = powerprep5allbandsleft_p1.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p1['Block'] = 'B5'
dfpowerprep5allbandsleft_p1['Hand'] = 'left'
dfpowerprep5allbandsleft_p1['Participant'] = '1'
print(dfpowerprep5allbandsleft_p1)

dfpowerprep5allbandsright_p1 = powerprep5allbandsright_p1.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p1['Block'] = 'B5'
dfpowerprep5allbandsright_p1['Hand'] = 'right'
dfpowerprep5allbandsright_p1['Participant'] = '1'
print(dfpowerprep5allbandsright_p1)



#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p1 = powerfeedback1allbandsleft_p1.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p1['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p1['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p1['Participant'] = '1'
print(dfpowerfeedback1allbandsleft_p1)

dfpowerfeedback1allbandsright_p1 = powerfeedback1allbandsright_p1.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p1['Block'] = 'B1'
dfpowerfeedback1allbandsright_p1['Hand'] = 'right'
dfpowerfeedback1allbandsright_p1['Participant'] = '1'
print(dfpowerfeedback1allbandsright_p1)

#b5
dfpowerfeedback5allbandsleft_p1 = powerfeedback5allbandsleft_p1.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p1['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p1['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p1['Participant'] = '1'
print(dfpowerfeedback5allbandsleft_p1)

dfpowerfeedback5allbandsright_p1 = powerfeedback5allbandsright_p1.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p1['Block'] = 'B5'
dfpowerfeedback5allbandsright_p1['Hand'] = 'right'
dfpowerfeedback5allbandsright_p1['Participant'] = '1'
print(dfpowerfeedback5allbandsright_p1)


#------------#
#PARTICIPANT 2
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p2 = powerprep1allbandsleft_p2.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p2['Block'] = 'B1'
dfpowerprep1allbandsleft_p2['Hand'] = 'left'
dfpowerprep1allbandsleft_p2['Participant'] = '2'
print(dfpowerprep1allbandsleft_p2)

dfpowerprep1allbandsright_p2 = powerprep1allbandsright_p2.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p2['Block'] = 'B1'
dfpowerprep1allbandsright_p2['Hand'] = 'right'
dfpowerprep1allbandsright_p2['Participant'] = '2'
print(dfpowerprep1allbandsright_p2)

#b5
dfpowerprep5allbandsleft_p2 = powerprep5allbandsleft_p2.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p2['Block'] = 'B5'
dfpowerprep5allbandsleft_p2['Hand'] = 'left'
dfpowerprep5allbandsleft_p2['Participant'] = '2'
print(dfpowerprep5allbandsleft_p2)

dfpowerprep5allbandsright_p2 = powerprep5allbandsright_p2.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p2['Block'] = 'B5'
dfpowerprep5allbandsright_p2['Hand'] = 'right'
dfpowerprep5allbandsright_p2['Participant'] = '2'
print(dfpowerprep5allbandsright_p2)


#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p2 = powerfeedback1allbandsleft_p2.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p2['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p2['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p2['Participant'] = '2'
print(dfpowerfeedback1allbandsleft_p2)

dfpowerfeedback1allbandsright_p2 = powerfeedback1allbandsright_p2.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p2['Block'] = 'B1'
dfpowerfeedback1allbandsright_p2['Hand'] = 'right'
dfpowerfeedback1allbandsright_p2['Participant'] = '2'
print(dfpowerfeedback1allbandsright_p2)

#b5
dfpowerfeedback5allbandsleft_p2 = powerfeedback5allbandsleft_p2.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p2['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p2['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p2['Participant'] = '2'
print(dfpowerfeedback5allbandsleft_p2)

dfpowerfeedback5allbandsright_p2 = powerfeedback5allbandsright_p2.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p2['Block'] = 'B5'
dfpowerfeedback5allbandsright_p2['Hand'] = 'right'
dfpowerfeedback5allbandsright_p2['Participant'] = '2'
print(dfpowerfeedback5allbandsright_p2)


#------------#
#PARTICIPANT 3
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p3 = powerprep1allbandsleft_p3.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p3['Block'] = 'B1'
dfpowerprep1allbandsleft_p3['Hand'] = 'left'
dfpowerprep1allbandsleft_p3['Participant'] = '3'
print(dfpowerprep1allbandsleft_p3)

dfpowerprep1allbandsright_p3 = powerprep1allbandsright_p3.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p3['Block'] = 'B1'
dfpowerprep1allbandsright_p3['Hand'] = 'right'
dfpowerprep1allbandsright_p3['Participant'] = '3'
print(dfpowerprep1allbandsright_p3)

#b5
dfpowerprep5allbandsleft_p3 = powerprep5allbandsleft_p3.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p3['Block'] = 'B5'
dfpowerprep5allbandsleft_p3['Hand'] = 'left'
dfpowerprep5allbandsleft_p3['Participant'] = '3'
print(dfpowerprep5allbandsleft_p3)

dfpowerprep5allbandsright_p3 = powerprep5allbandsright_p3.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p3['Block'] = 'B5'
dfpowerprep5allbandsright_p3['Hand'] = 'right'
dfpowerprep5allbandsright_p3['Participant'] = '3'
print(dfpowerprep5allbandsright_p3)


#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p3 = powerfeedback1allbandsleft_p3.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p3['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p3['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p3['Participant'] = '3'
print(dfpowerfeedback1allbandsleft_p3)

dfpowerfeedback1allbandsright_p3 = powerfeedback1allbandsright_p3.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p3['Block'] = 'B1'
dfpowerfeedback1allbandsright_p3['Hand'] = 'right'
dfpowerfeedback1allbandsright_p3['Participant'] = '3'
print(dfpowerfeedback1allbandsright_p3)

#b5
dfpowerfeedback5allbandsleft_p3 = powerfeedback5allbandsleft_p3.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p3['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p3['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p3['Participant'] = '3'
print(dfpowerfeedback5allbandsleft_p3)

dfpowerfeedback5allbandsright_p3 = powerfeedback5allbandsright_p3.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p3['Block'] = 'B5'
dfpowerfeedback5allbandsright_p3['Hand'] = 'right'
dfpowerfeedback5allbandsright_p3['Participant'] = '3'
print(dfpowerfeedback5allbandsright_p3)


#------------#
#PARTICIPANT 5
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p5 = powerprep1allbandsleft_p5.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p5['Block'] = 'B1'
dfpowerprep1allbandsleft_p5['Hand'] = 'left'
dfpowerprep1allbandsleft_p5['Participant'] = '5'
print(dfpowerprep1allbandsleft_p5)

dfpowerprep1allbandsright_p5 = powerprep1allbandsright_p5.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p5['Block'] = 'B1'
dfpowerprep1allbandsright_p5['Hand'] = 'right'
dfpowerprep1allbandsright_p5['Participant'] = '5'
print(dfpowerprep1allbandsright_p5)

#b5
dfpowerprep5allbandsleft_p5 = powerprep5allbandsleft_p5.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p5['Block'] = 'B5'
dfpowerprep5allbandsleft_p5['Hand'] = 'left'
dfpowerprep5allbandsleft_p5['Participant'] = '5'
print(dfpowerprep5allbandsleft_p5)

dfpowerprep5allbandsright_p5 = powerprep5allbandsright_p5.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p5['Block'] = 'B5'
dfpowerprep5allbandsright_p5['Hand'] = 'right'
dfpowerprep5allbandsright_p5['Participant'] = '5'
print(dfpowerprep5allbandsright_p5)


#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p5 = powerfeedback1allbandsleft_p5.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p5['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p5['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p5['Participant'] = '5'
print(dfpowerfeedback1allbandsleft_p5)

dfpowerfeedback1allbandsright_p5 = powerfeedback1allbandsright_p5.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p5['Block'] = 'B1'
dfpowerfeedback1allbandsright_p5['Hand'] = 'right'
dfpowerfeedback1allbandsright_p5['Participant'] = '5'
print(dfpowerfeedback1allbandsright_p5)

#b5
dfpowerfeedback5allbandsleft_p5 = powerfeedback5allbandsleft_p5.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p5['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p5['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p5['Participant'] = '5'
print(dfpowerfeedback5allbandsleft_p5)

dfpowerfeedback5allbandsright_p5 = powerfeedback5allbandsright_p5.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p5['Block'] = 'B5'
dfpowerfeedback5allbandsright_p5['Hand'] = 'right'
dfpowerfeedback5allbandsright_p5['Participant'] = '5'
print(dfpowerfeedback5allbandsright_p5)

#------------#
#PARTICIPANT 6
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p6 = powerprep1allbandsleft_p6.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p6['Block'] = 'B1'
dfpowerprep1allbandsleft_p6['Hand'] = 'left'
dfpowerprep1allbandsleft_p6['Participant'] = '6'
print(dfpowerprep1allbandsleft_p6)

dfpowerprep1allbandsright_p6 = powerprep1allbandsright_p6.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p6['Block'] = 'B1'
dfpowerprep1allbandsright_p6['Hand'] = 'right'
dfpowerprep1allbandsright_p6['Participant'] = '6'
print(dfpowerprep1allbandsright_p6)

#b5
dfpowerprep5allbandsleft_p6 = powerprep5allbandsleft_p6.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p6['Block'] = 'B5'
dfpowerprep5allbandsleft_p6['Hand'] = 'left'
dfpowerprep5allbandsleft_p6['Participant'] = '6'
print(dfpowerprep5allbandsleft_p6)

dfpowerprep5allbandsright_p6 = powerprep5allbandsright_p6.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p6['Block'] = 'B5'
dfpowerprep5allbandsright_p6['Hand'] = 'right'
dfpowerprep5allbandsright_p6['Participant'] = '6'
print(dfpowerprep5allbandsright_p6)



#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p6 = powerfeedback1allbandsleft_p6.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p6['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p6['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p6['Participant'] = '6'
print(dfpowerfeedback1allbandsleft_p6)

dfpowerfeedback1allbandsright_p6 = powerfeedback1allbandsright_p6.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p6['Block'] = 'B1'
dfpowerfeedback1allbandsright_p6['Hand'] = 'right'
dfpowerfeedback1allbandsright_p6['Participant'] = '6'
print(dfpowerfeedback1allbandsright_p6)

#b5
dfpowerfeedback5allbandsleft_p6 = powerfeedback5allbandsleft_p6.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p6['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p6['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p6['Participant'] = '6'
print(dfpowerfeedback5allbandsleft_p6)

dfpowerfeedback5allbandsright_p6 = powerfeedback5allbandsright_p6.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p6['Block'] = 'B5'
dfpowerfeedback5allbandsright_p6['Hand'] = 'right'
dfpowerfeedback5allbandsright_p6['Participant'] = '6'
print(dfpowerfeedback5allbandsright_p6)

#------------#
#PARTICIPANT 7
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p7 = powerprep1allbandsleft_p7.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p7['Block'] = 'B1'
dfpowerprep1allbandsleft_p7['Hand'] = 'left'
dfpowerprep1allbandsleft_p7['Participant'] = '7'
print(dfpowerprep1allbandsleft_p7)

dfpowerprep1allbandsright_p7 = powerprep1allbandsright_p7.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p7['Block'] = 'B1'
dfpowerprep1allbandsright_p7['Hand'] = 'right'
dfpowerprep1allbandsright_p7['Participant'] = '7'
print(dfpowerprep1allbandsright_p7)

#b5
dfpowerprep5allbandsleft_p7 = powerprep5allbandsleft_p7.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p7['Block'] = 'B5'
dfpowerprep5allbandsleft_p7['Hand'] = 'left'
dfpowerprep5allbandsleft_p7['Participant'] = '7'
print(dfpowerprep5allbandsleft_p7)

dfpowerprep5allbandsright_p7 = powerprep5allbandsright_p7.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p7['Block'] = 'B5'
dfpowerprep5allbandsright_p7['Hand'] = 'right'
dfpowerprep5allbandsright_p7['Participant'] = '7'
print(dfpowerprep5allbandsright_p7)


#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p7 = powerfeedback1allbandsleft_p7.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p7['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p7['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p7['Participant'] = '7'
print(dfpowerfeedback1allbandsleft_p7)

dfpowerfeedback1allbandsright_p7 = powerfeedback1allbandsright_p7.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p7['Block'] = 'B1'
dfpowerfeedback1allbandsright_p7['Hand'] = 'right'
dfpowerfeedback1allbandsright_p7['Participant'] = '7'
print(dfpowerfeedback1allbandsright_p7)

#b5
dfpowerfeedback5allbandsleft_p7 = powerfeedback5allbandsleft_p7.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p7['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p7['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p7['Participant'] = '7'
print(dfpowerfeedback5allbandsleft_p7)

dfpowerfeedback5allbandsright_p7 = powerfeedback5allbandsright_p7.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p7['Block'] = 'B5'
dfpowerfeedback5allbandsright_p7['Hand'] = 'right'
dfpowerfeedback5allbandsright_p7['Participant'] = '7'
print(dfpowerfeedback5allbandsright_p7)

#------------#
#PARTICIPANT 8
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p8 = powerprep1allbandsleft_p8.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p8['Block'] = 'B1'
dfpowerprep1allbandsleft_p8['Hand'] = 'left'
dfpowerprep1allbandsleft_p8['Participant'] = '8'
print(dfpowerprep1allbandsleft_p8)

dfpowerprep1allbandsright_p8 = powerprep1allbandsright_p8.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p8['Block'] = 'B1'
dfpowerprep1allbandsright_p8['Hand'] = 'right'
dfpowerprep1allbandsright_p8['Participant'] = '8'
print(dfpowerprep1allbandsright_p8)

#b5
dfpowerprep5allbandsleft_p8 = powerprep5allbandsleft_p8.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p8['Block'] = 'B5'
dfpowerprep5allbandsleft_p8['Hand'] = 'left'
dfpowerprep5allbandsleft_p8['Participant'] = '8'
print(dfpowerprep5allbandsleft_p8)

dfpowerprep5allbandsright_p8 = powerprep5allbandsright_p8.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p8['Block'] = 'B5'
dfpowerprep5allbandsright_p8['Hand'] = 'right'
dfpowerprep5allbandsright_p8['Participant'] = '8'
print(dfpowerprep5allbandsright_p8)


#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p8 = powerfeedback1allbandsleft_p8.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p8['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p8['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p8['Participant'] = '8'
print(dfpowerfeedback1allbandsleft_p8)

dfpowerfeedback1allbandsright_p8 = powerfeedback1allbandsright_p8.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p8['Block'] = 'B1'
dfpowerfeedback1allbandsright_p8['Hand'] = 'right'
dfpowerfeedback1allbandsright_p8['Participant'] = '8'
print(dfpowerfeedback1allbandsright_p8)

#b5
dfpowerfeedback5allbandsleft_p8 = powerfeedback5allbandsleft_p8.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p8['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p8['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p8['Participant'] = '8'
print(dfpowerfeedback5allbandsleft_p8)

dfpowerfeedback5allbandsright_p8 = powerfeedback5allbandsright_p8.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p8['Block'] = 'B5'
dfpowerfeedback5allbandsright_p8['Hand'] = 'right'
dfpowerfeedback5allbandsright_p8['Participant'] = '8'
print(dfpowerfeedback5allbandsright_p8)

#------------#
#PARTICIPANT 9
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p9 = powerprep1allbandsleft_p9.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p9['Block'] = 'B1'
dfpowerprep1allbandsleft_p9['Hand'] = 'left'
dfpowerprep1allbandsleft_p9['Participant'] = '9'
print(dfpowerprep1allbandsleft_p9)

dfpowerprep1allbandsright_p9 = powerprep1allbandsright_p9.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p9['Block'] = 'B1'
dfpowerprep1allbandsright_p9['Hand'] = 'right'
dfpowerprep1allbandsright_p9['Participant'] = '9'
print(dfpowerprep1allbandsright_p9)

#b5
dfpowerprep5allbandsleft_p9 = powerprep5allbandsleft_p9.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p9['Block'] = 'B5'
dfpowerprep5allbandsleft_p9['Hand'] = 'left'
dfpowerprep5allbandsleft_p9['Participant'] = '9'
print(dfpowerprep5allbandsleft_p9)

dfpowerprep5allbandsright_p9 = powerprep5allbandsright_p9.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p9['Block'] = 'B5'
dfpowerprep5allbandsright_p9['Hand'] = 'right'
dfpowerprep5allbandsright_p9['Participant'] = '9'
print(dfpowerprep5allbandsright_p9)


#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p9 = powerfeedback1allbandsleft_p9.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p9['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p9['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p9['Participant'] = '9'
print(dfpowerfeedback1allbandsleft_p9)

dfpowerfeedback1allbandsright_p9 = powerfeedback1allbandsright_p9.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p9['Block'] = 'B1'
dfpowerfeedback1allbandsright_p9['Hand'] = 'right'
dfpowerfeedback1allbandsright_p9['Participant'] = '9'
print(dfpowerfeedback1allbandsright_p9)

#b5
dfpowerfeedback5allbandsleft_p9 = powerfeedback5allbandsleft_p9.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p9['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p9['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p9['Participant'] = '9'
print(dfpowerfeedback5allbandsleft_p9)

dfpowerfeedback5allbandsright_p9 = powerfeedback5allbandsright_p9.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p9['Block'] = 'B5'
dfpowerfeedback5allbandsright_p9['Hand'] = 'right'
dfpowerfeedback5allbandsright_p9['Participant'] = '9'
print(dfpowerfeedback5allbandsright_p9)

#------------#
#PARTICIPANT 10
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p10 = powerprep1allbandsleft_p10.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p10['Block'] = 'B1'
dfpowerprep1allbandsleft_p10['Hand'] = 'left'
dfpowerprep1allbandsleft_p10['Participant'] = '10'
print(dfpowerprep1allbandsleft_p10)

dfpowerprep1allbandsright_p10 = powerprep1allbandsright_p10.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p10['Block'] = 'B1'
dfpowerprep1allbandsright_p10['Hand'] = 'right'
dfpowerprep1allbandsright_p10['Participant'] = '10'
print(dfpowerprep1allbandsright_p10)

#b5
dfpowerprep5allbandsleft_p10 = powerprep5allbandsleft_p10.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p10['Block'] = 'B5'
dfpowerprep5allbandsleft_p10['Hand'] = 'left'
dfpowerprep5allbandsleft_p10['Participant'] = '10'
print(dfpowerprep5allbandsleft_p10)

dfpowerprep5allbandsright_p10 = powerprep5allbandsright_p10.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p10['Block'] = 'B5'
dfpowerprep5allbandsright_p10['Hand'] = 'right'
dfpowerprep5allbandsright_p10['Participant'] = '10'
print(dfpowerprep5allbandsright_p10)


#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p10 = powerfeedback1allbandsleft_p10.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p10['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p10['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p10['Participant'] = '10'
print(dfpowerfeedback1allbandsleft_p10)

dfpowerfeedback1allbandsright_p10 = powerfeedback1allbandsright_p10.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p10['Block'] = 'B1'
dfpowerfeedback1allbandsright_p10['Hand'] = 'right'
dfpowerfeedback1allbandsright_p10['Participant'] = '10'
print(dfpowerfeedback1allbandsright_p10)

#b5
dfpowerfeedback5allbandsleft_p10 = powerfeedback5allbandsleft_p10.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p10['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p10['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p10['Participant'] = '10'
print(dfpowerfeedback5allbandsleft_p10)

dfpowerfeedback5allbandsright_p10 = powerfeedback5allbandsright_p10.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p10['Block'] = 'B5'
dfpowerfeedback5allbandsright_p10['Hand'] = 'right'
dfpowerfeedback5allbandsright_p10['Participant'] = '10'
print(dfpowerfeedback5allbandsright_p10)

#------------#
#PARTICIPANT 11
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p11 = powerprep1allbandsleft_p11.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p11['Block'] = 'B1'
dfpowerprep1allbandsleft_p11['Hand'] = 'left'
dfpowerprep1allbandsleft_p11['Participant'] = '11'
print(dfpowerprep1allbandsleft_p11)

dfpowerprep1allbandsright_p11 = powerprep1allbandsright_p11.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p11['Block'] = 'B1'
dfpowerprep1allbandsright_p11['Hand'] = 'right'
dfpowerprep1allbandsright_p11['Participant'] = '11'
print(dfpowerprep1allbandsright_p11)

#b5
dfpowerprep5allbandsleft_p11 = powerprep5allbandsleft_p11.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p11['Block'] = 'B5'
dfpowerprep5allbandsleft_p11['Hand'] = 'left'
dfpowerprep5allbandsleft_p11['Participant'] = '11'
print(dfpowerprep5allbandsleft_p11)

dfpowerprep5allbandsright_p11 = powerprep5allbandsright_p11.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p11['Block'] = 'B5'
dfpowerprep5allbandsright_p11['Hand'] = 'right'
dfpowerprep5allbandsright_p11['Participant'] = '11'
print(dfpowerprep5allbandsright_p11)


#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p11 = powerfeedback1allbandsleft_p11.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p11['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p11['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p11['Participant'] = '11'
print(dfpowerfeedback1allbandsleft_p11)

dfpowerfeedback1allbandsright_p11 = powerfeedback1allbandsright_p11.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p11['Block'] = 'B1'
dfpowerfeedback1allbandsright_p11['Hand'] = 'right'
dfpowerfeedback1allbandsright_p11['Participant'] = '11'
print(dfpowerfeedback1allbandsright_p11)

#b5
dfpowerfeedback5allbandsleft_p11 = powerfeedback5allbandsleft_p11.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p11['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p11['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p11['Participant'] = '11'
print(dfpowerfeedback5allbandsleft_p11)

dfpowerfeedback5allbandsright_p11 = powerfeedback5allbandsright_p11.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p11['Block'] = 'B5'
dfpowerfeedback5allbandsright_p11['Hand'] = 'right'
dfpowerfeedback5allbandsright_p11['Participant'] = '11'
print(dfpowerfeedback5allbandsright_p11)

#------------#
#PARTICIPANT 12
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p12 = powerprep1allbandsleft_p12.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p12['Block'] = 'B1'
dfpowerprep1allbandsleft_p12['Hand'] = 'left'
dfpowerprep1allbandsleft_p12['Participant'] = '12'
print(dfpowerprep1allbandsleft_p12)

dfpowerprep1allbandsright_p12 = powerprep1allbandsright_p12.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p12['Block'] = 'B1'
dfpowerprep1allbandsright_p12['Hand'] = 'right'
dfpowerprep1allbandsright_p12['Participant'] = '12'
print(dfpowerprep1allbandsright_p12)

#b5
dfpowerprep5allbandsleft_p12 = powerprep5allbandsleft_p12.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p12['Block'] = 'B5'
dfpowerprep5allbandsleft_p12['Hand'] = 'left'
dfpowerprep5allbandsleft_p12['Participant'] = '12'
print(dfpowerprep5allbandsleft_p12)

dfpowerprep5allbandsright_p12 = powerprep5allbandsright_p12.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p12['Block'] = 'B5'
dfpowerprep5allbandsright_p12['Hand'] = 'right'
dfpowerprep5allbandsright_p12['Participant'] = '12'
print(dfpowerprep5allbandsright_p12)


#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p12 = powerfeedback1allbandsleft_p12.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p12['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p12['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p12['Participant'] = '12'
print(dfpowerfeedback1allbandsleft_p12)

dfpowerfeedback1allbandsright_p12 = powerfeedback1allbandsright_p12.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p12['Block'] = 'B1'
dfpowerfeedback1allbandsright_p12['Hand'] = 'right'
dfpowerfeedback1allbandsright_p12['Participant'] = '12'
print(dfpowerfeedback1allbandsright_p12)

#b5
dfpowerfeedback5allbandsleft_p12 = powerfeedback5allbandsleft_p12.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p12['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p12['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p12['Participant'] = '12'
print(dfpowerfeedback5allbandsleft_p12)

dfpowerfeedback5allbandsright_p12 = powerfeedback5allbandsright_p12.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p12['Block'] = 'B5'
dfpowerfeedback5allbandsright_p12['Hand'] = 'right'
dfpowerfeedback5allbandsright_p12['Participant'] = '12'
print(dfpowerfeedback5allbandsright_p12)

#------------#
#PARTICIPANT 14
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p14 = powerprep1allbandsleft_p14.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p14['Block'] = 'B1'
dfpowerprep1allbandsleft_p14['Hand'] = 'left'
dfpowerprep1allbandsleft_p14['Participant'] = '14'
print(dfpowerprep1allbandsleft_p14)

dfpowerprep1allbandsright_p14 = powerprep1allbandsright_p14.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p14['Block'] = 'B1'
dfpowerprep1allbandsright_p14['Hand'] = 'right'
dfpowerprep1allbandsright_p14['Participant'] = '14'
print(dfpowerprep1allbandsright_p14)

#b5
dfpowerprep5allbandsleft_p14 = powerprep5allbandsleft_p14.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p14['Block'] = 'B5'
dfpowerprep5allbandsleft_p14['Hand'] = 'left'
dfpowerprep5allbandsleft_p14['Participant'] = '14'
print(dfpowerprep5allbandsleft_p14)

dfpowerprep5allbandsright_p14 = powerprep5allbandsright_p14.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p14['Block'] = 'B5'
dfpowerprep5allbandsright_p14['Hand'] = 'right'
dfpowerprep5allbandsright_p14['Participant'] = '14'
print(dfpowerprep5allbandsright_p14)


#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p14 = powerfeedback1allbandsleft_p14.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p14['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p14['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p14['Participant'] = '14'
print(dfpowerfeedback1allbandsleft_p14)

dfpowerfeedback1allbandsright_p14 = powerfeedback1allbandsright_p14.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p14['Block'] = 'B1'
dfpowerfeedback1allbandsright_p14['Hand'] = 'right'
dfpowerfeedback1allbandsright_p14['Participant'] = '14'
print(dfpowerfeedback1allbandsright_p14)

#b5
dfpowerfeedback5allbandsleft_p14 = powerfeedback5allbandsleft_p14.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p14['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p14['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p14['Participant'] = '14'
print(dfpowerfeedback5allbandsleft_p14)

dfpowerfeedback5allbandsright_p14 = powerfeedback5allbandsright_p14.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p14['Block'] = 'B5'
dfpowerfeedback5allbandsright_p14['Hand'] = 'right'
dfpowerfeedback5allbandsright_p14['Participant'] = '14'
print(dfpowerfeedback5allbandsright_p14)

#------------#
#PARTICIPANT 15
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p15 = powerprep1allbandsleft_p15.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p15['Block'] = 'B1'
dfpowerprep1allbandsleft_p15['Hand'] = 'left'
dfpowerprep1allbandsleft_p15['Participant'] = '15'
print(dfpowerprep1allbandsleft_p15)

dfpowerprep1allbandsright_p15 = powerprep1allbandsright_p15.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p15['Block'] = 'B1'
dfpowerprep1allbandsright_p15['Hand'] = 'right'
dfpowerprep1allbandsright_p15['Participant'] = '15'
print(dfpowerprep1allbandsright_p15)

#b5
dfpowerprep5allbandsleft_p15 = powerprep5allbandsleft_p15.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p15['Block'] = 'B5'
dfpowerprep5allbandsleft_p15['Hand'] = 'left'
dfpowerprep5allbandsleft_p15['Participant'] = '15'
print(dfpowerprep5allbandsleft_p15)

dfpowerprep5allbandsright_p15 = powerprep5allbandsright_p15.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p15['Block'] = 'B5'
dfpowerprep5allbandsright_p15['Hand'] = 'right'
dfpowerprep5allbandsright_p15['Participant'] = '15'
print(dfpowerprep5allbandsright_p15)



#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p15 = powerfeedback1allbandsleft_p15.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p15['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p15['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p15['Participant'] = '15'
print(dfpowerfeedback1allbandsleft_p15)

dfpowerfeedback1allbandsright_p15 = powerfeedback1allbandsright_p15.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p15['Block'] = 'B1'
dfpowerfeedback1allbandsright_p15['Hand'] = 'right'
dfpowerfeedback1allbandsright_p15['Participant'] = '15'
print(dfpowerfeedback1allbandsright_p15)

#b5
dfpowerfeedback5allbandsleft_p15 = powerfeedback5allbandsleft_p15.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p15['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p15['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p15['Participant'] = '15'
print(dfpowerfeedback5allbandsleft_p15)

dfpowerfeedback5allbandsright_p15 = powerfeedback5allbandsright_p15.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p15['Block'] = 'B5'
dfpowerfeedback5allbandsright_p15['Hand'] = 'right'
dfpowerfeedback5allbandsright_p15['Participant'] = '15'
print(dfpowerfeedback5allbandsright_p15)

#------------#
#PARTICIPANT 16
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p16 = powerprep1allbandsleft_p16.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p16['Block'] = 'B1'
dfpowerprep1allbandsleft_p16['Hand'] = 'left'
dfpowerprep1allbandsleft_p16['Participant'] = '16'
print(dfpowerprep1allbandsleft_p16)

dfpowerprep1allbandsright_p16 = powerprep1allbandsright_p16.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p16['Block'] = 'B1'
dfpowerprep1allbandsright_p16['Hand'] = 'right'
dfpowerprep1allbandsright_p16['Participant'] = '16'
print(dfpowerprep1allbandsright_p16)

#b5
dfpowerprep5allbandsleft_p16 = powerprep5allbandsleft_p16.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p16['Block'] = 'B5'
dfpowerprep5allbandsleft_p16['Hand'] = 'left'
dfpowerprep5allbandsleft_p16['Participant'] = '16'
print(dfpowerprep5allbandsleft_p16)

dfpowerprep5allbandsright_p16 = powerprep5allbandsright_p16.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p16['Block'] = 'B5'
dfpowerprep5allbandsright_p16['Hand'] = 'right'
dfpowerprep5allbandsright_p16['Participant'] = '16'
print(dfpowerprep5allbandsright_p16)


#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p16 = powerfeedback1allbandsleft_p16.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p16['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p16['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p16['Participant'] = '16'
print(dfpowerfeedback1allbandsleft_p16)

dfpowerfeedback1allbandsright_p16 = powerfeedback1allbandsright_p16.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p16['Block'] = 'B1'
dfpowerfeedback1allbandsright_p16['Hand'] = 'right'
dfpowerfeedback1allbandsright_p16['Participant'] = '16'
print(dfpowerfeedback1allbandsright_p16)

#b5
dfpowerfeedback5allbandsleft_p16 = powerfeedback5allbandsleft_p16.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p16['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p16['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p16['Participant'] = '16'
print(dfpowerfeedback5allbandsleft_p16)

dfpowerfeedback5allbandsright_p16 = powerfeedback5allbandsright_p16.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p16['Block'] = 'B5'
dfpowerfeedback5allbandsright_p16['Hand'] = 'right'
dfpowerfeedback5allbandsright_p16['Participant'] = '16'
print(dfpowerfeedback5allbandsright_p16)

#------------#
#PARTICIPANT 17
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p17 = powerprep1allbandsleft_p17.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p17['Block'] = 'B1'
dfpowerprep1allbandsleft_p17['Hand'] = 'left'
dfpowerprep1allbandsleft_p17['Participant'] = '17'
print(dfpowerprep1allbandsleft_p17)

dfpowerprep1allbandsright_p17 = powerprep1allbandsright_p17.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p17['Block'] = 'B1'
dfpowerprep1allbandsright_p17['Hand'] = 'right'
dfpowerprep1allbandsright_p17['Participant'] = '17'
print(dfpowerprep1allbandsright_p17)

#b5
dfpowerprep5allbandsleft_p17 = powerprep5allbandsleft_p17.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p17['Block'] = 'B5'
dfpowerprep5allbandsleft_p17['Hand'] = 'left'
dfpowerprep5allbandsleft_p17['Participant'] = '17'
print(dfpowerprep5allbandsleft_p17)

dfpowerprep5allbandsright_p17 = powerprep5allbandsright_p17.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p17['Block'] = 'B5'
dfpowerprep5allbandsright_p17['Hand'] = 'right'
dfpowerprep5allbandsright_p17['Participant'] = '17'
print(dfpowerprep5allbandsright_p17)



#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p17 = powerfeedback1allbandsleft_p17.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p17['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p17['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p17['Participant'] = '17'
print(dfpowerfeedback1allbandsleft_p17)

dfpowerfeedback1allbandsright_p17 = powerfeedback1allbandsright_p17.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p17['Block'] = 'B1'
dfpowerfeedback1allbandsright_p17['Hand'] = 'right'
dfpowerfeedback1allbandsright_p17['Participant'] = '17'
print(dfpowerfeedback1allbandsright_p17)

#b5
dfpowerfeedback5allbandsleft_p17 = powerfeedback5allbandsleft_p17.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p17['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p17['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p17['Participant'] = '17'
print(dfpowerfeedback5allbandsleft_p17)

dfpowerfeedback5allbandsright_p17 = powerfeedback5allbandsright_p17.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p17['Block'] = 'B5'
dfpowerfeedback5allbandsright_p17['Hand'] = 'right'
dfpowerfeedback5allbandsright_p17['Participant'] = '17'
print(dfpowerfeedback5allbandsright_p17)

#------------#
#PARTICIPANT 18
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p18 = powerprep1allbandsleft_p18.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p18['Block'] = 'B1'
dfpowerprep1allbandsleft_p18['Hand'] = 'left'
dfpowerprep1allbandsleft_p18['Participant'] = '18'
print(dfpowerprep1allbandsleft_p18)

dfpowerprep1allbandsright_p18 = powerprep1allbandsright_p18.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p18['Block'] = 'B1'
dfpowerprep1allbandsright_p18['Hand'] = 'right'
dfpowerprep1allbandsright_p18['Participant'] = '18'
print(dfpowerprep1allbandsright_p18)

#b5
dfpowerprep5allbandsleft_p18 = powerprep5allbandsleft_p18.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p18['Block'] = 'B5'
dfpowerprep5allbandsleft_p18['Hand'] = 'left'
dfpowerprep5allbandsleft_p18['Participant'] = '18'
print(dfpowerprep5allbandsleft_p18)

dfpowerprep5allbandsright_p18 = powerprep5allbandsright_p18.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p18['Block'] = 'B5'
dfpowerprep5allbandsright_p18['Hand'] = 'right'
dfpowerprep5allbandsright_p18['Participant'] = '18'
print(dfpowerprep5allbandsright_p18)


#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p18 = powerfeedback1allbandsleft_p18.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p18['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p18['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p18['Participant'] = '18'
print(dfpowerfeedback1allbandsleft_p18)

dfpowerfeedback1allbandsright_p18 = powerfeedback1allbandsright_p18.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p18['Block'] = 'B1'
dfpowerfeedback1allbandsright_p18['Hand'] = 'right'
dfpowerfeedback1allbandsright_p18['Participant'] = '18'
print(dfpowerfeedback1allbandsright_p18)

#b5
dfpowerfeedback5allbandsleft_p18 = powerfeedback5allbandsleft_p18.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p18['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p18['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p18['Participant'] = '18'
print(dfpowerfeedback5allbandsleft_p18)

dfpowerfeedback5allbandsright_p18 = powerfeedback5allbandsright_p18.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p18['Block'] = 'B5'
dfpowerfeedback5allbandsright_p18['Hand'] = 'right'
dfpowerfeedback5allbandsright_p18['Participant'] = '18'
print(dfpowerfeedback5allbandsright_p18)

#------------#
#PARTICIPANT 19
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p19 = powerprep1allbandsleft_p19.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p19['Block'] = 'B1'
dfpowerprep1allbandsleft_p19['Hand'] = 'left'
dfpowerprep1allbandsleft_p19['Participant'] = '19'
print(dfpowerprep1allbandsleft_p19)

dfpowerprep1allbandsright_p19 = powerprep1allbandsright_p19.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p19['Block'] = 'B1'
dfpowerprep1allbandsright_p19['Hand'] = 'right'
dfpowerprep1allbandsright_p19['Participant'] = '19'
print(dfpowerprep1allbandsright_p19)

#b5
dfpowerprep5allbandsleft_p19 = powerprep5allbandsleft_p19.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p19['Block'] = 'B5'
dfpowerprep5allbandsleft_p19['Hand'] = 'left'
dfpowerprep5allbandsleft_p19['Participant'] = '19'
print(dfpowerprep5allbandsleft_p19)

dfpowerprep5allbandsright_p19 = powerprep5allbandsright_p19.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p19['Block'] = 'B5'
dfpowerprep5allbandsright_p19['Hand'] = 'right'
dfpowerprep5allbandsright_p19['Participant'] = '19'
print(dfpowerprep5allbandsright_p19)




#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p19 = powerfeedback1allbandsleft_p19.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p19['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p19['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p19['Participant'] = '19'
print(dfpowerfeedback1allbandsleft_p19)

dfpowerfeedback1allbandsright_p19 = powerfeedback1allbandsright_p19.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p19['Block'] = 'B1'
dfpowerfeedback1allbandsright_p19['Hand'] = 'right'
dfpowerfeedback1allbandsright_p19['Participant'] = '19'
print(dfpowerfeedback1allbandsright_p19)

#b5
dfpowerfeedback5allbandsleft_p19 = powerfeedback5allbandsleft_p19.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p19['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p19['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p19['Participant'] = '19'
print(dfpowerfeedback5allbandsleft_p19)

dfpowerfeedback5allbandsright_p19 = powerfeedback5allbandsright_p19.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p19['Block'] = 'B5'
dfpowerfeedback5allbandsright_p19['Hand'] = 'right'
dfpowerfeedback5allbandsright_p19['Participant'] = '19'
print(dfpowerfeedback5allbandsright_p19)

#------------#
#PARTICIPANT 22
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p22 = powerprep1allbandsleft_p22.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p22['Block'] = 'B1'
dfpowerprep1allbandsleft_p22['Hand'] = 'left'
dfpowerprep1allbandsleft_p22['Participant'] = '22'
print(dfpowerprep1allbandsleft_p22)

dfpowerprep1allbandsright_p22 = powerprep1allbandsright_p22.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p22['Block'] = 'B1'
dfpowerprep1allbandsright_p22['Hand'] = 'right'
dfpowerprep1allbandsright_p22['Participant'] = '22'
print(dfpowerprep1allbandsright_p22)

#b5
dfpowerprep5allbandsleft_p22 = powerprep5allbandsleft_p22.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p22['Block'] = 'B5'
dfpowerprep5allbandsleft_p22['Hand'] = 'left'
dfpowerprep5allbandsleft_p22['Participant'] = '22'
print(dfpowerprep5allbandsleft_p22)

dfpowerprep5allbandsright_p22 = powerprep5allbandsright_p22.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p22['Block'] = 'B5'
dfpowerprep5allbandsright_p22['Hand'] = 'right'
dfpowerprep5allbandsright_p22['Participant'] = '22'
print(dfpowerprep5allbandsright_p22)



#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p22 = powerfeedback1allbandsleft_p22.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p22['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p22['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p22['Participant'] = '22'
print(dfpowerfeedback1allbandsleft_p22)

dfpowerfeedback1allbandsright_p22 = powerfeedback1allbandsright_p22.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p22['Block'] = 'B1'
dfpowerfeedback1allbandsright_p22['Hand'] = 'right'
dfpowerfeedback1allbandsright_p22['Participant'] = '22'
print(dfpowerfeedback1allbandsright_p22)

#b5
dfpowerfeedback5allbandsleft_p22 = powerfeedback5allbandsleft_p22.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p22['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p22['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p22['Participant'] = '22'
print(dfpowerfeedback5allbandsleft_p22)

dfpowerfeedback5allbandsright_p22 = powerfeedback5allbandsright_p22.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p22['Block'] = 'B5'
dfpowerfeedback5allbandsright_p22['Hand'] = 'right'
dfpowerfeedback5allbandsright_p22['Participant'] = '22'
print(dfpowerfeedback5allbandsright_p22)

#------------#
#PARTICIPANT 23
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p23 = powerprep1allbandsleft_p23.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p23['Block'] = 'B1'
dfpowerprep1allbandsleft_p23['Hand'] = 'left'
dfpowerprep1allbandsleft_p23['Participant'] = '23'
print(dfpowerprep1allbandsleft_p23)

dfpowerprep1allbandsright_p23 = powerprep1allbandsright_p23.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p23['Block'] = 'B1'
dfpowerprep1allbandsright_p23['Hand'] = 'right'
dfpowerprep1allbandsright_p23['Participant'] = '23'
print(dfpowerprep1allbandsright_p23)

#b5
dfpowerprep5allbandsleft_p23 = powerprep5allbandsleft_p23.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p23['Block'] = 'B5'
dfpowerprep5allbandsleft_p23['Hand'] = 'left'
dfpowerprep5allbandsleft_p23['Participant'] = '23'
print(dfpowerprep5allbandsleft_p23)

dfpowerprep5allbandsright_p23 = powerprep5allbandsright_p23.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p23['Block'] = 'B5'
dfpowerprep5allbandsright_p23['Hand'] = 'right'
dfpowerprep5allbandsright_p23['Participant'] = '23'
print(dfpowerprep5allbandsright_p23)


#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p23 = powerfeedback1allbandsleft_p23.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p23['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p23['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p23['Participant'] = '23'
print(dfpowerfeedback1allbandsleft_p23)

dfpowerfeedback1allbandsright_p23 = powerfeedback1allbandsright_p23.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p23['Block'] = 'B1'
dfpowerfeedback1allbandsright_p23['Hand'] = 'right'
dfpowerfeedback1allbandsright_p23['Participant'] = '23'
print(dfpowerfeedback1allbandsright_p23)

#b5
dfpowerfeedback5allbandsleft_p23 = powerfeedback5allbandsleft_p23.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p23['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p23['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p23['Participant'] = '23'
print(dfpowerfeedback5allbandsleft_p23)

dfpowerfeedback5allbandsright_p23 = powerfeedback5allbandsright_p23.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p23['Block'] = 'B5'
dfpowerfeedback5allbandsright_p23['Hand'] = 'right'
dfpowerfeedback5allbandsright_p23['Participant'] = '23'
print(dfpowerfeedback5allbandsright_p23)


#------------#
#PARTICIPANT 24
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p24 = powerprep1allbandsleft_p24.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p24['Block'] = 'B1'
dfpowerprep1allbandsleft_p24['Hand'] = 'left'
dfpowerprep1allbandsleft_p24['Participant'] = '24'
print(dfpowerprep1allbandsleft_p24)

dfpowerprep1allbandsright_p24 = powerprep1allbandsright_p24.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p24['Block'] = 'B1'
dfpowerprep1allbandsright_p24['Hand'] = 'right'
dfpowerprep1allbandsright_p24['Participant'] = '24'
print(dfpowerprep1allbandsright_p24)

#b5
dfpowerprep5allbandsleft_p24 = powerprep5allbandsleft_p24.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p24['Block'] = 'B5'
dfpowerprep5allbandsleft_p24['Hand'] = 'left'
dfpowerprep5allbandsleft_p24['Participant'] = '24'
print(dfpowerprep5allbandsleft_p24)

dfpowerprep5allbandsright_p24 = powerprep5allbandsright_p24.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p24['Block'] = 'B5'
dfpowerprep5allbandsright_p24['Hand'] = 'right'
dfpowerprep5allbandsright_p24['Participant'] = '24'
print(dfpowerprep5allbandsright_p24)


#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p24 = powerfeedback1allbandsleft_p24.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p24['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p24['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p24['Participant'] = '24'
print(dfpowerfeedback1allbandsleft_p24)

dfpowerfeedback1allbandsright_p24 = powerfeedback1allbandsright_p24.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p24['Block'] = 'B1'
dfpowerfeedback1allbandsright_p24['Hand'] = 'right'
dfpowerfeedback1allbandsright_p24['Participant'] = '24'
print(dfpowerfeedback1allbandsright_p24)

#b5
dfpowerfeedback5allbandsleft_p24 = powerfeedback5allbandsleft_p24.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p24['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p24['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p24['Participant'] = '24'
print(dfpowerfeedback5allbandsleft_p24)

dfpowerfeedback5allbandsright_p24 = powerfeedback5allbandsright_p24.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p24['Block'] = 'B5'
dfpowerfeedback5allbandsright_p24['Hand'] = 'right'
dfpowerfeedback5allbandsright_p24['Participant'] = '24'
print(dfpowerfeedback5allbandsright_p24)

#------------#
#PARTICIPANT 25
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p25 = powerprep1allbandsleft_p25.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p25['Block'] = 'B1'
dfpowerprep1allbandsleft_p25['Hand'] = 'left'
dfpowerprep1allbandsleft_p25['Participant'] = '25'
print(dfpowerprep1allbandsleft_p25)

dfpowerprep1allbandsright_p25 = powerprep1allbandsright_p25.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p25['Block'] = 'B1'
dfpowerprep1allbandsright_p25['Hand'] = 'right'
dfpowerprep1allbandsright_p25['Participant'] = '25'
print(dfpowerprep1allbandsright_p25)

#b5
dfpowerprep5allbandsleft_p25 = powerprep5allbandsleft_p25.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p25['Block'] = 'B5'
dfpowerprep5allbandsleft_p25['Hand'] = 'left'
dfpowerprep5allbandsleft_p25['Participant'] = '25'
print(dfpowerprep5allbandsleft_p25)

dfpowerprep5allbandsright_p25 = powerprep5allbandsright_p25.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p25['Block'] = 'B5'
dfpowerprep5allbandsright_p25['Hand'] = 'right'
dfpowerprep5allbandsright_p25['Participant'] = '25'
print(dfpowerprep5allbandsright_p25)


#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p25 = powerfeedback1allbandsleft_p25.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p25['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p25['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p25['Participant'] = '25'
print(dfpowerfeedback1allbandsleft_p25)

dfpowerfeedback1allbandsright_p25 = powerfeedback1allbandsright_p25.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p25['Block'] = 'B1'
dfpowerfeedback1allbandsright_p25['Hand'] = 'right'
dfpowerfeedback1allbandsright_p25['Participant'] = '25'
print(dfpowerfeedback1allbandsright_p25)

#b5
dfpowerfeedback5allbandsleft_p25 = powerfeedback5allbandsleft_p25.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p25['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p25['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p25['Participant'] = '25'
print(dfpowerfeedback5allbandsleft_p25)

dfpowerfeedback5allbandsright_p25 = powerfeedback5allbandsright_p25.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p25['Block'] = 'B5'
dfpowerfeedback5allbandsright_p25['Hand'] = 'right'
dfpowerfeedback5allbandsright_p25['Participant'] = '25'
print(dfpowerfeedback5allbandsright_p25)

#------------#
#PARTICIPANT 26
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p26 = powerprep1allbandsleft_p26.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p26['Block'] = 'B1'
dfpowerprep1allbandsleft_p26['Hand'] = 'left'
dfpowerprep1allbandsleft_p26['Participant'] = '26'
print(dfpowerprep1allbandsleft_p26)

dfpowerprep1allbandsright_p26 = powerprep1allbandsright_p26.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p26['Block'] = 'B1'
dfpowerprep1allbandsright_p26['Hand'] = 'right'
dfpowerprep1allbandsright_p26['Participant'] = '26'
print(dfpowerprep1allbandsright_p26)

#b5
dfpowerprep5allbandsleft_p26 = powerprep5allbandsleft_p26.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p26['Block'] = 'B5'
dfpowerprep5allbandsleft_p26['Hand'] = 'left'
dfpowerprep5allbandsleft_p26['Participant'] = '26'
print(dfpowerprep5allbandsleft_p26)

dfpowerprep5allbandsright_p26 = powerprep5allbandsright_p26.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p26['Block'] = 'B5'
dfpowerprep5allbandsright_p26['Hand'] = 'right'
dfpowerprep5allbandsright_p26['Participant'] = '26'
print(dfpowerprep5allbandsright_p26)


#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p26 = powerfeedback1allbandsleft_p26.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p26['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p26['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p26['Participant'] = '26'
print(dfpowerfeedback1allbandsleft_p26)

dfpowerfeedback1allbandsright_p26 = powerfeedback1allbandsright_p26.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p26['Block'] = 'B1'
dfpowerfeedback1allbandsright_p26['Hand'] = 'right'
dfpowerfeedback1allbandsright_p26['Participant'] = '26'
print(dfpowerfeedback1allbandsright_p26)

#b5
dfpowerfeedback5allbandsleft_p26 = powerfeedback5allbandsleft_p26.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p26['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p26['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p26['Participant'] = '26'
print(dfpowerfeedback5allbandsleft_p26)

dfpowerfeedback5allbandsright_p26 = powerfeedback5allbandsright_p26.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p26['Block'] = 'B5'
dfpowerfeedback5allbandsright_p26['Hand'] = 'right'
dfpowerfeedback5allbandsright_p26['Participant'] = '26'
print(dfpowerfeedback5allbandsright_p26)

#------------#
#PARTICIPANT 27
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p27 = powerprep1allbandsleft_p27.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p27['Block'] = 'B1'
dfpowerprep1allbandsleft_p27['Hand'] = 'left'
dfpowerprep1allbandsleft_p27['Participant'] = '27'
print(dfpowerprep1allbandsleft_p27)

dfpowerprep1allbandsright_p27 = powerprep1allbandsright_p27.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p27['Block'] = 'B1'
dfpowerprep1allbandsright_p27['Hand'] = 'right'
dfpowerprep1allbandsright_p27['Participant'] = '27'
print(dfpowerprep1allbandsright_p27)

#b5
dfpowerprep5allbandsleft_p27 = powerprep5allbandsleft_p27.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p27['Block'] = 'B5'
dfpowerprep5allbandsleft_p27['Hand'] = 'left'
dfpowerprep5allbandsleft_p27['Participant'] = '27'
print(dfpowerprep5allbandsleft_p27)

dfpowerprep5allbandsright_p27 = powerprep5allbandsright_p27.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p27['Block'] = 'B5'
dfpowerprep5allbandsright_p27['Hand'] = 'right'
dfpowerprep5allbandsright_p27['Participant'] = '27'
print(dfpowerprep5allbandsright_p27)


#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p27 = powerfeedback1allbandsleft_p27.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p27['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p27['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p27['Participant'] = '27'
print(dfpowerfeedback1allbandsleft_p27)

dfpowerfeedback1allbandsright_p27 = powerfeedback1allbandsright_p27.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p27['Block'] = 'B1'
dfpowerfeedback1allbandsright_p27['Hand'] = 'right'
dfpowerfeedback1allbandsright_p27['Participant'] = '27'
print(dfpowerfeedback1allbandsright_p27)

#b5
dfpowerfeedback5allbandsleft_p27 = powerfeedback5allbandsleft_p27.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p27['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p27['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p27['Participant'] = '27'
print(dfpowerfeedback5allbandsleft_p27)

dfpowerfeedback5allbandsright_p27 = powerfeedback5allbandsright_p27.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p27['Block'] = 'B5'
dfpowerfeedback5allbandsright_p27['Hand'] = 'right'
dfpowerfeedback5allbandsright_p27['Participant'] = '27'
print(dfpowerfeedback5allbandsright_p27)

#------------#
#PARTICIPANT 28
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p28 = powerprep1allbandsleft_p28.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p28['Block'] = 'B1'
dfpowerprep1allbandsleft_p28['Hand'] = 'left'
dfpowerprep1allbandsleft_p28['Participant'] = '28'
print(dfpowerprep1allbandsleft_p28)

dfpowerprep1allbandsright_p28 = powerprep1allbandsright_p28.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p28['Block'] = 'B1'
dfpowerprep1allbandsright_p28['Hand'] = 'right'
dfpowerprep1allbandsright_p28['Participant'] = '28'
print(dfpowerprep1allbandsright_p28)

#b5
dfpowerprep5allbandsleft_p28 = powerprep5allbandsleft_p28.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p28['Block'] = 'B5'
dfpowerprep5allbandsleft_p28['Hand'] = 'left'
dfpowerprep5allbandsleft_p28['Participant'] = '28'
print(dfpowerprep5allbandsleft_p28)

dfpowerprep5allbandsright_p28 = powerprep5allbandsright_p28.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p28['Block'] = 'B5'
dfpowerprep5allbandsright_p28['Hand'] = 'right'
dfpowerprep5allbandsright_p28['Participant'] = '28'
print(dfpowerprep5allbandsright_p28)


#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p28 = powerfeedback1allbandsleft_p28.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p28['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p28['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p28['Participant'] = '28'
print(dfpowerfeedback1allbandsleft_p28)

dfpowerfeedback1allbandsright_p28 = powerfeedback1allbandsright_p28.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p28['Block'] = 'B1'
dfpowerfeedback1allbandsright_p28['Hand'] = 'right'
dfpowerfeedback1allbandsright_p28['Participant'] = '28'
print(dfpowerfeedback1allbandsright_p28)

#b5
dfpowerfeedback5allbandsleft_p28 = powerfeedback5allbandsleft_p28.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p28['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p28['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p28['Participant'] = '28'
print(dfpowerfeedback5allbandsleft_p28)

dfpowerfeedback5allbandsright_p28 = powerfeedback5allbandsright_p28.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p28['Block'] = 'B5'
dfpowerfeedback5allbandsright_p28['Hand'] = 'right'
dfpowerfeedback5allbandsright_p28['Participant'] = '28'
print(dfpowerfeedback5allbandsright_p28)

#------------#
#PARTICIPANT 29
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p29 = powerprep1allbandsleft_p29.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p29['Block'] = 'B1'
dfpowerprep1allbandsleft_p29['Hand'] = 'left'
dfpowerprep1allbandsleft_p29['Participant'] = '29'
print(dfpowerprep1allbandsleft_p29)

dfpowerprep1allbandsright_p29 = powerprep1allbandsright_p29.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p29['Block'] = 'B1'
dfpowerprep1allbandsright_p29['Hand'] = 'right'
dfpowerprep1allbandsright_p29['Participant'] = '29'
print(dfpowerprep1allbandsright_p29)

#b5
dfpowerprep5allbandsleft_p29 = powerprep5allbandsleft_p29.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p29['Block'] = 'B5'
dfpowerprep5allbandsleft_p29['Hand'] = 'left'
dfpowerprep5allbandsleft_p29['Participant'] = '29'
print(dfpowerprep5allbandsleft_p29)

dfpowerprep5allbandsright_p29 = powerprep5allbandsright_p29.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p29['Block'] = 'B5'
dfpowerprep5allbandsright_p29['Hand'] = 'right'
dfpowerprep5allbandsright_p29['Participant'] = '29'
print(dfpowerprep5allbandsright_p29)


#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p29 = powerfeedback1allbandsleft_p29.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p29['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p29['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p29['Participant'] = '29'
print(dfpowerfeedback1allbandsleft_p29)

dfpowerfeedback1allbandsright_p29 = powerfeedback1allbandsright_p29.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p29['Block'] = 'B1'
dfpowerfeedback1allbandsright_p29['Hand'] = 'right'
dfpowerfeedback1allbandsright_p29['Participant'] = '29'
print(dfpowerfeedback1allbandsright_p29)

#b5
dfpowerfeedback5allbandsleft_p29 = powerfeedback5allbandsleft_p29.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p29['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p29['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p29['Participant'] = '29'
print(dfpowerfeedback5allbandsleft_p29)

dfpowerfeedback5allbandsright_p29 = powerfeedback5allbandsright_p29.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p29['Block'] = 'B5'
dfpowerfeedback5allbandsright_p29['Hand'] = 'right'
dfpowerfeedback5allbandsright_p29['Participant'] = '29'
print(dfpowerfeedback5allbandsright_p29)

#------------#
#PARTICIPANT 30
#------------#

#------prep------#
#b1
dfpowerprep1allbandsleft_p30 = powerprep1allbandsleft_p30.to_data_frame(time_format=None)
dfpowerprep1allbandsleft_p30['Block'] = 'B1'
dfpowerprep1allbandsleft_p30['Hand'] = 'left'
dfpowerprep1allbandsleft_p30['Participant'] = '30'
print(dfpowerprep1allbandsleft_p30)

dfpowerprep1allbandsright_p30 = powerprep1allbandsright_p30.to_data_frame(time_format=None)
dfpowerprep1allbandsright_p30['Block'] = 'B1'
dfpowerprep1allbandsright_p30['Hand'] = 'right'
dfpowerprep1allbandsright_p30['Participant'] = '30'
print(dfpowerprep1allbandsright_p30)

#b5
dfpowerprep5allbandsleft_p30 = powerprep5allbandsleft_p30.to_data_frame(time_format=None)
dfpowerprep5allbandsleft_p30['Block'] = 'B5'
dfpowerprep5allbandsleft_p30['Hand'] = 'left'
dfpowerprep5allbandsleft_p30['Participant'] = '30'
print(dfpowerprep5allbandsleft_p30)

dfpowerprep5allbandsright_p30 = powerprep5allbandsright_p30.to_data_frame(time_format=None)
dfpowerprep5allbandsright_p30['Block'] = 'B5'
dfpowerprep5allbandsright_p30['Hand'] = 'right'
dfpowerprep5allbandsright_p30['Participant'] = '30'
print(dfpowerprep5allbandsright_p30)

#------feedback------#
#b1
dfpowerfeedback1allbandsleft_p30 = powerfeedback1allbandsleft_p30.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft_p30['Block'] = 'B1'
dfpowerfeedback1allbandsleft_p30['Hand'] = 'left'
dfpowerfeedback1allbandsleft_p30['Participant'] = '30'
print(dfpowerfeedback1allbandsleft_p30)

dfpowerfeedback1allbandsright_p30 = powerfeedback1allbandsright_p30.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright_p30['Block'] = 'B1'
dfpowerfeedback1allbandsright_p30['Hand'] = 'right'
dfpowerfeedback1allbandsright_p30['Participant'] = '30'
print(dfpowerfeedback1allbandsright_p30)

#b5
dfpowerfeedback5allbandsleft_p30 = powerfeedback5allbandsleft_p30.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft_p30['Block'] = 'B5'
dfpowerfeedback5allbandsleft_p30['Hand'] = 'left'
dfpowerfeedback5allbandsleft_p30['Participant'] = '30'
print(dfpowerfeedback5allbandsleft_p30)

dfpowerfeedback5allbandsright_p30 = powerfeedback5allbandsright_p30.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright_p30['Block'] = 'B5'
dfpowerfeedback5allbandsright_p30['Hand'] = 'right'
dfpowerfeedback5allbandsright_p30['Participant'] = '30'
print(dfpowerfeedback5allbandsright_p30)

dfpowerprepindivB1_p1 = pd.concat([dfpowerprep1allbandsright_p1, dfpowerprep1allbandsleft_p1], axis=0)
dfpowerprepindivB1_p1.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p1.csv', index = False)
dfpowerprepindivB5_p1 = pd.concat([dfpowerprep5allbandsright_p1, dfpowerprep5allbandsleft_p1], axis=0)
dfpowerprepindivB5_p1.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p1.csv', index = False)


dfpowerfeedbackindivB1_p1 = pd.concat([dfpowerfeedback1allbandsright_p1, dfpowerfeedback1allbandsleft_p1], axis=0)
dfpowerfeedbackindivB1_p1.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p1.csv', index = False)
dfpowerfeedbackindivB5_p1 = pd.concat([dfpowerfeedback5allbandsright_p1, dfpowerfeedback5allbandsleft_p1], axis=0)
dfpowerfeedbackindivB5_p1.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p1.csv', index = False)


dfpowerprepindivB1_p2 = pd.concat([dfpowerprep1allbandsright_p2, dfpowerprep1allbandsleft_p2], axis=0)
dfpowerprepindivB1_p2.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p2.csv', index = False)
dfpowerprepindivB5_p2 = pd.concat([dfpowerprep5allbandsright_p2, dfpowerprep5allbandsleft_p2], axis=0)
dfpowerprepindivB5_p2.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p2.csv', index = False)


dfpowerfeedbackindivB1_p2 = pd.concat([dfpowerfeedback1allbandsright_p2, dfpowerfeedback1allbandsleft_p2], axis=0)
dfpowerfeedbackindivB1_p2.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p2.csv', index = False)
dfpowerfeedbackindivB5_p2 = pd.concat([dfpowerfeedback5allbandsright_p2, dfpowerfeedback5allbandsleft_p2], axis=0)
dfpowerfeedbackindivB5_p2.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p2.csv', index = False)

dfpowerprepindivB1_p3 = pd.concat([dfpowerprep1allbandsright_p3, dfpowerprep1allbandsleft_p3], axis=0)
dfpowerprepindivB1_p3.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p3.csv', index = False)
dfpowerprepindivB5_p3 = pd.concat([dfpowerprep5allbandsright_p3, dfpowerprep5allbandsleft_p3], axis=0)
dfpowerprepindivB5_p3.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p3.csv', index = False)


dfpowerfeedbackindivB1_p3 = pd.concat([dfpowerfeedback1allbandsright_p3, dfpowerfeedback1allbandsleft_p3], axis=0)
dfpowerfeedbackindivB1_p3.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p3.csv', index = False)
dfpowerfeedbackindivB5_p3 = pd.concat([dfpowerfeedback5allbandsright_p3, dfpowerfeedback5allbandsleft_p3], axis=0)
dfpowerfeedbackindivB5_p3.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p3.csv', index = False)

dfpowerprepindivB1_p5 = pd.concat([dfpowerprep1allbandsright_p5, dfpowerprep1allbandsleft_p5], axis=0)
dfpowerprepindivB1_p5.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p5.csv', index = False)
dfpowerprepindivB5_p5 = pd.concat([dfpowerprep5allbandsright_p5, dfpowerprep5allbandsleft_p5], axis=0)
dfpowerprepindivB5_p5.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p5.csv', index = False)


dfpowerfeedbackindivB1_p5 = pd.concat([dfpowerfeedback1allbandsright_p5, dfpowerfeedback1allbandsleft_p5], axis=0)
dfpowerfeedbackindivB1_p5.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p5.csv', index = False)
dfpowerfeedbackindivB5_p5 = pd.concat([dfpowerfeedback5allbandsright_p5, dfpowerfeedback5allbandsleft_p5], axis=0)
dfpowerfeedbackindivB5_p5.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p5.csv', index = False)

dfpowerprepindivB1_p6 = pd.concat([dfpowerprep1allbandsright_p6, dfpowerprep1allbandsleft_p6], axis=0)
dfpowerprepindivB1_p6.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p6.csv', index = False)
dfpowerprepindivB5_p6 = pd.concat([dfpowerprep5allbandsright_p6, dfpowerprep5allbandsleft_p6], axis=0)
dfpowerprepindivB5_p6.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p6.csv', index = False)

dfpowerfeedbackindivB1_p6 = pd.concat([dfpowerfeedback1allbandsright_p6, dfpowerfeedback1allbandsleft_p6], axis=0)
dfpowerfeedbackindivB1_p6.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p6.csv', index = False)
dfpowerfeedbackindivB5_p6 = pd.concat([dfpowerfeedback5allbandsright_p6, dfpowerfeedback5allbandsleft_p6], axis=0)
dfpowerfeedbackindivB5_p6.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p6.csv', index = False)

dfpowerprepindivB1_p7 = pd.concat([dfpowerprep1allbandsright_p7, dfpowerprep1allbandsleft_p7], axis=0)
dfpowerprepindivB1_p7.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p7.csv', index = False)
dfpowerprepindivB5_p7 = pd.concat([dfpowerprep5allbandsright_p7, dfpowerprep5allbandsleft_p7], axis=0)
dfpowerprepindivB5_p7.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p7.csv', index = False)

dfpowerfeedbackindivB1_p7 = pd.concat([dfpowerfeedback1allbandsright_p7, dfpowerfeedback1allbandsleft_p7], axis=0)
dfpowerfeedbackindivB1_p7.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p7.csv', index = False)
dfpowerfeedbackindivB5_p7 = pd.concat([dfpowerfeedback5allbandsright_p7, dfpowerfeedback5allbandsleft_p7], axis=0)
dfpowerfeedbackindivB5_p7.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p7.csv', index = False)

dfpowerprepindivB1_p8 = pd.concat([dfpowerprep1allbandsright_p8, dfpowerprep1allbandsleft_p8], axis=0)
dfpowerprepindivB1_p8.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p8.csv', index = False)
dfpowerprepindivB5_p8 = pd.concat([dfpowerprep5allbandsright_p8, dfpowerprep5allbandsleft_p8], axis=0)
dfpowerprepindivB5_p8.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p8.csv', index = False)

dfpowerfeedbackindivB1_p8 = pd.concat([dfpowerfeedback1allbandsright_p8, dfpowerfeedback1allbandsleft_p8], axis=0)
dfpowerfeedbackindivB1_p8.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p8.csv', index = False)
dfpowerfeedbackindivB5_p8 = pd.concat([dfpowerfeedback5allbandsright_p8, dfpowerfeedback5allbandsleft_p8], axis=0)
dfpowerfeedbackindivB5_p8.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p8.csv', index = False)

dfpowerprepindivB1_p9 = pd.concat([dfpowerprep1allbandsright_p9, dfpowerprep1allbandsleft_p9], axis=0)
dfpowerprepindivB1_p9.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p9.csv', index = False)
dfpowerprepindivB5_p9 = pd.concat([dfpowerprep5allbandsright_p9, dfpowerprep5allbandsleft_p9], axis=0)
dfpowerprepindivB5_p9.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p9.csv', index = False)

dfpowerfeedbackindivB1_p9 = pd.concat([dfpowerfeedback1allbandsright_p9, dfpowerfeedback1allbandsleft_p9], axis=0)
dfpowerfeedbackindivB1_p9.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p9.csv', index = False)
dfpowerfeedbackindivB5_p9 = pd.concat([dfpowerfeedback5allbandsright_p9, dfpowerfeedback5allbandsleft_p9], axis=0)
dfpowerfeedbackindivB5_p9.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p9.csv', index = False)

dfpowerprepindivB1_p10 = pd.concat([dfpowerprep1allbandsright_p10, dfpowerprep1allbandsleft_p10], axis=0)
dfpowerprepindivB1_p10.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p10.csv', index = False)
dfpowerprepindivB5_p10 = pd.concat([dfpowerprep5allbandsright_p10, dfpowerprep5allbandsleft_p10], axis=0)
dfpowerprepindivB5_p10.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p10.csv', index = False)


dfpowerfeedbackindivB1_p10 = pd.concat([dfpowerfeedback1allbandsright_p10, dfpowerfeedback1allbandsleft_p10], axis=0)
dfpowerfeedbackindivB1_p10.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p10.csv', index = False)
dfpowerfeedbackindivB5_p10 = pd.concat([dfpowerfeedback5allbandsright_p10, dfpowerfeedback5allbandsleft_p10], axis=0)
dfpowerfeedbackindivB5_p10.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p10.csv', index = False)


dfpowerprepindivB1_p11 = pd.concat([dfpowerprep1allbandsright_p11, dfpowerprep1allbandsleft_p11], axis=0)
dfpowerprepindivB1_p11.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p11.csv', index = False)
dfpowerprepindivB5_p11 = pd.concat([dfpowerprep5allbandsright_p11, dfpowerprep5allbandsleft_p11], axis=0)
dfpowerprepindivB5_p11.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p11.csv', index = False)

dfpowerfeedbackindivB1_p11 = pd.concat([dfpowerfeedback1allbandsright_p11, dfpowerfeedback1allbandsleft_p11], axis=0)
dfpowerfeedbackindivB1_p11.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p11.csv', index = False)
dfpowerfeedbackindivB5_p11 = pd.concat([dfpowerfeedback5allbandsright_p11, dfpowerfeedback5allbandsleft_p11], axis=0)
dfpowerfeedbackindivB5_p11.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p11.csv', index = False)

dfpowerprepindivB1_p12 = pd.concat([dfpowerprep1allbandsright_p12, dfpowerprep1allbandsleft_p12], axis=0)
dfpowerprepindivB1_p12.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p12.csv', index = False)
dfpowerprepindivB5_p12 = pd.concat([dfpowerprep5allbandsright_p12, dfpowerprep5allbandsleft_p12], axis=0)
dfpowerprepindivB5_p12.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p12.csv', index = False)

dfpowerfeedbackindivB1_p12 = pd.concat([dfpowerfeedback1allbandsright_p12, dfpowerfeedback1allbandsleft_p12], axis=0)
dfpowerfeedbackindivB1_p12.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p12.csv', index = False)
dfpowerfeedbackindivB5_p12 = pd.concat([dfpowerfeedback5allbandsright_p12, dfpowerfeedback5allbandsleft_p12], axis=0)
dfpowerfeedbackindivB5_p12.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p12.csv', index = False)

dfpowerprepindivB1_p14 = pd.concat([dfpowerprep1allbandsright_p14, dfpowerprep1allbandsleft_p14], axis=0)
dfpowerprepindivB1_p14.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p14.csv', index = False)
dfpowerprepindivB5_p14 = pd.concat([dfpowerprep5allbandsright_p14, dfpowerprep5allbandsleft_p14], axis=0)
dfpowerprepindivB5_p14.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p14.csv', index = False)

dfpowerfeedbackindivB1_p14 = pd.concat([dfpowerfeedback1allbandsright_p14, dfpowerfeedback1allbandsleft_p14], axis=0)
dfpowerfeedbackindivB1_p14.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p14.csv', index = False)
dfpowerfeedbackindivB5_p14 = pd.concat([dfpowerfeedback5allbandsright_p14, dfpowerfeedback5allbandsleft_p14], axis=0)
dfpowerfeedbackindivB5_p14.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p14.csv', index = False)

dfpowerprepindivB1_p15 = pd.concat([dfpowerprep1allbandsright_p15, dfpowerprep1allbandsleft_p15], axis=0)
dfpowerprepindivB1_p15.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p15.csv', index = False)
dfpowerprepindivB5_p15 = pd.concat([dfpowerprep5allbandsright_p15, dfpowerprep5allbandsleft_p15], axis=0)
dfpowerprepindivB5_p15.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p15.csv', index = False)

dfpowerfeedbackindivB1_p15 = pd.concat([dfpowerfeedback1allbandsright_p15, dfpowerfeedback1allbandsleft_p15], axis=0)
dfpowerfeedbackindivB1_p15.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p15.csv', index = False)
dfpowerfeedbackindivB5_p15 = pd.concat([dfpowerfeedback5allbandsright_p15, dfpowerfeedback5allbandsleft_p15], axis=0)
dfpowerfeedbackindivB5_p15.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p15.csv', index = False)

dfpowerprepindivB1_p16 = pd.concat([dfpowerprep1allbandsright_p16, dfpowerprep1allbandsleft_p16], axis=0)
dfpowerprepindivB1_p16.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p16.csv', index = False)
dfpowerprepindivB5_p16 = pd.concat([dfpowerprep5allbandsright_p16, dfpowerprep5allbandsleft_p16], axis=0)
dfpowerprepindivB5_p16.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p16.csv', index = False)

dfpowerfeedbackindivB1_p16 = pd.concat([dfpowerfeedback1allbandsright_p16, dfpowerfeedback1allbandsleft_p16], axis=0)
dfpowerfeedbackindivB1_p16.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p16.csv', index = False)
dfpowerfeedbackindivB5_p16 = pd.concat([dfpowerfeedback5allbandsright_p16, dfpowerfeedback5allbandsleft_p16], axis=0)
dfpowerfeedbackindivB5_p16.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p16.csv', index = False)

dfpowerprepindivB1_p17 = pd.concat([dfpowerprep1allbandsright_p17, dfpowerprep1allbandsleft_p17], axis=0)
dfpowerprepindivB1_p17.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p17.csv', index = False)
dfpowerprepindivB5_p17 = pd.concat([dfpowerprep5allbandsright_p17, dfpowerprep5allbandsleft_p17], axis=0)
dfpowerprepindivB5_p17.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p17.csv', index = False)

dfpowerfeedbackindivB1_p17 = pd.concat([dfpowerfeedback1allbandsright_p17, dfpowerfeedback1allbandsleft_p17], axis=0)
dfpowerfeedbackindivB1_p17.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p17.csv', index = False)
dfpowerfeedbackindivB5_p17 = pd.concat([dfpowerfeedback5allbandsright_p17, dfpowerfeedback5allbandsleft_p17], axis=0)
dfpowerfeedbackindivB5_p17.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p17.csv', index = False)

dfpowerprepindivB1_p18 = pd.concat([dfpowerprep1allbandsright_p18, dfpowerprep1allbandsleft_p18], axis=0)
dfpowerprepindivB1_p18.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p18.csv', index = False)
dfpowerprepindivB5_p18 = pd.concat([dfpowerprep5allbandsright_p18, dfpowerprep5allbandsleft_p18], axis=0)
dfpowerprepindivB5_p18.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p18.csv', index = False)


dfpowerfeedbackindivB1_p18 = pd.concat([dfpowerfeedback1allbandsright_p18, dfpowerfeedback1allbandsleft_p18], axis=0)
dfpowerfeedbackindivB1_p18.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p18.csv', index = False)
dfpowerfeedbackindivB5_p18 = pd.concat([dfpowerfeedback5allbandsright_p18, dfpowerfeedback5allbandsleft_p18], axis=0)
dfpowerfeedbackindivB5_p18.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p18.csv', index = False)

dfpowerprepindivB1_p19 = pd.concat([dfpowerprep1allbandsright_p19, dfpowerprep1allbandsleft_p19], axis=0)
dfpowerprepindivB1_p19.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p19.csv', index = False)
dfpowerprepindivB5_p19 = pd.concat([dfpowerprep5allbandsright_p19, dfpowerprep5allbandsleft_p19], axis=0)
dfpowerprepindivB5_p19.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p19.csv', index = False)

dfpowerfeedbackindivB1_p19 = pd.concat([dfpowerfeedback1allbandsright_p19, dfpowerfeedback1allbandsleft_p19], axis=0)
dfpowerfeedbackindivB1_p19.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p19.csv', index = False)
dfpowerfeedbackindivB5_p19 = pd.concat([dfpowerfeedback5allbandsright_p19, dfpowerfeedback5allbandsleft_p19], axis=0)
dfpowerfeedbackindivB5_p19.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p19.csv', index = False)

dfpowerprepindivB1_p22 = pd.concat([dfpowerprep1allbandsright_p22, dfpowerprep1allbandsleft_p22], axis=0)
dfpowerprepindivB1_p22.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p22.csv', index = False)
dfpowerprepindivB5_p22 = pd.concat([dfpowerprep5allbandsright_p22, dfpowerprep5allbandsleft_p22], axis=0)
dfpowerprepindivB5_p22.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p22.csv', index = False)

dfpowerfeedbackindivB1_p22 = pd.concat([dfpowerfeedback1allbandsright_p22, dfpowerfeedback1allbandsleft_p22], axis=0)
dfpowerfeedbackindivB1_p22.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p22.csv', index = False)
dfpowerfeedbackindivB5_p22 = pd.concat([dfpowerfeedback5allbandsright_p22, dfpowerfeedback5allbandsleft_p22], axis=0)
dfpowerfeedbackindivB5_p22.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p22.csv', index = False)


dfpowerprepindivB1_p23 = pd.concat([dfpowerprep1allbandsright_p23, dfpowerprep1allbandsleft_p23], axis=0)
dfpowerprepindivB1_p23.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p23.csv', index = False)
dfpowerprepindivB5_p23 = pd.concat([dfpowerprep5allbandsright_p23, dfpowerprep5allbandsleft_p23], axis=0)
dfpowerprepindivB5_p23.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p23.csv', index = False)

dfpowerfeedbackindivB1_p23 = pd.concat([dfpowerfeedback1allbandsright_p23, dfpowerfeedback1allbandsleft_p23], axis=0)
dfpowerfeedbackindivB1_p23.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p23.csv', index = False)
dfpowerfeedbackindivB5_p23 = pd.concat([dfpowerfeedback5allbandsright_p23, dfpowerfeedback5allbandsleft_p23], axis=0)
dfpowerfeedbackindivB5_p23.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p23.csv', index = False)


dfpowerprepindivB1_p24 = pd.concat([dfpowerprep1allbandsright_p24, dfpowerprep1allbandsleft_p24], axis=0)
dfpowerprepindivB1_p24.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p24.csv', index = False)
dfpowerprepindivB5_p24 = pd.concat([dfpowerprep5allbandsright_p24, dfpowerprep5allbandsleft_p24], axis=0)
dfpowerprepindivB5_p24.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p24.csv', index = False)

dfpowerfeedbackindivB1_p24 = pd.concat([dfpowerfeedback1allbandsright_p24, dfpowerfeedback1allbandsleft_p24], axis=0)
dfpowerfeedbackindivB1_p24.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p24.csv', index = False)
dfpowerfeedbackindivB5_p24 = pd.concat([dfpowerfeedback5allbandsright_p24, dfpowerfeedback5allbandsleft_p24], axis=0)
dfpowerfeedbackindivB5_p24.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p24.csv', index = False)

dfpowerprepindivB1_p25 = pd.concat([dfpowerprep1allbandsright_p25, dfpowerprep1allbandsleft_p25], axis=0)
dfpowerprepindivB1_p25.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p25.csv', index = False)
dfpowerprepindivB5_p25 = pd.concat([dfpowerprep5allbandsright_p25, dfpowerprep5allbandsleft_p25], axis=0)
dfpowerprepindivB5_p25.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p25.csv', index = False)

dfpowerfeedbackindivB1_p25 = pd.concat([dfpowerfeedback1allbandsright_p25, dfpowerfeedback1allbandsleft_p25], axis=0)
dfpowerfeedbackindivB1_p25.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p25.csv', index = False)
dfpowerfeedbackindivB5_p25 = pd.concat([dfpowerfeedback5allbandsright_p25, dfpowerfeedback5allbandsleft_p25], axis=0)
dfpowerfeedbackindivB5_p25.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p25.csv', index = False)


dfpowerprepindivB1_p26 = pd.concat([dfpowerprep1allbandsright_p26, dfpowerprep1allbandsleft_p26], axis=0)
dfpowerprepindivB1_p26.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p26.csv', index = False)
dfpowerprepindivB5_p26 = pd.concat([dfpowerprep5allbandsright_p26, dfpowerprep5allbandsleft_p26], axis=0)
dfpowerprepindivB5_p26.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p26.csv', index = False)

dfpowerfeedbackindivB1_p26 = pd.concat([dfpowerfeedback1allbandsright_p26, dfpowerfeedback1allbandsleft_p26], axis=0)
dfpowerfeedbackindivB1_p26.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p26.csv', index = False)
dfpowerfeedbackindivB5_p26 = pd.concat([dfpowerfeedback5allbandsright_p26, dfpowerfeedback5allbandsleft_p26], axis=0)
dfpowerfeedbackindivB5_p26.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p26.csv', index = False)

dfpowerprepindivB1_p27 = pd.concat([dfpowerprep1allbandsright_p27, dfpowerprep1allbandsleft_p27], axis=0)
dfpowerprepindivB1_p27.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p27.csv', index = False)
dfpowerprepindivB5_p27 = pd.concat([dfpowerprep5allbandsright_p27, dfpowerprep5allbandsleft_p27], axis=0)
dfpowerprepindivB5_p27.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p27.csv', index = False)

dfpowerfeedbackindivB1_p27 = pd.concat([dfpowerfeedback1allbandsright_p27, dfpowerfeedback1allbandsleft_p27], axis=0)
dfpowerfeedbackindivB1_p27.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p27.csv', index = False)
dfpowerfeedbackindivB5_p27 = pd.concat([dfpowerfeedback5allbandsright_p27, dfpowerfeedback5allbandsleft_p27], axis=0)
dfpowerfeedbackindivB5_p27.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p27.csv', index = False)

dfpowerprepindivB1_p28 = pd.concat([dfpowerprep1allbandsright_p28, dfpowerprep1allbandsleft_p28], axis=0)
dfpowerprepindivB1_p28.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p28.csv', index = False)
dfpowerprepindivB5_p28 = pd.concat([dfpowerprep5allbandsright_p28, dfpowerprep5allbandsleft_p28], axis=0)
dfpowerprepindivB5_p28.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p28.csv', index = False)

dfpowerfeedbackindivB1_p28 = pd.concat([dfpowerfeedback1allbandsright_p28, dfpowerfeedback1allbandsleft_p28], axis=0)
dfpowerfeedbackindivB1_p28.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p28.csv', index = False)
dfpowerfeedbackindivB5_p28 = pd.concat([dfpowerfeedback5allbandsright_p28, dfpowerfeedback5allbandsleft_p28], axis=0)
dfpowerfeedbackindivB5_p28.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p28.csv', index = False)

dfpowerprepindivB1_p29 = pd.concat([dfpowerprep1allbandsright_p29, dfpowerprep1allbandsleft_p29], axis=0)
dfpowerprepindivB1_p29.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p29.csv', index = False)
dfpowerprepindivB5_p29 = pd.concat([dfpowerprep5allbandsright_p29, dfpowerprep5allbandsleft_p29], axis=0)
dfpowerprepindivB5_p29.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p29.csv', index = False)

dfpowerfeedbackindivB1_p29 = pd.concat([dfpowerfeedback1allbandsright_p29, dfpowerfeedback1allbandsleft_p29], axis=0)
dfpowerfeedbackindivB1_p29.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p29.csv', index = False)
dfpowerfeedbackindivB5_p29 = pd.concat([dfpowerfeedback5allbandsright_p29, dfpowerfeedback5allbandsleft_p29], axis=0)
dfpowerfeedbackindivB5_p29.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p29.csv', index = False)

dfpowerprepindivB1_p30 = pd.concat([dfpowerprep1allbandsright_p30, dfpowerprep1allbandsleft_p30], axis=0)
dfpowerprepindivB1_p30.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB1_p30.csv', index = False)
dfpowerprepindivB5_p30 = pd.concat([dfpowerprep5allbandsright_p30, dfpowerprep5allbandsleft_p30], axis=0)
dfpowerprepindivB5_p30.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprepindivB5_p30.csv', index = False)

dfpowerfeedbackindivB1_p30 = pd.concat([dfpowerfeedback1allbandsright_p30, dfpowerfeedback1allbandsleft_p30], axis=0)
dfpowerfeedbackindivB1_p30.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB1_p30.csv', index = False)
dfpowerfeedbackindivB5_p30 = pd.concat([dfpowerfeedback5allbandsright_p30, dfpowerfeedback5allbandsleft_p30], axis=0)
dfpowerfeedbackindivB5_p30.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedbackindivB5_p30.csv', index = False)

#-----------------------------------------------------------------------------#
#                        MORLET AVERAGE OF ALL PARTICIPANTS
#-----------------------------------------------------------------------------#

#-------------------PREPARATION----------------#

#-------------------PREPARATION ALL BANDS------------------#
#----B1----#
powerprep1allbandsleft, itc321 = tfr_morlet(concatpreparationb1left, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allbandsright, itc322 = tfr_morlet(concatpreparationb1right, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
#----B5----#

powerprep5allbandsleft, itc323 = tfr_morlet(concatpreparationb5left, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep5allbandsright, itc324 = tfr_morlet(concatpreparationb5right, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#-------------------PREPARATION ALL CHANNELS ALL BANDS------------------#

#(for computing topo maps of average power)

#----B1----#
powerprep1allleft, itc325 = tfr_morlet(concatpreparationallb1left, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerprep1allright, itc326 = tfr_morlet(concatpreparationallb1right, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerprep5allleft, itc327 = tfr_morlet(concatpreparationallb5left, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerprep5allright, itc328 = tfr_morlet(concatpreparationallb5right, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)



#-------------------FEEDBACK----------------#

#-------------------FEEDBACK ALL BANDS------------------#
#----B1----#
powerfeedback1allbandsleft, itc337 = tfr_morlet(concatfeedbackb1left, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allbandsright, itc338 = tfr_morlet(concatfeedbackb1right, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
#----B5----#

powerfeedback5allbandsleft, itc339 = tfr_morlet(concatfeedbackb5left, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback5allbandsright, itc340 = tfr_morlet(concatfeedbackb5right, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#-------------------FEEDBACK ALL CHANNELS ALL BANDS------------------#

#(for computing topo maps of average power)

#----B1----#
powerfeedback1allleft, itc341 = tfr_morlet(concatfeedbackallb1left, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
powerfeedback1allright, itc342 = tfr_morlet(concatfeedbackallb1right, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

#----B5----#
powerfeedback5allleft, itc343 = tfr_morlet(concatfeedbackallb5left, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)

powerfeedback5allright, itc344 = tfr_morlet(concatfeedbackallb5right, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)







#exporting to dataframe
#prep
#b1
dfpowerprep1allbandsleft = powerprep1allbandsleft.to_data_frame(time_format=None)
dfpowerprep1allbandsleft['Block'] = 'B1'
dfpowerprep1allbandsleft['Hand'] = 'left'
print(dfpowerprep1allbandsleft)

dfpowerprep1allbandsright = powerprep1allbandsright.to_data_frame(time_format=None)
dfpowerprep1allbandsright['Block'] = 'B1'
dfpowerprep1allbandsright['Hand'] = 'right'
print(dfpowerprep1allbandsright)

#b5
dfpowerprep5allbandsleft = powerprep5allbandsleft.to_data_frame(time_format=None)
dfpowerprep5allbandsleft['Block'] = 'B5'
dfpowerprep5allbandsleft['Hand'] = 'left'
print(dfpowerprep5allbandsleft)

dfpowerprep5allbandsright = powerprep5allbandsright.to_data_frame(time_format=None)
dfpowerprep5allbandsright['Block'] = 'B5'
dfpowerprep5allbandsright['Hand'] = 'right'
print(dfpowerprep1allbandsright)


dfpowerprep = pd.concat([dfpowerprep1allbandsleft, dfpowerprep5allbandsleft, dfpowerprep1allbandsright, dfpowerprep5allbandsright], axis=0)
dfpowerprep.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerprep.csv', index = False)


#feedback
#b1
dfpowerfeedback1allbandsleft = powerfeedback1allbandsleft.to_data_frame(time_format=None)
dfpowerfeedback1allbandsleft['Block'] = 'B1'
dfpowerfeedback1allbandsleft['Hand'] = 'left'
print(dfpowerfeedback1allbandsleft)

dfpowerfeedback1allbandsright = powerfeedback1allbandsright.to_data_frame(time_format=None)
dfpowerfeedback1allbandsright['Block'] = 'B1'
dfpowerfeedback1allbandsright['Hand'] = 'right'
print(dfpowerfeedback1allbandsright)

#b5
dfpowerfeedback5allbandsleft = powerfeedback5allbandsleft.to_data_frame(time_format=None)
dfpowerfeedback5allbandsleft['Block'] = 'B5'
dfpowerfeedback5allbandsleft['Hand'] = 'left'
print(dfpowerfeedback5allbandsleft)

dfpowerfeedback5allbandsright = powerfeedback5allbandsright.to_data_frame(time_format=None)
dfpowerfeedback5allbandsright['Block'] = 'B5'
dfpowerfeedback5allbandsright['Hand'] = 'right'
print(dfpowerfeedback1allbandsright)


dfpowerfeedback = pd.concat([dfpowerfeedback1allbandsleft, dfpowerfeedback5allbandsleft, dfpowerfeedback1allbandsright, dfpowerfeedback5allbandsright], axis=0)
dfpowerfeedback.to_csv(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\XLS files for master thesis\powerfeedback.csv', index = False)



# Computing a topomap of average power for all channels to guide which channels should be picked
#LEFT HAND
#prep
powerprep1allleft.plot_topo(tmin=-0.5, tmax=1.5, baseline=baseline, mode='percent', vmin= -2, vmax=2, title='Average power prep1 left ')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\topo\prep\powerprep1alltopoleft')

powerprep5allleft.plot_topo(tmin=-0.5, tmax=1.5, baseline=baseline, mode='percent', vmin= -2, vmax=2, title='Average power prep5 left')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\topo\prep\powerprep5alltopoleft')

#feedback
powerfeedback1allleft.plot_topo(tmin=-0.5, tmax=2.5, baseline=baseline3, mode='percent', vmin= -2, vmax=2, title='Average power feedback1 left')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\topo\feedback\powerfeedback1alltopoleft')


powerfeedback5allleft.plot_topo(tmin=-0.5, tmax=2.5, baseline=baseline3, mode='percent', vmin= -2, vmax=2, title='Average power feedback5 left')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\topo\feedback\powerfeedback5alltopoleft')


#Showing the morlet wavelets in plots per frequency band
vmin, vmax = -2, 2
#----PREPARATION----#

#C3
powerprep1allbandsleft.plot(['C3'], tmin=-0.5, tmax=1.5, baseline=baseline, mode='percent', vmin=vmin, vmax=vmax, title='prep1 C3 left')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\wavelets\prep\C3\powerprep1C3left')


powerprep5allbandsleft.plot(['C3'], tmin=-0.5, tmax=1.5, baseline=baseline, mode='percent', vmin=vmin, vmax=vmax, title='prep5 C3 left')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\wavelets\prep\C3\powerprep5C3left')

#C4
powerprep1allbandsleft.plot(['C4'], tmin=-0.5, tmax=1.5, baseline=baseline, mode='percent', vmin=vmin, vmax=vmax, title='prep1 C4 left')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\wavelets\prep\C4\powerprep1C4left')


powerprep5allbandsleft.plot(['C4'], tmin=-0.5, tmax=1.5, baseline=baseline, mode='percent', vmin=vmin, vmax=vmax, title='prep5 C4 left')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\wavelets\prep\C4\powerprep5C4left')


#----FEEDBACK----#

#C3
powerfeedback1allbandsleft.plot(['C3'], tmin=-0.5, tmax=2.5, baseline=baseline3, mode='percent', vmin=vmin, vmax=vmax, title='feedback1 C3 left')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\wavelets\feedback\C3\powerfeedback1C3left')


powerfeedback5allbandsleft.plot(['C3'], tmin=-0.5, tmax=2.5, baseline=baseline3, mode='percent', vmin=vmin, vmax=vmax, title='feedback5 C3 left')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\wavelets\feedback\C3\powerfeedback5C3left')

#C4
powerfeedback1allbandsleft.plot(['C4'], tmin=-0.5, tmax=2.5, baseline=baseline3, mode='percent', vmin=vmin, vmax=vmax, title='feedback1 C4 left')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\wavelets\feedback\C4\powerfeedback1C4left')


powerfeedback5allbandsleft.plot(['C4'], tmin=-0.5, tmax=2.5, baseline=baseline3, mode='percent', vmin=vmin, vmax=vmax, title='feedback5 C4 left')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\wavelets\feedback\C4\powerfeedback5C4left')


#RIGHT HAND

# Computing a topomap of average power for all channels to guide which channels should be picked
#prep
powerprep1allright.plot_topo(tmin=-0.5, tmax=1.5, baseline=baseline, mode='percent', vmin= -2, vmax=2, title='Average power prep1 right ')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\topo\prep\powerprep1alltoporight')

powerprep5allright.plot_topo(tmin=-0.5, tmax=1.5, baseline=baseline, mode='percent', vmin= -2, vmax=2, title='Average power prep5 right')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\topo\prep\powerprep5alltoporight')

#feedback
powerfeedback1allright.plot_topo(tmin=-0.5, tmax=2.5, baseline=baseline3, mode='percent', vmin= -2, vmax=2, title='Average power feedback1 right')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\topo\feedback\powerfeedback1alltoporight')


powerfeedback5allright.plot_topo(tmin=-0.5, tmax=2.5, baseline=baseline3, mode='percent', vmin= -2, vmax=2, title='Average power feedback5 right')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\topo\feedback\powerfeedback5alltoporight')


#Showing the morlet wavelets in plots per frequency band
vmin, vmax = -2, 2
#----PREPARATION----#

#C3/
powerprep1allbandsright.plot(['C3'], tmin=-0.5, tmax=1.5, baseline=baseline, mode='percent', vmin=vmin, vmax=vmax, title='prep1 C3 right')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\wavelets\prep\C3\powerprep1C3right')


powerprep5allbandsright.plot(['C3'], tmin=-0.5, tmax=1.5, baseline=baseline, mode='percent', vmin=vmin, vmax=vmax, title='prep5 C3 right')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\wavelets\prep\C3\powerprep5C3right')

#C4
powerprep1allbandsright.plot(['C4'], tmin=-0.5, tmax=1.5, baseline=baseline, mode='percent', vmin=vmin, vmax=vmax, title='prep1 C4 right')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\wavelets\prep\C4\powerprep1C4right')


powerprep5allbandsright.plot(['C4'], tmin=-0.5, tmax=1.5, baseline=baseline, mode='percent', vmin=vmin, vmax=vmax, title='prep5 C4 right')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\wavelets\prep\C4\powerprep5C4right')



#----FEEDBACK----#

#C3
powerfeedback1allbandsright.plot(['C3'], tmin=-0.5, tmax=2.5, baseline=baseline3, mode='percent', vmin=vmin, vmax=vmax, title='feedback1 C3 right')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\wavelets\feedback\C3\powerfeedback1C3right')


powerfeedback5allbandsright.plot(['C3'], tmin=-0.5, tmax=2.5, baseline=baseline3, mode='percent', vmin=vmin, vmax=vmax, title='feedback5 C3 right')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\wavelets\feedback\C3\powerfeedback5C3right')

#C4
powerfeedback1allbandsright.plot(['C4'], tmin=-0.5, tmax=2.5, baseline=baseline3, mode='percent', vmin=vmin, vmax=vmax, title='feedback1 C4 right')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\wavelets\feedback\C4\powerfeedback1C4right')


powerfeedback5allbandsright.plot(['C4'], tmin=-0.5, tmax=2.5, baseline=baseline3, mode='percent', vmin=vmin, vmax=vmax, title='feedback5 C4 right')
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\wavelets\feedback\C4\powerfeedback5C4right')




#Topoplots over time
#LEFT HAND
#prep b1 left

fig, axis = plt.subplots(1, 3, figsize=(30, 30))
powerprep1allleft.plot_topomap(ch_type='eeg', tmin=-1.1, tmax=-1, fmin=19, fmax=23, vmin=vmin, vmax=vmax,
                   baseline=baseline, mode='percent', axes=axis[0], show=False)
powerprep1allleft.plot_topomap(ch_type='eeg', tmin=-1, tmax=-0.9, fmin=19, fmax=23, vmin=vmin, vmax=vmax,
                   baseline=baseline, mode='percent', axes=axis[1], show=False)
powerprep1allleft.plot_topomap(ch_type='eeg', tmin=-0.9, tmax=-0.8, fmin=19, fmax=23, vmin=vmin, vmax=vmax,
                   baseline=baseline, mode='percent', axes=axis[2], show=False)

plt.suptitle('Left hand - Block 1',fontsize=20, X=0.1, y=0.9)
mne.viz.tight_layout()
plt.show()
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\topomap\prep\powerprep1b2alltopomapleft')



#prep b5 left
fig2, axis2 = plt.subplots(1, 3, figsize=(30, 30))
powerprep5allleft.plot_topomap(ch_type='eeg', tmin=-1.1, tmax=-1, fmin=19, fmax=23, vmin=vmin, vmax=vmax,
                   baseline=baseline, mode='percent', axes=axis2[0], show=False)
powerprep5allleft.plot_topomap(ch_type='eeg', tmin=-1, tmax=-0.9, fmin=19, fmax=23, vmin=vmin, vmax=vmax,
                   baseline=baseline, mode='percent', axes=axis2[1], show=False)
powerprep5allleft.plot_topomap(ch_type='eeg', tmin=-0.9, tmax=-0.8, fmin=19, fmax=23, vmin=vmin, vmax=vmax,
                   baseline=baseline, mode='percent', axes=axis2[2], show=False)

plt.suptitle('Left hand - Block 5',fontsize=20, X=0.1, y=0.9)
mne.viz.tight_layout()
plt.show()
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\topomap\prep\powerprep5b2alltopomapleft')


#feedbackb1 left
fig3, axis3 = plt.subplots(1, 4, figsize=(20, 20))
powerfeedback1allleft.plot_topomap(ch_type='eeg', tmin=0.7, tmax=0.8, fmin=12, fmax=18, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis3[0], show=False)
powerfeedback1allleft.plot_topomap(ch_type='eeg', tmin=0.8, tmax=0.9, fmin=12, fmax=18, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis3[1], show=False)
powerfeedback1allleft.plot_topomap(ch_type='eeg', tmin=1.1, tmax=1.2, fmin=12, fmax=18, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis3[2], show=False)
powerfeedback1allleft.plot_topomap(ch_type='eeg', tmin=1.2, tmax=1.3, fmin=12, fmax=18, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis3[3], show=False)

plt.suptitle('Left hand - Block 1',fontsize=20, X=0.1, y=0.8)
mne.viz.tight_layout()
plt.show()
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\topomap\feedback\powerfeedback1b1alltopomapleft')



#feedbackb5 left
fig4, axis4 = plt.subplots(1, 4, figsize=(30, 30))
powerfeedback5allleft.plot_topomap(ch_type='eeg', tmin=0.7, tmax=0.8, fmin=12, fmax=18, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis4[0], show=False)
powerfeedback5allleft.plot_topomap(ch_type='eeg', tmin=0.8, tmax=0.9, fmin=12, fmax=18, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis4[1], show=False)
powerfeedback5allleft.plot_topomap(ch_type='eeg', tmin=1.1, tmax=1.2, fmin=12, fmax=18, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis4[2], show=False)
powerfeedback5allleft.plot_topomap(ch_type='eeg', tmin=1.2, tmax=1.3, fmin=12, fmax=18, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis4[3], show=False)

plt.suptitle('Left hand - Block 5',fontsize=20, X=0.1, y=0.8)
mne.viz.tight_layout()
plt.show()
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\topomap\feedback\powerfeedback5b1alltopomapleft')

#feedbackb1 left
fig5, axis5 = plt.subplots(1, 4, figsize=(30, 30))
powerfeedback1allleft.plot_topomap(ch_type='eeg', tmin=0.7, tmax=0.8, fmin=19, fmax=23, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis5[0], show=False)
powerfeedback1allleft.plot_topomap(ch_type='eeg', tmin=1.0, tmax=1.1, fmin=19, fmax=23, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis5[1], show=False)
powerfeedback1allleft.plot_topomap(ch_type='eeg', tmin=1.1, tmax=1.2, fmin=19, fmax=23, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis5[2], show=False)
powerfeedback1allleft.plot_topomap(ch_type='eeg', tmin=1.2, tmax=1.3, fmin=19, fmax=23, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis5[3], show=False)

plt.suptitle('Left hand - Block 1',fontsize=20, X=0.1, y=0.8)
mne.viz.tight_layout()
plt.show()
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\topomap\feedback\powerfeedback1b2alltopomapleft')

#feedbackb5 left
fig6, axis6 = plt.subplots(1, 4, figsize=(30, 30))
powerfeedback5allleft.plot_topomap(ch_type='eeg', tmin=0.7, tmax=0.8, fmin=19, fmax=23, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis6[0], show=False)
powerfeedback5allleft.plot_topomap(ch_type='eeg', tmin=1.0, tmax=1.1, fmin=19, fmax=23, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis6[1], show=False)
powerfeedback5allleft.plot_topomap(ch_type='eeg', tmin=1.1, tmax=1.2, fmin=19, fmax=23, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis6[2], show=False)
powerfeedback5allleft.plot_topomap(ch_type='eeg', tmin=1.2, tmax=1.3, fmin=19, fmax=23, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis6[3], show=False)

plt.suptitle('Left hand - Block 5',fontsize=20, X=0.1, y=0.8)
mne.viz.tight_layout()
plt.show()
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\topomap\feedback\powerfeedback5b2alltopomapleft')




#feedbackb1 left
fig7, axis7 = plt.subplots(1, 5, figsize=(30, 30))
powerfeedback1allleft.plot_topomap(ch_type='eeg', tmin=0, tmax=0.1, fmin=24, fmax=29, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis7[0], show=False)
powerfeedback1allleft.plot_topomap(ch_type='eeg', tmin=0.7, tmax=0.8, fmin=24, fmax=29, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis7[1], show=False)
powerfeedback1allleft.plot_topomap(ch_type='eeg', tmin=0.8, tmax=0.9, fmin=24, fmax=29, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis7[2], show=False)
powerfeedback1allleft.plot_topomap(ch_type='eeg', tmin=1, tmax=1.1, fmin=24, fmax=29, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis7[3], show=False)
powerfeedback1allleft.plot_topomap(ch_type='eeg', tmin=1.1, tmax=1.2, fmin=24, fmax=29, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis7[4], show=False)

plt.suptitle('Left hand - Block 1',fontsize=20, X=0.1, y=0.8)
mne.viz.tight_layout()
plt.show()
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\topomap\feedback\powerfeedback1b3alltopomapleft')


#feedbackb5 left
fig8, axis8 = plt.subplots(1, 5, figsize=(30, 30))
powerfeedback5allleft.plot_topomap(ch_type='eeg', tmin=0, tmax=0.1, fmin=24, fmax=29, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis8[0], show=False)
powerfeedback5allleft.plot_topomap(ch_type='eeg', tmin=0.7, tmax=0.8, fmin=24, fmax=29, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis8[1], show=False)
powerfeedback5allleft.plot_topomap(ch_type='eeg', tmin=0.8, tmax=0.9, fmin=24, fmax=29, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis8[2], show=False)
powerfeedback5allleft.plot_topomap(ch_type='eeg', tmin=1, tmax=1.1, fmin=24, fmax=29, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis8[3], show=False)
powerfeedback5allleft.plot_topomap(ch_type='eeg', tmin=1.1, tmax=1.2, fmin=24, fmax=29, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis8[4], show=False)

plt.suptitle('Left hand - Block 5',fontsize=20, X=0.1, y=0.8)
mne.viz.tight_layout()
plt.show()
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\topomap\feedback\powerfeedback5b3alltopomapleft')


#Topoplots over time
#RIGHT HAND

#prep b1 right

fig9, axis9 = plt.subplots(1, 3, figsize=(30, 30))
powerprep1allright.plot_topomap(ch_type='eeg', tmin=-1.1, tmax=-1, fmin=19, fmax=23, vmin=vmin, vmax=vmax,
                   baseline=baseline, mode='percent', axes=axis9[0], show=False)
powerprep1allright.plot_topomap(ch_type='eeg', tmin=-0.8, tmax=-0.7, fmin=19, fmax=23, vmin=vmin, vmax=vmax,
                   baseline=baseline, mode='percent', axes=axis9[1], show=False)
powerprep1allright.plot_topomap(ch_type='eeg', tmin=-0.4, tmax=-0.3, fmin=19, fmax=23, vmin=vmin, vmax=vmax,
                   baseline=baseline, mode='percent', axes=axis9[2], show=False)

plt.suptitle('Right hand - Block 1',fontsize=20, X=0.1, y=0.9)
mne.viz.tight_layout()
plt.show()
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\topomap\prep\powerprep1b2alltopomapright')



#prep b5 right
fig10, axis10 = plt.subplots(1, 3, figsize=(30, 30))
powerprep5allright.plot_topomap(ch_type='eeg', tmin=-1.1, tmax=-1, fmin=19, fmax=23, vmin=vmin, vmax=vmax,
                   baseline=baseline, mode='percent', axes=axis10[0], show=False)
powerprep5allright.plot_topomap(ch_type='eeg', tmin=-0.8, tmax=-0.7, fmin=19, fmax=23, vmin=vmin, vmax=vmax,
                   baseline=baseline, mode='percent', axes=axis10[1], show=False)
powerprep5allright.plot_topomap(ch_type='eeg', tmin=-0.4, tmax=-0.3, fmin=19, fmax=23, vmin=vmin, vmax=vmax,
                   baseline=baseline, mode='percent', axes=axis10[2], show=False)

plt.suptitle('Right hand - Block 5',fontsize=20, X=0.1, y=0.9)
mne.viz.tight_layout()
plt.show()
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\topomap\prep\powerprep5b2alltopomapright')


#feedbackb1 right
fig11, axis11 = plt.subplots(1, 2, figsize=(30, 30))
powerfeedback1allright.plot_topomap(ch_type='eeg', tmin=2, tmax=2.1, fmin=12, fmax=18, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis11[0], show=False)
powerfeedback1allright.plot_topomap(ch_type='eeg', tmin=2.1, tmax=2.2, fmin=12, fmax=18, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis11[1], show=False)

plt.suptitle('Right hand - Block 1',fontsize=20, X=0.1, y=1)
mne.viz.tight_layout()
plt.show()
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\topomap\feedback\powerfeedback1b1alltopomapright')


#feedbackb5 right
fig12, axis12 = plt.subplots(1, 2, figsize=(30, 30))
powerfeedback5allright.plot_topomap(ch_type='eeg', tmin=2, tmax=2.1, fmin=12, fmax=18, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis12[0], show=False)
powerfeedback5allright.plot_topomap(ch_type='eeg', tmin=2.1, tmax=2.2, fmin=12, fmax=18, vmin=vmin, vmax=vmax,
                   baseline=baseline3, mode='percent', axes=axis12[1], show=False)

plt.suptitle('Right hand - Block 5',fontsize=20, X=0.1, y=1)
mne.viz.tight_layout()
plt.show()
plt.savefig(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\plots\morlet\topomap\feedback\powerfeedback5b1alltopomapright')




