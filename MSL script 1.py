# -*- coding: utf-8 -*-
"""
Created on Thu May 13 12:44:23 2021

@author: Daphne Titsing
"""

#-----------------------------------------------------------------------------
#                           LOADING PACKAGES
#-----------------------------------------------------------------------------

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
#                               PARTICIPANT 14
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_14_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID1\Training\Part14\6key\part14_SeqL_ERD_6_B1.vhdr'
Part_14_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID1\Training\Part14\6key\part14_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p14 = mne.io.read_raw_brainvision(Part_14_1, preload = True) 

#--------B5--------#
raw5_p14 = mne.io.read_raw_brainvision(Part_14_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p14.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p14.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p14.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p14.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p14.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p14.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p14 = mne.io.add_reference_channels(raw_p14, 'TP8')   #reference channel

#--------B5-------#
raw5_p14.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p14.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p14.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p14.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p14.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p14.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p14 = mne.io.add_reference_channels(raw5_p14, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p14.plot_psd(fmax = 250)#maximum frequency 
raw_p14.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p14.plot()

#--------B5-------#
raw5_p14.plot_psd(fmax = 250)#maximum frequency 
raw5_p14.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p14.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p14.set_montage(montage)
raw5_p14.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p14, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p14, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p14.plot_sensors(kind='topomap', show_names=True)
raw5_p14.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p14.info)
print(raw5_p14.info)


#plot to show the waves and their source on the head
raw_p14.plot_psd(fmax = 250)
raw5_p14.plot_psd(fmax = 250)




#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica = mne.preprocessing.ICA()

raw_p14 = raw_p14.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica.fit(raw_p14)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices, eog_scores = ica.find_bads_eog(raw_p14, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica.exclude = eog_indices#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica.plot_scores(eog_scores)

# plot diagnostics
ica.plot_properties(raw_p14, picks=eog_indices)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica.plot_sources(raw_p14)


#check if raw data has been cleaned 
raw_p14.plot()


#visual presentation ICA components on head
ica.plot_components()

#creating eog epochs
eog_evoked = create_eog_epochs(raw_p14).average()
eog_evoked.apply_baseline(baseline=(None, -0.2))
eog_evoked.plot_joint()

ica.plot_properties(raw_p14, [0])
ica.exclude =[0]


#since ica.apply changes raw we are making a copy
reconst_raw_p14 = raw_p14.copy()
ica.apply(reconst_raw_p14)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p14.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p14.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part14_SeqL_ERD_6_B1.fif', overwrite=True)


#--------B5-------#

ica5 = mne.preprocessing.ICA()

raw5_p14 = raw5_p14.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5.fit(raw5_p14)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5, eog_scores5 = ica5.find_bads_eog(raw5_p14, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5.exclude = eog_indices5#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5.plot_scores(eog_scores5)

# plot diagnostics
ica5.plot_properties(raw5_p14, picks=eog_indices5)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5.plot_sources(raw5_p14)

#check if raw data has been cleaned 
raw5_p14.plot()

#visual presentation ICA components on head
ica5.plot_components()

#creating eog epochs
eog_evoked5 = create_eog_epochs(raw5_p14).average()
eog_evoked5.apply_baseline(baseline=(None, -0.2))
eog_evoked5.plot_joint()

ica5.plot_properties(raw5_p14, [0,1,2])
ica5.exclude =[0,1,2]

reconst_raw5_p14 = raw5_p14.copy()
ica5.apply(reconst_raw5_p14)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p14.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p14.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part14_SeqL_ERD_6_B5.fif', overwrite=True)




#-----------------------------------------------------------------------------#
#                               PARTICIPANT 1
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_1_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID1\Training\Part1\6 key\part1_SeqL_ERD_6_B1.vhdr'
Part_1_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID1\Training\Part1\6 key\part1_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p1 = mne.io.read_raw_brainvision(Part_1_1, preload = True) 

#--------B5--------#
raw5_p1 = mne.io.read_raw_brainvision(Part_1_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p1.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p1.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p1.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p1.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p1.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p1.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p1 = mne.io.add_reference_channels(raw_p1, 'TP8')   #reference channel


#--------B5-------#
raw5_p1.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p1.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p1.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p1.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p1.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p1.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p1 = mne.io.add_reference_channels(raw5_p1, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p1.plot_psd(fmax = 250)#maximum frequency 
raw_p1.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p1.plot()

#--------B5-------#
raw5_p1.plot_psd(fmax = 250)#maximum frequency 
raw5_p1.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p1.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p1.set_montage(montage)
raw5_p1.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p1, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p1, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p1.plot_sensors(kind='topomap', show_names=True)
raw5_p1.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p1.info)
print(raw5_p1.info)


#plot to show the waves and their source on the head
raw_p1.plot_psd(fmax = 250)
raw5_p1.plot_psd(fmax = 250)




#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica_p1 = mne.preprocessing.ICA()

raw_p1 = raw_p1.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica_p1.fit(raw_p1)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices_p1, eog_scores_p1 = ica_p1.find_bads_eog(raw_p1, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica_p1.exclude = eog_indices_p1#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica_p1.plot_scores(eog_scores_p1)

# plot diagnostics
ica_p1.plot_properties(raw_p1, picks=eog_indices_p1)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica_p1.plot_sources(raw_p1)


#check if raw data has been cleaned 
raw_p1.plot()


#visual presentation ICA components on head
ica_p1.plot_components()

#creating eog epochs
eog_evoked_p1 = create_eog_epochs(raw_p1).average()
eog_evoked_p1.apply_baseline(baseline=(None, -0.2))
eog_evoked_p1.plot_joint()

ica_p1.plot_properties(raw_p1, [0,1])
ica_p1.exclude =[0,1]


#since ica.apply changes raw we are making a copy
reconst_raw_p1 = raw_p1.copy()
ica_p1.apply(reconst_raw_p1)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p1.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p1.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part1_SeqL_ERD_6_B1.fif', overwrite=True)



#--------B5-------#

ica5_p1 = mne.preprocessing.ICA()

raw5_p1 = raw5_p1.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5_p1.fit(raw5_p1)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5_p1, eog_scores5_p1 = ica5_p1.find_bads_eog(raw5_p1, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5.exclude = eog_indices5#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5_p1.plot_scores(eog_scores5_p1)

# plot diagnostics
ica5_p1.plot_properties(raw5_p1, picks=eog_indices5_p1)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5_p1.plot_sources(raw5_p1)

#check if raw data has been cleaned 
raw5_p1.plot()

#visual presentation ICA components on head
ica5_p1.plot_components()

#creating eog epochs
eog_evoked5_p1 = create_eog_epochs(raw5_p1).average()
eog_evoked5_p1.apply_baseline(baseline=(None, -0.2))
eog_evoked5_p1.plot_joint()

ica5_p1.plot_properties(raw5_p1, [0,1,2])
ica5_p1.exclude =[0,1,2]

reconst_raw5_p1 = raw5_p1.copy()
ica5.apply(reconst_raw5_p1)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p1.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p1.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part1_SeqL_ERD_6_B5.fif', overwrite=True)



#-----------------------------------------------------------------------------#
#                               PARTICIPANT 9
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_9_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID1\Training\Part9\6key\part9_SeqL_ERD_6_B1.vhdr'
Part_9_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID1\Training\Part9\6key\part9_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p9 = mne.io.read_raw_brainvision(Part_9_1, preload = True) 

#--------B5--------#
raw5_p9 = mne.io.read_raw_brainvision(Part_9_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p9.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p9.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p9.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p9.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p9.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p9.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p9 = mne.io.add_reference_channels(raw_p9, 'TP8')   #reference channel


#--------B5-------#
raw5_p9.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p9.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p9.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p9.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p9.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p9.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p9 = mne.io.add_reference_channels(raw5_p9, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p9.plot_psd(fmax = 250)#maximum frequency 
raw_p9.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p9.plot()

#--------B5-------#
raw5_p9.plot_psd(fmax = 250)#maximum frequency 
raw5_p9.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p9.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p9.set_montage(montage)
raw5_p9.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p9, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p9, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p9.plot_sensors(kind='topomap', show_names=True)
raw5_p9.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p9.info)
print(raw5_p9.info)


#plot to show the waves and their source on the head
raw_p9.plot_psd(fmax = 250)
raw5_p9.plot_psd(fmax = 250)



#-----------------------------------------------------------------------------#
#                               PARTICIPANT 24
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_24_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID1\Training\Part24\6key\part24_SeqL_ERD_6_B1.vhdr'
Part_24_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID1\Training\Part24\6key\part24_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p24 = mne.io.read_raw_brainvision(Part_24_1, preload = True) 


#--------B5--------#
raw5_p24 = mne.io.read_raw_brainvision(Part_24_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p24.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p24.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p24.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p24.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p24.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p24.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p24 = mne.io.add_reference_channels(raw_p24, 'TP8')   #reference channel

#--------B5-------#
raw5_p24.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw5_p24.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p24.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p24.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p24.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p24.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p24 = mne.io.add_reference_channels(raw5_p24, 'TP8')  #=reference channel


raw5_p24.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p24.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p24.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p24.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p24.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p24.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p24 = mne.io.add_reference_channels(raw5_p24, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p24.plot_psd(fmax = 250)#maximum frequency 
raw_p24.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p24.plot()

#--------B3-------#
raw3_p24.plot_psd(fmax = 250)#maximum frequency 
raw3_p24.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw3_p24.plot()

#--------B1-------#
raw_p24.plot_psd(fmax = 250)#maximum frequency 
raw_p24.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p24.plot()

#--------B5-------#
raw5_p24.plot_psd(fmax = 250)#maximum frequency 
raw5_p24.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p24.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p24.set_montage(montage)
raw5_p24.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p24, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p24, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p24.plot_sensors(kind='topomap', show_names=True)
raw5_p24.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p24.info)
print(raw5_p24.info)


#plot to show the waves and their source on the head
raw_p24.plot_psd(fmax = 250)
raw5_p24.plot_psd(fmax = 250)



#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica_p24 = mne.preprocessing.ICA()

raw_p24 = raw_p24.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica_p24.fit(raw_p24)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices_p24, eog_scores_p24 = ica_p24.find_bads_eog(raw_p24, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica_p24.exclude = eog_indices_p24#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica_p24.plot_scores(eog_scores_p24)

# plot diagnostics
ica_p24.plot_properties(raw_p24, picks=eog_indices_p24)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica_p24.plot_sources(raw_p24)


#check if raw data has been cleaned 
raw_p24.plot()


#visual presentation ICA components on head
ica_p24.plot_components()

#creating eog epochs
eog_evoked_p24 = create_eog_epochs(raw_p24).average()
eog_evoked_p24.apply_baseline(baseline=(None, -0.2))
eog_evoked_p24.plot_joint()

ica_p24.plot_properties(raw_p24, [0])
ica_p24.exclude =[0]


#since ica.apply changes raw we are making a copy
reconst_raw_p24 = raw_p24.copy()
ica_p24.apply(reconst_raw_p24)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p24.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p24.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part24_SeqL_ERD_6_B1.fif', overwrite=True)




#--------B5-------#

ica5_p24 = mne.preprocessing.ICA()

raw5_p24 = raw5_p24.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5_p24.fit(raw5_p24)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5_p24, eog_scores5_p24 = ica5_p24.find_bads_eog(raw5_p24, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5_p24.exclude = eog_indices5_p24#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5_p24.plot_scores(eog_scores5_p24)

# plot diagnostics
ica5_p24.plot_properties(raw5_p24, picks=eog_indices5_p24)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5_p24.plot_sources(raw5_p24)

#check if raw data has been cleaned 
raw5_p24.plot()

#visual presentation ICA components on head
ica5_p24.plot_components()

#creating eog epochs
eog_evoked5_p24 = create_eog_epochs(raw5_p24).average()
eog_evoked5_p24.apply_baseline(baseline=(None, -0.2))
eog_evoked5_p24.plot_joint()

ica5_p24.plot_properties(raw5_p24, [1,3])

ica5_p24.exclude =[1,3]

reconst_raw5_p24 = raw5_p24.copy()
ica5_p24.apply(reconst_raw5_p24)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p24.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p24.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part24_SeqL_ERD_6_B5.fif', overwrite=True)



#-----------------------------------------------------------------------------#
#                                   ID2
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
#                               PARTICIPANT 2
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_2_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID2\Training\Part2\6 key\part2_SeqL_ERD_6_B1.vhdr'
Part_2_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID2\Training\Part2\6 key\part2_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p2 = mne.io.read_raw_brainvision(Part_2_1, preload = True) 

#--------B5--------#
raw5_p2 = mne.io.read_raw_brainvision(Part_2_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p2.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p2.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p2.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p2.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p2.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p2.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p2 = mne.io.add_reference_channels(raw_p2, 'TP8')   #reference channel


#--------B5-------#
raw5_p2.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p2.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p2.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p2.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p2.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p2.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p2 = mne.io.add_reference_channels(raw5_p2, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p2.plot_psd(fmax = 250)#maximum frequency 
raw_p2.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p2.plot()


#--------B5-------#
raw5_p2.plot_psd(fmax = 250)#maximum frequency 
raw5_p2.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p2.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p2.set_montage(montage)
raw5_p2.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p2, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p2, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p2.plot_sensors(kind='topomap', show_names=True)
raw5_p2.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p2.info)
print(raw5_p2.info)


#plot to show the waves and their source on the head
raw_p2.plot_psd(fmax = 250)
raw5_p2.plot_psd(fmax = 250)




#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica_p2 = mne.preprocessing.ICA()

raw_p2 = raw_p2.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica_p2.fit(raw_p2)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices_p2, eog_scores_p2 = ica_p2.find_bads_eog(raw_p2, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica_p2.exclude = eog_indices_p2#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica_p2.plot_scores(eog_scores_p2)

# plot diagnostics
ica_p2.plot_properties(raw_p2, picks=eog_indices_p2)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica_p2.plot_sources(raw_p2)


#check if raw data has been cleaned 
raw_p9.plot()


#visual presentation ICA components on head
ica_p2.plot_components()

#creating eog epochs
eog_evoked_p2 = create_eog_epochs(raw_p2).average()
eog_evoked_p2.apply_baseline(baseline=(None, -0.2))
eog_evoked_p2.plot_joint()

ica_p2.plot_properties(raw_p2, [0,3,4])
ica_p2.exclude =[0,3,4]


#since ica.apply changes raw we are making a copy
reconst_raw_p2 = raw_p2.copy()
ica_p2.apply(reconst_raw_p2)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p2.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p2.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part2_SeqL_ERD_6_B1.fif', overwrite=True)




#--------B5-------#

ica5_p2 = mne.preprocessing.ICA()

raw5_p2 = raw5_p2.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5_p2.fit(raw5_p2)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5_p2, eog_scores5_p2 = ica5_p2.find_bads_eog(raw5_p2, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5_p2.exclude = eog_indices5_p2#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5_p2.plot_scores(eog_scores5_p2)

# plot diagnostics
ica5_p2.plot_properties(raw5_p2, picks=eog_indices5_p2)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5_p2.plot_sources(raw5_p2)

#check if raw data has been cleaned 
raw5_p2.plot()

#visual presentation ICA components on head
ica5_p2.plot_components()

#creating eog epochs
eog_evoked5_p2 = create_eog_epochs(raw5_p2).average()
eog_evoked5_p2.apply_baseline(baseline=(None, -0.2))
eog_evoked5_p2.plot_joint()

ica5_p2.plot_properties(raw5_p2, [0,1])
ica5_p2.exclude =[0,1]

reconst_raw5_p2 = raw5_p2.copy()
ica5_p2.apply(reconst_raw5_p2)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p2.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p2.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part2_SeqL_ERD_6_B5.fif', overwrite=True)




#-----------------------------------------------------------------------------#
#                               PARTICIPANT 10
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_10_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID2\Training\Part10\6key\part10_SeqL_ERD_6_B1.vhdr'
Part_10_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID2\Training\Part10\6key\part10_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p10 = mne.io.read_raw_brainvision(Part_10_1, preload = True) 

#--------B5--------#
raw5_p10 = mne.io.read_raw_brainvision(Part_10_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p10.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p10.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p10.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p10.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p10.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p10.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p10 = mne.io.add_reference_channels(raw_p10, 'TP8')   #reference channel



#--------B5-------#
raw5_p10.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p10.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p10.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p10.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p10.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p10.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p10 = mne.io.add_reference_channels(raw5_p10, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p10.plot_psd(fmax = 250)#maximum frequency 
raw_p10.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p10.plot()


#--------B5-------#
raw5_p10.plot_psd(fmax = 250)#maximum frequency 
raw5_p10.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p10.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p10.set_montage(montage)

raw5_p10.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p10, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p10, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p10.plot_sensors(kind='topomap', show_names=True)
raw5_p10.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p10.info)
print(raw5_p10.info)


#plot to show the waves and their source on the head
raw_p10.plot_psd(fmax = 250)
raw5_p10.plot_psd(fmax = 250)




#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica = mne.preprocessing.ICA()

raw_p10 = raw_p10.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica.fit(raw_p10)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices, eog_scores = ica.find_bads_eog(raw_p10, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica.exclude = eog_indices#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica.plot_scores(eog_scores)

# plot diagnostics
ica.plot_properties(raw_p10, picks=eog_indices)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica.plot_sources(raw_p10)


#check if raw data has been cleaned 
raw_p10.plot()


#visual presentation ICA components on head
ica.plot_components()

#creating eog epochs
eog_evoked = create_eog_epochs(raw_p10).average()
eog_evoked.apply_baseline(baseline=(None, -0.2))
eog_evoked.plot_joint()

ica.plot_properties(raw_p10, [0,1,2,3,4])
ica.exclude =[0,1,2,3,4]


#since ica.apply changes raw we are making a copy
reconst_raw_p10 = raw_p10.copy()
ica.apply(reconst_raw_p10)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p10.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p10.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part10_SeqL_ERD_6_B1.fif', overwrite=True)



#--------B5-------#

ica5 = mne.preprocessing.ICA()

raw5_p10 = raw5_p10.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5.fit(raw5_p10)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5, eog_scores5 = ica5.find_bads_eog(raw5_p10, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5.exclude = eog_indices5#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5.plot_scores(eog_scores5)

# plot diagnostics
ica5.plot_properties(raw5_p10, picks=eog_indices5)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5.plot_sources(raw5_p10)

#check if raw data has been cleaned 
raw5_p10.plot()

#visual presentation ICA components on head
ica5.plot_components()

#creating eog epochs
eog_evoked5 = create_eog_epochs(raw5_p10).average()
eog_evoked5.apply_baseline(baseline=(None, -0.2))
eog_evoked5.plot_joint()

ica5.plot_properties(raw5_p10, [1,3])
ica5.exclude =[1,3]

reconst_raw5_p10 = raw5_p10.copy()
ica5.apply(reconst_raw5_p10)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p10.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p10.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part10_SeqL_ERD_6_B5.fif', overwrite=True)



#-----------------------------------------------------------------------------#
#                               PARTICIPANT 1
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_18_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID2\Training\Part18\6key\part18_SeqL_ERD_6_B1.vhdr'
Part_18_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID2\Training\Part18\6key\part18_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p18 = mne.io.read_raw_brainvision(Part_18_1, preload = True) 

#--------B5--------#
raw5_p18 = mne.io.read_raw_brainvision(Part_18_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p18.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p18.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p18.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p18.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p18.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p18.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p18 = mne.io.add_reference_channels(raw_p18, 'TP8')   #reference channel


#--------B5-------#
raw5_p18.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p18.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p18.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p18.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p18.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p18.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p18 = mne.io.add_reference_channels(raw5_p18, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p18.plot_psd(fmax = 250)#maximum frequency 
raw_p18.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p18.plot()


#--------B5-------#
raw5_p18.plot_psd(fmax = 250)#maximum frequency 
raw5_p18.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p18.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p18.set_montage(montage)
raw5_p18.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p18, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p18, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p18.plot_sensors(kind='topomap', show_names=True)
raw5_p18.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p18.info)
print(raw5_p18.info)


#plot to show the waves and their source on the head
raw_p18.plot_psd(fmax = 250)
raw5_p18.plot_psd(fmax = 250)



#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica_p18 = mne.preprocessing.ICA()

raw_p18 = raw_p18.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica_p18.fit(raw_p18)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices_p18, eog_scores_p18 = ica_p18.find_bads_eog(raw_p18, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica_p18.exclude = eog_indices_p18#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica_p18.plot_scores(eog_scores_p18)

# plot diagnostics
ica_p18.plot_properties(raw_p18, picks=eog_indices_p18)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica_p18.plot_sources(raw_p18)


#check if raw data has been cleaned 
raw_p18.plot()


#visual presentation ICA components on head
ica_p18.plot_components()

#creating eog epochs
eog_evoked_p18 = create_eog_epochs(raw_p18).average()
eog_evoked_p18.apply_baseline(baseline=(None, -0.2))
eog_evoked_p18.plot_joint()

ica_p18.plot_properties(raw_p18, [0,5])
ica_p18.exclude =[0,5]


#since ica.apply changes raw we are making a copy
reconst_raw_p18 = raw_p18.copy()
ica_p18.apply(reconst_raw_p18)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p18.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p18.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part18_SeqL_ERD_6_B1.fif', overwrite=True)


#--------B5-------#

ica5_p18 = mne.preprocessing.ICA()

raw5_p18 = raw5_p18.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5_p18.fit(raw5_p18)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5_p18, eog_scores5_p18 = ica5_p18.find_bads_eog(raw5_p18, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5.exclude = eog_indices5#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5_p18.plot_scores(eog_scores5_p18)

# plot diagnostics
ica5_p18.plot_properties(raw5_p18, picks=eog_indices5_p18)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5_p18.plot_sources(raw5_p18)

#check if raw data has been cleaned 
raw5_p18.plot()

#visual presentation ICA components on head
ica5_p18.plot_components()

#creating eog epochs
eog_evoked5_p18 = create_eog_epochs(raw5_p18).average()
eog_evoked5_p18.apply_baseline(baseline=(None, -0.2))
eog_evoked5_p18.plot_joint()

ica5_p18.plot_properties(raw5_p18, [0,1,2,4,9])
ica5_p18.exclude =[0,1,2,4,9]

reconst_raw5_p18 = raw5_p18.copy()
ica5.apply(reconst_raw5_p18)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p18.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p18.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part18_SeqL_ERD_6_B5.fif', overwrite=True)




#-----------------------------------------------------------------------------#
#                               PARTICIPANT 25
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_25_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID2\Training\Part25\6key\part25_SeqL_ERD_6_B1.vhdr'
Part_25_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID2\Training\Part25\6key\part25_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p25 = mne.io.read_raw_brainvision(Part_25_1, preload = True) 

#--------B5--------#
raw5_p25 = mne.io.read_raw_brainvision(Part_25_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p25.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p25.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p25.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p25.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p25.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p25.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p25 = mne.io.add_reference_channels(raw_p25, 'TP8')   #reference channel


#--------B5-------#
raw5_p25.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p25.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p25.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p25.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p25.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p25.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p25 = mne.io.add_reference_channels(raw5_p25, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p25.plot_psd(fmax = 250)#maximum frequency 
raw_p25.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p25.plot()

#--------B5-------#
raw5_p25.plot_psd(fmax = 250)#maximum frequency 
raw5_p25.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p25.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p25.set_montage(montage)
raw5_p25.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p25, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p25, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p25.plot_sensors(kind='topomap', show_names=True)
raw5_p25.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p25.info)
print(raw5_p25.info)


#plot to show the waves and their source on the head
raw_p25.plot_psd(fmax = 250)
raw5_p25.plot_psd(fmax = 250)




#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica_p25 = mne.preprocessing.ICA()

raw_p25 = raw_p25.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica_p25.fit(raw_p25)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices_p25, eog_scores_p25 = ica_p25.find_bads_eog(raw_p25, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica_p25.exclude = eog_indices_p25#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica_p25.plot_scores(eog_scores_p25)

# plot diagnostics
ica_p25.plot_properties(raw_p25, picks=eog_indices_p25)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica_p25.plot_sources(raw_p25)


#check if raw data has been cleaned 
raw_p25.plot()


#visual presentation ICA components on head
ica_p25.plot_components()

#creating eog epochs
eog_evoked_p25 = create_eog_epochs(raw_p25).average()
eog_evoked_p25.apply_baseline(baseline=(None, -0.2))
eog_evoked_p25.plot_joint()

ica_p25.plot_properties(raw_p25, [1,3])
ica_p25.exclude =[1,3]


#since ica.apply changes raw we are making a copy
reconst_raw_p25 = raw_p25.copy()
ica_p25.apply(reconst_raw_p25)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p25.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p25.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part25_SeqL_ERD_6_B1.fif', overwrite=True)



#--------B5-------#

ica5_p25 = mne.preprocessing.ICA()

raw5_p25 = raw5_p25.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5_p25.fit(raw5_p25)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5_p25, eog_scores5_p25 = ica5_p25.find_bads_eog(raw5_p25, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5_p25.exclude = eog_indices5_p25#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5_p25.plot_scores(eog_scores5_p25)

# plot diagnostics
ica5_p25.plot_properties(raw5_p25, picks=eog_indices5_p25)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5_p25.plot_sources(raw5_p25)

#check if raw data has been cleaned 
raw5_p25.plot()

#visual presentation ICA components on head
ica5_p25.plot_components()

#creating eog epochs
eog_evoked5_p25 = create_eog_epochs(raw5_p25).average()
eog_evoked5_p25.apply_baseline(baseline=(None, -0.2))
eog_evoked5_p25.plot_joint()

ica5_p25.plot_properties(raw5_p25, [1])
ica5_p25.exclude =[2]

reconst_raw5_p25 = raw5_p25.copy()
ica5_p25.apply(reconst_raw5_p25)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p25.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p25.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part25_SeqL_ERD_6_B5.fif', overwrite=True)





#-----------------------------------------------------------------------------#
#                                   ID3
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
#                               PARTICIPANT 3
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_3_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID3\Training\Part3\6 key\part3_SeqL_ERD_6_B1.vhdr'
Part_3_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID3\Training\Part3\6 key\part3_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p3 = mne.io.read_raw_brainvision(Part_3_1, preload = True) 

#--------B5--------#
raw5_p3 = mne.io.read_raw_brainvision(Part_3_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p3.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p3.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p3.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p3.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p3.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p3.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p3 = mne.io.add_reference_channels(raw_p3, 'TP8')   #reference channel

#--------B5-------#
raw5_p3.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p3.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p3.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p3.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p3.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p3.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p3 = mne.io.add_reference_channels(raw5_p3, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p3.plot_psd(fmax = 250)#maximum frequency 
raw_p3.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p3.plot()


#--------B5-------#
raw5_p3.plot_psd(fmax = 250)#maximum frequency 
raw5_p3.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p3.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p3.set_montage(montage)
raw5_p3.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p3, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p3, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p3.plot_sensors(kind='topomap', show_names=True)
raw5_p3.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p3.info)
print(raw5_p3.info)


#plot to show the waves and their source on the head
raw_p3.plot_psd(fmax = 250)
raw5_p3.plot_psd(fmax = 250)


#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica_p3 = mne.preprocessing.ICA()

raw_p3 = raw_p3.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica_p3.fit(raw_p3)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices_p3, eog_scores_p3 = ica_p3.find_bads_eog(raw_p3, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica_p3.exclude = eog_indices_p3#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica_p3.plot_scores(eog_scores_p3)

# plot diagnostics
ica_p3.plot_properties(raw_p3, picks=eog_indices_p3)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica_p3.plot_sources(raw_p3)


#check if raw data has been cleaned 
raw_p3.plot()


#visual presentation ICA components on head
ica_p3.plot_components()

#creating eog epochs
eog_evoked_p3 = create_eog_epochs(raw_p3).average()
eog_evoked_p3.apply_baseline(baseline=(None, -0.2))
eog_evoked_p3.plot_joint()

ica_p3.plot_properties(raw_p3, [0,1])
ica_p3.exclude =[0, 1]


#since ica.apply changes raw we are making a copy
reconst_raw_p3 = raw_p3.copy()
ica_p3.apply(reconst_raw_p3)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p3.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p3.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part3_SeqL_ERD_6_B1.fif', overwrite=True)




#--------B5-------#

ica5_p3 = mne.preprocessing.ICA()

raw5_p3 = raw5_p3.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5_p3.fit(raw5_p3)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5_p3, eog_scores5_p3 = ica5_p3.find_bads_eog(raw5_p3, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5_p3.exclude = eog_indices5_p3#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5_p3.plot_scores(eog_scores5_p3)

# plot diagnostics
ica5_p3.plot_properties(raw5_p3, picks=eog_indices5_p3)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5_p3.plot_sources(raw5_p3)

#check if raw data has been cleaned 
raw5_p3.plot()

#visual presentation ICA components on head
ica5_p3.plot_components()

#creating eog epochs
eog_evoked5_p3 = create_eog_epochs(raw5_p3).average()
eog_evoked5_p3.apply_baseline(baseline=(None, -0.2))
eog_evoked5_p3.plot_joint()

ica5_p3.plot_properties(raw5_p3, [0,1])

ica5_p3.exclude =[0,1]

reconst_raw5_p3 = raw5_p3.copy()
ica5_p3.apply(reconst_raw5_p3)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p3.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p3.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part3_SeqL_ERD_6_B5.fif', overwrite=True)



#-----------------------------------------------------------------------------#
#                               PARTICIPANT 11
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_11_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID3\Training\Part11\6key\part11_SeqL_ERD_6_B1.vhdr'
Part_11_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID3\Training\Part11\6key\part11_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p11 = mne.io.read_raw_brainvision(Part_11_1, preload = True) 

#--------B5--------#
raw5_p11 = mne.io.read_raw_brainvision(Part_11_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p11.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p11.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p11.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p11.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p11.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p11.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p11 = mne.io.add_reference_channels(raw_p11, 'TP8')   #reference channel


#--------B5-------#
raw5_p11.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p11.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p11.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p11.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p11.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p11.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p11 = mne.io.add_reference_channels(raw5_p11, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p11.plot_psd(fmax = 250)#maximum frequency 
raw_p11.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p11.plot()


#--------B5-------#
raw5_p11.plot_psd(fmax = 250)#maximum frequency 
raw5_p11.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p11.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p11.set_montage(montage)
raw5_p11.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p11, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p11, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p11.plot_sensors(kind='topomap', show_names=True)
raw5_p11.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p11.info)
print(raw5_p11.info)


#plot to show the waves and their source on the head
raw_p11.plot_psd(fmax = 250)
raw5_p11.plot_psd(fmax = 250)



#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica_p11 = mne.preprocessing.ICA()

raw_p11 = raw_p11.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica_p11.fit(raw_p11)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices_p11, eog_scores_p11 = ica_p11.find_bads_eog(raw_p11, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica_p11.exclude = eog_indices_p11#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica_p11.plot_scores(eog_scores_p11)

# plot diagnostics
ica_p11.plot_properties(raw_p11, picks=eog_indices_p11)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica_p11.plot_sources(raw_p11)


#check if raw data has been cleaned 
raw_p11.plot()


#visual presentation ICA components on head
ica_p11.plot_components()

#creating eog epochs
eog_evoked_p11 = create_eog_epochs(raw_p11).average()
eog_evoked_p11.apply_baseline(baseline=(None, -0.2))
eog_evoked_p11.plot_joint()

ica_p11.plot_properties(raw_p11, [0,1,2])
ica_p11.exclude =[0,1,2]


#since ica.apply changes raw we are making a copy
reconst_raw_p11 = raw_p11.copy()
ica_p11.apply(reconst_raw_p11)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p11.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p11.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part11_SeqL_ERD_6_B1.fif', overwrite=True)




#--------B5-------#

ica5_p11 = mne.preprocessing.ICA()

raw5_p11 = raw5_p11.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5_p11.fit(raw5_p11)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5_p11, eog_scores5_p11 = ica5_p11.find_bads_eog(raw5_p11, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5_p11.exclude = eog_indices5_p11#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5_p11.plot_scores(eog_scores5_p11)

# plot diagnostics
ica5_p11.plot_properties(raw5_p11, picks=eog_indices5_p11)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5_p11.plot_sources(raw5_p11)

#check if raw data has been cleaned 
raw5_p11.plot()

#visual presentation ICA components on head
ica5_p11.plot_components()

#creating eog epochs
eog_evoked5_p11 = create_eog_epochs(raw5_p11).average()
eog_evoked5_p11.apply_baseline(baseline=(None, -0.2))
eog_evoked5_p11.plot_joint()

ica5_p11.plot_properties(raw5_p11, [0,1])
ica5_p11.exclude =[0,1]

reconst_raw5_p11 = raw5_p11.copy()
ica5_p11.apply(reconst_raw5_p11)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p11.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p11.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part11_SeqL_ERD_6_B5.fif', overwrite=True)





#-----------------------------------------------------------------------------#
#                               PARTICIPANT 19
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_19_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID3\Training\Part19\6key\part19_SeqL_ERD_6_B1.vhdr'
Part_19_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID3\Training\Part19\6key\part19_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p19 = mne.io.read_raw_brainvision(Part_19_1, preload = True) 

#--------B5--------#
raw5_p19 = mne.io.read_raw_brainvision(Part_19_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p19.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p19.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p19.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p19.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p19.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p19.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p19 = mne.io.add_reference_channels(raw_p19, 'TP8')   #reference channel


#--------B5-------#
raw5_p19.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p19.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p19.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p19.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p19.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p19.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p19 = mne.io.add_reference_channels(raw5_p19, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p19.plot_psd(fmax = 250)#maximum frequency 
raw_p19.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p19.plot()

#--------B5-------#
raw5_p19.plot_psd(fmax = 250)#maximum frequency 
raw5_p19.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p19.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p19.set_montage(montage)
raw5_p19.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p19, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p19, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p19.plot_sensors(kind='topomap', show_names=True)
raw5_p19.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p19.info)
print(raw5_p19.info)


#plot to show the waves and their source on the head
raw_p19.plot_psd(fmax = 250)
raw5_p19.plot_psd(fmax = 250)



#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica = mne.preprocessing.ICA()

raw_p19 = raw_p19.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica.fit(raw_p19)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices, eog_scores = ica.find_bads_eog(raw_p19, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica.exclude = eog_indices#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica.plot_scores(eog_scores)

# plot diagnostics
ica.plot_properties(raw_p19, picks=eog_indices)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica.plot_sources(raw_p19)


#check if raw data has been cleaned 
raw_p19.plot()


#visual presentation ICA components on head
ica.plot_components()

#creating eog epochs
eog_evoked = create_eog_epochs(raw_p19).average()
eog_evoked.apply_baseline(baseline=(None, -0.2))
eog_evoked.plot_joint()

ica.plot_properties(raw_p19, [1,2])
ica.exclude =[1,2]


#since ica.apply changes raw we are making a copy
reconst_raw_p19 = raw_p19.copy()
ica.apply(reconst_raw_p19)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p19.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p19.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part19_SeqL_ERD_6_B1.fif', overwrite=True)




#--------B5-------#

ica5 = mne.preprocessing.ICA()

raw5_p19 = raw5_p19.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5.fit(raw5_p19)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5, eog_scores5 = ica5.find_bads_eog(raw5_p19, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5.exclude = eog_indices5#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5.plot_scores(eog_scores5)

# plot diagnostics
ica5.plot_properties(raw5_p19, picks=eog_indices5)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5.plot_sources(raw5_p19)

#check if raw data has been cleaned 
raw5_p19.plot()

#visual presentation ICA components on head
ica5.plot_components()

#creating eog epochs
eog_evoked5 = create_eog_epochs(raw5_p19).average()
eog_evoked5.apply_baseline(baseline=(None, -0.2))
eog_evoked5.plot_joint()

ica5.plot_properties(raw5_p19, [0,1])
ica5.exclude =[0,1]

reconst_raw5_p19 = raw5_p19.copy()
ica5.apply(reconst_raw5_p19)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p19.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p19.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part19_SeqL_ERD_6_B5.fif', overwrite=True)




#-----------------------------------------------------------------------------#
#                               PARTICIPANT 28
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_28_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID3\Training\Part28\6key\part29_SeqL_ERD_6_B1.vhdr'
Part_28_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID3\Training\Part28\6key\part29_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p28 = mne.io.read_raw_brainvision(Part_28_1, preload = True) 

#--------B5--------#
raw5_p28 = mne.io.read_raw_brainvision(Part_28_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p28.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p28.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p28.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p28.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p28.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p28.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p28 = mne.io.add_reference_channels(raw_p28, 'TP8')   #reference channel

#--------B5-------#
raw5_p28.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p28.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p28.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p28.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p28.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p28.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p28 = mne.io.add_reference_channels(raw5_p28, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p28.plot_psd(fmax = 250)#maximum frequency 
raw_p28.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p28.plot()

#--------B5-------#
raw5_p28.plot_psd(fmax = 250)#maximum frequency 
raw5_p28.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p28.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p28.set_montage(montage)
raw5_p28.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p28, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p28, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p28.plot_sensors(kind='topomap', show_names=True)
raw5_p28.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p28.info)
print(raw5_p28.info)


#plot to show the waves and their source on the head
raw_p28.plot_psd(fmax = 250)
raw5_p28.plot_psd(fmax = 250)



#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica_p28 = mne.preprocessing.ICA()

raw_p28 = raw_p28.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica_p28.fit(raw_p28)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices_p28, eog_scores_p28 = ica_p28.find_bads_eog(raw_p28, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica_p28.exclude = eog_indices_p28#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica_p28.plot_scores(eog_scores_p28)

# plot diagnostics
ica_p28.plot_properties(raw_p28, picks=eog_indices_p28)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica_p28.plot_sources(raw_p28)


#check if raw data has been cleaned 
raw_p28.plot()


#visual presentation ICA components on head
ica_p28.plot_components()

#creating eog epochs
eog_evoked_p28 = create_eog_epochs(raw_p28).average()
eog_evoked_p28.apply_baseline(baseline=(None, -0.2))
eog_evoked_p28.plot_joint()

ica_p28.plot_properties(raw_p28, [0,1])
ica_p28.exclude =[0,1]


#since ica.apply changes raw we are making a copy
reconst_raw_p28 = raw_p28.copy()
ica_p28.apply(reconst_raw_p28)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p28.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p28.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part28_SeqL_ERD_6_B1.fif', overwrite=True)




#--------B5-------#

ica5_p28 = mne.preprocessing.ICA()

raw5_p28 = raw5_p28.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5_p28.fit(raw5_p28)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5_p28, eog_scores5_p28 = ica5_p28.find_bads_eog(raw5_p28, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5_p28.exclude = eog_indices5_p28#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5_p28.plot_scores(eog_scores5_p28)

# plot diagnostics
ica5_p28.plot_properties(raw5_p28, picks=eog_indices5_p28)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5_p28.plot_sources(raw5_p28)

#check if raw data has been cleaned 
raw5_p28.plot()

#visual presentation ICA components on head
ica5_p28.plot_components()

#creating eog epochs
eog_evoked5_p28 = create_eog_epochs(raw5_p28).average()
eog_evoked5_p28.apply_baseline(baseline=(None, -0.2))
eog_evoked5_p28.plot_joint()

ica5_p28.plot_properties(raw5_p28, [0,1])
ica5_p28.exclude =[0,1]

reconst_raw5_p28 = raw5_p28.copy()
ica5_p28.apply(reconst_raw5_p28)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p28.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p28.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part28_SeqL_ERD_6_B5.fif', overwrite=True)




#-----------------------------------------------------------------------------#
#                               PARTICIPANT 4
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_4_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID4\Training\Part4\6key\part4_SeqL_ERD_6_B1.vhdr'
Part_4_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID4\Training\Part4\6key\part4_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p4 = mne.io.read_raw_brainvision(Part_4_1, preload = True) 

#--------B5--------#
raw5_p4 = mne.io.read_raw_brainvision(Part_4_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p4.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p4.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p4.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p4.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p4.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p4.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p4 = mne.io.add_reference_channels(raw_p4, 'TP8')   #reference channel


#--------B5-------#
raw5_p4.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p4.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p4.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p4.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p4.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p4.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p4 = mne.io.add_reference_channels(raw5_p4, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p4.plot_psd(fmax = 250)#maximum frequency 
raw_p4.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p4.plot()


#--------B5-------#
raw5_p4.plot_psd(fmax = 250)#maximum frequency 
raw5_p4.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p4.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p4.set_montage(montage)
raw5_p4.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p4, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p4, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p4.plot_sensors(kind='topomap', show_names=True)
raw5_p4.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p4.info)
print(raw5_p4.info)


#plot to show the waves and their source on the head
raw_p4.plot_psd(fmax = 250)
raw5_p4.plot_psd(fmax = 250)




#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica_p4 = mne.preprocessing.ICA()

raw_p4 = raw_p4.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica_p4.fit(raw_p4)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices_p4, eog_scores_p4 = ica_p4.find_bads_eog(raw_p4, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica_p4.exclude = eog_indices_p4#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica_p4.plot_scores(eog_scores_p4)

# plot diagnostics
ica_p4.plot_properties(raw_p4, picks=eog_indices_p4)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica_p4.plot_sources(raw_p4)


#check if raw data has been cleaned 
raw_p4.plot()


#visual presentation ICA components on head
ica_p4.plot_components()

#creating eog epochs
eog_evoked_p4 = create_eog_epochs(raw_p4).average()
eog_evoked_p4.apply_baseline(baseline=(None, -0.2))
eog_evoked_p4.plot_joint()

ica_p4.plot_properties(raw_p4, [0,2])
ica_p4.exclude =[0,2]


#since ica.apply changes raw we are making a copy
reconst_raw_p4 = raw_p4.copy()
ica_p4.apply(reconst_raw_p4)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p4.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p4.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part4_SeqL_ERD_6_B1.fif', overwrite=True)




#--------B5-------#

ica5_p4 = mne.preprocessing.ICA()

raw5_p4 = raw5_p4.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5_p4.fit(raw5_p4)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5_p4, eog_scores5_p4 = ica5_p4.find_bads_eog(raw5_p4, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5_p4.exclude = eog_indices5_p4#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5_p4.plot_scores(eog_scores5_p4)

# plot diagnostics
ica5_p4.plot_properties(raw5_p4, picks=eog_indices5_p4)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5_p4.plot_sources(raw5_p4)

#check if raw data has been cleaned 
raw5_p4.plot()

#visual presentation ICA components on head
ica5_p4.plot_components()

#creating eog epochs
eog_evoked5_p4 = create_eog_epochs(raw5_p4).average()
eog_evoked5_p4.apply_baseline(baseline=(None, -0.2))
eog_evoked5_p4.plot_joint()

ica5_p4.plot_properties(raw5_p4, [0])
ica5_p4.exclude =[0]

reconst_raw5_p4 = raw5_p4.copy()
ica5_p4.apply(reconst_raw5_p4)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p4.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p4.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part4_SeqL_ERD_6_B5.fif', overwrite=True)




#-----------------------------------------------------------------------------#
#                               PARTICIPANT 12
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_12_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID4\Training\Part12\6key\part12_SeqL_ERD_6_B1.vhdr'
Part_12_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID4\Training\Part12\6key\part12_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p12 = mne.io.read_raw_brainvision(Part_12_1, preload = True) 

#--------B5--------#
raw5_p12 = mne.io.read_raw_brainvision(Part_12_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p12.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p12.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p12.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p12.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p12.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p12.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p12 = mne.io.add_reference_channels(raw_p12, 'TP8')   #reference channel

#--------B5-------#
raw5_p12.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p12.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p12.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p12.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p12.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p12.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p12 = mne.io.add_reference_channels(raw5_p12, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p12.plot_psd(fmax = 250)#maximum frequency 
raw_p12.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p12.plot()

#--------B5-------#
raw5_p12.plot_psd(fmax = 250)#maximum frequency 
raw5_p12.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p12.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p12.set_montage(montage)
raw5_p12.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p12, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p12, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p12.plot_sensors(kind='topomap', show_names=True)
raw5_p12.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p12.info)
print(raw5_p12.info)


#plot to show the waves and their source on the head
raw_p12.plot_psd(fmax = 250)
raw5_p12.plot_psd(fmax = 250)



#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica_p12 = mne.preprocessing.ICA()

raw_p12 = raw_p12.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica_p12.fit(raw_p12)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices_p12, eog_scores_p12 = ica_p12.find_bads_eog(raw_p12, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica_p12.exclude = eog_indices_p12#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica_p12.plot_scores(eog_scores_p12)

# plot diagnostics
ica_p12.plot_properties(raw_p12, picks=eog_indices_p12)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica_p12.plot_sources(raw_p12)


#check if raw data has been cleaned 
raw_p12.plot()


#visual presentation ICA components on head
ica_p12.plot_components()

#creating eog epochs
eog_evoked_p12 = create_eog_epochs(raw_p12).average()
eog_evoked_p12.apply_baseline(baseline=(None, -0.2))
eog_evoked_p12.plot_joint()

ica_p12.plot_properties(raw_p12, [1,2])
ica_p12.exclude =[1,2]


#since ica.apply changes raw we are making a copy
reconst_raw_p12 = raw_p12.copy()
ica_p12.apply(reconst_raw_p12)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p12.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p12.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part12_SeqL_ERD_6_B1.fif', overwrite=True)



#--------B5-------#

ica5_p12 = mne.preprocessing.ICA()

raw5_p12 = raw5_p12.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5_p12.fit(raw5_p12)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5_p12, eog_scores5_p12 = ica5_p12.find_bads_eog(raw5_p12, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5_p12.exclude = eog_indices5_p12#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5_p12.plot_scores(eog_scores5_p12)

# plot diagnostics
ica5_p12.plot_properties(raw5_p12, picks=eog_indices5_p12)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5_p12.plot_sources(raw5_p12)

#check if raw data has been cleaned 
raw5_p12.plot()

#visual presentation ICA components on head
ica5_p12.plot_components()

#creating eog epochs
eog_evoked5_p12 = create_eog_epochs(raw5_p12).average()
eog_evoked5_p12.apply_baseline(baseline=(None, -0.2))
eog_evoked5_p12.plot_joint()

ica5_p12.plot_properties(raw5_p12, [0,1])

ica5_p12.exclude =[0,1]

reconst_raw5_p12 = raw5_p12.copy()
ica5_p12.apply(reconst_raw5_p12)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p12.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p12.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part12_SeqL_ERD_6_B5.fif', overwrite=True)




#-----------------------------------------------------------------------------#
#                               PARTICIPANT 26
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_26_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID4\Training\Part26\6key\part26_SeqL_ERD_6_B1.vhdr'
Part_26_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID4\Training\Part26\6key\part26_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p26 = mne.io.read_raw_brainvision(Part_26_1, preload = True) 

#--------B5--------#
raw5_p26 = mne.io.read_raw_brainvision(Part_26_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p26.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p26.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p26.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p26.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p26.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p26.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p26 = mne.io.add_reference_channels(raw_p26, 'TP8')   #reference channel

#--------B5-------#
raw5_p26.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p26.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p26.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p26.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p26.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p26.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p26 = mne.io.add_reference_channels(raw5_p26, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p26.plot_psd(fmax = 250)#maximum frequency 
raw_p26.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p26.plot()

#--------B5-------#
raw5_p26.plot_psd(fmax = 250)#maximum frequency 
raw5_p26.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p26.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p26.set_montage(montage)
raw5_p26.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p26, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p26, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p26.plot_sensors(kind='topomap', show_names=True)
raw5_p26.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p26.info)
print(raw5_p26.info)


#plot to show the waves and their source on the head
raw_p26.plot_psd(fmax = 250)
raw5_p26.plot_psd(fmax = 250)




#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica_p26 = mne.preprocessing.ICA()

raw_p26 = raw_p26.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica_p26.fit(raw_p26)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices_p26, eog_scores_p26 = ica_p26.find_bads_eog(raw_p26, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica_p26.exclude = eog_indices_p26#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica_p26.plot_scores(eog_scores_p26)

# plot diagnostics
ica_p26.plot_properties(raw_p26, picks=eog_indices_p26)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica_p26.plot_sources(raw_p26)


#check if raw data has been cleaned 
raw_p4.plot()


#visual presentation ICA components on head
ica_p26.plot_components()

#creating eog epochs
eog_evoked_p26 = create_eog_epochs(raw_p26).average()
eog_evoked_p26.apply_baseline(baseline=(None, -0.2))
eog_evoked_p26.plot_joint()

ica_p26.plot_properties(raw_p26, [0,3])
ica_p26.exclude =[0,3]


#since ica.apply changes raw we are making a copy
reconst_raw_p26 = raw_p26.copy()
ica_p26.apply(reconst_raw_p26)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p26.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p26.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part26_SeqL_ERD_6_B1.fif', overwrite=True)



#--------B5-------#

ica5_p26 = mne.preprocessing.ICA()

raw5_p26 = raw5_p26.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5_p26.fit(raw5_p26)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5_p26, eog_scores5_p26 = ica5_p26.find_bads_eog(raw5_p26, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5_p26.exclude = eog_indices5_p26#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5_p26.plot_scores(eog_scores5_p26)

# plot diagnostics
ica5_p26.plot_properties(raw5_p26, picks=eog_indices5_p26)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5_p26.plot_sources(raw5_p26)

#check if raw data has been cleaned 
raw5_p26.plot()

#visual presentation ICA components on head
ica5_p26.plot_components()

#creating eog epochs
eog_evoked5_p26 = create_eog_epochs(raw5_p26).average()
eog_evoked5_p26.apply_baseline(baseline=(None, -0.2))
eog_evoked5_p26.plot_joint()

ica5_p26.plot_properties(raw5_p26, [0,2])
ica5_p26.exclude =[0,2]

reconst_raw5_p26 = raw5_p26.copy()
ica5_p26.apply(reconst_raw5_p26)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p26.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p26.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part26_SeqL_ERD_6_B5.fif', overwrite=True)




#-----------------------------------------------------------------------------#
#                               PARTICIPANT 30
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_30_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID4\Training\Part30\6key\part31_SeqL_6_B1.vhdr'
Part_30_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID4\Training\Part30\6key\part31_SeqL_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p30 = mne.io.read_raw_brainvision(Part_30_1, preload = True) 

#--------B5--------#
raw5_p30 = mne.io.read_raw_brainvision(Part_30_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p30.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p30.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p30.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p30.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p30.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p30.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p30 = mne.io.add_reference_channels(raw_p30, 'TP8')   #reference channel

#--------B5-------#
raw5_p30.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p30.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p30.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p30.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p30.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p30.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p30 = mne.io.add_reference_channels(raw5_p30, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p30.plot_psd(fmax = 250)#maximum frequency 
raw_p30.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p30.plot()


#--------B5-------#
raw5_p30.plot_psd(fmax = 250)#maximum frequency 
raw5_p30.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p30.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p30.set_montage(montage)
raw5_p30.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p30, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p30, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p30.plot_sensors(kind='topomap', show_names=True)
raw5_p30.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p30.info)
print(raw5_p30.info)


#plot to show the waves and their source on the head
raw_p30.plot_psd(fmax = 250)
raw5_p30.plot_psd(fmax = 250)



#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica = mne.preprocessing.ICA()

raw_p30 = raw_p30.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica.fit(raw_p30)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices, eog_scores = ica.find_bads_eog(raw_p30, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica.exclude = eog_indices#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica.plot_scores(eog_scores)

# plot diagnostics
ica.plot_properties(raw_p30, picks=eog_indices)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica.plot_sources(raw_p30)


#check if raw data has been cleaned 
raw_p30.plot()


#visual presentation ICA components on head
ica.plot_components()

#creating eog epochs
eog_evoked = create_eog_epochs(raw_p30).average()
eog_evoked.apply_baseline(baseline=(None, -0.2))
eog_evoked.plot_joint()

ica.plot_properties(raw_p30, [0,1,3])
ica.exclude =[0,1,3]


#since ica.apply changes raw we are making a copy
reconst_raw_p30 = raw_p30.copy()
ica.apply(reconst_raw_p30)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p30.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p30.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part30_SeqL_ERD_6_B1.fif', overwrite=True)



#--------B5-------#

ica5 = mne.preprocessing.ICA()

raw5_p30 = raw5_p30.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5.fit(raw5_p30)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5, eog_scores5 = ica5.find_bads_eog(raw5_p30, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5.exclude = eog_indices5#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5.plot_scores(eog_scores5)

# plot diagnostics
ica5.plot_properties(raw5_p30, picks=eog_indices5)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5.plot_sources(raw5_p30)

#check if raw data has been cleaned 
raw5_p30.plot()

#visual presentation ICA components on head
ica5.plot_components()

#creating eog epochs
eog_evoked5 = create_eog_epochs(raw5_p30).average()
eog_evoked5.apply_baseline(baseline=(None, -0.2))
eog_evoked5.plot_joint()

ica5.plot_properties(raw5_p30, [0,1,2])
ica5.exclude =[0,1,2]

reconst_raw5_p30 = raw5_p30.copy()
ica5.apply(reconst_raw5_p30)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p30.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p30.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part30_SeqL_ERD_6_B5.fif', overwrite=True)



#-----------------------------------------------------------------------------#
#                               PARTICIPANT 5
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_5_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID5\Training\Part5\6key\part5_SeqL_ERD_6_B1.vhdr'
Part_5_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID5\Training\Part5\6key\part5_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p5 = mne.io.read_raw_brainvision(Part_5_1, preload = True) 

#--------B5--------#
raw5_p5 = mne.io.read_raw_brainvision(Part_5_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p5.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p5.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p5.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p5.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p5.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p5.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p5 = mne.io.add_reference_channels(raw_p5, 'TP8')   #reference channel


#--------B5-------#
raw5_p5.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p5.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p5.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p5.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p5.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p5.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p5 = mne.io.add_reference_channels(raw5_p5, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p5.plot_psd(fmax = 250)#maximum frequency 
raw_p5.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p5.plot()

#--------B5-------#
raw5_p5.plot_psd(fmax = 250)#maximum frequency 
raw5_p5.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p5.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p5.set_montage(montage)
raw5_p5.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p5, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p5, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p5.plot_sensors(kind='topomap', show_names=True)
raw5_p5.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p5.info)
print(raw5_p5.info)


#plot to show the waves and their source on the head
raw_p5.plot_psd(fmax = 250)
raw5_p5.plot_psd(fmax = 250)



#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica_p5 = mne.preprocessing.ICA()

raw_p5 = raw_p5.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica_p5.fit(raw_p5)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices_p5, eog_scores_p5 = ica_p5.find_bads_eog(raw_p5, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica_p5.exclude = eog_indices_p5#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica_p5.plot_scores(eog_scores_p5)

# plot diagnostics
ica_p5.plot_properties(raw_p5, picks=eog_indices_p5)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica_p5.plot_sources(raw_p5)


#check if raw data has been cleaned 
raw_p5.plot()


#visual presentation ICA components on head
ica_p5.plot_components()

#creating eog epochs
eog_evoked_p5 = create_eog_epochs(raw_p5).average()
eog_evoked_p5.apply_baseline(baseline=(None, -0.2))
eog_evoked_p5.plot_joint()

ica_p5.plot_properties(raw_p5, [0,3])
ica_p5.exclude =[0,3]


#since ica.apply changes raw we are making a copy
reconst_raw_p5 = raw_p5.copy()
ica_p5.apply(reconst_raw_p5)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p5.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p5.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part5_SeqL_ERD_6_B1.fif', overwrite=True)




#--------B5-------#

ica5_p5 = mne.preprocessing.ICA()

raw5_p5 = raw5_p5.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5_p5.fit(raw5_p5)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5_p5, eog_scores5_p5 = ica5_p5.find_bads_eog(raw5_p5, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5.exclude = eog_indices5#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5_p5.plot_scores(eog_scores5_p5)

# plot diagnostics
ica5_p5.plot_properties(raw5_p5, picks=eog_indices5_p5)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5_p5.plot_sources(raw5_p5)

#check if raw data has been cleaned 
raw5_p5.plot()

#visual presentation ICA components on head
ica5_p5.plot_components()

#creating eog epochs
eog_evoked5_p5 = create_eog_epochs(raw5_p5).average()
eog_evoked5_p5.apply_baseline(baseline=(None, -0.2))
eog_evoked5_p5.plot_joint()

ica5_p5.plot_properties(raw5_p5, [2])
ica5_p5.exclude =[2]

reconst_raw5_p5 = raw5_p5.copy()
ica5.apply(reconst_raw5_p5)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p5.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p5.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part5_SeqL_ERD_6_B5.fif', overwrite=True)




#-----------------------------------------------------------------------------#
#                               PARTICIPANT 15
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_15_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID5\Training\Part15\6 key\part15_SeqL_ERD_6_B1.vhdr'
Part_15_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID5\Training\Part15\6 key\part15_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p15 = mne.io.read_raw_brainvision(Part_15_1, preload = True) 

#--------B5--------#
raw5_p15 = mne.io.read_raw_brainvision(Part_15_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p15.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p15.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p15.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p15.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p15.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p15.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p15 = mne.io.add_reference_channels(raw_p15, 'TP8')   #reference channel

#--------B5-------#
raw5_p15.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p15.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p15.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p15.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p15.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p15.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p15 = mne.io.add_reference_channels(raw5_p15, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p15.plot_psd(fmax = 250)#maximum frequency 
raw_p15.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p15.plot()

#--------B5-------#
raw5_p15.plot_psd(fmax = 250)#maximum frequency 
raw5_p15.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p15.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p15.set_montage(montage)
raw5_p15.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p15, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p15, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p15.plot_sensors(kind='topomap', show_names=True)
raw5_p15.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p15.info)
print(raw5_p15.info)


#plot to show the waves and their source on the head
raw_p15.plot_psd(fmax = 250)
raw5_p15.plot_psd(fmax = 250)



#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica_p15 = mne.preprocessing.ICA()

raw_p15 = raw_p15.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica_p15.fit(raw_p15)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices_p15, eog_scores_p15 = ica_p15.find_bads_eog(raw_p15, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica_p15.exclude = eog_indices_p15#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica_p15.plot_scores(eog_scores_p15)

# plot diagnostics
ica_p15.plot_properties(raw_p15, picks=eog_indices_p15)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica_p15.plot_sources(raw_p15)


#check if raw data has been cleaned 
raw_p15.plot()


#visual presentation ICA components on head
ica_p15.plot_components()

#creating eog epochs
eog_evoked_p15 = create_eog_epochs(raw_p15).average()
eog_evoked_p15.apply_baseline(baseline=(None, -0.2))
eog_evoked_p15.plot_joint()

ica_p15.plot_properties(raw_p15, [0,1,2])
ica_p15.exclude =[0,1,2]


#since ica.apply changes raw we are making a copy
reconst_raw_p15 = raw_p15.copy()
ica_p15.apply(reconst_raw_p15)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p15.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p15.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part15_SeqL_ERD_6_B1.fif', overwrite=True)



#--------B5-------#

ica5_p15 = mne.preprocessing.ICA()

raw5_p15 = raw5_p15.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5_p15.fit(raw5_p15)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5_p15, eog_scores5_p15 = ica5_p15.find_bads_eog(raw5_p15, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5_p15.exclude = eog_indices5_p15#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5_p15.plot_scores(eog_scores5_p15)

# plot diagnostics
ica5_p15.plot_properties(raw5_p15, picks=eog_indices5_p15)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5_p15.plot_sources(raw5_p15)

#check if raw data has been cleaned 
raw5_p15.plot()

#visual presentation ICA components on head
ica5_p15.plot_components()

#creating eog epochs
eog_evoked5_p15 = create_eog_epochs(raw5_p15).average()
eog_evoked5_p15.apply_baseline(baseline=(None, -0.2))
eog_evoked5_p15.plot_joint()

ica5_p15.plot_properties(raw5_p15, [0,1])
ica5_p15.exclude =[0,1]

reconst_raw5_p15 = raw5_p15.copy()
ica5_p15.apply(reconst_raw5_p15)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p15.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p15.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part15_SeqL_ERD_6_B5.fif', overwrite=True)





#-----------------------------------------------------------------------------#
#                               PARTICIPANT 20
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_20_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID5\Training\Part20\6key\part20_SeqL_ERD_6_B1.vhdr'
Part_20_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID5\Training\Part20\6key\part20_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p20 = mne.io.read_raw_brainvision(Part_20_1, preload = True) 

#--------B5--------#
raw5_p20 = mne.io.read_raw_brainvision(Part_20_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p20.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p20.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p20.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p20.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p20.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p20.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p20 = mne.io.add_reference_channels(raw_p20, 'TP8')   #reference channel


#--------B5-------#
raw5_p20.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p20.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p20.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p20.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p20.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p20.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p20 = mne.io.add_reference_channels(raw5_p20, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p20.plot_psd(fmax = 250)#maximum frequency 
raw_p20.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p20.plot()

#--------B5-------#
raw5_p20.plot_psd(fmax = 250)#maximum frequency 
raw5_p20.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p20.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p20.set_montage(montage)
raw5_p20.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p20, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p20, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p20.plot_sensors(kind='topomap', show_names=True)
raw5_p20.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p20.info)
print(raw5_p20.info)


#plot to show the waves and their source on the head
raw_p20.plot_psd(fmax = 250)
raw5_p20.plot_psd(fmax = 250)



#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica_p20 = mne.preprocessing.ICA()

raw_p20 = raw_p20.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica_p20.fit(raw_p20)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices_p20, eog_scores_p20 = ica_p20.find_bads_eog(raw_p20, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica_p20.exclude = eog_indices_p20#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica_p20.plot_scores(eog_scores_p20)

# plot diagnostics
ica_p20.plot_properties(raw_p20, picks=eog_indices_p20)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica_p20.plot_sources(raw_p20)


#check if raw data has been cleaned 
raw_p20.plot()


#visual presentation ICA components on head
ica_p20.plot_components()

#creating eog epochs
eog_evoked_p20 = create_eog_epochs(raw_p20).average()
eog_evoked_p20.apply_baseline(baseline=(None, -0.2))
eog_evoked_p20.plot_joint()

ica_p20.plot_properties(raw_p20, [0,1])
ica_p20.exclude =[0, 1]


#since ica.apply changes raw we are making a copy
reconst_raw_p20 = raw_p20.copy()
ica_p20.apply(reconst_raw_p20)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p20.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p20.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part20_SeqL_ERD_6_B1.fif', overwrite=True)




#--------B5-------#

ica5_p20 = mne.preprocessing.ICA()

raw5_p20 = raw5_p20.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5_p20.fit(raw5_p20)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5_p20, eog_scores5_p20 = ica5_p20.find_bads_eog(raw5_p20, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5_p20.exclude = eog_indices5_p20#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5_p20.plot_scores(eog_scores5_p20)

# plot diagnostics
ica5_p20.plot_properties(raw5_p20, picks=eog_indices5_p20)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5_p20.plot_sources(raw5_p20)

#check if raw data has been cleaned 
raw5_p20.plot()

#visual presentation ICA components on head
ica5_p20.plot_components()

#creating eog epochs
eog_evoked5_p20 = create_eog_epochs(raw5_p20).average()
eog_evoked5_p20.apply_baseline(baseline=(None, -0.2))
eog_evoked5_p20.plot_joint()

ica5_p20.plot_properties(raw5_p20, [0,1])

ica5_p20.exclude =[0,1]

reconst_raw5_p20 = raw5_p20.copy()
ica5_p20.apply(reconst_raw5_p20)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p20.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p20.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part20_SeqL_ERD_6_B5.fif', overwrite=True)


#-----------------------------------------------------------------------------#
#                               PARTICIPANT 27
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_27_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID5\Training\Part27\6key\part27_SeqL_ERD_6_B1.vhdr'
Part_27_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID5\Training\Part27\6key\part27_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p27 = mne.io.read_raw_brainvision(Part_27_1, preload = True) 

#--------B5--------#
raw5_p27 = mne.io.read_raw_brainvision(Part_27_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p27.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p27.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p27.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p27.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p27.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p27.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p27 = mne.io.add_reference_channels(raw_p27, 'TP8')   #reference channel

#--------B5-------#
raw5_p27.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p27.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p27.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p27.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p27.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p27.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p27 = mne.io.add_reference_channels(raw5_p27, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p27.plot_psd(fmax = 250)#maximum frequency 
raw_p27.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p27.plot()


#--------B5-------#
raw5_p27.plot_psd(fmax = 250)#maximum frequency 
raw5_p27.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p27.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p27.set_montage(montage)
raw5_p27.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p27, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p27, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p27.plot_sensors(kind='topomap', show_names=True)
raw5_p27.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p27.info)
print(raw5_p27.info)


#plot to show the waves and their source on the head
raw_p27.plot_psd(fmax = 250)
raw5_p27.plot_psd(fmax = 250)




#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica_p27 = mne.preprocessing.ICA()

raw_p27 = raw_p27.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica_p27.fit(raw_p27)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices_p27, eog_scores_p27 = ica_p27.find_bads_eog(raw_p27, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica_p27.exclude = eog_indices_p27#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica_p27.plot_scores(eog_scores_p27)

# plot diagnostics
ica_p27.plot_properties(raw_p27, picks=eog_indices_p27)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica_p27.plot_sources(raw_p27)


#check if raw data has been cleaned 
raw_p27.plot()


#visual presentation ICA components on head
ica_p27.plot_components()

#creating eog epochs
eog_evoked_p27 = create_eog_epochs(raw_p27).average()
eog_evoked_p27.apply_baseline(baseline=(None, -0.2))
eog_evoked_p27.plot_joint()

ica_p27.plot_properties(raw_p27, [1,2,3])
ica_p27.exclude =[1,2,3]


#since ica.apply changes raw we are making a copy
reconst_raw_p27 = raw_p27.copy()
ica_p27.apply(reconst_raw_p27)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p27.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p27.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part27_SeqL_ERD_6_B1.fif', overwrite=True)




#--------B5-------#

ica5_p27 = mne.preprocessing.ICA()

raw5_p27 = raw5_p27.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5_p27.fit(raw5_p27)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5_p27, eog_scores5_p27 = ica5_p27.find_bads_eog(raw5_p27, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5_p27.exclude = eog_indices5_p27#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5_p27.plot_scores(eog_scores5_p27)

# plot diagnostics
ica5_p27.plot_properties(raw5_p27, picks=eog_indices5_p27)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5_p27.plot_sources(raw5_p27)

#check if raw data has been cleaned 
raw5_p27.plot()

#visual presentation ICA components on head
ica5_p27.plot_components()

#creating eog epochs
eog_evoked5_p27 = create_eog_epochs(raw5_p27).average()
eog_evoked5_p27.apply_baseline(baseline=(None, -0.2))
eog_evoked5_p27.plot_joint()

ica5_p27.plot_properties(raw5_p27, [2,3,4])
ica5_p27.exclude =[2,3,4]

reconst_raw5_p27 = raw5_p27.copy()
ica5_p27.apply(reconst_raw5_p27)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p27.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p27.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part27_SeqL_ERD_6_B5.fif', overwrite=True)





#-----------------------------------------------------------------------------#
#                               PARTICIPANT 6
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_6_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID6\Training\Part6\6key\part6_SeqL_ERD_6_B1.vhdr'
Part_6_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID6\Training\Part6\6key\part6_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p6 = mne.io.read_raw_brainvision(Part_6_1, preload = True) 

#--------B5--------#
raw5_p6 = mne.io.read_raw_brainvision(Part_6_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p6.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p6.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p6.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p6.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p6.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p6.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p6 = mne.io.add_reference_channels(raw_p6, 'TP8')   #reference channel

#--------B5-------#
raw5_p6.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p6.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p6.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p6.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p6.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p6.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p6 = mne.io.add_reference_channels(raw5_p6, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p6.plot_psd(fmax = 250)#maximum frequency 
raw_p6.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p6.plot()

#--------B5-------#
raw5_p6.plot_psd(fmax = 250)#maximum frequency 
raw5_p6.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p6.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p6.set_montage(montage)
raw5_p6.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p6, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p6, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p6.plot_sensors(kind='topomap', show_names=True)
raw5_p6.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p6.info)
print(raw5_p6.info)


#plot to show the waves and their source on the head
raw_p6.plot_psd(fmax = 250)
raw5_p6.plot_psd(fmax = 250)



#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica = mne.preprocessing.ICA()

raw_p6 = raw_p6.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica.fit(raw_p6)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices, eog_scores = ica.find_bads_eog(raw_p6, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica.exclude = eog_indices#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica.plot_scores(eog_scores)

# plot diagnostics
ica.plot_properties(raw_p6, picks=eog_indices)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica.plot_sources(raw_p6)


#check if raw data has been cleaned 
raw_p6.plot()


#visual presentation ICA components on head
ica.plot_components()

#creating eog epochs
eog_evoked = create_eog_epochs(raw_p6).average()
eog_evoked.apply_baseline(baseline=(None, -0.2))
eog_evoked.plot_joint()

ica.plot_properties(raw_p6, [0,3])
ica.exclude =[0,3]


#since ica.apply changes raw we are making a copy
reconst_raw_p6 = raw_p6.copy()
ica.apply(reconst_raw_p6)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p6.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p6.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part6_SeqL_ERD_6_B1.fif', overwrite=True)



#--------B5-------#

ica5 = mne.preprocessing.ICA()

raw5_p6 = raw5_p6.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5.fit(raw5_p6)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5, eog_scores5 = ica5.find_bads_eog(raw5_p6, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5.exclude = eog_indices5#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5.plot_scores(eog_scores5)

# plot diagnostics
ica5.plot_properties(raw5_p6, picks=eog_indices5)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5.plot_sources(raw5_p6)

#check if raw data has been cleaned 
raw5_p6.plot()

#visual presentation ICA components on head
ica5.plot_components()

#creating eog epochs
eog_evoked5 = create_eog_epochs(raw5_p6).average()
eog_evoked5.apply_baseline(baseline=(None, -0.2))
eog_evoked5.plot_joint()

ica5.plot_properties(raw5_p6, [0,2])
ica5.exclude =[0,2]

reconst_raw5_p6 = raw5_p6.copy()
ica5.apply(reconst_raw5_p6)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p6.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p6.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part6_SeqL_ERD_6_B5.fif', overwrite=True)




#-----------------------------------------------------------------------------#
#                               PARTICIPANT 16
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_16_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID6\Training\Part16\6key\part16_SeqL_ERD_6_B1.vhdr'
Part_16_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID6\Training\Part16\6key\part16_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p16 = mne.io.read_raw_brainvision(Part_16_1, preload = True) 
#--------B5--------#
raw5_p16 = mne.io.read_raw_brainvision(Part_16_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p16.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p16.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p16.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p16.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p16.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p16.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p16 = mne.io.add_reference_channels(raw_p16, 'TP8')   #reference channel


#--------B5-------#
raw5_p16.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p16.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p16.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p16.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p16.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p16.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p16 = mne.io.add_reference_channels(raw5_p16, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p16.plot_psd(fmax = 250)#maximum frequency 
raw_p16.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p16.plot()

#--------B5-------#
raw5_p16.plot_psd(fmax = 250)#maximum frequency 
raw5_p16.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p16.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p16.set_montage(montage)
raw5_p16.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p16, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p16, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p16.plot_sensors(kind='topomap', show_names=True)
raw5_p16.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p16.info)
print(raw5_p16.info)


#plot to show the waves and their source on the head
raw_p16.plot_psd(fmax = 250)
raw5_p16.plot_psd(fmax = 250)



#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica_p16 = mne.preprocessing.ICA()

raw_p16 = raw_p16.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica_p16.fit(raw_p16)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices_p16, eog_scores_p16 = ica_p16.find_bads_eog(raw_p16, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica_p16.exclude = eog_indices_p16#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica_p16.plot_scores(eog_scores_p16)

# plot diagnostics
ica_p16.plot_properties(raw_p16, picks=eog_indices_p16)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica_p16.plot_sources(raw_p16)


#check if raw data has been cleaned 
raw_p16.plot()


#visual presentation ICA components on head
ica_p16.plot_components()

#creating eog epochs
eog_evoked_p16 = create_eog_epochs(raw_p16).average()
eog_evoked_p16.apply_baseline(baseline=(None, -0.2))
eog_evoked_p16.plot_joint()

ica_p16.plot_properties(raw_p16, [0,1])
ica_p16.exclude =[0,1]


#since ica.apply changes raw we are making a copy
reconst_raw_p16 = raw_p16.copy()
ica_p16.apply(reconst_raw_p16)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p16.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p16.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part16_SeqL_ERD_6_B1.fif', overwrite=True)




#--------B5-------#

ica5_p16 = mne.preprocessing.ICA()

raw5_p16 = raw5_p16.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5_p16.fit(raw5_p16)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5_p16, eog_scores5_p16 = ica5_p16.find_bads_eog(raw5_p16, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5.exclude = eog_indices5#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5_p16.plot_scores(eog_scores5_p16)

# plot diagnostics
ica5_p16.plot_properties(raw5_p16, picks=eog_indices5_p16)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5_p16.plot_sources(raw5_p16)

#check if raw data has been cleaned 
raw5_p16.plot()

#visual presentation ICA components on head
ica5_p16.plot_components()

#creating eog epochs
eog_evoked5_p16 = create_eog_epochs(raw5_p16).average()
eog_evoked5_p16.apply_baseline(baseline=(None, -0.2))
eog_evoked5_p16.plot_joint()

ica5_p16.plot_properties(raw5_p16, [4])
ica5_p16.exclude =[4]

reconst_raw5_p16 = raw5_p16.copy()
ica5.apply(reconst_raw5_p16)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p16.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p16.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part16_SeqL_ERD_6_B5.fif', overwrite=True)




#-----------------------------------------------------------------------------#
#                               PARTICIPANT 21
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_21_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID6\Training\Part21\6key\part21_SeqL_ERD_6_B1.vhdr'
Part_21_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID6\Training\Part21\6key\part21_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p21 = mne.io.read_raw_brainvision(Part_21_1, preload = True) 

#--------B5--------#
raw5_p21 = mne.io.read_raw_brainvision(Part_21_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p21.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p21.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p21.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p21.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p21.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p21.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p21 = mne.io.add_reference_channels(raw_p21, 'TP8')   #reference channel


#--------B5-------#
raw5_p21.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p21.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p21.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p21.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p21.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p21.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p21 = mne.io.add_reference_channels(raw5_p21, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p21.plot_psd(fmax = 250)#maximum frequency 
raw_p21.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p21.plot()

#--------B5-------#
raw5_p21.plot_psd(fmax = 250)#maximum frequency 
raw5_p21.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p21.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p21.set_montage(montage)
raw5_p21.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p21, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p21, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p21.plot_sensors(kind='topomap', show_names=True)
raw5_p21.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p21.info)
print(raw5_p21.info)


#plot to show the waves and their source on the head
raw_p21.plot_psd(fmax = 250)
raw5_p21.plot_psd(fmax = 250)



#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica_p21 = mne.preprocessing.ICA()

raw_p21 = raw_p21.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica_p21.fit(raw_p21)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices_p21, eog_scores_p21 = ica_p21.find_bads_eog(raw_p21, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica_p21.exclude = eog_indices_p21#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica_p21.plot_scores(eog_scores_p21)

# plot diagnostics
ica_p21.plot_properties(raw_p21, picks=eog_indices_p21)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica_p21.plot_sources(raw_p21)


#check if raw data has been cleaned 
raw_p21.plot()


#visual presentation ICA components on head
ica_p21.plot_components()

#creating eog epochs
eog_evoked_p21 = create_eog_epochs(raw_p21).average()
eog_evoked_p21.apply_baseline(baseline=(None, -0.2))
eog_evoked_p21.plot_joint()

ica_p21.plot_properties(raw_p21, [0,2])
ica_p21.exclude =[0,2]


#since ica.apply changes raw we are making a copy
reconst_raw_p21 = raw_p21.copy()
ica_p21.apply(reconst_raw_p21)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p21.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p21.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part21_SeqL_ERD_6_B1.fif', overwrite=True)




#--------B5-------#

ica5_p21 = mne.preprocessing.ICA()

raw5_p21 = raw5_p21.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5_p21.fit(raw5_p21)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5_p21, eog_scores5_p21 = ica5_p21.find_bads_eog(raw5_p21, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5_p21.exclude = eog_indices5_p21#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5_p21.plot_scores(eog_scores5_p21)

# plot diagnostics
ica5_p21.plot_properties(raw5_p21, picks=eog_indices5_p21)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5_p21.plot_sources(raw5_p21)

#check if raw data has been cleaned 
raw5_p21.plot()

#visual presentation ICA components on head
ica5_p21.plot_components()

#creating eog epochs
eog_evoked5_p21 = create_eog_epochs(raw5_p21).average()
eog_evoked5_p21.apply_baseline(baseline=(None, -0.2))
eog_evoked5_p21.plot_joint()

ica5_p21.plot_properties(raw5_p21, [0,1])
ica5_p21.exclude =[0,1]

reconst_raw5_p21 = raw5_p21.copy()
ica5_p21.apply(reconst_raw5_p21)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p21.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p21.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part21_SeqL_ERD_6_B5.fif', overwrite=True)




#-----------------------------------------------------------------------------#
#                               PARTICIPANT 7
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_7_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID7\Training\Part7\6 key\part7_SeqL_ERD_6_B1.vhdr'
Part_7_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID7\Training\Part7\6 key\part7_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p7 = mne.io.read_raw_brainvision(Part_7_1, preload = True) 

#--------B5--------#
raw5_p7 = mne.io.read_raw_brainvision(Part_7_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p7.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p7.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p7.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p7.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p7.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p7.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p7 = mne.io.add_reference_channels(raw_p7, 'TP8')   #reference channel

#--------B5-------#
raw5_p7.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p7.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p7.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p7.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p7.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p7.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p7 = mne.io.add_reference_channels(raw5_p7, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p7.plot_psd(fmax = 250)#maximum frequency 
raw_p7.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p7.plot()

#--------B5-------#
raw5_p7.plot_psd(fmax = 250)#maximum frequency 
raw5_p7.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p7.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p7.set_montage(montage)
raw5_p7.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p7, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p7, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p7.plot_sensors(kind='topomap', show_names=True)
raw5_p7.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p7.info)
print(raw5_p7.info)


#plot to show the waves and their source on the head
raw_p7.plot_psd(fmax = 250)
raw5_p7.plot_psd(fmax = 250)




#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica_p7 = mne.preprocessing.ICA()

raw_p7 = raw_p7.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica_p7.fit(raw_p7)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices_p7, eog_scores_p7 = ica_p7.find_bads_eog(raw_p7, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica_p7.exclude = eog_indices_p7#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica_p7.plot_scores(eog_scores_p7)

# plot diagnostics
ica_p7.plot_properties(raw_p7, picks=eog_indices_p7)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica_p7.plot_sources(raw_p7)


#check if raw data has been cleaned 
raw_p7.plot()


#visual presentation ICA components on head
ica_p7.plot_components()

#creating eog epochs
eog_evoked_p7 = create_eog_epochs(raw_p7).average()
eog_evoked_p7.apply_baseline(baseline=(None, -0.2))
eog_evoked_p7.plot_joint()

ica_p7.plot_properties(raw_p7, [0,1])
ica_p7.exclude =[0,1]


#since ica.apply changes raw we are making a copy
reconst_raw_p7 = raw_p7.copy()
ica_p7.apply(reconst_raw_p7)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p7.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p7.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part7_SeqL_ERD_6_B1.fif', overwrite=True)



#--------B5-------#

ica5_p7 = mne.preprocessing.ICA()

raw5_p7 = raw5_p7.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5_p7.fit(raw5_p7)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5_p7, eog_scores5_p7 = ica5_p7.find_bads_eog(raw5_p7, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5_p7.exclude = eog_indices5_p7#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5_p7.plot_scores(eog_scores5_p7)

# plot diagnostics
ica5_p7.plot_properties(raw5_p7, picks=eog_indices5_p7)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5_p7.plot_sources(raw5_p7)

#check if raw data has been cleaned 
raw5_p7.plot()

#visual presentation ICA components on head
ica5_p7.plot_components()

#creating eog epochs
eog_evoked5_p7 = create_eog_epochs(raw5_p7).average()
eog_evoked5_p7.apply_baseline(baseline=(None, -0.2))
eog_evoked5_p7.plot_joint()

ica5_p7.plot_properties(raw5_p7, [0,1,2])

ica5_p7.exclude =[0,1,2]

reconst_raw5_p7 = raw5_p7.copy()
ica5_p7.apply(reconst_raw5_p7)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p7.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p7.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part7_SeqL_ERD_6_B5.fif', overwrite=True)




#-----------------------------------------------------------------------------#
#                               PARTICIPANT 17
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_17_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID7\Training\Part17\6key\part17_SeqL_ERD_6_B1.vhdr'
Part_17_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID7\Training\Part17\6key\part17_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p17 = mne.io.read_raw_brainvision(Part_17_1, preload = True) 


#--------B5--------#
raw5_p17 = mne.io.read_raw_brainvision(Part_17_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p17.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p17.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p17.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p17.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p17.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p17.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p17 = mne.io.add_reference_channels(raw_p17, 'TP8')   #reference channel

#--------B5-------#
raw5_p17.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p17.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p17.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p17.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p17.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p17.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p17 = mne.io.add_reference_channels(raw5_p17, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p17.plot_psd(fmax = 250)#maximum frequency 
raw_p17.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p17.plot()

#--------B5-------#
raw5_p17.plot_psd(fmax = 250)#maximum frequency 
raw5_p17.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p17.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p17.set_montage(montage)
raw5_p17.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p17, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p17, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p17.plot_sensors(kind='topomap', show_names=True)
raw5_p17.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p17.info)
print(raw5_p17.info)


#plot to show the waves and their source on the head
raw_p17.plot_psd(fmax = 250)
raw5_p17.plot_psd(fmax = 250)




#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica_p17 = mne.preprocessing.ICA()

raw_p17 = raw_p17.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica_p17.fit(raw_p17)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices_p17, eog_scores_p17 = ica_p17.find_bads_eog(raw_p17, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica_p17.exclude = eog_indices_p17#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica_p17.plot_scores(eog_scores_p17)

# plot diagnostics
ica_p17.plot_properties(raw_p17, picks=eog_indices_p17)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica_p17.plot_sources(raw_p17)


#check if raw data has been cleaned 
raw_p17.plot()


#visual presentation ICA components on head
ica_p17.plot_components()

#creating eog epochs
eog_evoked_p17 = create_eog_epochs(raw_p17).average()
eog_evoked_p17.apply_baseline(baseline=(None, -0.2))
eog_evoked_p17.plot_joint()

ica_p17.plot_properties(raw_p17, [0,1])
ica_p17.exclude =[0,1]


#since ica.apply changes raw we are making a copy
reconst_raw_p17 = raw_p17.copy()
ica_p17.apply(reconst_raw_p17)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p17.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p17.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part17_SeqL_ERD_6_B1.fif', overwrite=True)


#--------B5-------#

ica5_p17 = mne.preprocessing.ICA()

raw5_p17 = raw5_p17.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5_p17.fit(raw5_p17)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5_p17, eog_scores5_p17 = ica5_p17.find_bads_eog(raw5_p17, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5_p17.exclude = eog_indices5_p17#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5_p17.plot_scores(eog_scores5_p17)

# plot diagnostics
ica5_p17.plot_properties(raw5_p17, picks=eog_indices5_p17)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5_p17.plot_sources(raw5_p17)

#check if raw data has been cleaned 
raw5_p17.plot()

#visual presentation ICA components on head
ica5_p17.plot_components()

#creating eog epochs
eog_evoked5_p17 = create_eog_epochs(raw5_p17).average()
eog_evoked5_p17.apply_baseline(baseline=(None, -0.2))
eog_evoked5_p17.plot_joint()

ica5_p17.plot_properties(raw5_p17, [0,1])
ica5_p17.exclude =[0,1]

reconst_raw5_p17 = raw5_p17.copy()
ica5_p17.apply(reconst_raw5_p17)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p17.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p17.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part17_SeqL_ERD_6_B5.fif', overwrite=True)




#-----------------------------------------------------------------------------#
#                               PARTICIPANT 22
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_22_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID7\Training\Part22\6key\part22_SeqL_ERD_6_B1.vhdr'
Part_22_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID7\Training\Part22\6key\part22_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p22 = mne.io.read_raw_brainvision(Part_22_1, preload = True) 

#--------B5--------#
raw5_p22 = mne.io.read_raw_brainvision(Part_22_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p22.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p22.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p22.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p22.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p22.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p22.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p22 = mne.io.add_reference_channels(raw_p22, 'TP8')   #reference channel

#--------B5-------#
raw5_p22.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p22.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p22.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p22.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p22.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p22.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p22 = mne.io.add_reference_channels(raw5_p22, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p22.plot_psd(fmax = 250)#maximum frequency 
raw_p22.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p22.plot()

#--------B5-------#
raw5_p22.plot_psd(fmax = 250)#maximum frequency 
raw5_p22.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p22.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p22.set_montage(montage)
raw5_p22.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p22, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p22, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p22.plot_sensors(kind='topomap', show_names=True)
raw5_p22.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p22.info)
print(raw5_p22.info)


#plot to show the waves and their source on the head
raw_p22.plot_psd(fmax = 250)
raw5_p22.plot_psd(fmax = 250)



#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica = mne.preprocessing.ICA()

raw_p22 = raw_p22.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica.fit(raw_p22)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices, eog_scores = ica.find_bads_eog(raw_p22, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica.exclude = eog_indices#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica.plot_scores(eog_scores)

# plot diagnostics
ica.plot_properties(raw_p22, picks=eog_indices)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica.plot_sources(raw_p22)


#check if raw data has been cleaned 
raw_p22.plot()


#visual presentation ICA components on head
ica.plot_components()

#creating eog epochs
eog_evoked = create_eog_epochs(raw_p22).average()
eog_evoked.apply_baseline(baseline=(None, -0.2))
eog_evoked.plot_joint()

ica.plot_properties(raw_p22, [0,1,2,3])
ica.exclude =[0,1,2,3]


#since ica.apply changes raw we are making a copy
reconst_raw_p22 = raw_p22.copy()
ica.apply(reconst_raw_p22)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p22.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p22.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part22_SeqL_ERD_6_B1.fif', overwrite=True)


#--------B5-------#

ica5 = mne.preprocessing.ICA()

raw5_p22 = raw5_p22.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5.fit(raw5_p22)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5, eog_scores5 = ica5.find_bads_eog(raw5_p22, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5.exclude = eog_indices5#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5.plot_scores(eog_scores5)

# plot diagnostics
ica5.plot_properties(raw5_p22, picks=eog_indices5)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5.plot_sources(raw5_p22)

#check if raw data has been cleaned 
raw5_p22.plot()

#visual presentation ICA components on head
ica5.plot_components()

#creating eog epochs
eog_evoked5 = create_eog_epochs(raw5_p22).average()
eog_evoked5.apply_baseline(baseline=(None, -0.2))
eog_evoked5.plot_joint()

ica5.plot_properties(raw5_p22, [0,1,2])
ica5.exclude =[0,1,2]

reconst_raw5_p22 = raw5_p22.copy()
ica5.apply(reconst_raw5_p22)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p22.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p22.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part22_SeqL_ERD_6_B5.fif', overwrite=True)



#-----------------------------------------------------------------------------#
#                               PARTICIPANT 8
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_8_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID8\Training\Part8\6 key\part11_SeqL_ERD_6_B2.vhdr'
Part_8_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID8\Training\Part8\6 key\part11_SeqL_ERD_6_B10.vhdr'

#read file from folder

#--------B1--------#
raw_p8 = mne.io.read_raw_brainvision(Part_8_1, preload = True) 

#--------B5--------#
raw5_p8 = mne.io.read_raw_brainvision(Part_8_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p8.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p8.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p8.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p8.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p8.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p8.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p8 = mne.io.add_reference_channels(raw_p8, 'TP8')   #reference channel

#--------B5-------#
raw5_p8.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p8.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p8.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p8.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p8.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p8.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p8 = mne.io.add_reference_channels(raw5_p8, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p8.plot_psd(fmax = 250)#maximum frequency 
raw_p8.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p8.plot()

#--------B5-------#
raw5_p8.plot_psd(fmax = 250)#maximum frequency 
raw5_p8.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p8.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p8.set_montage(montage)
raw5_p8.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p8, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p8, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p8.plot_sensors(kind='topomap', show_names=True)
raw5_p8.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p8.info)
print(raw5_p8.info)


#plot to show the waves and their source on the head
raw_p8.plot_psd(fmax = 250)
raw5_p8.plot_psd(fmax = 250)



#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica_p8 = mne.preprocessing.ICA()

raw_p8 = raw_p8.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica_p8.fit(raw_p8)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices_p8, eog_scores_p8 = ica_p8.find_bads_eog(raw_p8, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica_p8.exclude = eog_indices_p8#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica_p8.plot_scores(eog_scores_p8)

# plot diagnostics
ica_p8.plot_properties(raw_p8, picks=eog_indices_p8)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica_p8.plot_sources(raw_p8)


#check if raw data has been cleaned 
raw_p8.plot()


#visual presentation ICA components on head
ica_p8.plot_components()

#creating eog epochs
eog_evoked_p8 = create_eog_epochs(raw_p8).average()
eog_evoked_p8.apply_baseline(baseline=(None, -0.2))
eog_evoked_p8.plot_joint()

ica_p8.plot_properties(raw_p8, [0,1])
ica_p8.exclude =[0,1]


#since ica.apply changes raw we are making a copy
reconst_raw_p8 = raw_p8.copy()
ica_p8.apply(reconst_raw_p8)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p8.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p8.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part8_SeqL_ERD_6_B1.fif', overwrite=True)



#--------B5-------#

ica5_p8 = mne.preprocessing.ICA()

raw5_p8 = raw5_p8.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5_p8.fit(raw5_p8)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5_p8, eog_scores5_p8 = ica5_p8.find_bads_eog(raw5_p8, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5.exclude = eog_indices5#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5_p8.plot_scores(eog_scores5_p8)

# plot diagnostics
ica5_p8.plot_properties(raw5_p8, picks=eog_indices5_p8)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5_p8.plot_sources(raw5_p8)

#check if raw data has been cleaned 
raw5_p8.plot()

#visual presentation ICA components on head
ica5_p8.plot_components()

#creating eog epochs
eog_evoked5_p8 = create_eog_epochs(raw5_p8).average()
eog_evoked5_p8.apply_baseline(baseline=(None, -0.2))
eog_evoked5_p8.plot_joint()

ica5_p8.plot_properties(raw5_p8, [0,1,4])
ica5_p8.exclude =[0,1,4]

reconst_raw5_p8 = raw5_p8.copy()
ica5.apply(reconst_raw5_p8)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p8.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p8.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part8_SeqL_ERD_6_B5.fif', overwrite=True)





#-----------------------------------------------------------------------------#
#                               PARTICIPANT 23
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_23_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID8\Training\Part23\6key\part23_SeqL_ERD_6_B1.vhdr'
Part_23_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID8\Training\Part23\6key\part23_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p23 = mne.io.read_raw_brainvision(Part_23_1, preload = True) 


#--------B5--------#
raw5_p23 = mne.io.read_raw_brainvision(Part_23_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p23.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p23.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p23.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p23.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p23.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p23.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p23 = mne.io.add_reference_channels(raw_p23, 'TP8')   #reference channel


#--------B5-------#
raw5_p23.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p23.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p23.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p23.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p23.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p23.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p23 = mne.io.add_reference_channels(raw5_p23, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p23.plot_psd(fmax = 250)#maximum frequency 
raw_p23.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p23.plot()

#--------B5-------#
raw5_p23.plot_psd(fmax = 250)#maximum frequency 
raw5_p23.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p23.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p23.set_montage(montage)
raw5_p23.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p23, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p23, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p23.plot_sensors(kind='topomap', show_names=True)
raw5_p23.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p23.info)
print(raw5_p23.info)


#plot to show the waves and their source on the head
raw_p23.plot_psd(fmax = 250)
raw5_p23.plot_psd(fmax = 250)




#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica_p23 = mne.preprocessing.ICA()

raw_p23 = raw_p23.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica_p23.fit(raw_p23)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices_p23, eog_scores_p23 = ica_p23.find_bads_eog(raw_p23, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica_p23.exclude = eog_indices_p23#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica_p23.plot_scores(eog_scores_p23)

# plot diagnostics
ica_p23.plot_properties(raw_p23, picks=eog_indices_p23)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica_p23.plot_sources(raw_p23)


#check if raw data has been cleaned 
raw_p23.plot()


#visual presentation ICA components on head
ica_p23.plot_components()

#creating eog epochs
eog_evoked_p23 = create_eog_epochs(raw_p23).average()
eog_evoked_p23.apply_baseline(baseline=(None, -0.2))
eog_evoked_p23.plot_joint()

ica_p23.plot_properties(raw_p23, [1,4,5])
ica_p23.exclude =[1,4,5]


#since ica.apply changes raw we are making a copy
reconst_raw_p23 = raw_p23.copy()
ica_p23.apply(reconst_raw_p23)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p23.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p23.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part23_SeqL_ERD_6_B1.fif', overwrite=True)



#--------B5-------#

ica5_p23 = mne.preprocessing.ICA()

raw5_p23 = raw5_p23.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5_p23.fit(raw5_p23)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5_p23, eog_scores5_p23 = ica5_p23.find_bads_eog(raw5_p23, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5_p23.exclude = eog_indices5_p23#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5_p23.plot_scores(eog_scores5_p23)

# plot diagnostics
ica5_p23.plot_properties(raw5_p23, picks=eog_indices5_p23)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5_p23.plot_sources(raw5_p23)

#check if raw data has been cleaned 
raw5_p23.plot()

#visual presentation ICA components on head
ica5_p23.plot_components()

#creating eog epochs
eog_evoked5_p23 = create_eog_epochs(raw5_p23).average()
eog_evoked5_p23.apply_baseline(baseline=(None, -0.2))
eog_evoked5_p23.plot_joint()

ica5_p23.plot_properties(raw5_p23, [0,1])

ica5_p23.exclude =[0,1]

reconst_raw5_p23 = raw5_p23.copy()
ica5_p23.apply(reconst_raw5_p23)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p23.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p23.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part23_SeqL_ERD_6_B5.fif', overwrite=True)




#-----------------------------------------------------------------------------#
#                               PARTICIPANT 29
#-----------------------------------------------------------------------------#
#give file a name and save it#

Part_29_1 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID8\Training\Part29\6key\part30_SeqL_ERD_6_B1.vhdr'
Part_29_5 = r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\Seql_ERD_copy\ID8\Training\Part29\6key\part30_SeqL_ERD_6_B5.vhdr'

#read file from folder

#--------B1--------#
raw_p29 = mne.io.read_raw_brainvision(Part_29_1, preload = True) 

#--------B5--------#
raw5_p29 = mne.io.read_raw_brainvision(Part_29_5, preload = True) 



#give channel the right type (=eeg an eog)

#--------B1---------#
raw_p29.pick_types (meg=False, eeg=True, eog=True, ecg=False)
raw_p29.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw_p29.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw_p29.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw_p29.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  
raw_p29.drop_channels(['hEOG', 'vEOG'])               #not used
raw_p29 = mne.io.add_reference_channels(raw_p29, 'TP8')   #reference channel

#--------B5-------#
raw5_p29.pick_types (meg=False, eeg=True, eog=True, ecg=False) 
raw5_p29.set_channel_types(mapping={'vEOG_L' : 'eog'}) #ocular signals 
raw5_p29.set_channel_types(mapping={'vEOG_U' : 'eog'}) #ocular signals 
raw5_p29.set_channel_types(mapping={'hEOG_L' : 'eog'}) #ocular signals  
raw5_p29.set_channel_types(mapping={'hEOG_R' : 'eog'}) #ocular signals  '
raw5_p29.drop_channels(['hEOG', 'vEOG'])               #not used
raw5_p29 = mne.io.add_reference_channels(raw5_p29, 'TP8')  #reference channel


#plot figure of raw

#--------B1-------#

raw_p29.plot_psd(fmax = 250)#maximum frequency 
raw_p29.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw_p29.plot()

#--------B5-------#
raw5_p29.plot_psd(fmax = 250)#maximum frequency 
raw5_p29.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 
raw5_p29.plot()


#set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 
raw_p29.set_montage(montage)
raw5_p29.set_montage(montage)


#setting bipolar reference

raw_bip_ref = mne.set_bipolar_reference(raw_p29, anode=['TP8'],
                                        cathode=['TP7'])
raw_bip_ref = mne.set_bipolar_reference(raw5_p29, anode=['TP8'],
                                        cathode=['TP7'])


#plot that shows the channel locations on the head
raw_p29.plot_sensors(kind='topomap', show_names=True)
raw5_p29.plot_sensors(kind='topomap', show_names=True) 


#check raw information  
print(raw_p29.info)
print(raw5_p29.info)


#plot to show the waves and their source on the head
raw_p29.plot_psd(fmax = 250)
raw5_p29.plot_psd(fmax = 250)




#-----------------------------------------------------------------------------#
#                                       ICA
#-----------------------------------------------------------------------------#


#--------B1-------#

ica_p29 = mne.preprocessing.ICA()

raw_p29 = raw_p29.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica_p29.fit(raw_p29)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices_p29, eog_scores_p29 = ica_p29.find_bads_eog(raw_p29, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica_p29.exclude = eog_indices_p29#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica_p29.plot_scores(eog_scores_p29)

# plot diagnostics
ica_p29.plot_properties(raw_p29, picks=eog_indices_p29)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica_p29.plot_sources(raw_p29)


#check if raw data has been cleaned 
raw_p29.plot()


#visual presentation ICA components on head
ica_p29.plot_components()

#creating eog epochs
eog_evoked_p29 = create_eog_epochs(raw_p29).average()
eog_evoked_p29.apply_baseline(baseline=(None, -0.2))
eog_evoked_p29.plot_joint()

ica_p29.plot_properties(raw_p29, [0,1,2,3,4,5,6,8,10])
ica_p29.exclude =[0,1,2,3,4,5,6,8,10]


#since ica.apply changes raw we are making a copy
reconst_raw_p29 = raw_p29.copy()
ica_p29.apply(reconst_raw_p29)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw_p29.plot()#final check of raw data, here the data should be full cleaned

reconst_raw_p29.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part29_SeqL_ERD_6_B1.fif', overwrite=True)



#--------B5-------#

ica5_p29 = mne.preprocessing.ICA()

raw5_p29 = raw5_p29.filter(0.1, 39)#band-pass filtering in the range 0.1Hz to 39Hz

ica5_p29.fit(raw5_p29)

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices5_p29, eog_scores5_p29 = ica5_p29.find_bads_eog(raw5_p29, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica5_p29.exclude = eog_indices5_p29#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica5_p29.plot_scores(eog_scores5_p29)

# plot diagnostics
ica5_p29.plot_properties(raw5_p29, picks=eog_indices5_p29)

# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica5_p29.plot_sources(raw5_p29)

#check if raw data has been cleaned 
raw5_p29.plot()

#visual presentation ICA components on head
ica5_p29.plot_components()

#creating eog epochs
eog_evoked5_p29 = create_eog_epochs(raw5_p29).average()
eog_evoked5_p29.apply_baseline(baseline=(None, -0.2))
eog_evoked5_p29.plot_joint()

ica5_p29.plot_properties(raw5_p29, [0,1,3,4])
ica5_p29.exclude =[0,1,3,4]

reconst_raw5_p29 = raw5_p29.copy()
ica5_p29.apply(reconst_raw5_p29)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw5_p29.plot()#final check of raw data, here the data should be full cleaned

reconst_raw5_p29.save(r'C:\Users\daphn\OneDrive\Bureaublad\Master Thesis\new_part29_SeqL_ERD_6_B5.fif', overwrite=True)

