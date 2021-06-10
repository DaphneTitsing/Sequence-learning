# -*- coding: utf-8 -*-
"""
Created on Thu May 13 12:44:23 2021

@author: Daphne Titsing
"""
import os
import numpy as np
import mne
from mne.event import define_target_events 
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs, corrmap
from mne.connectivity import spectral_connectivity 
from mne.datasets import sample 
from mne.viz import plot_sensors_connectivity 

#give file a name and save it

Part_14 = r'C:\Users\Russell\Desktop\Master Thesis\Seql_ERD_copy\ID1\Training\Part14\6key\part14_SeqL_ERD_6_B1.vhdr'

#read file from folder 

raw = mne.io.read_raw_brainvision(Part_14) 

raw = mne.io.read_raw_brainvision(Part_14, preload = True) 

events, _ = mne.events_from_annotations(raw)
print (events)


 #give channel the right type (=eeg an eog) 

raw.pick_types (meg=False, eeg=True, eog=True, ecg=True) #not sure about ecg

raw.set_channel_types(mapping={'vEOG_L' : 'eog'})#ocular signals 

raw.set_channel_types(mapping={'vEOG_U' : 'eog'})#ocular signals 

raw.set_channel_types(mapping={'hEOG_L' : 'eog'})#ocular signals  

raw.set_channel_types(mapping={'hEOG_R' : 'eog'})#ocular signals  '

raw.drop_channels(['hEOG', 'vEOG']) #these channels are not used

#adding reference channel

raw = mne.io.add_reference_channels(raw, 'TP8')


#plot figure of raw

raw.plot_psd(fmax = 250)#maximum frequency 

raw.plot(duration = 4, n_channels = 30)#duration (in  seconds) & channels shown in graph 

raw.plot()



##set electrode location (extended 10-20system) through montage 

montage = mne.channels.make_standard_montage('standard_1020') 

raw.set_montage(montage)

#setting bipolar reference
raw_bip_ref = mne.set_bipolar_reference(raw, anode=['TP8'],
                                        cathode=['TP7'])

#plot that shows the channel locations on the head
raw.plot_sensors(kind='topomap', show_names=True)
    
print(raw.info)

#plot to show the waves and their source on the head
raw.plot_psd(fmax = 250)#maximum frequency 


#get and save stimuli times --> make an event  
events, _ = mne.events_from_annotations(raw)

new_events = mne.make_fixed_length_events(raw, id=34, start=5.5, stop=None, duration=2.25, first_samp=False, overlap=0.25)

events, _ = mne.events_from_annotations(raw)
events, _ = mne.events_from_annotations(raw, event_id={'Stimulus/S  5': 5,'Stimulus/S  6': 6,
                                                       
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
                                                                 
                                                       'Stimulus/S 27': 27, 
                                                       
                                                       'Stimulus/S 29': 29, 'New Segment/': 34}) 

#get stimuli times --> form event markers which later the epochs will be created around 
print (events)


#making a event dictionary that is needed for showing the frequency of events in the plot
#depending on the block, some events are presented for one block but not for the other 
#in principle the sequence indicator events are not needed as they are specified elsewhere
#if needed it can be looked up in the escel file and put in later

#stim position including!!! keep for later and make new one with orignals
#event_dict = { 'r stim position k': 10, 

  #            'r stim position l': 11, 'r stim position ;': 12, 

 #        'l response a': 14, 
                                                       
  #            'l response s': 15, 'l response d': 16,
                                                       
  #            'l response f': 17, 'r response j': 18,
                                                         
  #            'r response k': 19, 'r response l': 20,
                                                           
  #            'r response ;': 21,
  #            'before nogo': 24,
                                                               
   #           'error response': 25, 'good response': 26,
                                                                 
     #         'go': 27, 
                                                       
     #         'Setting variables': 29}

event_dict = {'5': 5,'6': 6,
                                                       
              '7': 7,'8': 8, 
                                                       
              '9': 9, '10': 10, 

              '11': 11, '12': 12, 

              '14': 14, 
                                                       
              '15': 15, '16': 16,
                                                       
              '17': 17, '18': 18,
                                                         
              '19': 19, '20': 20,
                                                           
              '21': 21,
                                                             
              '24': 24,
                                                               
              '25': 25, '26': 26,
                                                                 
              '27': 27, 
                                                       
              '29': 29, '34': 34}


#create plot showing at what times selected stimuli are 

fig = mne.viz.plot_events(events, event_id=event_dict, 

                         sfreq=raw.info['sfreq']) 

fig.subplots_adjust(right=0.6)#to make room for legend(description)<- smaller number bigger legend 


        ##ICA##
#ica tutorial:https://mne.tools/stable/auto_tutorials/preprocessing/plot_40_artifact_correction_ica.html

#data decomposition using ICA --> estimates independent components from raw data 
#first step is just initializing ica settings and does nothing    
 
ica = mne.preprocessing.ICA()#default setting: (n_components=None, *, max_pca_components=None, n_pca_components=None, noise_cov=None, random_state=None, method='fastica', fit_params=None, max_iter=200, allow_ref_meg=False, verbose=None)
#Additional Info: 
#   1) n_components = None: => n_pca_components (deprecated) or 0.999999 (will become the default in 0.23) will be used, whichever results in fewer components. This is done to avoid numerical stability problems when whitening, particularly when working with rank-deficient data.
#   2) random_state = None: => seed will be obtained from the operating system
raw = raw.filter(1, 35)#band-pass filtering in the range 1Hz to 30Hz
ica.fit(raw)#procheeds in two steps: 1) Whitening the data by means of a pre-whitening step (here:SD of each channel type) followed by PCA
#                                    2) Passing the n_components largest-variance components to the ICA algorithm to obtain the unmixing matrix

#instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
eog_indices, eog_scores = ica.find_bads_eog(raw, ['vEOG_U', 'vEOG_L'])#automatically find the ICs that best match the EOG signal 
ica.exclude = eog_indices#excludes artefacts matching eog signals

#barpolt of ICA component "EOG" match scores
ica.plot_scores(eog_scores)
# plot diagnostics
ica.plot_properties(raw, picks=eog_indices)
# plot ICs applied to raw data, with EOG matches highlighted + allows for further exclusion of components
ica.plot_sources(raw)
#check if raw data has been cleaned 
raw.plot()

#visual presentation ICA components on head
ica.plot_components()

#creating eog epochs
eog_evoked = create_eog_epochs(raw).average()
eog_evoked.apply_baseline(baseline=(None, -0.2))
eog_evoked.plot_joint()

ica.plot_properties(raw, [0,1])
ica.exclude =[0,1]


#since ica.apply changes raw we are making a copy
reconst_raw = raw.copy()
ica.apply(reconst_raw)#proceeds in 4 steps: 1)Unmixes the data with the unmixing matrix
#                                            2)Includes ICA components based on ica.exclude
#                                            3)Re-mixes the data with mixing_matrix
#                                            4)Restores any data not passed to the ICA algorithm (i.e. PCA components between n_components & n_pca_components)
reconst_raw.plot()#final check of raw data, here the data should be full cleaned

reconst_raw.save(r'C:\Users\Russell\Desktop\daphne\new_part14_SeqL_ERD_6_B1.fif', overwrite=True)

# band-pass filtering in the range 1 Hz - 50 Hz
raw.filter(1, 35., fir_design='firwin')

reject_criteria = dict(eeg=100e-6) #100uV 

flat_criteria = dict(eeg=1e-6)#1uV 

tmin, tmax = -1, 1  #new marker 2250 and then from that marker and then -200 and +2250
#making epochs
epochs = mne.Epochs(raw, events, event_id=event_dict, 

                    tmin=tmin, tmax=tmax, reject_tmax=0, 

                    reject=None, flat=None, detrend=1, reject_by_annotation=True, preload=True) #detrending is set here)

#plot to show the effect of detrending
raw.plot_psd(fmax = 250)#maximum frequency 
 
#to show which epochs are dropped 
print(epochs.drop_log) 

#graphic showing dropped epochs & shows the channels that caused the dropping  

epochs.plot_drop_log() 



#drop epochs later IF reject and/or flat criteria have already been provided by: 

epochs.drop_bad() 

#graphic of first 5 epochs + !shows all transient & sustained epochs!
epochs['27'].plot(events=events, event_id=event_dict, n_epochs=5)



 


#deleting bad channels by clicking

#deleting bad segments using:
#my_annot = mne.Annotations(onset=[3, 5, 7],  # in seconds
#                           duration=[1, 0.5, 0.25],  # in seconds, too
#                           description=['AAA', 'BBB', 'CCC'])
#print(my_annot)
#raw.set_annotations(my_annot)
#print(raw.annotations)

# convert meas_date (a tuple of seconds, microseconds) into a float:
#meas_date = raw.info['meas_date']
#orig_time = raw.annotations.orig_time
#print(meas_date == orig_time)

#time_of_first_sample = raw.first_samp / raw.info['sfreq']
#print(my_annot.onset + time_of_first_sample)
#print(raw.annotations.onset)

#time_format = '%Y-%m-%d %H:%M:%S.%f'
#new_orig_time = (meas_date + timedelta(seconds=50)).strftime(time_format)
#print(new_orig_time)

#later_annot = mne.Annotations(onset=[3, 5, 7],
#                              duration=[1, 0.5, 0.25],
#                              description=['DDD', 'EEE', 'FFF'],
 #                             orig_time=new_orig_time)

#raw2 = raw.copy().set_annotations(later_annot)
#print(later_annot.onset)
#print(raw2.annotations.onset)





