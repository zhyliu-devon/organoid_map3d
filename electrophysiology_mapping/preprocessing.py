import sys
# Append the directory containing the module to the Python path
sys.path.append('E:\\Desktop\\omipolar\\DataPlotting\\load_intan_rhs_format')
# Now you can import the module
try:
    import load_intan_rhs_format as ld
except ImportError:
    raise ImportError("Please download load_intan_rhs_format.py from Intan's website and place it in your working directory.")


import matplotlib.pyplot as plt 
import numpy as np
from scipy.signal import butter, lfilter, resample, filtfilt
import scipy.signal
from . import data as dt

def extract_raw_data(data_address_1, data_address_2, fs = 30000,
                       cutoff = 100, fs_new = 1000, window_length = 21, 
                       polyorder = 3, plot = 0, one_address = 0):
    '''
    fs: sampling frequency of the raw signal
    '''
    if one_address:
        data_1 = ld.read_data(data_address_1)
        raw_data = data_1['amplifier_data']
    else:
        data_1 = ld.read_data(data_address_1)
        data_2 = ld.read_data(data_address_2)
        raw_data_1 = data_1['amplifier_data']
        raw_data_2 = data_2['amplifier_data']
        raw_data = np.concatenate((raw_data_1,raw_data_2), axis = 1)

    return raw_data

def butter_lowpass(cutoff, fs, order = 5):
    nyq = 0.5*fs 
    normal_cutoff = cutoff/nyq
    b,a = butter(order, normal_cutoff, btype= 'low', analog= False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order = 5):
    b,a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b,a,data)
    return y

def extractFromIntan(data_address):
    data = ld.read_data(data_address)
    raw_data = data['amplifier_data']
    return raw_data

def processingPipline(data_address_1, data_address_2, fs = 30000,
                       cutoff = 1000, fs_new = 10000, plot = 0, one_address = 0):
    '''
    fs: sampling frequency of the raw signal
    '''
    if one_address:
        data_1 = ld.read_data(data_address_1)
        raw_data = data_1['amplifier_data']
    else:
        data_1 = ld.read_data(data_address_1)
        data_2 = ld.read_data(data_address_2)
        raw_data_1 = data_1['amplifier_data']
        raw_data_2 = data_2['amplifier_data']
        raw_data = np.concatenate((raw_data_1,raw_data_2), axis = 1)
    processed_data = np.zeros((32, int(len(raw_data[0])*fs_new/fs)))
    print('Filtering Data...')
    for i in range(raw_data.shape[0]):
        filtered = butter_lowpass_filter(raw_data[i], cutoff,fs)
        processed_data[i] = resample(filtered, int(len(raw_data[i])*fs_new/fs))
    
    smoothed_data = processed_data
    j = 0
    for i in range(smoothed_data.shape[0]):
        # Extract the data for the current channel and specified range
        #smoothed_data[i] = scipy.signal.savgol_filter(processed_data[i], window_length, polyorder)
        if(plot):
            plt.plot(smoothed_data[i][0:7000].T-j*100, label = f'Channel {i+1}')
            j = j+1
    if(plot):
        plt.legend()
        plt.show()   
    print('Filtering Done!')
    
    return processed_data

def processingPipline_WithHighpass(data_address_1, data_address_2, fs=30000,
                      cutoff=1000, fs_new=10000, plot=0, one_address=0, high_pass=None):
    '''
    fs: sampling frequency of the raw signal
    high_pass: cutoff frequency for high-pass filtering, if specified
    '''
    if one_address:
        data_1 = ld.read_data(data_address_1)
        raw_data = data_1['amplifier_data']
    else:
        data_1 = ld.read_data(data_address_1)
        data_2 = ld.read_data(data_address_2)
        raw_data_1 = data_1['amplifier_data']
        raw_data_2 = data_2['amplifier_data']
        raw_data = np.concatenate((raw_data_1, raw_data_2), axis=1)
    raw_data_ori = raw_data.copy()
    # Apply high-pass filter if specified
    if high_pass is not None:
        print(f'Applying high-pass filter at {high_pass} Hz...')
        for i in range(raw_data.shape[0]):
            raw_data[i] = butter_highpass_filter(raw_data_ori[i], high_pass, fs)
 
    
    processed_data = np.zeros((32, int(len(raw_data[0]) * fs_new / fs)))
    print('Filtering Data...')
    
    for i in range(raw_data.shape[0]):
        filtered = butter_lowpass_filter(raw_data[i], cutoff, fs)
        processed_data[i] = resample(filtered, int(len(raw_data[i]) * fs_new / fs))
    
    smoothed_data = processed_data
    j = 0
    for i in range(smoothed_data.shape[0]):
        # Extract the data for the current channel and specified range
        # smoothed_data[i] = scipy.signal.savgol_filter(processed_data[i], window_length, polyorder)
        if plot:
            plt.plot(smoothed_data[i][0:7000].T - j * 100, label=f'Channel {i + 1}')
            j = j + 1
    
    if plot:
        plt.legend()
        plt.show()
    
    print('Filtering Done!')
    return processed_data

def butter_highpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)


def ExtractingPipline(data_address_1, data_address_2, fs = 30000,
                       cutoff = 100, fs_new = 1000, window_length = 21, 
                       polyorder = 3, plot = 0, one_address = 0):
    '''
    fs: sampling frequency of the raw signal
    '''
    if one_address:
        data_1 = ld.read_data(data_address_1)
        raw_data = data_1['amplifier_data']
    else:
        data_1 = ld.read_data(data_address_1)
        data_2 = ld.read_data(data_address_2)
        raw_data_1 = data_1['amplifier_data']
        raw_data_2 = data_2['amplifier_data']
        raw_data = np.concatenate((raw_data_1,raw_data_2), axis = 1)

    return raw_data

def plotWaves(data,target_channel = dt.wanted_channel, fs = 1000, interval = 150, sample = 60000, start = 0):
    j = 0
    time_scale = np.linspace(0, sample/fs, num=sample)
    for i in np.array(target_channel):
        if len(data[i]) <= start + sample:
            print("Not enough sample, skip")
            return
        # Extract the data for the current channel and specified range
        plt.plot(time_scale,data[i][start:start + sample].T-j*interval, label = f'Channel {i+1}')
        j = j+1
    plt.ylabel('Voltage (micro volt)')
    plt.xlabel('Time (s)')

    #plt.legend()
    plt.show()   
    return


def noSmoothProcessingPipline(data_address_1, data_address_2, fs = 30000,
                       cutoff = 100, fs_new = 1000, window_length = 21, 
                       polyorder = 3, plot = 0, one_address = 0):
    '''
    fs: sampling frequency of the raw signal
    '''
    if one_address:
        data_1 = ld.read_data(data_address_1)
        raw_data = data_1['amplifier_data']
    else:
        data_1 = ld.read_data(data_address_1)
        data_2 = ld.read_data(data_address_2)
        raw_data_1 = data_1['amplifier_data']
        raw_data_2 = data_2['amplifier_data']
        raw_data = np.concatenate((raw_data_1,raw_data_2), axis = 1)
    processed_data = np.zeros((32, int(len(raw_data[0])*fs_new/fs)))
    print('Filtering Data...')
    for i in range(raw_data.shape[0]):
        filtered = butter_lowpass_filter(raw_data[i], cutoff,fs)
        processed_data[i] = resample(filtered, int(len(filtered)*fs_new/fs))
    
    smoothed_data = processed_data
    j = 0
    for i in range(smoothed_data.shape[0]):
        # Extract the data for the current channel and specified range
        smoothed_data[i] = processed_data[i] #= scipy.signal.savgol_filter(processed_data[i], window_length, polyorder)
        if(plot):
            plt.plot(smoothed_data[i][0:7000].T-j*100, label = f'Channel {i+1}')
            j = j+1
    if(plot):
        plt.legend()
        plt.show()   
    print('Filtering Done!')
    return smoothed_data
