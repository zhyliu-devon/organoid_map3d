import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
# Add necessary paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))

sys.path.append(os.path.abspath(parent_dir))


import requests
from tqdm import tqdm  # Progress bar for large downloads

# SharePoint direct download link (replace with the correct link)
file_url = "https://livejohnshopkins-my.sharepoint.com/personal/schoi84_jh_edu/_layouts/15/download.aspx?share=EarUY9jn8txDv-vRWM4aQMABdx-oBAW5DD-SnHxHfVHVTA"

# Define parent directory (modify as needed)
destination_folder = os.path.join(parent_dir, "ExampleData")
os.makedirs(destination_folder, exist_ok=True)
# Define file path
file_name = "CalciumMEARecording.rhs"
file_path = os.path.join(destination_folder, file_name)
# Function to download the file with a progress bar
def download_file(url, file_path):
    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
        return

    print(f"Downloading {file_name} to {file_path}...")

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        with open(file_path, "wb") as file, tqdm(
            desc="Downloading",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
                bar.update(len(chunk))
        print("Download complete!")
    else:
        print("Error downloading file. Check your link.")

# Download the file
download_file(file_url, file_path)


from electrophysiology_mapping import preprocessing

# Constants
SAMPLING_RATE = 10000
CUTOFF_FREQUENCY = 100
USE_RAW_DATA = 0

# Channel information
condition = 'JKCO6IsoCalcium'
save_dir = os.path.join(  script_dir, "results")


MANUAL_CHANNELS = [27, 28]  # Channels requiring manual LAT selection
# The selection are included. If new selection is intended, delete folder lat_selections
# Custom settings for FPD window and peak detection

# File paths
CalciumMEADataPath = os.path.join(parent_dir,"ExampleData", "CalciumMEARecording.rhs")

TARGET_INDICES = [0,1,2,3,4,5,8,9,10,11,12,13,14,15]

INTAN_CHANNELS = [0,1,2,3,4,5,24,25,26,27,28,29,30,31]

# Custom settings for FPD window and peak detection
CUSTOM_SETTINGS = {
    'Baseline': {
        0: {'fpd_window': (200, 300), 'find_max': True, 'detect_above': True, 'threshold_factor': 6},
        1: {'fpd_window': (200, 300), 'find_max': True, 'detect_above': True, 'threshold_factor': 6},
        2: {'fpd_window': (200, 400), 'find_max': True, 'detect_above': True, 'threshold_factor': 6},
        3: {'fpd_window': (150,300), 'find_max': True, 'detect_above': True, 'threshold_factor': 6},
        4: {'fpd_window': (150,300), 'find_max': True, 'detect_above': True, 'threshold_factor': 6},
        5: {'fpd_window': (150,300), 'find_max': True, 'detect_above': True, 'threshold_factor': 6},
        6: {'fpd_window': (40, 150), 'find_max': True, 'detect_above': True, 'threshold_factor': 4},
        7: {'fpd_window': (40, 250), 'find_max': True, 'detect_above': True, 'threshold_factor': 2},
        24: {'fpd_window': (150, 300), 'find_max': False, 'detect_above': True, 'threshold_factor': 6},
        25: {'fpd_window': (150, 300), 'find_max': True, 'detect_above': True, 'threshold_factor': 6},
        26: {'fpd_window': (150, 350), 'find_max': True, 'detect_above': True, 'threshold_factor': 6},
        27: {'fpd_window': (150, 400), 'find_max': False, 'detect_above': False, 'threshold_factor': 2},
        28: {'fpd_window': (150, 400), 'find_max': False, 'detect_above': False, 'threshold_factor': 3},
        29: {'fpd_window': (150, 400), 'find_max': False, 'detect_above': False, 'threshold_factor': 6},
        30: {'fpd_window': (150, 350), 'find_max': True, 'detect_above': True, 'threshold_factor': 6},
        31: {'fpd_window': (150, 400), 'find_max': True, 'detect_above': True, 'threshold_factor': 4},
    },
    'Drug': {
        0: {'fpd_window': (50, 300), 'find_max': True, 'detect_above': True, 'threshold_factor': 4},
        1: {'fpd_window': (50, 200), 'find_max': True, 'detect_above': True, 'threshold_factor': 6},
        2: {'fpd_window': (50, 200), 'find_max': True, 'detect_above': True, 'threshold_factor': 5},
        3: {'fpd_window': (50, 200), 'find_max': True, 'detect_above': True, 'threshold_factor': 6},
        4: {'fpd_window': (50, 200), 'find_max': True, 'detect_above': True, 'threshold_factor': 7},
        5: {'fpd_window': (50, 200), 'find_max': True, 'detect_above': True, 'threshold_factor': 7},
        6: {'fpd_window': (40, 150), 'find_max': True, 'detect_above': True, 'threshold_factor': 4},
        7: {'fpd_window': (40, 250), 'find_max': True, 'detect_above': True, 'threshold_factor': 3},
        24: {'fpd_window': (50, 200), 'find_max': False, 'detect_above': True, 'threshold_factor': 6},
        25: {'fpd_window': (50, 200), 'find_max': True, 'detect_above': True, 'threshold_factor': 5},
        26: {'fpd_window': (50, 250), 'find_max': True, 'detect_above': True, 'threshold_factor': 5},
        27: {'fpd_window': (50, 250), 'find_max': False, 'detect_above': True, 'threshold_factor': 3},
        28: {'fpd_window': (50, 250), 'find_max': False, 'detect_above': True, 'threshold_factor': 3},
        29: {'fpd_window': (50, 300), 'find_max': False, 'detect_above': True, 'threshold_factor': 4},
        30: {'fpd_window': (50, 250), 'find_max': True, 'detect_above': True, 'threshold_factor': 5},
        31: {'fpd_window': (40, 200), 'find_max': True, 'detect_above': False, 'threshold_factor': 4},
    }
}


def load_and_process_data(data_path, cutoff=CUTOFF_FREQUENCY, fs_new=SAMPLING_RATE):
    """Load and process data using the preprocessing pipeline."""
    processed_data = preprocessing.processingPipline(data_path, data_path, cutoff=cutoff, fs_new=fs_new, one_address=1)
    if USE_RAW_DATA:
        raw_data = preprocessing.extract_raw_data(data_path, data_path)
        return raw_data[TARGET_INDICES], 30000
    return processed_data[TARGET_INDICES], fs_new

def detect_spikes(data, fs, threshold_factor=4, min_distance=1, detect_above=True):
    """
    Detect spikes in the data.
    
    Args:
    data (numpy.array): The input signal data.
    fs (float): Sampling frequency of the data.
    threshold_factor (float): Factor to multiply the standard deviation for threshold calculation.
    min_distance (float): Minimum distance between peaks in seconds.
    detect_above (bool): If True, detect peaks above the threshold. If False, detect peaks below the threshold.
    
    Returns:
    numpy.array: Indices of detected spikes.
    """
    mean = np.mean(data)
    std = np.std(data)
    threshold = mean + threshold_factor * std if detect_above else mean - threshold_factor * std
    
    if detect_above:
        spikes, _ = find_peaks(data, height=threshold, distance=int(min_distance * fs))
    else:
        spikes, _ = find_peaks(-data, height=-threshold, distance=int(min_distance * fs))
    
    return spikes


def find_lat(data, peak_index, fs, window_ms=(-25, 25), custom_window=None):
    """Find LAT with optional custom window."""
    if custom_window is not None:
        start_samples = int(custom_window[0] * fs / 1000)
        end_samples = int(custom_window[1] * fs / 1000)
    else:
        start_samples = int(window_ms[0] * fs / 1000)
        end_samples = int(window_ms[1] * fs / 1000)
    
    start_index = max(0, peak_index + start_samples)
    end_index = min(len(data), peak_index + end_samples)
    window = data[start_index:end_index]
    slopes = np.diff(window)
    lat_relative_index = np.argmin(slopes)
    return start_index + lat_relative_index

def manual_lat_selection(data, spike_index, fs, save_path):
    """Interactive plot for manual LAT window selection."""
    plt.figure(figsize=(12, 6))
    time = (np.arange(len(data)) - spike_index) / fs * 1000
    plt.plot(time, data)
    plt.title(f"Click LEFT then RIGHT boundary for LAT search window (Beat {spike_index})")
    plt.xlabel("Time from Beat (ms)")
    plt.ylabel("Amplitude (µV)")
    
    points = plt.ginput(2, timeout=-1)
    plt.close()
    
    if len(points) != 2:
        return None
    
    window = sorted([points[0][0], points[1][0]])
    np.save(save_path, window)
    return window

def get_selection_path(save_dir, condition, channel, spike_index):
    """Get path for saving/loading manual selections."""
    selection_dir = os.path.join(save_dir, "lat_selections", condition, f"channel_{channel}")
    os.makedirs(selection_dir, exist_ok=True)
    return os.path.join(selection_dir, f"spike_{spike_index}.npy")

def load_manual_window(save_dir, condition, channel, spike_index):
    """Load existing manual window if available."""
    path = get_selection_path(save_dir, condition, channel, spike_index)
    return np.load(path) if os.path.exists(path) else None



def analyze_spikes(data, spikes, fs, fpd_window, find_max):
    """Analyze detected spikes for various parameters."""
    results = []
    for i, spike in enumerate(spikes):
        start = max(0, spike - int(0.001 * fs))
        end = min(len(data), spike + int(0.001 * fs))
        segment = data[start:end]
        lat = find_lat(data, spike, fs)
        amp_peak = start + np.argmax(segment)
        results.append({
            'spike_index': spike,
            'amp_peak': amp_peak,
            'lat': lat,
        })
    
    return results

def analyze_spikes(data, spikes, fs, fpd_window, find_max, condition, save_dir, intan_channel):
    """Analyze spikes with manual LAT selection for specific channels."""
    results = []
    lat_storage_dir = os.path.join(save_dir, "LAT_data", condition, f"channel_{intan_channel}")
    os.makedirs(lat_storage_dir, exist_ok=True)
    for i, spike in enumerate(spikes):
        spike_info = {'spike_index': spike}
        
        # Manual LAT selection for specific channels
        if intan_channel in MANUAL_CHANNELS:
            selection_path = get_selection_path(save_dir, condition, intan_channel, i)
            manual_window = load_manual_window(save_dir, condition, intan_channel, i)
            
            if manual_window is None:
                # Create plot data centered on spike
                plot_start = max(0, spike - int(0.3 * fs))  # 500ms before
                plot_end = min(len(data), spike + int(0.3 * fs))  # 2000ms after
                plot_data = data[plot_start:plot_end]
                plot_spike_index = spike - plot_start
                
                # Show interactive plot
                manual_window = manual_lat_selection(
                    plot_data, plot_spike_index, fs, selection_path
                )
            
            if manual_window is not None:
                lat = find_lat(data, spike, fs, custom_window=manual_window)
            else:
                lat = spike  # Fallback to spike index
        else:
            # Automatic LAT detection
            lat = find_lat(data, spike, fs)
            
        
        
        # Calculate amplitude peak
        start = max(0, spike - int(0.001 * fs))
        end = min(len(data), spike + int(0.001 * fs))
        segment = data[start:end]
        amp_peak = start + np.argmax(segment) if find_max else start + np.argmin(segment)
        
        results.append({
            'spike_index': spike,
            'amp_peak': amp_peak,
            'lat': lat,
            'lat_time': lat/fs  # Convert sample index to seconds
        })

                # Save individual LAT data point
        np.save(os.path.join(lat_storage_dir, f"spike_{i}_lat.npy"), {
            'spike_index': spike,
            'lat_index': lat,
            'lat_time': lat/fs,
            'channel': intan_channel,
            'condition': condition
        })

    
    return results

def visualize_spike_detection(data, spikes, fs, intan_channel, condition, save_dir, detect_above, threshold_factor):
    """Visualize the signal data with detected spikes and threshold."""
    time = np.arange(len(data)) / fs
    mean = np.mean(data)
    std = np.std(data)
    threshold = mean + threshold_factor * std if detect_above else mean - threshold_factor * std
    
    plt.figure(figsize=(15, 6))
    plt.plot(time, data, label='Signal')
    plt.plot(time[spikes], data[spikes], 'ro', label='Detected Spikes')
    plt.axhline(y=threshold, color='g', linestyle='--', label='Threshold')
    
    plt.title(f'{condition} - Intan Channel {intan_channel}\nDetect {"Above" if detect_above else "Below"} Threshold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.legend()
    
    plt.grid(True)
    plt.tight_layout()
    
    # Save the figure
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'{condition}_intan_channel_{intan_channel}_spike_detection.png'))
    plt.close()

def visualize_spike_details(data, spike_info, fs, intan_channel, condition, save_dir, fpd_window, find_max):
    """Visualize details of a single spike."""
    spike = spike_info['spike_index']
    start = max(0, spike - int(0.05 * fs))
    end = min(len(data), spike + int(0.5 * fs))
    segment = data[start:end]
    time = np.arange(len(segment)) / fs * 1000
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, segment)
    amp_start_time = time[spike_info['amp_peak'] - start]
    ax.axvline(x=amp_start_time, color='r', linestyle='--', label='Peak')
    
    # Mark LAT
    lat_time = time[spike_info['lat'] - start]
    lat_value = segment[spike_info['lat'] - start]
    ax.plot(lat_time, lat_value, 'go', label='LAT')
    
    
    
    ax.set_title(f'{condition} - Intan Channel {intan_channel} - Beat at {spike/fs:.3f}s\n'
                 f'FPD Window: {fpd_window}, Find Max: {find_max}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'{condition}_intan_channel_{intan_channel}_beat_{spike}.png'))
    plt.close()



        
def process_and_visualize_detailed(data, fs, condition, save_dir, custom_settings, min_distance = 1):
    """Process data, detect spikes, analyze, and visualize results."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    all_results = []
    
    for i, channel_data in enumerate(data):
        intan_channel = INTAN_CHANNELS[i]
        channel_settings = custom_settings[intan_channel]
        fpd_window = channel_settings['fpd_window']
        find_max = channel_settings['find_max']
        detect_above = channel_settings['detect_above']
        threshold_factor = channel_settings['threshold_factor']
        
        spikes = detect_spikes(channel_data, fs, threshold_factor=threshold_factor, 
                               detect_above=detect_above, min_distance= min_distance)
        spike_analysis = analyze_spikes(channel_data, spikes, fs, fpd_window, find_max)
        
        print(f"Condition: {condition}, Intan Channel: {intan_channel}")
        print(f"Number of beats detected: {len(spikes)}")
        print(f"FPD Window: {fpd_window}, Find Max: {find_max}")
        print(f"Detect Above: {detect_above}, Threshold Factor: {threshold_factor}")
        
        # Visualize spike detection results
        visualize_spike_detection(channel_data, spikes, fs, intan_channel, condition,
                                  os.path.join(save_dir, 'beat_detection_plots'),
                                  detect_above, threshold_factor)
        
        # Visualize details for the first 3 spikes (or fewer if less than 3 spikes)
        for j, spike_info in enumerate(spike_analysis):
            visualize_spike_details(channel_data, spike_info, fs, intan_channel, condition, 
                                    os.path.join(save_dir, 'beat_details_plots'),
                                    fpd_window, find_max)
    

def process_and_visualize_detailed(data, fs, condition, save_dir, custom_settings, min_distance=1):
    """Updated processing function with manual LAT handling."""
    os.makedirs(save_dir, exist_ok=True)
    all_results = []
    
    for i, channel_data in enumerate(data):
        intan_channel = INTAN_CHANNELS[i]
        channel_settings = custom_settings[intan_channel]
        
        # Spike detection remains the same
        spikes = detect_spikes(
            channel_data, fs,
            threshold_factor=channel_settings['threshold_factor'],
            detect_above=channel_settings['detect_above'],
            min_distance=min_distance
        )
        
        # Modified analyze_spikes call
        spike_analysis = analyze_spikes(
            channel_data, spikes, fs,
            channel_settings['fpd_window'],
            channel_settings['find_max'],
            condition,
            save_dir,
            intan_channel
        )
        # Inside process_and_visualize_detailed after spike analysis
        results_df = pd.DataFrame(spike_analysis)
        results_df.to_csv(os.path.join(save_dir, f"{condition}_channel_{intan_channel}_lat_results.csv"))
        # Rest of the processing remains unchanged
        # ... (visualization and saving code)
        intan_channel = INTAN_CHANNELS[i]
        channel_settings = custom_settings[intan_channel]
        fpd_window = channel_settings['fpd_window']
        find_max = channel_settings['find_max']
        detect_above = channel_settings['detect_above']
        threshold_factor = channel_settings['threshold_factor']
        
        print(f"Condition: {condition}, Intan Channel: {intan_channel}")
        print(f"Number of spikes detected: {len(spikes)}")
        print(f"FPD Window: {fpd_window}, Find Max: {find_max}")
        print(f"Detect Above: {detect_above}, Threshold Factor: {threshold_factor}")
        
        # Visualize spike detection results
        visualize_spike_detection(channel_data, spikes, fs, intan_channel, condition,
                                  os.path.join(save_dir, 'beat_detection_plots'),
                                  detect_above, threshold_factor)
        
        # Visualize details for the first 3 spikes (or fewer if less than 3 spikes)
        for j, spike_info in enumerate(spike_analysis):
            visualize_spike_details(channel_data, spike_info, fs, intan_channel, condition, 
                                    os.path.join(save_dir, 'beat_details_plots'),
                                    fpd_window, find_max)


def main():
    drug_data, fs = load_and_process_data(CalciumMEADataPath)
    process_and_visualize_detailed(drug_data, fs, condition, 
            save_dir, CUSTOM_SETTINGS['Drug'],min_distance= 2)


    for channel in INTAN_CHANNELS:
        print(f"Channel {channel}:")
        print()

if __name__ == "__main__":
    
    main()