import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.optimize import linear_sum_assignment

def detect_spike_peaks(data, threshold):
    """
    Detect peaks in spikes in a 1D numpy array based on a single threshold.
    
    Parameters:
    - data: numpy array, the 1D array to be analyzed.
    - threshold: float, the threshold for spike detection. If positive, detects peaks above it;
                 if negative, detects peaks below it.
    
    Returns:
    - peaks: list of tuples, each tuple contains the index and value of the peak.
    """
    peaks = []
    above_threshold = data > threshold if threshold > 0 else data < threshold
    
    # Identify the start and end indices of segments where the data crosses the threshold
    segments = []
    start_idx = None
    for i, is_above in enumerate(above_threshold):
        if is_above and start_idx is None:
            start_idx = i
        elif not is_above and start_idx is not None:
            segments.append((start_idx, i - 1))
            start_idx = None
    if start_idx is not None:  # Handle case where the last segment goes till the end of the array
        segments.append((start_idx, len(data) - 1))
    
    # Find the peak within each segment
    for start, end in segments:
        segment = data[start:end+1]
        if threshold > 0:
            peak_value = np.max(segment)
        else:
            peak_value = np.min(segment)
        peak_index = np.where(segment == peak_value)[0][0] + start
        peaks.append(peak_index)
    
    return peaks

def overlay_spike_peaks(data, peaks, window_size=1000, title = None, save = False):
    """
    Overlay plots of spike peaks, aligning each peak.
    
    Parameters:
    - data: numpy array, the 1D array that was analyzed.
    - peaks: list of tuples, each tuple contains the index and value of the peak.
    - window_size: int, the number of data points to include before and after each peak.
    """
    plt.figure(figsize=(10, 6))
    
    # Number of points to plot before and after the peak
    window_1 = int(window_size*0.3)
    window_2 = window_size-window_1
    all_segement = np.zeros((len(peaks), window_size+1))
    i = 0
    peaks = list(peaks)
    for peak_index in peaks:
        # Calculate the start and end indices of the segment around the peak
        start = int(max(0, peak_index - window_1))
        end = int(min(len(data), peak_index + window_2 + 1))
        
        # Extract the segment
        segment = data[start:end]
        
        if len(segment) < window_size+1:
            continue
        all_segement[i] = segment
        i = i+1
        # Adjust indices for plotting so that the peak aligns at the center
        
        plt.plot( segment,  color='gray', alpha=0.5)
    mean_peak = np.mean(all_segement,axis=0 )
    plt.plot(mean_peak,  color='blue')
    plt.axvline(x=window_1, color='red', linestyle='--', label='Aligned Peaks')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (uV)')
    if title is not None:
        plt.title('Overlay of Spike Peaks: ' + str(title))
    else:
        plt.title('Overlay of Spike Peaks')
    #plt.legend()
    if(save):
        plt.savefig("./SpikeOverlay/Overlay"+str(title)+'.png', dpi = 600)
    plt.show()

def overlay_spike_peaks_only_mean(data, peaks, window_size=1000, title = None, save = False):
    """
    Overlay plots of spike peaks, aligning each peak.
    
    Parameters:
    - data: numpy array, the 1D array that was analyzed.
    - peaks: list of tuples, each tuple contains the index and value of the peak.
    - window_size: int, the number of data points to include before and after each peak.
    """
    plt.figure(figsize=(10, 6))
    
    # Number of points to plot before and after the peak
    window_1 = int(window_size*0.3)
    window_2 = window_size-window_1
    all_segement = np.zeros((len(peaks), window_size+1))
    i = 0
    peaks = list(peaks)
    for peak_index in peaks:
        # Calculate the start and end indices of the segment around the peak
        start = int(max(0, peak_index - window_1))
        end = int(min(len(data), peak_index + window_2 + 1))
        
        # Extract the segment
        segment = data[start:end]
        
        if len(segment) < window_size+1:
            continue
        all_segement[i] = segment
        i = i+1
        # Adjust indices for plotting so that the peak aligns at the center
        
        #plt.plot( segment,  color='gray', alpha=0.5)
    mean_peak = np.mean(all_segement,axis=0 )
    plt.plot(mean_peak,  color='black')
    #plt.axvline(x=window_1, color='red', linestyle='--', label='Aligned Peaks')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    if title is not None:
        plt.title('Mean of the Overlay of Spike Peaks: ' + str(title))
    else:
        plt.title('Mean of the Overlay of Spike Peaks')
    #plt.legend()
    if(save):
        plt.savefig("./SpikeOverlay/Mean"+str(title)+'.png', dpi = 600)
    plt.show()


def find_largest_plateau_with_peaks(change_in_peaks, peak_counts):
    """
    Identify the largest plateau where actual peaks are detected.
    
    Parameters:
    - change_in_peaks: The change in the number of detected peaks across thresholds.
    - peak_counts: The number of peaks detected at each threshold.
    
    Returns:
    - start_index: The start index of the largest plateau.
    - end_index: The end index of the largest plateau.
    """
    plateaus = []
    start_index = 0
    for i in range(1, len(change_in_peaks)):
        if change_in_peaks[i] != change_in_peaks[i - 1] or peak_counts[i] == 0:
            if start_index < i - 1:
                plateaus.append((start_index, i - 1, i - 1 - start_index))
            start_index = i
    # Include the last plateau
    if start_index < len(change_in_peaks) - 1 and peak_counts[-1] != 0:
        plateaus.append((start_index, len(change_in_peaks) - 1, len(change_in_peaks) - 1 - start_index))
    
    # Sort plateaus by size (width) and then by starting index to find the largest, earliest plateau
    plateaus.sort(key=lambda x: (x[2], x[0]), reverse=True)
    
    if plateaus:
        largest_plateau = plateaus[0]
        return largest_plateau[0], largest_plateau[1]
    else:
        return 0, 0  # Default to the first threshold if no plateaus are found

def auto_threshold_detect_spike_peaks(data, plot=False):
    threshold_range = np.linspace(-300, 300, 601)
    peak_counts = []

    for threshold in threshold_range:
        peaks = detect_spike_peaks(data, threshold)
        peak_counts.append(len(peaks))

    changes = np.diff(peak_counts)
    plateau_start, plateau_end = find_largest_plateau_with_peaks(changes, peak_counts)

    optimal_threshold_index = (plateau_start + plateau_end) // 2
    optimal_threshold = threshold_range[optimal_threshold_index]

    peaks = detect_spike_peaks(data, optimal_threshold)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(threshold_range[:-1], changes, label='Change in Peak Count')
        plt.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Optimal Threshold: {optimal_threshold}')
        plt.xlabel('Threshold')
        plt.ylabel('Change in Detected Peaks')
        plt.title('Change in Detected Peaks vs. Threshold')
        plt.legend()
        plt.show()

    return peaks

def plotISI(peaks, title = None, save = False):
    isi = np.diff(peaks) # Calculate ISIs
    isi = isi[np.where(isi>0)]
    # Setting up the figure and axes for the plot
    plt.figure(figsize=(10, 6))

    # Create the KDE plot
    sns.kdeplot(isi, bw_adjust=0.5, fill=True, alpha=0.5, linewidth=2, color='blue')

    # Create the bar plot on the same axes
    counts, bins, bars = plt.hist(isi, bins=10, density=True, alpha=0.3, color='orange')

    # Adding labels and title
    if title is not None:
        plt.title('ISI Distribution with KDE: '+ str(title))
    else:
        plt.title('ISI Distribution with KDE')
    
    plt.xlabel('Interval Duration (ms)')
    plt.ylabel('Density')

    # Show the plot
    
    if(save):
        plt.savefig("./SpikeOverlay/ISI"+str(title)+'.png', dpi = 600)
    else:
        plt.show()
    return isi

def find_peak(x, a, b, max=1):
    # find the index of the max or min value in a 1d array at a:b
    # Return the index
    # Check if the range is valid
    if a < 0 or b >= len(x) or a > b:
        return -1  # Return -1 or some error indication for invalid range

    # Initialize the index and value of the peak
    peak_index = a
    peak_value = x[a]

    # Iterate over the range a to b
    for i in range(a + 1, b + 1):
        # Check if we are finding max or min
        if (max == 1 and x[i] > peak_value) or (max != 1 and x[i] < peak_value):
            peak_value = x[i]
            peak_index = i

    return peak_index

def detect_peak(x, sd=2, multi = 1):
    peaks = []
    std_dev = np.std(x)  # Calculate standard deviation

    # Iterate over the array, skipping the first and last element
    for i in range(1, len(x) - 1):
        # Check if the current element is a local maxima and exceeds the threshold
        if x[i]*multi > x[i - 1]*multi and x[i]*multi > x[i + 1]*multi and x[i]*multi > sd * std_dev:
            peaks.append(i)

    return peaks

def find_peak_based_on_reference_channel(data, reference_index, max = [1,1,-1,1,1,1,1,1,-1,-1,1,1], peak_range = (200,200), sd=2, multi = 1, reference_peak = None):
    """Find peaks in multiple channels based on a reference channel's peaks.

    Args:
        data (numpy.ndarray): Multi-channel time series data with shape (num_channels, time_points).
        reference_index (int): Index of the reference channel to use for initial peak detection.
        max (list, optional): List indicating peak direction for each channel (1 for maximum, -1 for minimum). 
            Defaults to [1,1,-1,1,1,1,1,1,-1,-1,1,1].
        peak_range (tuple, optional): Search window around reference peaks (samples_before, samples_after). 
            Defaults to (200,200).
        sd (int, optional): Standard deviation threshold for peak detection. Defaults to 2.
        multi (int, optional): Multiplier for peak detection threshold. Defaults to 1.
        reference_peak (numpy.ndarray, optional): Pre-defined reference peaks to use instead of detecting them. 
            Defaults to None.

    Returns:
        numpy.ndarray: Array of detected peaks with shape (num_channels, num_peaks) where each element 
            represents the time index of the peak in that channel.

    Notes:
        For each peak detected in the reference channel, the function searches within the specified 
        peak_range window in all other channels to find corresponding peaks. The direction of the peak 
        search (maximum or minimum) is determined by the max parameter for each channel.
    """
    peaks = detect_peak(data[reference_index], sd=sd, multi = multi)
    #print(peaks)
    if reference_peak is not None:
        peaks = reference_peak
    #peaks = auto_threshold_detect_spike_peaks(data[reference_index], plot=1)
    all_peak = np.zeros((data.shape[0],len(peaks)))
    j = 0 # Channel index
    k = -1 # Peak index
    for peak in peaks:
        j = 0
        k = k+1
        for i in range(data.shape[0]):
            all_peak[j,k] = find_peak(data[i], peak - peak_range[0], peak+peak_range[1],max[j] )
            j = j+1
    return all_peak

def find_peak_duration_simple(x, peak, threshold=0.2, detailed=False):
    peak = int(peak)
    positive = x[peak] / np.abs(x[peak])
    bar = np.abs(x[peak]) * threshold
    peak_start, peak_end = 0, x.shape[0] - 1  # Default values in case the loop doesn't find start/end

    for i in range(peak):
        if x[peak - i] * positive > bar and x[peak - i - 1] * positive <= bar:
            peak_start = peak - i
            break  # Exit the loop once the start is found

    for i in range(x.shape[0] - peak):
        if x[peak + i-1] * positive > bar and x[peak + i] * positive <= bar:
            peak_end = peak + i
            break  # Exit the loop once the end is found

    peak_duration = peak_end - peak_start


    # Return more detailed information if requested
    return peak_duration, peak_start, peak_end

def plot_peak_duration(x, peak_duration, peak_start, peak_end, window = 1000, start = 0.3):
    # Plot the waveform
    start_point = int(peak_start-window*start)
    end_point = int(start_point + window)
    peak_start = int(peak_start)
    peak_end = int(peak_end)
    plt.figure(figsize=(10, 6))
    plt.plot(x[start_point: end_point], label='Waveform')
    
    # Highlight the duration of the peak
    plt.axvspan(peak_start-start_point, peak_end-start_point, color='red', alpha=0.3, label='Peak Duration')
    
    # Mark the start and end of the peak
    plt.plot(peak_start-start_point, x[peak_start], 'go', label='Peak Start')  # Green dot at peak start
    plt.plot(peak_end-start_point, x[peak_end], 'ro', label='Peak End')  # Red dot at peak end
    
    # Annotating the peak duration
    #plt.text((peak_start + peak_end) / 2, max(x), f'Peak Duration: {peak_duration}', 
    #         horizontalalignment='center', verticalalignment='bottom')
    
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.title('Waveform and Peak Duration: '+str(peak_duration)+" (ms)")
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_peak_durations(peak_duration, hist = True):
    n_channels, n_peaks = peak_duration.shape
    
    # 1. Descriptive Statistics
    for i in range(n_channels):
        channel_durations = peak_duration[i, :]
        mean_duration = np.mean(channel_durations)
        median_duration = np.median(channel_durations)
        std_deviation = np.std(channel_durations)
        print(f"Channel {i+1}: Mean = {mean_duration:.2f}, Median = {median_duration}, Std Dev = {std_deviation:.2f}")
    
    # 2. Histograms
    if(hist):
        fig, axs = plt.subplots(n_channels, 1, figsize=(10, 6 * n_channels))
        for i in range(n_channels):
            axs[i].hist(peak_duration[i, :], bins=20, alpha=0.7, label=f'Channel {i+1}')
            axs[i].set_title(f'Peak Duration Distribution for Channel {i+1}')
            axs[i].set_xlabel('Duration')
            axs[i].set_ylabel('Frequency')
            axs[i].legend()
        plt.tight_layout()
        plt.show()

    # 3. Density Plots
    plt.figure(figsize=(10, 6))
    for i in range(n_channels):
        sns.kdeplot(peak_duration[i, :], label=f'Channel {i+1}')
    plt.title('Density Plot of Peak Durations')
    plt.xlabel('Duration')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # 4. Box Plots
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=peak_duration.T)
    plt.title('Box Plot of Peak Durations for Each Channel')
    plt.xlabel('Channel')
    plt.ylabel('Duration')
    plt.xticks(ticks=np.arange(n_channels), labels=[f'Channel {i+1}' for i in range(n_channels)])
    plt.show()

def find_peak_duration_cardiac(x, peak, threshold=0.2,threshold_2 = 0.1 ,detailed=False):
    peak = int(peak)

    positive = x[peak] / np.abs(x[peak])
    bottom = np.min(x*positive)
    bar = np.abs(x[peak]) * threshold
    bar_2 = np.abs(bottom) * threshold_2
    peak_start, peak_end = 0, x.shape[0] - 1  # Default values in case the loop doesn't find start/end

    for i in range(peak):
        if x[peak - i] * positive > bar and x[peak - i - 1] * positive <= bar:
            peak_start = peak - i
            break  # Exit the loop once the start is found

    for i in range(x.shape[0] - peak):
        if x[peak + i-1] * positive < -bar_2 and x[peak + i] * positive >= -bar_2:
            peak_end = peak + i
            break  # Exit the loop once the end is found

    peak_duration = peak_end - peak_start


    # Return more detailed information if requested
    return peak_duration, peak_start, peak_end

def peak_rank(peaks):
    # Calculate ranks for each position. The 'argsort' function sorts each column (position),
    # then 'argsort' again on the sorted indices to get ranks. Adding 1 so that ranks start from 1 instead of 0.
    ranks = np.argsort(np.argsort(peaks, axis=0), axis=0) + 1

    # Plotting
    plt.figure(figsize=(12, 8))
    for i in range(ranks.shape[0]):
        plt.plot(ranks[i, :], label=f'Channel {i+1}')

    plt.title('Ranking of Channels by Peak Times Across Positions')
    plt.xlabel('Position')
    plt.ylabel('Rank (1 = Earliest Peak)')
    plt.xticks(np.arange(19), np.arange(1, 20))
    plt.legend()
    plt.grid(True)
    plt.show()
    return ranks

def peak_rank_time(peaks,excluded_indices = []  ):
    # Calculate relative latency for each position
    # Subtract the minimum peak time at each position from each channel's peak time
    relative_latency = peaks - np.min(peaks, axis=0)


    # Assuming my_array is your original lis
    wanted_peak = np.delete(peaks, excluded_indices, axis=0)
    relative_latency = peaks - np.min(wanted_peak, axis=0)

    # Plotting
    plt.figure(figsize=(12, 8))
    for i in range(relative_latency.shape[0]):
        if i in [0,6]:
            continue
        plt.plot(relative_latency[i, :], label=f'Channel {i+1}')

    plt.title('Relative Latency of Peaks Across Channels and Positions')
    plt.xlabel('Position')
    plt.ylabel('Relative Latency (ms)')
    plt.xticks(np.arange(19), np.arange(1, 20))
    plt.legend()
    plt.grid(True)
    plt.show()
    return relative_latency

# def rank_confusion_matrix(peaks):
#     # Assuming all_peak is defined elsewhere in your code with shape (12, 19)
#     all_peak = peaks
#     num_spikes = all_peak.shape[1]
#     num_channels = all_peak.shape[0]

#     # Initialize the matrix to hold the firing order of channels for each spike
#     firing_order_matrix = np.zeros((num_spikes, num_channels), dtype=int)

#     # For each spike, find the channel that fires first, then second, and so on
#     for spike in range(num_spikes):
#         firing_order = np.argsort(all_peak[:, spike])
#         firing_order_matrix[spike, :] = firing_order

#     confusion_matrix_like = np.zeros((num_channels, num_channels))

#     for i in range(num_spikes):
#         for j in range(num_channels):
#             channel = firing_order_matrix[i, j]
#             confusion_matrix_like[j, channel] += 1

#     # Plotting the "confusion matrix"
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(confusion_matrix_like, annot=True, cmap="YlGnBu", fmt="g", 
#                 xticklabels=[f"Channel {i+1}" for i in range(num_channels)], 
#                 yticklabels=[f"Order {i+1}" for i in range(num_channels)])
#     plt.title('Channel Firing Order Across Spikes')
#     plt.xlabel('Channels')
#     plt.ylabel('Firing Order')
#     plt.show()

#     # Assuming confusion_matrix_like is defined as before

#     # Convert the problem to a maximization problem
#     cost_matrix = confusion_matrix_like.max() - confusion_matrix_like

#     # Apply the Hungarian algorithm to find the optimal assignment
#     row_ind, col_ind = linear_sum_assignment(cost_matrix)

#     # Create a new sorted confusion matrix
#     sorted_confusion_matrix = confusion_matrix_like[:, col_ind]

#     # You can also sort the rows based on the optimal assignment to maintain consistency
#     sorted_confusion_matrix = sorted_confusion_matrix[row_ind, :]
#     accuracy = np.trace(sorted_confusion_matrix) / np.sum(sorted_confusion_matrix)
#     # Plotting the sorted "confusion matrix"
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(sorted_confusion_matrix, annot=True, cmap="YlGnBu", fmt="g", 
#                 xticklabels=[f"Channel {i+1}" for i in col_ind], 
#                 yticklabels=[f"Order {i+1}" for i in row_ind])
#     plt.title('Sorted Channel Firing Order Across Spikes: 2nd orgnanoid. Accuracy = '+ str(accuracy))
#     plt.xlabel('Channels')
#     plt.ylabel('Firing Order')
#     plt.show()
#     return sorted_confusion_matrix

def overlay_different_spikes(peaks, data, target_electrodes = None, window_size=1000, target_spike = 1):
    if target_electrodes != None:
        print('Choosing Target Electrodes is still developing')
        return 
    plt.figure(figsize=(10, 6))
    
    # Number of points to plot before and after the peak
    window_1 = int(window_size*0.3)
    window_2 = window_size-window_1
    all_segement = np.zeros((len(peaks), window_size+1))
    i = 0
    peaks = list(peaks)
    for peak_index in peaks:
        # Calculate the start and end indices of the segment around the peak
        start = int(max(0, peak_index - window_1))
        end = int(min(len(data), peak_index + window_2 + 1))
        
        # Extract the segment
        segment = data[start:end]
        if len(segment) < window_size+1:
            continue
        all_segement[i] = segment
        i = i+1
        # Adjust indices for plotting so that the peak aligns at the center
        
        plt.plot( segment,  color='gray', alpha=0.5)
    mean_peak = np.mean(all_segement,axis=0 )
    plt.plot(mean_peak,  color='blue')
    plt.axvline(x=window_1, color='red', linestyle='--', label='Aligned Peaks')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Overlay of Spike Peaks')
    #plt.legend()
    plt.show()

    return


def overlay_different_spikes(peaks, data, leave_out = None, window_size=800, proportion = 0.4, target_spike = 1, stair = False, step = 40): 
    if leave_out != None:
        data = np.delete(data, leave_out, axis=0)
    plt.figure(figsize=(10, 6))
    #data_return = data.copy  
    if stair == True:     
        for i in range(data.shape[0]):
            data[i] = data[i]-step*i
    start = int(np.min(peaks[:,target_spike]) - window_size* proportion)
    print(start)
    plt.plot(data[:,start:start + window_size].T)
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (uV)')
    plt.title('Overlay of Spike Peaks')
    #plt.legend()
    plt.show()

    return data[:,start:start + window_size].T

def calculate_neo(signal):
    # Calculate the Nonlinear Energy Operator (NEO)
    neo = np.zeros(len(signal))
    for n in range(1, len(signal) - 1):
        neo[n] = signal[n]**2 - signal[n-1] * signal[n+1]
    return neo


def detect_beats(neo_signal, threshold, min_distance=1000):
    """
    Detect peaks in the NEO signal that are above a certain threshold,
    ensuring each detected peak is separated by at least min_distance samples.
    
    Parameters:
    - neo_signal: The NEO signal array.
    - threshold: The threshold value for peak detection.
    - min_distance: Minimum number of samples between successive peaks.
    
    Returns:
    - A list of indices corresponding to the detected peaks.
    - A list of NEO signal values at the detected peaks.
    """
    peak_indices = []
    peak_values = []  # Store peak values here
    last_peak = -min_distance  # Initialize to ensure the first peak is accepted if it meets the criteria
    
    for i, value in enumerate(neo_signal):
        if value > threshold and i - last_peak >= min_distance:
            peak_indices.append(i)
            peak_values.append(value)  # Append the peak value
            last_peak = i  # Update the last accepted peak position

    return peak_indices



def calculate_intervals(peaks, sampling_rate):
    # Calculate intervals in seconds between peaks
    intervals = np.diff(peaks) / sampling_rate
    return intervals

def process_electrodes(target_data, sampling_rate, plot = 0, threshold_val = [1,2]):
    beat_frequencies = []
    peaks_all = []
    for i in range(target_data.shape[0]):
        # Calculate NEO
        neo = calculate_neo(target_data[i])

        # Define threshold (this could be adaptive or fixed)
        threshold = np.mean(neo)* threshold_val[0] + threshold_val[1] * np.std(neo)  # Example threshold

        # Detect beats
        peaks = detect_beats(neo, threshold= threshold)
        peaks_all.append(peaks)
        # Calculate intervals and derive frequency
        intervals = calculate_intervals(peaks, sampling_rate)
        if len(intervals) > 0:
            # Calculate mean frequency (beats per second)
            mean_frequency = 1 / np.mean(intervals)
        else:
            mean_frequency = 0

        beat_frequencies.append(mean_frequency*60)

        # Optional: Plot NEO signal with detected peaks
        if plot:
            plt.figure()
            plt.plot(neo, label='NEO Signal')
            plt.plot(peaks, neo[peaks], 'rx', label='Detected Peaks')
            plt.title(f'Electrode {i+1} NEO and Detected Peaks')
            plt.legend()
            plt.show()

    return beat_frequencies, peaks_all


def process_electrodes_windowed(target_data, sampling_rate):
    window_size = 15 * sampling_rate  # 5 seconds window
    bpm_over_time = []

    for i in range(target_data.shape[0]):  # For each electrode
        electrode_bpm = []
        for start in range(0, target_data.shape[1], window_size):
            end = start + window_size
            segment = target_data[i, start:end]

            # Calculate NEO for the segment
            neo = calculate_neo(segment)

            # Threshold for peak detection, may need adjustment
            threshold = np.mean(neo) + 2 * np.std(neo)
            min_distance = int(sampling_rate * 0.2)  # Adjust based on your data
            
            # Detect peaks
            peaks = detect_beats(neo, threshold, min_distance)

            # Calculate BPM for the segment
            beats = len(peaks)
            bpm = (beats / (window_size / sampling_rate)) * 60  # Convert beats per window to beats per minute
            electrode_bpm.append(bpm)

        bpm_over_time.append(electrode_bpm)
    
    return bpm_over_time


def overlay_spike_peaks_multiple_second_delay(datas, peakses, window_size=1000, plot_all=1, titles=None, stages=None):
    plt.figure(figsize=(10, 6))
    text_obj = None  # Initialize a variable to keep track of the text object

    # Determine the starting index for the 4th stage
    # stages list is 1-indexed in context, but Python is 0-indexed, so "4" for the 4th stage means "3" in Python indexing
    start_stage_idx = 3  # This is for the 4th stage, adjust accordingly if you want to start from a different stage

    for idx, (data, peaks) in enumerate(zip(datas, peakses), start=1):
        # Skip stages before the 4th
        if idx < stages[start_stage_idx]:
            continue

        data = data[0]
        peaks = list(peaks[0])

        window_1 = int(window_size * 0.3)
        window_2 = window_size - window_1
        all_segment = np.zeros((len(peaks), window_size + 1))

        for i, peak_index in enumerate(peaks):
            start = int(max(0, peak_index - window_1))
            end = int(min(len(data), peak_index + window_2 + 1))
            segment = data[start:end]

            if len(segment) < window_size + 1:
                continue
            all_segment[i] = segment

            if plot_all:
                plt.plot(segment, color='gray', alpha=0.5)

        mean_peak = np.mean(all_segment, axis=0)
        plt.plot(mean_peak, label=str(idx))

        # Determine the current stage and prepare the title with time information
        current_stage_index = next((j for j, stage in enumerate(stages) if idx < stage), len(stages) - 1)
        previous_stage = stages[current_stage_index - 1] if current_stage_index > 0 else 1
        time_diff = idx - previous_stage + 1
        current_title = titles[current_stage_index] + " - Time: {} mins".format(time_diff)

        # Display the new title as text on the plot
        #text_obj = plt.text(0.5, 0.95, current_title, ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.pause(1)  # Pause for 1 second before plotting the next

    plt.axvline(x=window_1, color='red', linestyle='--', label='Aligned Peaks')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Overlay of Spike Peaks')
    plt.legend()
    plt.show()


def overlay_spike_peaks_multiple_gray_area_mean(datas, peakses, window_size=1000, plot_all=1, titles=None, stages=None):
    plt.figure(figsize=(10, 6))

    titles = titles[:4]
    stages = stages[:4] if len(stages) > 4 else stages

    stage_segments = {title: [] for title in titles}

    for idx, (data, peaks) in enumerate(zip(datas, peakses), start=1):
        current_stage_index = next((i for i, stage in enumerate(stages + [float('inf')]) if idx < stage), len(stages))
        
        if current_stage_index >= 4:
            continue

        current_stage_title = titles[current_stage_index]
        data = data[0]
        peaks = list(peaks[0])

        window_1 = int(window_size * 0.3)
        window_2 = window_size - window_1
        all_segment = np.zeros((len(peaks), window_size + 1))

        for i, peak_index in enumerate(peaks):
            start = max(0, peak_index - window_1)
            end = min(len(data), peak_index + window_2 + 1)
            segment = data[start:end]

            if len(segment) < window_size + 1:
                continue
            all_segment[i] = segment

        mean_peak = np.mean(all_segment, axis=0)
        stage_segments[current_stage_title].append(mean_peak)

    for title, segments in stage_segments.items():
        if segments:
            segments = np.array(segments)
            mean_segments = np.mean(segments, axis=0)
            std_segments = np.std(segments, axis=0)

            # Plot and get the color of the line
            line = plt.plot(mean_segments, label=title)
            color = line[0].get_color()

            # Use the same color for the fill_between area
            plt.fill_between(range(len(mean_segments)), mean_segments - std_segments, mean_segments + std_segments, color=color, alpha=0.5)

    plt.axvline(x=window_1, color='red', linestyle='--', label='Aligned Peaks')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Overlay of Spike Peaks by Stage (First Four Stages)')
    plt.legend()
    plt.savefig('overlay_gray.png', dpi = 600)
    plt.show()


from scipy.optimize import linear_sum_assignment

def rank_confusion_matrix(peaks):
    # Assuming peaks is defined elsewhere in your code with shape (number of channels, number of spikes)
    num_spikes = peaks.shape[1]
    num_channels = peaks.shape[0]

    # Initialize the matrix to hold the firing order of channels for each spike
    firing_order_matrix = np.zeros((num_spikes, num_channels), dtype=int)

    # For each spike, find the channel that fires first, then second, and so on
    for spike in range(num_spikes):
        firing_order = np.argsort(peaks[:, spike])
        firing_order_matrix[spike, :] = firing_order

    confusion_matrix_like = np.zeros((num_channels, num_channels))

    # Build the matrix where rows are firing orders and columns are channels
    for i in range(num_spikes):
        for j in range(num_channels):
            channel = firing_order_matrix[i, j]
            confusion_matrix_like[j, channel] += 1

    # Plotting the "confusion matrix"
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix_like, annot=True, cmap="YlGnBu", fmt="g",
                xticklabels=[f"Channel {i+1}" for i in range(num_channels)],
                yticklabels=[f"Order {i+1}" for i in range(num_channels)])
    plt.title('Channel Firing Order Across Spikes')
    plt.xlabel('Channels')
    plt.ylabel('Firing Order')
    plt.show()

    # Convert the problem to a maximization problem for Hungarian algorithm
    cost_matrix = confusion_matrix_like.max() - confusion_matrix_like

    # Apply the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create a new sorted confusion matrix based on column assignment
    sorted_confusion_matrix = confusion_matrix_like[:, col_ind]

    # Accuracy based on sorted columns and their original firing orders
    accuracy = np.trace(sorted_confusion_matrix) / np.sum(sorted_confusion_matrix)

    # Plotting the sorted "confusion matrix"
    plt.figure(figsize=(10, 8))
    sns.heatmap(sorted_confusion_matrix, annot=True, cmap="YlGnBu", fmt="g",
                xticklabels=[f"Channel {col+1}" for col in col_ind], 
                yticklabels=[f"Order {i+1}" for i in range(num_channels)])
    plt.title(f'Sorted Channel Firing Order Across Spikes: Accuracy = {accuracy:.2f}')
    plt.xlabel('Channels')
    plt.ylabel('Firing Order')
    plt.show()

    return sorted_confusion_matrix, col_ind  # Returning both the matrix and the sorted indices


def find_slop(x, a, b, max=1):
    # find the index of the max or min value in a 1d array at a:b
    # Return the index
    # Check if the range is valid
    if a < 0 or b >= len(x) or a > b:
        #print("Error when trying to find the slop",a,b)
        return -1  # Return -1 or some error indication for invalid range
    a = int(a)
    b = int(b)
    # Initialize the index and value of the peak
    peak_index = a
    peak_value = x[a] - x[a+1]

    # Iterate over the range a to b
    for i in range(a + 1, b + 1):
        # Check if we are finding max or min
        if (x[i] - x[i-1] < peak_value):
            peak_value = x[i] - x[i-1]
            peak_index = i

    return peak_index
def find_activation_slope(data, reference_index, peak_range = 
                          (200,200), sd=2, multi = 1, reference_peak = None):
    # In shape of #channels * # peaks
    peaks = detect_peak(data[reference_index], sd=sd, multi = multi)
    #print(peaks)
    if reference_peak is not None:
        peaks = reference_peak
    #peaks = auto_threshold_detect_spike_peaks(data[reference_index], plot=1)
    all_peak = np.zeros((data.shape[0],len(peaks)))
    j = 0 # Channel index
    k = -1 # Peak index
    for peak in peaks:
        j = 0
        k = k+1
        for i in range(data.shape[0]):
            all_peak[j,k] = find_slop(data[i], peak - peak_range[0], peak+peak_range[1] )
            j = j+1
    return all_peak

