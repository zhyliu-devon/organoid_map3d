import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level from the script directory
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
print(parent_dir)
sys.path.append(os.path.abspath(parent_dir))

configurations = [
    {
        "data_path": os.path.join(
                            parent_dir,
                            "ExampleData",
                            "FY CO 05062024 Si 500 micron unfolded_240515_141638.rhs"
                            ),
        "wanted_channel": list(range(0,16)),
        "target_indeces": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15],
        "plot_ind": 13,
        "window_before": 0.40,
        "window_after": 0.40,
        "flatdata": 1
    },
    {
        "data_path": os.path.join(
                            parent_dir,
                            "ExampleData",
                            "JK CO6 day after calcium imaging_240522_124841.rhs"
                            ),
        "wanted_channel": list(range(0,16)),
        "target_indeces": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "plot_ind": 0,
        "window_before": 0.40,
        "window_after": 0.40,
        "flatdata": 0
    }
]

import numpy as np
import matplotlib.pyplot as plt

from electrophysiology_mapping import preprocessing
import pandas as pd



def select_points_windowed(data, channel_index, target_indeces, sampling_rate=10000, window_duration=5):
    """
    Interactive function to select individual points from a specific channel in windowed segments.
    """
    total_samples = data.shape[1]
    window_samples = window_duration * sampling_rate
    selected_points = []
    scatter_plots = []

    def onclick(event):
        if event.inaxes is not None:
            x = event.xdata
            y = data[channel_index][int(x * sampling_rate)]
            selected_points.append(x)
            scatter = ax.scatter(x, y, color='red')
            scatter_plots.append(scatter)
            print(f"Point selected: {x:.2f} at amplitude {y:.2f}")
            fig.canvas.draw()

    def on_key(event):
        nonlocal start_sample, end_sample

        if event.key == 'backspace' and selected_points:
            removed_point = selected_points.pop()
            print(f"Point removed: {removed_point:.2f}")
            scatter = scatter_plots.pop()
            scatter.remove()
            fig.canvas.draw()
        elif event.key == 'enter':
            start_sample += window_samples
            end_sample += window_samples
            if start_sample >= total_samples:
                print("End of data reached.")
                return
            update_plot()

    def update_plot():
        while scatter_plots:
            scatter = scatter_plots.pop()
            scatter.remove()

        ax.clear()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Channel {target_indeces[channel_index]} - Window: {start_sample/sampling_rate:.2f}s to {end_sample/sampling_rate:.2f}s\nClick to select points, Backspace to remove last point, Enter to move to next window')
        
        time_window = np.arange(start_sample, min(end_sample, total_samples)) / sampling_rate
        ax.plot(time_window, data[channel_index][start_sample:end_sample])
        fig.canvas.draw()

    fig, ax = plt.subplots(figsize=(15, 6))
    start_sample = 0
    end_sample = window_samples
    update_plot()

    cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.show()
    
    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)

    return selected_points

def snr_analysis(data, selected_points, target_indeces, sampling_rate=10000, 
                 window_before=0.15, window_after=0.35, channel_to_plot=None, save_folder=None):
    """
    Perform SNR analysis using signal amplitude / noise std deviation,
    including SNR calculation for all channels, SNR bar plot, and signal plot for a specific channel.
    """
    def calculate_snr(signal, noise):
        signal_amplitude = np.max(signal) - np.min(signal)
        noise_std = np.std(noise)
        snr = signal_amplitude / noise_std
        return snr

    num_channels, total_samples = data.shape
    window_before_samples = int(window_before * sampling_rate)
    window_after_samples = int(window_after * sampling_rate)
    
    results = {}
    for i, channel in enumerate(target_indeces):
        signal_mask = np.zeros(total_samples, dtype=bool)
        for point in selected_points:
            start = max(0, int(point * sampling_rate) - window_before_samples)
            end = min(total_samples, int(point * sampling_rate) + window_after_samples)
            signal_mask[start:end] = True
            
        signal = data[i][signal_mask]
        noise = data[i][~signal_mask]
        snr = calculate_snr(signal, noise)
        results[channel] = {"SNR": snr}

    # Plot SNR results for all channels
    channels = list(results.keys())
    snr_values = [results[ch]["SNR"] for ch in channels]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(channels))
    width = 0.35
    ax.bar(x + width/2, snr_values, width, label='SNR')
    ax.set_xlabel('Channel')
    ax.set_ylabel('SNR (amplitude/std)')
    ax.set_title('SNR Across Channels')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Ch {i}' for i in channels])
    ax.legend()
    plt.grid()
    plt.tight_layout()
    
    if save_folder:
        snr_plot_filename = os.path.join(save_folder, 'snr_across_channels.png')
        plt.savefig(snr_plot_filename)
        print(f"Saved SNR plot to {snr_plot_filename}")
    plt.show()

    # Plot specific channel data if requested
    if channel_to_plot is not None:
        channel_index = target_indeces.index(channel_to_plot)
        channel_data = data[channel_index]
        time = np.arange(len(channel_data)) / sampling_rate
        
        plt.figure(figsize=(15, 6))
        plt.plot(time, channel_data, 'b', label='Channel Data')
        for point in selected_points:
            start = max(0, int(point * sampling_rate) - window_before_samples)
            end = min(len(channel_data), int(point * sampling_rate) + window_after_samples)
            plt.axvspan(start/sampling_rate, end/sampling_rate, color='red', alpha=0.3)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'Channel {channel_to_plot} Data with Signal Areas (In red shadow)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if save_folder:
            channel_plot_filename = os.path.join(save_folder, f'channel_{channel_to_plot}_data.png')
            plt.savefig(channel_plot_filename)
            print(f"Saved channel {channel_to_plot} plot to {channel_plot_filename}")
        plt.show()

    return results

# ... (keep all the function definitions as they were)

def select_or_load_points(target_data_copy, channel_index, target_indeces, sampling_rate, npy_filename):
    """
    Either load existing points or select new ones and save them.
    """
    if os.path.exists(npy_filename):
        user_input = input(f"File {npy_filename} already exists. Do you want to select points again? (y/n) (Press n if you are running a demo to see the original segement used): ")
        if user_input.lower() != 'y':
            selected_points = np.load(npy_filename)
            print(f"Loaded existing selected points from {npy_filename}")
            return selected_points
    
    selected_points = select_points_windowed(target_data_copy, channel_index=channel_index, target_indeces=target_indeces, sampling_rate=sampling_rate)
    np.save(npy_filename, selected_points)
    print(f"Saved new selected points to {npy_filename}")
    return selected_points

if __name__ == "__main__":
    sampling_rate = 10000
    cutoff_frequency = 1000

    for config in configurations:
        data_path = config["data_path"]
        wanted_channel = config["wanted_channel"]
        target_indeces = config["target_indeces"]
        plot_ind = config["plot_ind"]
        window_before = config["window_before"]
        window_after = config["window_after"]
        flatdata = config["flatdata"]

        processed_data = preprocessing.processingPipline(data_path, data_path, cutoff=cutoff_frequency, fs_new=sampling_rate, one_address=1)
        available_data = processed_data[wanted_channel]
        target_data = available_data[target_indeces]
        target_data_copy = target_data.copy()

        # Create 'selected_spikes' folder if it doesn't exist
        base_folder = 'E:\Desktop\omipolar\DataPlotting\SNRCalculation\selected_spikes'
        folder_name = 'flatdata' if flatdata else 'foldeddata'
        selected_spikes_folder = os.path.join(base_folder, folder_name)
        os.makedirs(selected_spikes_folder, exist_ok=True)

        # Create 'selected_points' folder
        selected_points_folder = os.path.join(base_folder, 'selected_points')
        os.makedirs(selected_points_folder, exist_ok=True)

        # Generate filename for selected points
        base_filename = os.path.basename(data_path).replace('.rhs', '')
        npy_filename = os.path.join(selected_points_folder, f"{base_filename}.npy")

        # Create a subfolder for other output files
        output_subfolder = os.path.join(selected_spikes_folder, base_filename)
        os.makedirs(output_subfolder, exist_ok=True)

        # Select or load points

        # Select or load points 
        selected_points = select_or_load_points(target_data_copy, plot_ind, target_indeces, sampling_rate, npy_filename) 

        results = snr_analysis( 
            data=target_data_copy, 
            selected_points=selected_points, 
            target_indeces=target_indeces, 
            sampling_rate=sampling_rate, 
            window_before=window_before, 
            window_after=window_after, 
            channel_to_plot=target_indeces[plot_ind], 
            save_folder=output_subfolder 
        ) 

        # Prepare data for Excel 
        excel_data = [] 
        for channel, snr in results.items(): 
            excel_data.append({ 
                'Channel': channel, 
                'SNR': snr['SNR']  # Updated to use single SNR value
            }) 

        # Create DataFrame 
        df = pd.DataFrame(excel_data) 

        # Generate Excel filename 
        excel_filename = os.path.join(output_subfolder, f"{base_filename}.xlsx") 

        # Save to Excel 
        df.to_excel(excel_filename, index=False) 
        print(f"Saved results to {excel_filename}") 

        # Print results to console 
        for channel, snr in results.items(): 
            print(f"Channel {channel}: SNR = {snr['SNR']:.2f}")  # Updated to show single SNR value

        print(f"Finished processing {base_filename}") 
        print("-" * 50)

