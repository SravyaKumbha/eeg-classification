import os
import pyedflib
import numpy as np

def extract_channels(input_folder, output_folder, channels):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".edf"):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename)
            
            f = pyedflib.EdfReader(input_filepath)
            available_channels = f.getSignalLabels()
            print(f"Available channels in {filename}: {available_channels}")
            
            try:
                channel_indices = [available_channels.index(channel) for channel in channels]
            except ValueError as e:
                print(f"Error: {e}. Skipping file {filename}.")
                f._close()
                continue
            
            extracted_data = [f.readSignal(idx) for idx in channel_indices]
            extracted_data = np.array(extracted_data)
            
            f._close()
            
            hdl = pyedflib.EdfWriter(output_filepath, len(channels), f.filetype)
            hdl.setSignalHeaders([f.getSignalHeader(idx) for idx in channel_indices])
            hdl.writeSamples(extracted_data)
            hdl.close()

def main():
    seizure_input_folder = r"C:\Users\KSUDHEER\Desktop\project2\SEIZURE_EXTRACTION"
    non_seizure_input_folder = r"C:\Users\KSUDHEER\Desktop\project2\NON_SEIZURE_EXTRACTION"
    seizure_output_folder = r"C:\Users\KSUDHEER\Desktop\project2\CMN_CHNLS_SEIZURE"
    non_seizure_output_folder = r"C:\Users\KSUDHEER\Desktop\project2\CMN_CHNLS_NON_SEIZURE"
    
    channels = [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", 
        "F3-C3", "C3-P3", "P3-O1", "FZ-CZ", "CZ-PZ", 
        "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", 
        "F8-T8", "T8-P8", "P8-O2"
    ]
    
    extract_channels(seizure_input_folder, seizure_output_folder, channels)
    extract_channels(non_seizure_input_folder, non_seizure_output_folder, channels)

if __name__ == "__main__":
    main()
