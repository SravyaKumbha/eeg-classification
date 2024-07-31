import os
import pyedflib
import numpy as np
import pandas as pd

def extract_segments(data_path, edf_file, start_time, end_time):
    file_path = os.path.join(data_path, edf_file)

    try:
        f = pyedflib.EdfReader(file_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None, None

    sfreq = f.getSampleFrequency(0)
    start_sample = int(start_time * sfreq)
    end_sample = int(end_time * sfreq)

    segment_data = []
    for i in range(f.signals_in_file):
        signal = f.readSignal(i)[start_sample:end_sample]
        segment_data.append(signal)

    f.close()

    segment_data = np.array(segment_data)

    return segment_data, sfreq, f

def save_as_edf(segment_data, sfreq, f, output_file):
    n_channels = segment_data.shape[0]
    signal_headers = f.getSignalHeaders()

    edf_writer = pyedflib.EdfWriter(output_file, n_channels, file_type=pyedflib.FILETYPE_EDFPLUS)

    for i in range(n_channels):
        min_value = segment_data[i].min()
        max_value = segment_data[i].max()

        if min_value == max_value:
            max_value += 1.0  

        signal_headers[i]['physical_min'] = min_value
        signal_headers[i]['physical_max'] = max_value
        signal_headers[i]['digital_min'] = -32768
        signal_headers[i]['digital_max'] = 32767
        signal_headers[i]['sample_frequency'] = sfreq

    edf_writer.setSignalHeaders(signal_headers)
    edf_writer.setPatientCode(f.getPatientCode())
    edf_writer.setPatientName(f.getPatientName())
    edf_writer.setStartdatetime(f.getStartdatetime())

    edf_writer.writeSamples(segment_data)
    edf_writer.close()

csv_file_path = r"C:\Users\KSUDHEER\Desktop\o1.csv"
csv_data = pd.read_csv(csv_file_path)

base_data_path = r"C:\Users\KSUDHEER\Desktop\chb-mit-scalp-eeg-database-1.0.0\chb-mit-scalp-eeg-database-1.0.0"
seizure_output_dir = r"C:\Users\KSUDHEER\Desktop\SEIZURE_EXTRACTION"
non_seizure_output_dir = r"C:\Users\KSUDHEER\Desktop\NON_SEIZURE_EXTRACTION"
os.makedirs(seizure_output_dir, exist_ok=True)
os.makedirs(non_seizure_output_dir, exist_ok=True)

for i in range(1, 25):
    folder_name = f'chb{str(i).zfill(2)}'
    folder_path = os.path.join(base_data_path, folder_name)

    for index, row in csv_data.iterrows():
        edf_file = row['File Name']
        if edf_file.startswith(folder_name):
            num_seizures = row['Number of Seizures']
            seizure_times = row['Seizure Times']

            if num_seizures > 0 and pd.notna(seizure_times):
                seizure_times = seizure_times.strip('()').replace(' ', '').split('),(')
                seizure_intervals = []
                for seizure_time in seizure_times:
                    times = seizure_time.split(',')
                    if len(times) == 2:
                        start_time_str, end_time_str = times

                        start_time_parts = start_time_str.split(':')
                        end_time_parts = end_time_str.split(':')

                        seizure_start_time = int(start_time_parts[0]) * 3600 + int(start_time_parts[1]) * 60 + int(start_time_parts[2])
                        seizure_end_time = int(end_time_parts[0]) * 3600 + int(end_time_parts[1]) * 60 + int(end_time_parts[2])

                        seizure_intervals.append((seizure_start_time, seizure_end_time))

                for j, (seizure_start_time, seizure_end_time) in enumerate(seizure_intervals):
                    print(f"Processing seizure {j + 1} from {seizure_start_time} to {seizure_end_time} in file {edf_file}")

                    seizure_data, sfreq, f = extract_segments(folder_path, edf_file, seizure_start_time, seizure_end_time)

                    if seizure_data is not None:
                        output_file_seizure = os.path.join(seizure_output_dir, f"{edf_file.replace('.edf', '')}_seizure{j + 1}.edf")
                        save_as_edf(seizure_data, sfreq, f, output_file_seizure)
                        print(f"Seizure segment saved as {output_file_seizure}")
                    else:
                        print(f"Failed to extract seizure segment from {edf_file}")

                
                for j, (seizure_start_time, seizure_end_time) in enumerate(seizure_intervals):
                   if seizure_start_time >= 75:
                    non_seizure_start_time = max(0, seizure_start_time - 75)
                    non_seizure_end_time = seizure_start_time
                   else:
                      non_seizure_start_time = seizure_end_time
                      non_seizure_end_time = min(seizure_end_time + 75, 3600)  
    
                   if all(non_seizure_start_time >= end or non_seizure_end_time <= start for start, end in seizure_intervals):
                    print(f"Processing non-seizure segment {j + 1} from {non_seizure_start_time} to {non_seizure_end_time} in file {edf_file}")

                    non_seizure_data, sfreq, f = extract_segments(folder_path, edf_file, non_seizure_start_time, non_seizure_end_time)

                    if non_seizure_data is not None:
                        output_file_non_seizure = os.path.join(non_seizure_output_dir, f"{edf_file.replace('.edf', '')}_non_seizure{j + 1}.edf")
                        save_as_edf(non_seizure_data, sfreq, f, output_file_non_seizure)
                        print(f"Non-seizure segment saved as {output_file_non_seizure}")
                    else:
                     print(f"Failed to extract non-seizure segment from {edf_file}")
                   else:
                      print(f"Non-seizure segment {j + 1} overlaps with seizure in file {edf_file}, skipping.")

print("Processing complete.")
