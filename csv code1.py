import os
import pandas as pd
from datetime import timedelta

folder_path = r"C:\Users\KSUDHEER\Desktop\chb-mit-scalp-eeg-database-1.0.0\chb-mit-scalp-eeg-database-1.0.0"

def seconds_to_hms(seconds):
    return str(timedelta(seconds=seconds))

def hms_to_seconds(hms):
    if hms is None or hms == 'None':
        return 0
    h, m, s = map(int, hms.split(':'))
    return h * 3600 + m * 60 + s

def parse_summary_file(file_path):
    data = []
    file_name = None
    start_time = 'NA'
    end_time = 'NA'

    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines_iter = iter(lines)
        for line in lines_iter:
            if line.startswith("File Name:"):
                file_name = line.split(": ")[1].strip()
            elif line.startswith("File Start Time:"):
                start_time = line.split(": ")[1].strip()
            elif line.startswith("File End Time:"):
                end_time = line.split(": ")[1].strip()
            elif line.startswith("Number of Seizures in File:"):
                num_seizures = int(line.split(": ")[1].strip())
                seizures = []
                seizure_durations = []
                if num_seizures > 0:
                    for i in range(num_seizures):
                        seizure_start = None
                        seizure_end = None
                        try:
                            line = next(lines_iter).strip()
                            if f"Seizure {i+1} Start Time:" in line or "Seizure Start Time:" in line:
                                seizure_start = int(line.split(": ")[1].strip().split()[0])
                            line = next(lines_iter).strip()
                            if f"Seizure {i+1} End Time:" in line or "Seizure End Time:" in line:
                                seizure_end = int(line.split(": ")[1].strip().split()[0])
                            
                            seizure_start_hms = seconds_to_hms(seizure_start)
                            seizure_end_hms = seconds_to_hms(seizure_end)
                            seizure_duration = seizure_end - seizure_start
                            seizure_duration_hms = seconds_to_hms(seizure_duration)
                            
                            seizures.append(f"({seizure_start_hms},{seizure_end_hms})")
                            seizure_durations.append(seizure_duration)
                        except Exception as e:
                            print(f"Error parsing seizures for file {file_name}: {e}")
                    seizures_str = ",".join(seizures)
                    seizure_durations_str = ",".join([seconds_to_hms(sd) for sd in seizure_durations])
                    average_seizure_duration = seconds_to_hms(sum(seizure_durations) // num_seizures)
                    if file_name:
                        data.append([file_name, start_time, end_time, num_seizures, seizures_str, seizure_durations_str, average_seizure_duration])
                    else:
                        print(f"Warning: Missing file name in file {file_path}")
                else:
                    if file_name:
                        data.append([file_name, start_time, end_time, num_seizures, None, None, None])
                    else:
                        print(f"Warning: Missing file name in file {file_path}")

    return data

all_data = []
for i in range(1, 25):
    file_num = str(i).zfill(2)
    file_name = f'chb{file_num}\chb{file_num}-summary.txt'
    file_path = os.path.join(folder_path, file_name)
    if os.path.exists(file_path):
        print(f"Parsing file: {file_path}")
        file_data = parse_summary_file(file_path)
        if file_data:
            print(f"Data parsed from file {file_name}: {file_data}")
        all_data.extend(file_data)
    else:
        print(f"File does not exist: {file_path}")

if all_data:
    df = pd.DataFrame(all_data, columns=['File Name', 'Start Time', 'End Time', 'Number of Seizures', 'Seizure Times', 'Seizure Durations', 'Average Seizure Duration'])
    
   
    o1_csv_path = 'o1.csv'
    df.to_csv(o1_csv_path, index=False)
    print(f"CSV file saved to {o1_csv_path}")

    
    df['Average Seizure Duration Seconds'] = df['Average Seizure Duration'].apply(lambda x: hms_to_seconds(x))
    total_avg_seizure_duration_seconds = df['Average Seizure Duration Seconds'].sum() / df[df['Number of Seizures'] > 0].shape[0]
    total_avg_seizure_duration = seconds_to_hms(total_avg_seizure_duration_seconds)
    print(f"Total average seizure duration for all files: {total_avg_seizure_duration}")
else:
    print("No data parsed. CSV file not created.")
