import os
import shutil
import random


non_seizure_dir = r"C:\Users\KSUDHEER\Desktop\2D rfft\non seizure"
seizure_dir = r"C:\Users\KSUDHEER\Desktop\2D rfft\seizure"
base_output_dir = r"C:\Users\KSUDHEER\Desktop\2D rfft\split"


sets = ['set1', 'set2', 'set3']
for s in sets:
    os.makedirs(os.path.join(base_output_dir, s, 'seizure'), exist_ok=True)
    os.makedirs(os.path.join(base_output_dir, s, 'non seizure'), exist_ok=True)


def move_files(file_list, destination_folder):
    for file in file_list:
        shutil.move(file, destination_folder)


non_seizure_files = [os.path.join(non_seizure_dir, f) for f in os.listdir(non_seizure_dir) if f.endswith('.npy')]
seizure_files = [os.path.join(seizure_dir, f) for f in os.listdir(seizure_dir) if f.endswith('.npy')]


chb_prefixes = set()
for file in non_seizure_files + seizure_files:
    chb_prefix = os.path.basename(file)[:5]
    chb_prefixes.add(chb_prefix)


chb_prefixes = list(chb_prefixes)
random.shuffle(chb_prefixes)
split1 = chb_prefixes[:len(chb_prefixes)//3]
split2 = chb_prefixes[len(chb_prefixes)//3:2*len(chb_prefixes)//3]
split3 = chb_prefixes[2*len(chb_prefixes)//3:]


def distribute_files(file_list, split, output_dir):
    for file in file_list:
        chb_prefix = os.path.basename(file)[:5]
        if chb_prefix in split:
            if 'non seizure' in file:
                move_files([file], os.path.join(output_dir, 'non seizure'))
            elif 'seizure' in file:
                move_files([file], os.path.join(output_dir, 'seizure'))


distribute_files(non_seizure_files, split1, os.path.join(base_output_dir, 'set1'))
distribute_files(seizure_files, split1, os.path.join(base_output_dir, 'set1'))

distribute_files(non_seizure_files, split2, os.path.join(base_output_dir, 'set2'))
distribute_files(seizure_files, split2, os.path.join(base_output_dir, 'set2'))

distribute_files(non_seizure_files, split3, os.path.join(base_output_dir, 'set3'))
distribute_files(seizure_files, split3, os.path.join(base_output_dir, 'set3'))

print("Files have been split into three sets successfully.")
