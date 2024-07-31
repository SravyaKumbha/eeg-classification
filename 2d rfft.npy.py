import os
import numpy as np


def extract_and_save(input_folder, output_folder):
    
    os.makedirs(output_folder, exist_ok=True)
    
   
    files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]
    
    for file in files:
        
        full_path = os.path.join(input_folder, file)
        data = np.load(full_path)
        
       
        num_arrays, y, z = data.shape
        
        
        for i in range(num_arrays):
            extracted_array = data[i]
            output_file = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_{i}.npy")
            np.save(output_file, extracted_array)
            print(f"Saved {output_file}")


input_folder1 = r"C:\Users\KSUDHEER\Desktop\transformed rfft\transformed_rfft_non_seizure"
output_folder1 = r"C:\Users\KSUDHEER\Desktop\2D rfft\non seizure"
input_folder2=r"C:\Users\KSUDHEER\Desktop\transformed rfft\transformed_rfft_seizure"
output_folder2=r"C:\Users\KSUDHEER\Desktop\2D rfft\seizure"

extract_and_save(input_folder1, output_folder1)
extract_and_save(input_folder2, output_folder2)
