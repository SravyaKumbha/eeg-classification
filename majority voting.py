#"C:\Users\nithy\Desktop\EEG analysis\model_fold_1.h5"
#"C:\Users\nithy\Downloads\model_fold_1.keras"
#"C:\Users\nithy\Desktop\MODEL_SAVE\model_fold_1.keras"

import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
sets = 3
data_dirs = [
    {'seizure': r"C:\Users\KSUDHEER\Desktop\2D rfft\split\set1\seizure", 'non_seizure': r"C:\Users\KSUDHEER\Desktop\2D rfft\split\set1\non seizure"},
    {'seizure': r"C:\Users\KSUDHEER\Desktop\2D rfft\split\set2\seizure", 'non_seizure': r"C:\Users\KSUDHEER\Desktop\2D rfft\split\set2\non seizure"},
    {'seizure': r"C:\Users\KSUDHEER\Desktop\2D rfft\split\set3\seizure", 'non_seizure': r"C:\Users\KSUDHEER\Desktop\2D rfft\split\set3\non seizure"}
]
model_paths = [r"C:\Users\KSUDHEER\Desktop\best_model_fold_1.h5", r"C:\Users\KSUDHEER\Desktop\best_model_fold_2.h5", r"C:\Users\KSUDHEER\Desktop\best_model_fold_3.h5"]
chunk_suffix = '_{}.npy'
models = [load_model(model_path) for model_path in model_paths]

def get_majority_vote(predictions):
    return 1 if np.sum(predictions) / len(predictions) > 0.5 else 0

def predict_with_majority_voting(model, main_file_base_path, n_chunks):
    chunk_predictions = []
    for chunk_index in range(n_chunks):
        chunk_file = f"{main_file_base_path}_{chunk_index}.npy"
        if os.path.exists(chunk_file):
            data = np.load(chunk_file)
            data = data.reshape((1, 18, 129, 1))
            prediction = model.predict(data)
            chunk_predictions.append(np.argmax(prediction, axis=1)[0])
    if chunk_predictions:
        return get_majority_vote(chunk_predictions)
    return None

for set_index in range(sets):
    print(f"Processing set {set_index + 1}...")
    model = models[set_index]
    data_dir = data_dirs[set_index]

    true_labels = []
    predicted_labels = []

    for file_name in os.listdir(data_dir['seizure']):
        if file_name.endswith('_0.npy'):
            base_name = file_name.rsplit('_', 1)[0]
            n_chunks = len([f for f in os.listdir(data_dir['seizure']) if f.startswith(os.path.basename(base_name))])
            true_labels.append(1)
            prediction = predict_with_majority_voting(model, os.path.join(data_dir['seizure'], base_name), n_chunks)
            if prediction is not None:
                predicted_labels.append(prediction)

    for file_name in os.listdir(data_dir['non_seizure']):
        if file_name.endswith('_0.npy'):
            base_name = file_name.rsplit('_', 1)[0]
            n_chunks = len([f for f in os.listdir(data_dir['non_seizure']) if f.startswith(os.path.basename(base_name))])
            true_labels.append(0)
            prediction = predict_with_majority_voting(model, os.path.join(data_dir['non_seizure'], base_name), n_chunks)
            if prediction is not None:
                predicted_labels.append(prediction)

    if len(true_labels) != len(predicted_labels):
        print(f"Warning: Mismatch between true labels and predicted labels in set {set_index + 1}.")
        continue

    fold_accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy for set {set_index + 1}: {fold_accuracy}")

    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix for Set {set_index + 1}")
    plt.show()