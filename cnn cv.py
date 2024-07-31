import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

base_dir = r"C:\Users\KSUDHEER\Desktop\2D rfft\split"
sets = ['set1', 'set2', 'set3']


def load_data():
    data = []
    labels = []
    for s in sets:
        for label, folder in enumerate(['non seizure', 'seizure']):
            folder_path = os.path.join(base_dir, s, folder)
            for file in os.listdir(folder_path):
                if file.endswith('.npy'):
                    file_path = os.path.join(folder_path, file)
                    array = np.load(file_path)
                    array = np.expand_dims(array, axis=-1)
                    data.append(array)
                    labels.append(label)
    data = np.array(data)
    labels = np.array(labels)
    return data, labels


data, labels = load_data()
labels = to_categorical(labels, num_classes=2)


def create_model(input_shape):
    inputs = Input(shape=input_shape)
    print(f"Input shape: {inputs.shape}")

    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    print(f"After Conv2D(32): {x.shape}")

    x = MaxPooling2D((2, 2))(x)
    print(f"After MaxPooling2D: {x.shape}")

    x = Conv2D(64, (3, 3), activation='relu')(x)
    print(f"After Conv2D(64): {x.shape}")

    x = MaxPooling2D((2, 2))(x)
    print(f"After second MaxPooling2D: {x.shape}")

    x = Conv2D(128, (3, 3), activation='relu')(x)
    print(f"After Conv2D(128): {x.shape}")

    x = MaxPooling2D((1, 2))(x)
    print(f"After third MaxPooling2D: {x.shape}")

    x = Flatten()(x)
    print(f"After Flatten: {x.shape}")

    x = Dense(128, activation='relu')(x)
    print(f"After Dense(128): {x.shape}")

    outputs = Dense(2, activation='softmax')(x)
    print(f"After Dense(2): {outputs.shape}")

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


kf = KFold(n_splits=3, shuffle=True, random_state=42)
input_shape = data.shape[1:]

train_losses = []
val_losses = []
val_accuracies = []
test_accuracies = []

for fold, (train_index, test_index) in enumerate(kf.split(data)):
    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    val_split = 0.2
    split_index = int(len(x_train) * (1 - val_split))
    x_train, x_val = x_train[:split_index], x_train[split_index:]
    y_train, y_val = y_train[:split_index], y_train[split_index:]

    model = create_model(input_shape)

    history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_val, y_val))

    # Save the model
    model_path = f'C:/Users/KSUDHEER/Desktop/MODEL_SAVE/model_with_3cnnlayers_epchs20_fold_{fold + 1}.keras'
    model.save(model_path)
    print(f"Model saved to {model_path}")

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    val_accuracy = max(history.history['val_accuracy'])
    val_loss = min(history.history['val_loss'])
    train_loss = min(history.history['loss'])

    test_accuracies.append(test_accuracy)
    val_accuracies.append(val_accuracy)
    val_losses.append(val_loss)
    train_losses.append(train_loss)

    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non Seizure', 'Seizure'])
    disp.plot()
    plt.title(f"Confusion Matrix for Set {fold + 1}")
    plt.show()

print(f"Test Accuracies: {test_accuracies}")
print(f"Validation Accuracies: {val_accuracies}")
print(f"Validation Losses: {val_losses}")
print(f"Train Losses: {train_losses}")

plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Fold')
plt.ylabel('Loss')
plt.legend()
plt.title('Train and Validation Loss')
plt.show()

plt.figure()
plt.plot(test_accuracies, label='Test Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Test and Validation Accuracy')
plt.show()