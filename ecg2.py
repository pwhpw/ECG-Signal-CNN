import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, ReLU, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from collections import Counter

dataset_path = 'C:/Users/USER/Downloads/ECG-acquisition-classification-master/training2017/'

checkpoint_dir = './checkpoints/'
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_path = os.path.join(checkpoint_dir, 'model_CNN_epoch_{epoch:02d}_val_acc_{val_accuracy:.2f}.keras')

# rekaman dataset
record_names = [f[:-4] for f in os.listdir(dataset_path) if f.endswith('.hea')]

signals = []
labels = []

sampling_rate = 300 
n_steps = 30 * sampling_rate 

# labeling dataset
for i, record_name in enumerate(record_names):
    file_path = os.path.join(dataset_path, record_name + '.hea')
    if os.path.exists(file_path):
        record = wfdb.rdrecord(os.path.join(dataset_path, record_name))
        signal = record.p_signal[:n_steps]
        signal = signal.flatten()

        signals.append(signal)
        
        if 'A' in record_name: 
            labels.append(1)  # atrial fibrillation
        elif 'T' in record_name:
            labels.append(2)  # trigeminy
        else:
            labels.append(0)  # normal

signals = pad_sequences(signals, maxlen=n_steps, padding='post', dtype='float32')
labels = np.array(labels)

# split dataset
X_train, X_val, y_train, y_val = train_test_split(signals, labels, test_size=0.2, random_state=42)

X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)

model = Sequential([
    Conv1D(32, kernel_size=3, padding='same', input_shape=(n_steps, 1)),
    BatchNormalization(),
    ReLU(),
    
    Conv1D(32, kernel_size=3, padding='same'),
    BatchNormalization(),
    ReLU(),
    Dropout(0.3),
    MaxPooling1D(pool_size=2, padding='same'),  

    *[layer for _ in range(5) for layer in [  
        Conv1D(64, kernel_size=3, padding='same'),
        BatchNormalization(),
        ReLU(),
        Conv1D(64, kernel_size=3, padding='same'),
        BatchNormalization(),
        ReLU(),
        Dropout(0.3),
        MaxPooling1D(pool_size=2, padding='same'),  
    ]],

    Flatten(),
    Dense(64),
    BatchNormalization(),
    ReLU(),
    Dense(3, activation='softmax') 
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint_callback, early_stopping])

plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# klasifikasi set-level
def majority_voting(predictions):
    predicted_classes = np.argmax(predictions, axis=1)
    majority_class = Counter(predicted_classes).most_common(1)[0][0]
    return majority_class

# evaluasi Set-Level
def evaluate_set_level(model, X_val, y_val):
    set_level_predictions = []
    for i in range(len(X_val)):
        signal = X_val[i]
        predictions = model.predict(np.expand_dims(signal, axis=0))
        set_prediction = majority_voting(predictions)
        set_level_predictions.append(set_prediction)
    
    accuracy = np.mean(np.array(set_level_predictions) == y_val)
    print(f'Set-level Accuracy: {accuracy * 100:.2f}%')

evaluate_set_level(model, X_val, y_val)

print("Hasil Prediksi Penyakit Jantung untuk 20 Dataset Pertama:")

disease_mapping = {0: 'Normal', 1: 'Atrial Fibrillation', 2: 'Penyakit Tipe T'}

for i in range(20):
    signal = np.expand_dims(X_val[i], axis=0)
    prediction = model.predict(signal)
    predicted_class = np.argmax(prediction, axis=1)[0]

    print(f"Dataset {i+1}: Prediksi Tipe Penyakit Jantung - {disease_mapping[predicted_class]}")

normal_indices = np.where(y_val == 0)[0]
normal_signals = X_val[normal_indices]

print("\nHasil Prediksi untuk Sinyal Jantung Normal:")
for i, idx in enumerate(normal_indices[:20]):
    signal = np.expand_dims(normal_signals[i], axis=0)
    prediction = model.predict(signal)
    predicted_class = np.argmax(prediction, axis=1)[0]
    print(f"Dataset Normal ke-{i+1}: Prediksi - {disease_mapping[predicted_class]}")

final_model_path = os.path.join(checkpoint_dir, 'model_CNN_final.keras')
model.save(final_model_path)
print(f"\nModel final telah disimpan di {final_model_path}")
