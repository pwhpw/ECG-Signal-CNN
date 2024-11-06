import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import wfdb

# Path dataset
dataset_path = 'C:/Users/USER/Downloads/ECG-acquisition-classification-master/training2017/'
checkpoint_path = './checkpoints/model_CNN_final.keras'

# Memuat model dari file checkpoint
model = tf.keras.models.load_model(checkpoint_path)
print(f"Model berhasil dimuat dari {checkpoint_path}")

# Pemetaan label ke tipe penyakit jantung untuk pembacaan
disease_mapping = {
    0: 'Normal',
    1: 'Atrial Fibrillation/Flutter (AF)',
    2: 'Atrioventricular Block (AVB)',
    3: 'Ectopic Atrial Rhythm (EAR)',
    4: 'Idioventricular Rhythm (IVR)',
    5: 'Supraventricular Tachycardia (SVT)',
    6: 'Ventricular Tachycardia (VT)',
    7: 'Sinus Bradycardia',
    8: 'Sinus Tachycardia',
    9: 'Premature Atrial Contraction',
    10: 'Premature Ventricular Contraction',
    11: 'Wandering Atrial Pacemaker'
}

# Mengambil nama file rekaman dari dataset
record_names = [f[:-4] for f in os.listdir(dataset_path) if f.endswith('.hea')]

# Persiapan data sinyal dan labeling (hanya 30 dataset pertama)
signals = []
labels = []
sampling_rate = 300 
n_steps = 30 * sampling_rate 

for i, record_name in enumerate(record_names[:30]):
    file_path = os.path.join(dataset_path, record_name + '.hea')
    if os.path.exists(file_path):
        record = wfdb.rdrecord(os.path.join(dataset_path, record_name))
        signal = record.p_signal[:n_steps].flatten()
        signals.append(signal)

        # Mengatur label berdasarkan nama file atau karakteristik sinyal
        if 'AF' in record_name: 
            labels.append(1)  # Atrial Fibrillation/Flutter
        elif 'AVB' in record_name:
            labels.append(2)  # Atrioventricular Block
        elif 'EAR' in record_name:
            labels.append(3)  # Ectopic Atrial Rhythm
        elif 'IVR' in record_name:
            labels.append(4)  # Idioventricular Rhythm
        elif 'SVT' in record_name:
            labels.append(5)  # Supraventricular Tachycardia
        elif 'VT' in record_name:
            labels.append(6)  # Ventricular Tachycardia
        elif 'SB' in record_name:
            labels.append(7)  # Sinus Bradycardia
        elif 'ST' in record_name:
            labels.append(8)  # Sinus Tachycardia
        elif 'PAC' in record_name:
            labels.append(9)  # Premature Atrial Contraction
        elif 'PVC' in record_name:
            labels.append(10)  # Premature Ventricular Contraction
        elif 'WAP' in record_name:
            labels.append(11)  # Wandering Atrial Pacemaker
        else:
            labels.append(0)  # Default untuk Normal jika tidak ada klasifikasi lain

signals = pad_sequences(signals, maxlen=n_steps, padding='post', dtype='float32')
signals = np.expand_dims(signals, axis=-1)  # Tambahkan dimensi untuk channel

# Fungsi untuk menampilkan sinyal dengan shading jika terindikasi tidak normal
def plot_signal_with_prediction(X_sample, prediction, true_label, dataset_index):
    disease_name = disease_mapping[prediction]
    
    # Plot sinyal
    plt.figure(figsize=(10, 4))
    plt.plot(X_sample, label="ECG Signal", color='blue')
    
    # Jika sinyal terindikasi tidak normal, shading area sinyal
    if prediction != 0:  # Hanya shading jika prediksi bukan "Normal"
        plt.fill_between(range(len(X_sample)), X_sample, color='red', alpha=0.3, label=f"Detected: {disease_name}")
    
    # Informasi tambahan
    plt.title(f"Dataset {dataset_index + 1}: Predicted - {disease_name} | Actual - {disease_mapping[true_label]}")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

# Prediksi dan plot untuk dataset pertama hingga ke-30
for i in range(30):
    X_sample = signals[i].flatten()  # Sinyal sample
    true_label = labels[i]  # Label sebenarnya

    # Prediksi untuk sampel
    prediction = np.argmax(model.predict(np.expand_dims(signals[i], axis=0)), axis=1)[0]

    # Plot sinyal dengan shading untuk prediksi penyakit (hanya jika tidak normal)
    plot_signal_with_prediction(X_sample, prediction, true_label, i)
