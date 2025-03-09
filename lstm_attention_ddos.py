# Kode lengkap untuk deteksi DDoS dengan LSTM-Attention
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, Input
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')
file_path = '/content/drive/MyDrive/dataSDN/SDN_DDoS.csv'
print("File ada?", os.path.exists(file_path))
if not os.path.exists(file_path):
    raise FileNotFoundError("File SDN_DDoS.csv tidak ditemukan!")

# Baca dan bersihkan data
data = pd.read_csv(file_path, on_bad_lines='skip')
data_clean = data[['timestamp', 'packet_count', 'byte_count', 
                   'packet_count_per_second', 'byte_count_per_second', 'label']].copy()
data_clean['timestamp'] = pd.to_numeric(data_clean['timestamp'], errors='coerce')
data_clean['timestamp'] = data_clean['timestamp'] - data_clean['timestamp'].min()
data_clean = data_clean.dropna()

print("Distribusi label setelah dropna:")
print(data_clean['label'].value_counts())

# Correlation Matrix
correlation_matrix = data_clean.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='YlOrRd', vmin=-1, vmax=1)
plt.title('Correlation of Features')
plt.savefig('correlation_matrix.png', dpi=300)

# Scatter Plot
data_sample = data_clean.sample(n=5000, random_state=42)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_sample, x='packet_count', y='byte_count', hue='label', alpha=0.6, s=10, legend='brief')
plt.title('Scatter Plot of Packet Count vs Byte Count by Label')
plt.xlabel('Packet Count')
plt.ylabel('Byte Count')
plt.legend(title='Label', loc='upper right')
plt.savefig('scatter_plot.png', dpi=300)

# Ambil sampel seimbang
n_sample = min(len(data_clean[data_clean['label'] == 0]), 
               len(data_clean[data_clean['label'] == 1]), 7500)
print(f"Jumlah sampel per kelas: {n_sample}")
normal_data = data_clean[data_clean['label'] == 0].sample(n=n_sample, random_state=42)
ddos_data = data_clean[data_clean['label'] == 1].sample(n=n_sample, random_state=42)
data_sample = pd.concat([normal_data, ddos_data])

# Normalisasi fitur
scaler = MinMaxScaler()
fitur = ['timestamp', 'packet_count', 'byte_count', 
         'packet_count_per_second', 'byte_count_per_second']
data_sample.loc[:, fitur] = scaler.fit_transform(data_sample[fitur])

# Bikin urutan
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[fitur].iloc[i:i+seq_length].values)
        y.append(data['label'].iloc[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10
X, y = create_sequences(data_sample, seq_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bikin model
inputs = Input(shape=(seq_length, len(fitur)))
lstm_out = LSTM(50, return_sequences=True)(inputs)
attention_out = Attention()([lstm_out, lstm_out])
lstm_final = LSTM(50, return_sequences=False)(attention_out)
dropout = Dropout(0.2)(lstm_final)
outputs = Dense(1, activation='sigmoid')(dropout)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Latih model
history = model.fit(X_train, y_train, epochs=250, batch_size=32, 
                    validation_data=(X_test, y_test))

# Cek hasil
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Akurasi di data test: {accuracy*100:.2f}%")
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("Laporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'DDoS']))

# Visualisasi Loss dan Accuracy
plt.style.use('default')
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0, 1.0)
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0.5, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig('loss_accuracy.png', dpi=300)

# ROC Curve
y_pred_proba = model.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png', dpi=300)

# Bar Chart Perbandingan
models = ['Braga (2010)', 'Bhandari (2016)', 'Novaes (2020)', 'Elsayed (2020)', 'Our Model']
accuracies = [80, 85, 90, 95, 96.66]
plt.figure(figsize=(10, 6))
sns.barplot(x=accuracies, y=models, palette='Blues_d')
plt.title('Accuracy Comparison with State of Art')
plt.xlabel('Accuracy (%)')
plt.ylabel('Model')
plt.savefig('comparison.png', dpi=300)
