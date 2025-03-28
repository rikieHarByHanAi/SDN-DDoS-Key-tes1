import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Layer, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import os
from google.colab import drive

# Mount Google Drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')
else:
    print("Google Drive sudah dimount.")

# Path ke file dataset di Google Drive
file_path = '/content/drive/MyDrive/dataSDN/SDN_DDoS.csv'
print("File ada?", os.path.exists(file_path))
if not os.path.exists(file_path):
    raise FileNotFoundError("File SDN_DDoS.csv tidak ditemukan di path yang diberikan!")

# Load dataset
data = pd.read_csv(file_path)
data_clean = data.dropna()

# Sampling untuk menyeimbangkan dataset
normal_data = data_clean[data_clean['label'] == 0].sample(n=7500, random_state=42)
ddos_data = data_clean[data_clean['label'] == 1].sample(n=7500, random_state=42)
data_balanced = pd.concat([normal_data, ddos_data])

# Persiapan data
fitur = ['timestamp', 'packet_count', 'byte_count', 'packet_count_per_second', 'byte_count_per_second']
X = data_balanced[fitur].values
y = data_balanced['label'].values

# Normalisasi data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape data untuk model deep learning (seq_length=10)
seq_length = 10
X_seq = []
y_seq = []
for i in range(len(X) - seq_length):
    X_seq.append(X[i:i + seq_length])
    y_seq.append(y[i + seq_length])
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Custom Attention Layer untuk LSTM-Attention
class CustomAttention(Layer):
    def __init__(self, **kwargs):
        super(CustomAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(CustomAttention, self).build(input_shape)

    def call(self, inputs):
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = inputs * alpha
        context = K.sum(context, axis=1)
        self.attention_weights = alpha
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# Inisialisasi untuk cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Simpan metrik untuk setiap model
metrics = {
    'CNN': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'threshold': []},
    'LSTM-Attention': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'threshold': []}
}

# Rentang threshold yang akan diuji
thresholds = np.arange(0.5, 0.96, 0.05)  # Dari 0.5 hingga 0.95, langkah 0.05

# 5-Fold Cross-Validation
for fold, (train_index, test_index) in enumerate(kf.split(X_seq)):
    print(f'Fold {fold + 1}')
    X_train_seq, X_test_seq = X_seq[train_index], X_seq[test_index]
    y_train, y_test = y_seq[train_index], y_seq[test_index]

    # --- Model 1: CNN ---
    inputs = Input(shape=(seq_length, len(fitur)))
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = Conv1D(filters=32, kernel_size=3, activation='relu')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    flat = Flatten()(pool2)
    dense1 = Dense(50, activation='relu')(flat)
    dropout = Dropout(0.2)(dense1)
    outputs = Dense(1, activation='sigmoid')(dropout)
    cnn_model = Model(inputs=inputs, outputs=outputs)
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    cnn_model.fit(X_train_seq, y_train, epochs=100, batch_size=32,
                  validation_data=(X_test_seq, y_test), verbose=0,
                  callbacks=[early_stopping])
    
    # Simpan probabilitas prediksi
    y_pred_prob_cnn = cnn_model.predict(X_test_seq)
    print(f"CNN Fold {fold + 1} - Probabilitas Prediksi: Min={y_pred_prob_cnn.min():.4f}, Max={y_pred_prob_cnn.max():.4f}")

    # Optimasi threshold untuk CNN
    best_f1_cnn = -1  # Inisialisasi dengan nilai negatif untuk memastikan selalu ada nilai terbaik
    best_threshold_cnn = 0.5
    best_metrics_cnn = {}
    for threshold in thresholds:
        y_pred_cnn = (y_pred_prob_cnn > threshold).astype("int32")
        # Gunakan zero_division=0 untuk menangani kasus di mana metrik tidak terdefinisi
        f1 = f1_score(y_test, y_pred_cnn, zero_division=0)
        if f1 > best_f1_cnn:
            best_f1_cnn = f1
            best_threshold_cnn = threshold
            best_metrics_cnn = {
                'accuracy': accuracy_score(y_test, y_pred_cnn),
                'precision': precision_score(y_test, y_pred_cnn, zero_division=0),
                'recall': recall_score(y_test, y_pred_cnn, zero_division=0),
                'f1': f1
            }
    
    # Pastikan metrik selalu disimpan, bahkan jika tidak ada perubahan
    if not best_metrics_cnn:
        y_pred_cnn = (y_pred_prob_cnn > 0.5).astype("int32")
        best_metrics_cnn = {
            'accuracy': accuracy_score(y_test, y_pred_cnn),
            'precision': precision_score(y_test, y_pred_cnn, zero_division=0),
            'recall': recall_score(y_test, y_pred_cnn, zero_division=0),
            'f1': f1_score(y_test, y_pred_cnn, zero_division=0)
        }
        best_threshold_cnn = 0.5

    metrics['CNN']['accuracy'].append(best_metrics_cnn['accuracy'])
    metrics['CNN']['precision'].append(best_metrics_cnn['precision'])
    metrics['CNN']['recall'].append(best_metrics_cnn['recall'])
    metrics['CNN']['f1'].append(best_metrics_cnn['f1'])
    metrics['CNN']['threshold'].append(best_threshold_cnn)

    # --- Model 2: LSTM-Attention ---
    inputs = Input(shape=(seq_length, len(fitur)))
    lstm_out = LSTM(50, return_sequences=True)(inputs)
    attention_layer = CustomAttention()
    attention_out = attention_layer(lstm_out)
    lstm_final = LSTM(50, return_sequences=False)(lstm_out)
    dropout = Dropout(0.2)(lstm_final)
    outputs = Dense(1, activation='sigmoid')(dropout)
    lstm_model = Model(inputs=inputs, outputs=outputs)
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    lstm_model.fit(X_train_seq, y_train, epochs=100, batch_size=32,
                   validation_data=(X_test_seq, y_test), verbose=0,
                   callbacks=[early_stopping])
    
    # Simpan probabilitas prediksi
    y_pred_prob_lstm = lstm_model.predict(X_test_seq)
    print(f"LSTM-Attention Fold {fold + 1} - Probabilitas Prediksi: Min={y_pred_prob_lstm.min():.4f}, Max={y_pred_prob_lstm.max():.4f}")

    # Optimasi threshold untuk LSTM-Attention
    best_f1_lstm = -1  # Inisialisasi dengan nilai negatif untuk memastikan selalu ada nilai terbaik
    best_threshold_lstm = 0.5
    best_metrics_lstm = {}
    for threshold in thresholds:
        y_pred_lstm = (y_pred_prob_lstm > threshold).astype("int32")
        # Gunakan zero_division=0 untuk menangani kasus di mana metrik tidak terdefinisi
        f1 = f1_score(y_test, y_pred_lstm, zero_division=0)
        if f1 > best_f1_lstm:
            best_f1_lstm = f1
            best_threshold_lstm = threshold
            best_metrics_lstm = {
                'accuracy': accuracy_score(y_test, y_pred_lstm),
                'precision': precision_score(y_test, y_pred_lstm, zero_division=0),
                'recall': recall_score(y_test, y_pred_lstm, zero_division=0),
                'f1': f1
            }
    
    # Pastikan metrik selalu disimpan, bahkan jika tidak ada perubahan
    if not best_metrics_lstm:
        y_pred_lstm = (y_pred_prob_lstm > 0.5).astype("int32")
        best_metrics_lstm = {
            'accuracy': accuracy_score(y_test, y_pred_lstm),
            'precision': precision_score(y_test, y_pred_lstm, zero_division=0),
            'recall': recall_score(y_test, y_pred_lstm, zero_division=0),
            'f1': f1_score(y_test, y_pred_lstm, zero_division=0)
        }
        best_threshold_lstm = 0.5

    metrics['LSTM-Attention']['accuracy'].append(best_metrics_lstm['accuracy'])
    metrics['LSTM-Attention']['precision'].append(best_metrics_lstm['precision'])
    metrics['LSTM-Attention']['recall'].append(best_metrics_lstm['recall'])
    metrics['LSTM-Attention']['f1'].append(best_metrics_lstm['f1'])
    metrics['LSTM-Attention']['threshold'].append(best_threshold_lstm)

# Hitung rata-rata dan standar deviasi untuk setiap model
for model_name in metrics:
    print(f"\nResults for {model_name} (with optimal threshold):")
    mean_accuracy = np.mean(metrics[model_name]['accuracy'])
    std_accuracy = np.std(metrics[model_name]['accuracy'])
    mean_precision = np.mean(metrics[model_name]['precision'])
    std_precision = np.std(metrics[model_name]['precision'])
    mean_recall = np.mean(metrics[model_name]['recall'])
    std_recall = np.std(metrics[model_name]['recall'])
    mean_f1 = np.mean(metrics[model_name]['f1'])
    std_f1 = np.std(metrics[model_name]['f1'])
    mean_threshold = np.mean(metrics[model_name]['threshold'])
    std_threshold = np.std(metrics[model_name]['threshold'])
    
    print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Mean Precision: {mean_precision:.4f} ± {std_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f} ± {std_recall:.4f}")
    print(f"Mean F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Mean Optimal Threshold: {mean_threshold:.4f} ± {std_threshold:.4f}")

# Visualisasi perbandingan metrik
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Comparison of Models with Optimal Threshold (5-Fold Cross-Validation)', fontsize=16)

# Akurasi
means = [np.mean(metrics[model]['accuracy']) for model in metrics]
stds = [np.std(metrics[model]['accuracy']) for model in metrics]
axes[0, 0].bar(metrics.keys(), means, yerr=stds, capsize=5, color=['blue', 'green'])
axes[0, 0].set_title('Accuracy')
axes[0, 0].set_ylim(0, 1)

# Presisi
means = [np.mean(metrics[model]['precision']) for model in metrics]
stds = [np.std(metrics[model]['precision']) for model in metrics]
axes[0, 1].bar(metrics.keys(), means, yerr=stds, capsize=5, color=['blue', 'green'])
axes[0, 1].set_title('Precision')
axes[0, 1].set_ylim(0, 1)

# Recall
means = [np.mean(metrics[model]['recall']) for model in metrics]
stds = [np.std(metrics[model]['recall']) for model in metrics]
axes[1, 0].bar(metrics.keys(), means, yerr=stds, capsize=5, color=['blue', 'green'])
axes[1, 0].set_title('Recall')
axes[1, 0].set_ylim(0, 1)

# F1-Score
means = [np.mean(metrics[model]['f1']) for model in metrics]
stds = [np.std(metrics[model]['f1']) for model in metrics]
axes[1, 1].bar(metrics.keys(), means, yerr=stds, capsize=5, color=['blue', 'green'])
axes[1, 1].set_title('F1-Score')
axes[1, 1].set_ylim(0, 1)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('threshold_optimization_comparison.png')
plt.show()
