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
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add
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
    'CNN': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
    'LSTM-Attention': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
    'Transformer': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
}

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
    
    y_pred_cnn = (cnn_model.predict(X_test_seq) > 0.5).astype("int32")
    metrics['CNN']['accuracy'].append(accuracy_score(y_test, y_pred_cnn))
    metrics['CNN']['precision'].append(precision_score(y_test, y_pred_cnn))
    metrics['CNN']['recall'].append(recall_score(y_test, y_pred_cnn))
    metrics['CNN']['f1'].append(f1_score(y_test, y_pred_cnn))

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
    
    y_pred_lstm = (lstm_model.predict(X_test_seq) > 0.5).astype("int32")
    metrics['LSTM-Attention']['accuracy'].append(accuracy_score(y_test, y_pred_lstm))
    metrics['LSTM-Attention']['precision'].append(precision_score(y_test, y_pred_lstm))
    metrics['LSTM-Attention']['recall'].append(recall_score(y_test, y_pred_lstm))
    metrics['LSTM-Attention']['f1'].append(f1_score(y_test, y_pred_lstm))

    # --- Model 3: Transformer ---
    inputs = Input(shape=(seq_length, len(fitur)))
    # Positional encoding (sederhana, menggunakan Dense layer)
    x = Dense(len(fitur))(inputs)
    # Transformer block
    attention_output = MultiHeadAttention(num_heads=2, key_dim=len(fitur))(x, x)
    attention_output = Dropout(0.1)(attention_output)
    residual = Add()([x, attention_output])
    norm1 = LayerNormalization(epsilon=1e-6)(residual)
    ff_output = Dense(64, activation='relu')(norm1)
    ff_output = Dense(len(fitur))(ff_output)
    ff_output = Dropout(0.1)(ff_output)
    residual = Add()([norm1, ff_output])
    norm2 = LayerNormalization(epsilon=1e-6)(residual)
    # Flatten dan Dense untuk klasifikasi
    flat = Flatten()(norm2)
    dense = Dense(50, activation='relu')(flat)
    dropout = Dropout(0.2)(dense)
    outputs = Dense(1, activation='sigmoid')(dropout)
    transformer_model = Model(inputs=inputs, outputs=outputs)
    transformer_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    transformer_model.fit(X_train_seq, y_train, epochs=100, batch_size=32,
                         validation_data=(X_test_seq, y_test), verbose=0,
                         callbacks=[early_stopping])
    
    y_pred_transformer = (transformer_model.predict(X_test_seq) > 0.5).astype("int32")
    metrics['Transformer']['accuracy'].append(accuracy_score(y_test, y_pred_transformer))
    metrics['Transformer']['precision'].append(precision_score(y_test, y_pred_transformer))
    metrics['Transformer']['recall'].append(recall_score(y_test, y_pred_transformer))
    metrics['Transformer']['f1'].append(f1_score(y_test, y_pred_transformer))

# Hitung rata-rata dan standar deviasi untuk setiap model
for model_name in metrics:
    print(f"\nResults for {model_name}:")
    mean_accuracy = np.mean(metrics[model_name]['accuracy'])
    std_accuracy = np.std(metrics[model_name]['accuracy'])
    mean_precision = np.mean(metrics[model_name]['precision'])
    std_precision = np.std(metrics[model_name]['precision'])
    mean_recall = np.mean(metrics[model_name]['recall'])
    std_recall = np.std(metrics[model_name]['recall'])
    mean_f1 = np.mean(metrics[model_name]['f1'])
    std_f1 = np.std(metrics[model_name]['f1'])
    
    print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Mean Precision: {mean_precision:.4f} ± {std_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f} ± {std_recall:.4f}")
    print(f"Mean F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")

# Visualisasi perbandingan metrik
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Comparison of Deep Learning Models (5-Fold Cross-Validation)', fontsize=16)

# Akurasi
means = [np.mean(metrics[model]['accuracy']) for model in metrics]
stds = [np.std(metrics[model]['accuracy']) for model in metrics]
axes[0, 0].bar(metrics.keys(), means, yerr=stds, capsize=5, color=['blue', 'green', 'orange'])
axes[0, 0].set_title('Accuracy')
axes[0, 0].set_ylim(0, 1)

# Presisi
means = [np.mean(metrics[model]['precision']) for model in metrics]
stds = [np.std(metrics[model]['precision']) for model in metrics]
axes[0, 1].bar(metrics.keys(), means, yerr=stds, capsize=5, color=['blue', 'green', 'orange'])
axes[0, 1].set_title('Precision')
axes[0, 1].set_ylim(0, 1)

# Recall
means = [np.mean(metrics[model]['recall']) for model in metrics]
stds = [np.std(metrics[model]['recall']) for model in metrics]
axes[1, 0].bar(metrics.keys(), means, yerr=stds, capsize=5, color=['blue', 'green', 'orange'])
axes[1, 0].set_title('Recall')
axes[1, 0].set_ylim(0, 1)

# F1-Score
means = [np.mean(metrics[model]['f1']) for model in metrics]
stds = [np.std(metrics[model]['f1']) for model in metrics]
axes[1, 1].bar(metrics.keys(), means, yerr=stds, capsize=5, color=['blue', 'green', 'orange'])
axes[1, 1].set_title('F1-Score')
axes[1, 1].set_ylim(0, 1)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('deep_learning_comparison.png')
plt.show()
