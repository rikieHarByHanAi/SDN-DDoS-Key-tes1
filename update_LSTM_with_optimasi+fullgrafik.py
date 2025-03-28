import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Layer
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
        return context, alpha  # Kembalikan context dan attention weights

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1], 1)]

# Inisialisasi untuk cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Simpan metrik untuk LSTM-Attention
metrics = {
    'LSTM-Attention': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'threshold': []}
}

# Simpan history pelatihan untuk learning curve
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Rentang threshold yang akan diuji
thresholds = np.arange(0.5, 0.96, 0.01)  # Langkah 0.01 untuk hasil yang lebih halus

# Simpan probabilitas prediksi dan label sebenarnya untuk ROC dan distribusi
all_y_test = []
all_y_pred_prob = []

# Simpan attention weights untuk visualisasi
all_attention_weights = []

# 5-Fold Cross-Validation
for fold, (train_index, test_index) in enumerate(kf.split(X_seq)):
    print(f'Fold {fold + 1}')
    X_train_seq, X_test_seq = X_seq[train_index], X_seq[test_index]
    y_train, y_test = y_seq[train_index], y_seq[test_index]

    # --- Model: LSTM-Attention (Dioptimalkan) ---
    inputs = Input(shape=(seq_length, len(fitur)))
    lstm_out = LSTM(32, return_sequences=True)(inputs)  # Kurangi unit menjadi 32, hanya 1 layer LSTM
    attention_layer = CustomAttention()
    attention_out, attention_weights = attention_layer(lstm_out)  # Dapatkan context dan attention weights
    dropout = Dropout(0.3)(attention_out)  # Tambah dropout untuk mencegah overfitting
    outputs = Dense(1, activation='sigmoid')(dropout)
    lstm_model = Model(inputs=inputs, outputs=outputs)

    # Buat model sementara untuk mendapatkan attention weights
    attention_model = Model(inputs=inputs, outputs=[outputs, attention_weights])

    # Compile model dengan learning rate yang lebih kecil
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    lstm_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Latih model
    history = lstm_model.fit(X_train_seq, y_train, epochs=100, batch_size=64,  # Batch size lebih besar
                             validation_data=(X_test_seq, y_test), verbose=0,
                             callbacks=[early_stopping])
    
    # Simpan history untuk learning curve
    train_losses.append(history.history['loss'])
    val_losses.append(history.history['val_loss'])
    train_accuracies.append(history.history['accuracy'])
    val_accuracies.append(history.history['val_accuracy'])

    # Simpan probabilitas prediksi dan attention weights
    y_pred_prob_lstm, attention_weights = attention_model.predict(X_test_seq)
    print(f"LSTM-Attention Fold {fold + 1} - Probabilitas Prediksi: Min={y_pred_prob_lstm.min():.4f}, Max={y_pred_prob_lstm.max():.4f}")

    # Simpan untuk ROC dan distribusi
    all_y_test.extend(y_test)
    all_y_pred_prob.extend(y_pred_prob_lstm.flatten())

    # Simpan attention weights
    all_attention_weights.append(attention_weights)

    # Optimasi threshold untuk LSTM-Attention
    best_f1_lstm = -1
    best_threshold_lstm = 0.5
    best_metrics_lstm = {}
    for threshold in thresholds:
        y_pred_lstm = (y_pred_prob_lstm > threshold).astype("int32")
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
    
    # Pastikan metrik selalu disimpan
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

# Hitung rata-rata dan standar deviasi untuk metrik
model_name = 'LSTM-Attention'
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

# --- Grafik Wajib 1: Perbandingan Metrik ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('LSTM-Attention Metrics (5-Fold Cross-Validation)', fontsize=16)

# Akurasi
axes[0, 0].bar(['LSTM-Attention'], [mean_accuracy], yerr=[std_accuracy], capsize=5, color='green')
axes[0, 0].set_title('Accuracy')
axes[0, 0].set_ylim(0, 1)

# Presisi
axes[0, 1].bar(['LSTM-Attention'], [mean_precision], yerr=[std_precision], capsize=5, color='green')
axes[0, 1].set_title('Precision')
axes[0, 1].set_ylim(0, 1)

# Recall
axes[1, 0].bar(['LSTM-Attention'], [mean_recall], yerr=[std_recall], capsize=5, color='green')
axes[1, 0].set_title('Recall')
axes[1, 0].set_ylim(0, 1)

# F1-Score
axes[1, 1].bar(['LSTM-Attention'], [mean_f1], yerr=[std_f1], capsize=5, color='green')
axes[1, 1].set_title('F1-Score')
axes[1, 1].set_ylim(0, 1)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('lstm_attention_metrics.png')
plt.show()

# --- Grafik Wajib 2: Distribusi Probabilitas Prediksi ---
plt.figure(figsize=(10, 6))
sns.histplot(all_y_pred_prob, bins=50, kde=True, color='green')
plt.title('Distribution of Predicted Probabilities (LSTM-Attention)')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.savefig('lstm_attention_prob_distribution.png')
plt.show()

# --- Grafik Opsional 1: Attention Weights ---
# Rata-rata attention weights di seluruh sampel
avg_attention_weights = np.mean(np.concatenate(all_attention_weights, axis=0), axis=0).flatten()  # Shape: (seq_length,)
plt.figure(figsize=(10, 6))
plt.bar(range(seq_length), avg_attention_weights, color='green')
plt.title('Average Attention Weights per Timestep (LSTM-Attention)')
plt.xlabel('Timestep')
plt.ylabel('Attention Weight')
plt.savefig('lstm_attention_weights.png')
plt.show()

# --- Grafik Opsional 2: Learning Curve ---
# Rata-rata loss dan akurasi di seluruh fold
min_epochs = min(len(loss) for loss in train_losses)
avg_train_loss = np.mean([loss[:min_epochs] for loss in train_losses], axis=0)
avg_val_loss = np.mean([loss[:min_epochs] for loss in val_losses], axis=0)
avg_train_acc = np.mean([acc[:min_epochs] for acc in train_accuracies], axis=0)
avg_val_acc = np.mean([acc[:min_epochs] for acc in val_accuracies], axis=0)

# Plot learning curve (loss)
plt.figure(figsize=(10, 6))
plt.plot(avg_train_loss, label='Training Loss', color='blue')
plt.plot(avg_val_loss, label='Validation Loss', color='red')
plt.title('Learning Curve - Loss (LSTM-Attention)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('lstm_attention_learning_curve_loss.png')
plt.show()

# Plot learning curve (accuracy)
plt.figure(figsize=(10, 6))
plt.plot(avg_train_acc, label='Training Accuracy', color='blue')
plt.plot(avg_val_acc, label='Validation Accuracy', color='red')
plt.title('Learning Curve - Accuracy (LSTM-Attention)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('lstm_attention_learning_curve_accuracy.png')
plt.show()

# --- Grafik Opsional 3: ROC Curve dan AUC ---
fpr, tpr, _ = roc_curve(all_y_test, all_y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='green', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (LSTM-Attention)')
plt.legend(loc="lower right")
plt.savefig('lstm_attention_roc_curve.png')
plt.show()

# --- Grafik Opsional 4: Confusion Matrix ---
# Gunakan threshold optimal untuk menghitung confusion matrix
y_pred_final = (np.array(all_y_pred_prob) > mean_threshold).astype("int32")
cm = confusion_matrix(all_y_test, y_pred_final)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['Normal', 'DDoS'], yticklabels=['Normal', 'DDoS'])
plt.title('Confusion Matrix (LSTM-Attention)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('lstm_attention_confusion_matrix.png')
plt.show()
