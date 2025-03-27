# Simpan kode ke file Python
code = """
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns
import os
from google.colab import drive

# Mount Google Drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')
else:
    print("Google Drive sudah dimount.")

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
plt.show()

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

# 10-fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []
precisions = []
recalls = []
f1_scores = []
histories = []

fold_no = 1
for train_index, test_index in kf.split(X):
    print(f'Fold {fold_no}...')
    
    # Split data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
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
                        validation_data=(X_test, y_test), verbose=1)
    histories.append(history.history)
    
    # Cek hasil
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Fold {fold_no} - Akurasi di data test: {accuracy*100:.2f}%")
    
    # Prediksi dan hitung metrik
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    
    print(f"Fold {fold_no} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    fold_no += 1

# Calculate mean and standard deviation of metrics
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
mean_precision = np.mean(precisions)
std_precision = np.std(precisions)
mean_recall = np.mean(recalls)
std_recall = np.std(recalls)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

# Print final results
print("\n10-Fold Cross-Validation Results:")
print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"Mean Precision: {mean_precision:.4f} ± {std_precision:.4f}")
print(f"Mean Recall: {mean_recall:.4f} ± {std_recall:.4f}")
print(f"Mean F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")

# Plot the cross-validation metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
means = [mean_accuracy, mean_precision, mean_recall, mean_f1]
stds = [std_accuracy, std_precision, std_recall, std_f1]

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, means, yerr=stds, capsize=5, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], edgecolor='black')
plt.title('10-Fold Cross-Validation Metrics for LSTM-Attention Model', fontsize=14, pad=15)
plt.ylabel('Score', fontsize=12)
plt.ylim(0, 1.1)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.4f}', ha='center', va='bottom', fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('cross_validation_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualisasi Loss dan Accuracy (Rata-rata dari semua fold)
plt.style.use('default')
plt.figure(figsize=(12, 4))

# Rata-rata Loss
avg_loss = np.mean([h['loss'] for h in histories], axis=0)
avg_val_loss = np.mean([h['val_loss'] for h in histories], axis=0)
plt.subplot(1, 2, 1)
plt.plot(avg_loss, label='Average Loss', color='blue')
plt.plot(avg_val_loss, label='Average Validation Loss', color='red')
plt.title('Average Training and Validation Loss Across Folds')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0, 1.0)
plt.legend()

# Rata-rata Accuracy
avg_accuracy = np.mean([h['accuracy'] for h in histories], axis=0)
avg_val_accuracy = np.mean([h['val_accuracy'] for h in histories], axis=0)
plt.subplot(1, 2, 2)
plt.plot(avg_accuracy, label='Average Accuracy', color='blue')
plt.plot(avg_val_accuracy, label='Average Validation Accuracy', color='red')
plt.title('Average Training and Validation Accuracy Across Folds')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0.5, 1.0)
plt.legend()

plt.tight_layout()
plt.show()

# ROC Curve (Menggunakan fold terakhir untuk contoh)
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
plt.title('Receiver Operating Characteristic (ROC) Curve - Last Fold')
plt.legend(loc="lower right")
plt.show()
"""

# Tulis kode ke file
with open('lstm_attention_ddos_with_cross_validation.py', 'w') as f:
    f.write(code)
print("File 'lstm_attention_ddos_with_cross_validation.py' telah disimpan.")
