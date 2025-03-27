import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Layer
from tensorflow.keras.layers import Attention
import tensorflow.keras.backend as K
import shap

# Load dataset (sesuaikan path dengan dataset Anda)
data = pd.read_csv('/content/SDN_DDoS_dataset.csv')
data_clean = data.dropna()

# Distribusi label (sebelum sampling)
print("Distribusi label setelah dropna:")
print(data_clean['label'].value_counts())

# --- Grafik 1: Distribusi Label (Opsional) ---
plt.figure(figsize=(6, 4))
data_clean['label'].value_counts().plot(kind='bar', color=['blue', 'red'])
plt.xticks(ticks=[0, 1], labels=['Normal', 'DDoS'], rotation=0)
plt.title('Distribusi Label')
plt.xlabel('Kelas')
plt.ylabel('Jumlah Sampel')
plt.show()

# Sampling untuk menyeimbangkan dataset
normal_data = data_clean[data_clean['label'] == 0].sample(n=7500, random_state=42)
ddos_data = data_clean[data_clean['label'] == 1].sample(n=7500, random_state=42)
data_balanced = pd.concat([normal_data, ddos_data])

# --- Grafik 2: Heatmap Korelasi Fitur (Opsional) ---
correlation_matrix = data_balanced.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='YlOrRd', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Features')
plt.show()

# Persiapan data
fitur = ['timestamp', 'packet_count', 'byte_count', 'packet_count_per_second', 'byte_count_per_second']
X = data_balanced[fitur].values
y = data_balanced['label'].values

# Normalisasi data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape data untuk LSTM (seq_length=10)
seq_length = 10
X_seq = []
y_seq = []
for i in range(len(X) - seq_length):
    X_seq.append(X[i:i + seq_length])
    y_seq.append(y[i + seq_length])
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Custom Attention Layer untuk mengakses bobot attention
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

    def get_attention_weights(self):
        return self.attention_weights

# Inisialisasi untuk cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies, precisions, recalls, f1_scores = [], [], [], []
histories = []
fprs, tprs, aucs = [], [], []
cms = []

# 10-Fold Cross-Validation
for fold, (train_index, test_index) in enumerate(kf.split(X_seq)):
    print(f'Fold {fold + 1}')
    X_train, X_test = X_seq[train_index], X_seq[test_index]
    y_train, y_test = y_seq[train_index], y_seq[test_index]

    # Definisikan model LSTM-Attention
    inputs = Input(shape=(seq_length, len(fitur)))
    lstm_out = LSTM(50, return_sequences=True)(inputs)
    attention_layer = CustomAttention()
    attention_out = attention_layer(lstm_out)
    lstm_final = LSTM(50, return_sequences=False)(lstm_out)
    dropout = Dropout(0.2)(lstm_final)
    outputs = Dense(1, activation='sigmoid')(dropout)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Latih model
    history = model.fit(X_train, y_train, epochs=250, batch_size=32,
                        validation_data=(X_test, y_test), verbose=1)
    histories.append(history.history)

    # Evaluasi model
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    # Hitung ROC curve untuk fold ini
    y_pred_proba = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    fprs.append(fpr)
    tprs.append(tpr)
    aucs.append(auc(fpr, tpr))

    # Hitung confusion matrix untuk fold ini
    cm = confusion_matrix(y_test, y_pred)
    cms.append(cm)

# Hitung rata-rata dan standar deviasi metrik
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
mean_precision = np.mean(precisions)
std_precision = np.std(precisions)
mean_recall = np.mean(recalls)
std_recall = np.std(recalls)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print(f"10-Fold Cross-Validation Results:")
print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"Mean Precision: {mean_precision:.4f} ± {std_precision:.4f}")
print(f"Mean Recall: {mean_recall:.4f} ± {std_recall:.4f}")
print(f"Mean F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")

# --- Grafik 3: Training dan Validation Loss Plot (Wajib) ---
avg_loss = np.mean([h['loss'] for h in histories], axis=0)
avg_val_loss = np.mean([h['val_loss'] for h in histories], axis=0)
std_loss = np.std([h['loss'] for h in histories], axis=0)
std_val_loss = np.std([h['val_loss'] for h in histories], axis=0)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(avg_loss, label='Average Loss', color='blue')
plt.plot(avg_val_loss, label='Average Validation Loss', color='red')
plt.fill_between(range(len(avg_loss)), avg_loss - std_loss, avg_loss + std_loss, color='blue', alpha=0.1)
plt.fill_between(range(len(avg_val_loss)), avg_val_loss - std_val_loss, avg_val_loss + std_val_loss, color='red', alpha=0.1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Average Loss Across Folds')
plt.legend()

# --- Grafik 4: Training dan Validation Accuracy Plot (Wajib) ---
avg_accuracy = np.mean([h['accuracy'] for h in histories], axis=0)
avg_val_accuracy = np.mean([h['val_accuracy'] for h in histories], axis=0)
std_accuracy = np.std([h['accuracy'] for h in histories], axis=0)
std_val_accuracy = np.std([h['val_accuracy'] for h in histories], axis=0)

plt.subplot(1, 2, 2)
plt.plot(avg_accuracy, label='Average Accuracy', color='blue')
plt.plot(avg_val_accuracy, label='Average Validation Accuracy', color='red')
plt.fill_between(range(len(avg_accuracy)), avg_accuracy - std_accuracy, avg_accuracy + std_accuracy, color='blue', alpha=0.1)
plt.fill_between(range(len(avg_val_accuracy)), avg_val_accuracy - std_val_accuracy, avg_val_accuracy + std_val_accuracy, color='red', alpha=0.1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Average Accuracy Across Folds')
plt.legend()

plt.tight_layout()
plt.show()

# --- Grafik 5: Bar Chart Metrik Cross-Validation (Wajib) ---
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
means = [mean_accuracy, mean_precision, mean_recall, mean_f1]
stds = [std_accuracy, std_precision, std_recall, std_f1]

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, means, yerr=stds, capsize=5, color=['blue', 'green', 'orange', 'red'])
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Cross-Validation Metrics (Mean ± Std)')
plt.ylim(0, 1)

# Tambahkan nilai numerik di atas setiap bar
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.05, f'{means[i]:.4f}\n±{stds[i]:.4f}', ha='center')

plt.show()

# --- Grafik 6: Mean ROC Curve (Wajib) ---
mean_fpr = np.linspace(0, 1, 100)
mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fprs, tprs)], axis=0)
mean_auc = np.mean(aucs)
std_auc = np.std(aucs)

plt.figure(figsize=(6, 6))
plt.plot(mean_fpr, mean_tpr, label=f'Mean ROC curve (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mean ROC Curve Across 10 Folds')
plt.legend()
plt.show()

# --- Grafik 7: Total Confusion Matrix (Opsional) ---
total_cm = np.sum(cms, axis=0)

plt.figure(figsize=(6, 4))
sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'DDoS'], yticklabels=['Normal', 'DDoS'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Total Confusion Matrix Across 10 Folds')
plt.show()

# --- Grafik 8: Attention Weights Visualization (Opsional) ---
attention_weights = attention_layer.get_attention_weights()
attention_weights = attention_weights.numpy()  # Untuk TensorFlow 2.x

plt.figure(figsize=(10, 6))
plt.bar(range(seq_length), attention_weights[0, :, 0])
plt.title('Attention Weights for First Sample')
plt.xlabel('Timestep')
plt.ylabel('Attention Weight')
plt.show()

# --- Grafik 9: Feature Importance menggunakan SHAP (Opsional) ---
# Catatan: SHAP membutuhkan instalasi tambahan (pip install shap)
explainer = shap.DeepExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=fitur, plot_type="bar")
plt.title('Feature Importance using SHAP')
plt.show()

# Simpan semua grafik dalam satu file (opsional)
plt.savefig('cross_validation_metrics.png')
