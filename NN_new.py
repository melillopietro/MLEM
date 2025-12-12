import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from keras import regularizers
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

tf.keras.utils.set_random_seed(123)

# Carica e prepara i dati (rimane uguale)
df = pd.read_csv('final_ml_dataset_encoded.csv')
counts = df['label_gang'].value_counts()
valid_classes = counts[counts > 5].index
df = df[df['label_gang'].isin(valid_classes)].reset_index(drop=True)

X = df.drop(columns='label_gang')
y = df['label_gang']

label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y)
index_to_label = dict(enumerate(label_encoder.classes_))

onehot_encoder = OneHotEncoder(sparse_output=False)
y_onehot = onehot_encoder.fit_transform(y_int.reshape(-1, 1))

# Suddivisione dataset con stratificazione
X_train, X_temp, y_train, y_temp, y_int_train, y_int_temp = train_test_split(
    X, y_onehot, y_int, test_size=0.3, random_state=42, stratify=y_int
)
X_val, X_test, y_val, y_test, y_int_val, y_int_test = train_test_split(
    X_temp, y_temp, y_int_temp, test_size=0.5, random_state=42, stratify=y_int_temp
)

# Normalizzazione dei dati
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Calcolo class weights per gestire lo sbilanciamento
class_weights = compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
class_weights = dict(enumerate(class_weights))

class FeedforwardNN():
    def __init__(self, input_dim, output_dim):
        input_layer = Input(shape=(input_dim,))
        
        # Architettura migliorata
        x = Dense(256, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        
        x = Dense(128, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.4)(x)
        
        x = Dense(64, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        
        x = Dense(output_dim, kernel_initializer='he_normal')(x)
        output_layer = Activation('softmax')(x)
        
        self.classifier = Model(inputs=input_layer, outputs=output_layer)

    def summary(self):
        self.classifier.summary()

    def train(self, x, y, x_val, y_val, class_weights):
        optimizer = Adam(learning_rate=0.001, weight_decay=1e-4)
        self.classifier.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
        ]

        history = self.classifier.fit(
            x, y,
            validation_data=(x_val, y_val),
            epochs=200,
            batch_size=128,
            class_weight=class_weights,
            callbacks=callbacks,
            shuffle=True,
            verbose=2
        )

        # Plot della loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot dell'accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

        return pd.DataFrame(history.history)

    def predict(self, x_evaluation):
        probs = self.classifier.predict(x_evaluation)
        return np.argmax(probs, axis=1)

# Istanzia e allena il modello
input_dim = X_train.shape[1]
output_dim = y_onehot.shape[1]

nn = FeedforwardNN(input_dim, output_dim)
nn.summary()
history_df = nn.train(X_train, y_train, X_val, y_val, class_weights)

# Valutazione finale
pred_int = nn.predict(X_test)
pred_labels = [index_to_label[i] for i in pred_int]
true_labels = [index_to_label[i] for i in y_int_test]

print(classification_report(true_labels, pred_labels, output_dict=True))
nn.classifier.save("FFNN_model.keras")
print("Model saved as FFNN_model.keras")