import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Normalization
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import os

# ------------------- Load Dataset ------------------- #
house_df = pd.read_csv("/Users/rahmani/Documents/Assets_ML/Datasets-master/house price.csv")

# One-hot encode 'city'
encoder = OneHotEncoder(sparse_output=False)
city_encoded = encoder.fit_transform(house_df[["city"]])
city_df = pd.DataFrame(city_encoded, columns=encoder.get_feature_names_out(["city"]))

# Combine back with other features
house_df = pd.concat([house_df.drop(columns=["city"]), city_df], axis=1)

# Drop irrelevant columns
house_df = house_df.drop(['street','statezip','country','date'], axis=1)

# Shuffle dataset
house_shuffled = house_df.sample(frac=1).reset_index(drop=True)

# Features & Target
X = tf.cast(house_shuffled.iloc[:, 1:].values, tf.float32)
y = tf.cast(house_shuffled.iloc[:, 0].values, tf.float32)
y = tf.expand_dims(y, -1)

# Normalize target
y_mean = tf.reduce_mean(y)
y_std = tf.math.reduce_std(y)
y_norm = (y - y_mean) / y_std

# Train-validation split
train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y_norm[:train_size]
X_val, y_val = X[train_size:], y_norm[train_size:]

# ------------------- TensorFlow Dataset ------------------- #
BUFFER_SIZE = 4600
BATCH_SIZE = 64

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ------------------- Normalization ------------------- #
normalizer = Normalization()
normalizer.adapt(X_train)

# ------------------- R2 Metric ------------------- #
def r2_score(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / ss_tot

# ------------------- Model ------------------- #
model = tf.keras.Sequential([
    Input(shape=(X_train.shape[1],)),
    normalizer,
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer=Adam(0.001),
    loss='mean_absolute_error',
    metrics=['mean_squared_error', r2_score]
)

# ------------------- Callbacks ------------------- #
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-6
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# ------------------- Train Model ------------------- #
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=150,
    callbacks=[reduce_lr, early_stop]
)

# ------------------- Model Summary ------------------- #
model.summary()

# ------------------- Plot Training ------------------- #
plt.plot(history.history['loss'], label='Train Loss (MAE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MAE)')
plt.xlabel('Epoch')
plt.ylabel('MAE Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

# ------------------- Predict & Denormalize ------------------- #
y_pred_norm = model.predict(X_val)
y_pred = y_pred_norm * y_std + y_mean