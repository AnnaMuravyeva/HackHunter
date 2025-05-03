import numpy as np
from sklearn.model_selection import train_test_split

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

dataset = 'keypoint.csv'
model_save_path = 'keypoint_classifier.hdf5'
tflite_save_path = 'keypoint_classifier.tflite'
NUM_CLASSES = 26


def normalize_keypoints(pts):
    """
    Center on wrist (landmark 0) and scale so the max distance = 1.

    Args:
        pts (np.array): shape (21,2) of raw (x,y) landmarks
    Returns:
        np.array: flattened shape (42,) normalized landmarks
    """
    wrist = pts[0]
    centered = pts - wrist
    scale = np.max(np.linalg.norm(centered, axis=1)) + 1e-6
    return (centered / scale).flatten()


def augment_keypoints(pts_norm):
    """
    Generate simple augmentations: rotations, flip, jitter.

    Args:
        pts_norm (np.array): flattened (42,) normalized landmarks
    Returns:
        List[np.array]: list of flattened augmented arrays
    """
    pts = pts_norm.reshape(-1, 2)
    aug = []
    # rotate ±15°
    for angle in (-15, 15):
        theta = np.deg2rad(angle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        aug.append((pts @ R.T).flatten())
    # horizontal flip
    f = pts.copy()
    f[:, 0] *= -1
    aug.append(f.flatten())
    # jitter
    noise = np.random.normal(0, 0.01, pts.shape)
    aug.append((pts + noise).flatten())
    return aug


# ——— Load & preprocess data ———
raw = np.loadtxt(dataset, delimiter=',', dtype='float32',
                 usecols=list(range(1, (21 * 2) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0,))

# Normalize
X_norm = np.array([
    normalize_keypoints(raw[i].reshape(21, 2))
    for i in range(len(raw))
])

# Augment
X_all, y_all = [], []
for x, y in zip(X_norm, y_dataset):
    X_all.append(x)
    y_all.append(y)
    for aug in augment_keypoints(x):
        X_all.append(aug)
        y_all.append(y)
X_all = np.array(X_all)
y_all = np.array(y_all)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, train_size=0.8, random_state=RANDOM_SEED
)


# ——— Build & compile model ———
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(21 * 2,)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True, verbose=1
    )
]

# ——— Train ———
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=callbacks
)

# ——— Evaluate ———
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")

# ——— Save models ———
model.save(model_save_path)

# Convert to TFLite
tflite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = tflite_converter.convert()
with open(tflite_save_path, 'wb') as f:
    f.write(tflite_model)



