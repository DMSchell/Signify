import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATASET_DIR = "dataset"
MODEL_PATH = "hand_model.keras"
LABELS_PATH = "labels.npy"

EPOCHS = 35
BATCH = 32
SEED = 42

def load_dataset(dataset_dir):
    X_list, y_list = [], []
    npz_files = glob.glob(os.path.join(dataset_dir, "*", "*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz clips found in {dataset_dir}/<label>/")

    for path in npz_files:
        data = np.load(path, allow_pickle=True)
        X = data["X"].astype(np.float32)     # [T,126]
        y = str(data["y"])
        X_list.append(X)
        y_list.append(y)

    X_all = np.stack(X_list, axis=0)  # [N,T,126]
    y_all = np.array(y_list)
    return X_all, y_all

def build_model(T, F, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(T, F)),
        tf.keras.layers.GRU(128, return_sequences=False),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    X, y_text = load_dataset(DATASET_DIR)
    N, T, F = X.shape
    print("Loaded:", X.shape, "labels:", len(set(y_text)))

    le = LabelEncoder()
    y = le.fit_transform(y_text)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    model = build_model(T, F, num_classes=len(le.classes_))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True
        )
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=callbacks
    )

    model.save(MODEL_PATH)
    np.save(LABELS_PATH, le.classes_)
    print("Saved:", MODEL_PATH, "and", LABELS_PATH)
    print("Classes:", list(le.classes_))

if __name__ == "__main__":
    main()
