# model_train.py
import os, cv2, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils
from sklearn.model_selection import train_test_split
from collections import Counter

CSV = "data.csv"
IMG_SIZE = (64,64)
BATCH = 64
EPOCHS = 20
MODEL_OUT = "gaze_cnn.h5"
SEED = 42

def load_img(p):
    x = cv2.imread(p)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return cv2.resize(x, IMG_SIZE).astype("float32")/255.0

def augment_pair(L,R):
    if np.random.rand() < 0.5:
        delta = (np.random.rand()*0.2 - 0.1)
        L = np.clip(L+delta, 0, 1); R = np.clip(R+delta, 0, 1)
    if np.random.rand() < 0.3:
        L = cv2.GaussianBlur(L,(3,3),0); R = cv2.GaussianBlur(R,(3,3),0)
    return L,R

def make_dataset(df, num_classes, batch=BATCH, shuffle=True, augment=False):
    pathsL = df["left"].values
    pathsR = df["right"].values
    ys = df["label_idx"].values

    def gen():
        for l,r,y in zip(pathsL, pathsR, ys):
            L = load_img(l); R = load_img(r)
            if augment: L,R = augment_pair(L,R)
            X = np.concatenate([L,R], axis=-1) # (H,W,6)
            yield X.astype("float32"), y

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE[0],IMG_SIZE[1],6), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    if shuffle: ds = ds.shuffle(1000, seed=SEED)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    ds = ds.map(lambda x,y: (x, tf.one_hot(y, num_classes)))
    return ds

def build(input_shape=(64,64,6), num_classes=6):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp); x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x);  x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128,3, padding="same", activation="relu")(x);  x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x); x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

def main():
    df = pd.read_csv(CSV)
    labels = sorted(df["label"].unique())
    l2i = {l:i for i,l in enumerate(labels)}
    df["label_idx"] = df["label"].map(l2i)
    num_classes = len(labels)

    tr, va = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df["label_idx"])

    # class weights for imbalance
    counts = Counter(tr["label_idx"].values)
    total = sum(counts.values())
    class_weight = {cls: total/(len(counts)*cnt) for cls,cnt in counts.items()}
    print("labels:", labels)
    print("train counts:", dict(counts))
    print("class_weight:", class_weight)

    train_ds = make_dataset(tr, num_classes, augment=True)
    val_ds   = make_dataset(va, num_classes, shuffle=False, augment=False)

    model = build(num_classes=num_classes)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    ck = callbacks.ModelCheckpoint(MODEL_OUT, monitor="val_loss", save_best_only=True)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=[es, ck],
        verbose=1
    )

    print("saved ->", MODEL_OUT)
    with open("labels.txt","w") as f: f.write("\n".join(labels))
    print("labels -> labels.txt :", labels)

if __name__ == "__main__":
    main()
