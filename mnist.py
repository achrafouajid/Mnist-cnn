import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


train = pd.read_csv("train.csv")  
Y_train = train["label"]
X_train = train.drop(labels=["label"], axis=1)

X_train = X_train.values.reshape(-1, 28, 28, 1)
X_train = X_train / 255.0
Y_train = to_categorical(Y_train, num_classes=10)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=2)


def build_cnn():
    model = Sequential([
        Conv2D(32, (5,5), activation='relu', padding='same', input_shape=(28,28,1)),
        BatchNormalization(),
        Conv2D(32, (5,5), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.10,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train)

lr_reducer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, verbose=1, min_lr=1e-5)


nets = 5
models = [build_cnn() for _ in range(nets)]

for i, m in enumerate(models):
    print(f"\n🔹 Training model {i+1}/{nets}...\n")
    checkpoint = ModelCheckpoint(
        f"model_{i+1}.h5",
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    m.fit(
        datagen.flow(X_train, Y_train, batch_size=64),
        epochs=45,
        validation_data=(X_val, Y_val),
        callbacks=[lr_reducer, checkpoint],
        steps_per_epoch=X_train.shape[0] // 64,
        verbose=1
    )


models = [load_model(f"model_{i+1}.h5") for i in range(nets)]

test = pd.read_csv("test.csv")
X_test = test.values.reshape(-1, 28, 28, 1)
X_test = X_test / 255.0

ensemble_pred = np.zeros((X_test.shape[0], 10))
for m in models:
    ensemble_pred += m.predict(X_test, verbose=1)
ensemble_pred /= nets

results = np.argmax(ensemble_pred, axis=1)
submission = pd.DataFrame({
    "ImageId": range(1, len(results) + 1),
    "Label": results
})
submission.to_csv("mnist_ensemble_submission.csv", index=False)
print("\n✅ Saved ensemble predictions to mnist_ensemble_submission.csv")


best_model = build_cnn()
avg_weights = [np.mean([m.get_weights()[i] for m in models], axis=0) for i in range(len(models[0].get_weights()))]
best_model.set_weights(avg_weights)
best_model.save("best_mnist_cnn.h5")
print("\n💾 Saved ensemble averaged model as best_mnist_cnn.h5")
