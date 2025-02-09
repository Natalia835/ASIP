import tensorflow as tf
from keras._tf_keras.keras.applications import ResNet50
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.applications.efficientnet import preprocess_input
from keras import layers, models, optimizers, callbacks
import numpy as np
import matplotlib.pyplot as plt

# Załadowanie danych CIFAR-100
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

# Konwersja etykiet na one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 100)
y_test = tf.keras.utils.to_categorical(y_test, 100)

# Augmentacja danych
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2
)
train_datagen.fit(x_train)

# Dane treningowe i walidacyjne
train_data_gen = train_datagen.flow(x_train, y_train, batch_size=128, subset="training")
valid_data_gen = train_datagen.flow(x_train, y_train, batch_size=128, subset="validation")

# Inicjalizacja modelu ResNet50
base_model = ResNet50(include_top=False, input_shape=(32, 32, 3), weights='imagenet')
base_model.trainable = False  # Na początku zamrażamy całą sieć

model = models.Sequential([
    base_model,
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(100, activation='softmax')
])

optimizer = optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

lr_reduction = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint('best_resnet50.keras', save_best_only=True, monitor='val_accuracy', mode='max')

# Trenowanie modelu dla pierwszych 20 epok
epochs_initial = 20
history_initial = model.fit(train_data_gen, epochs=epochs_initial, validation_data=valid_data_gen,
                            callbacks=[lr_reduction, early_stopping, checkpoint])

# Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# Dodatkowe trenowanie
epochs_finetune = 70
history_finetune = model.fit(train_data_gen, epochs=epochs_finetune, validation_data=valid_data_gen,
                             callbacks=[lr_reduction, early_stopping, checkpoint])

# Ocena modelu
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Final Test Accuracy: {test_acc * 100:.2f}%')

def combine_histories(hist1, hist2, key):
    return hist1.history[key] + hist2.history[key]

full_accuracy = combine_histories(history_initial, history_finetune, 'accuracy')
full_val_accuracy = combine_histories(history_initial, history_finetune, 'val_accuracy')
full_loss = combine_histories(history_initial, history_finetune, 'loss')
full_val_loss = combine_histories(history_initial, history_finetune, 'val_loss')

# Wykresy dla fine-tuningu
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_finetune.history['accuracy'], label='Dokładność treningu')
plt.plot(history_finetune.history['val_accuracy'], label='Dokładność walidacji')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.title('Dokładność - Fine-tuning')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_finetune.history['loss'], label='Strata treningu')
plt.plot(history_finetune.history['val_loss'], label='Strata walidacji')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.title('Strata - Fine-tuning')
plt.legend()

plt.tight_layout()
plt.show()

# Wykresy dla całego okresu trenowania
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(full_accuracy, label='Dokładność treningu')
plt.plot(full_val_accuracy, label='Dokładność walidacji')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.title('Dokładność - Całe trenowanie')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(full_loss, label='Strata treningu')
plt.plot(full_val_loss, label='Strata walidacji')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.title('Strata - Całe trenowanie')
plt.legend()

plt.tight_layout()
plt.show()

# Ocena jakości modelu na zbiorze testowym
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Accuracy on test data: {test_acc*100:.2f}%')

# Wyświetlenie 20 losowych obrazków testowych i ich predykcji
class_names = ["class_" + str(i) for i in range(100)]
random_indices = np.random.choice(len(x_test), 20, replace=False)
random_images = x_test[random_indices]
random_labels = np.argmax(y_test[random_indices], axis=1)
predictions = np.argmax(model.predict(random_images), axis=1)

plt.figure(figsize=(20, 8))
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(random_images[i])
    plt.title(f"True: {class_names[random_labels[i]]}\nPred: {class_names[predictions[i]]}")
    plt.axis('off')
plt.show()