import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator 

# Załadowanie CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalizacja wartości pikseli do zakresu [0,1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Konwersja etykiet na one-hot encoding - 10 klas
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Augmentacja danych
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# Definicja modelu CNN
def create_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', 
                      kernel_regularizer=keras.regularizers.l2(0.001), 
                      input_shape=(32,32,3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(64, (3,3), activation='relu', padding='same', 
                      kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(128, (3,3), activation='relu', padding='same', 
                      kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

model = create_model()
model.summary()

lr_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

# Trenowanie modelu
epochs = 50
history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                    epochs=epochs, validation_data=(x_test, y_test),
                    callbacks=[lr_reduction, early_stopping])

# Wizualizacja wyników
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Dokładność treningu')
plt.plot(history.history['val_accuracy'], label='Dokładność walidacji')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Strata treningu')
plt.plot(history.history['val_loss'], label='Strata walidacji')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()

plt.show()

# Ocena jakości modelu na zbiorze testowym
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Accuracy on test data: {test_acc*100:.2f}%')

# Wyświetlenie 20 losowych obrazków testowych i ich predykcji
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
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
