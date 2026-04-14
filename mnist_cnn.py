import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def main():
    # =========================================================================
    # 1. Data Loading and Preprocessing
    # =========================================================================
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize the pixel values to the range [0, 1] by dividing by 255.
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Reshape the images to (28, 28, 1) to include the single grayscale channel dimension.
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    # One-hot encode the labels using Keras's to_categorical function.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    # =========================================================================
    # 2. Data Augmentation
    # =========================================================================
    # Configure an ImageDataGenerator for the training set.
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    # Note: Augmentation is not applied to x_test/y_test. We will pass x_test directly to evaluation.

    # =========================================================================
    # 3. Model Architecture (LeNet-inspired)
    # =========================================================================
    model = models.Sequential([
        # Input Layer
        layers.Input(shape=(28, 28, 1)),

        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        # Output Layer
        layers.Dense(10, activation='softmax')
    ])
    
    model.summary()

    # =========================================================================
    # 4. Model Compilation and Training
    # =========================================================================
    # Compile the model using the Adam optimizer with learning rate of 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Implement a ReduceLROnPlateau callback
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        verbose=1, 
        min_lr=1e-6
    )

    epochs = 30
    batch_size = 64

    # Train the model using the augmented data generator
    print("\nStarting Training...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) // batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[reduce_lr]
    )

    # Save the trained model so the drawing app can use it
    model.save('mnist_model.keras')
    print("\nModel saved to 'mnist_model.keras'")

    # =========================================================================
    # 5. Evaluation and Visualization
    # =========================================================================
    # Print the final test accuracy and test loss
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nFinal Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}\n")

    # Plot graphs side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Graph 1: Training and Validation Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    axes[0].set_title('Training and Validation Accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Graph 2: Training and Validation Loss
    axes[1].plot(history.history['loss'], label='Train Loss', color='blue')
    axes[1].plot(history.history['val_loss'], label='Validation Loss', color='orange')
    axes[1].set_title('Training and Validation Loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # Generate predictions on the test set for the Confusion Matrix
    # Using np.argmax to convert one-hot encoded predictions and true labels back to integers
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Plot Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix for MNIST')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

if __name__ == "__main__":
    main()
