import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Paths to training, validation, and test data
train_dir = 'dataset (divided)/Train'  # Path to training data
val_dir = 'dataset (divided)/Val'      # Path to validation data
test_dir = 'dataset (divided)/Test'     # Path to test data

# Data preprocessing and augmentation
print("Preparing data generators...")

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Load the DenseNet121 model
print("Loading DenseNet121 model...")

base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers to retain pre-trained weights
base_model.trainable = False

# Add custom layers for glaucoma detection
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Dropout for regularization
output = Dense(1, activation='sigmoid')(x)  # Binary classification

# Final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("Starting initial training...")
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=epochs
)

# Print initial training accuracy
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
print(f"Initial Training Accuracy: {train_acc * 100:.2f}%")
print(f"Initial Validation Accuracy: {val_acc * 100:.2f}%")

# Fine-tune by unfreezing some layers of DenseNet
print("Starting fine-tuning...")
base_model.trainable = True

# Set fine-tuning from the 100th layer onwards
for layer in base_model.layers[:100]:
    layer.trainable = False

# Re-compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Continue training with fine-tuning
fine_tune_epochs = 10
total_epochs = epochs + fine_tune_epochs

history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1]
)

# Print fine-tuning accuracy
fine_tune_train_acc = history_fine.history['accuracy'][-1]
fine_tune_val_acc = history_fine.history['val_accuracy'][-1]
print(f"Fine-Tuned Training Accuracy: {fine_tune_train_acc * 100:.2f}%")
print(f"Fine-Tuned Validation Accuracy: {fine_tune_val_acc * 100:.2f}%")

# Optional: Evaluate on test data if available
if test_dir:
    print("Evaluating on test data...")
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model
model.save('glaucoma_detection_densenet1.h5')
print("Model saved as glaucoma_detection_densenet1.h5")
