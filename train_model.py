import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.utils import to_categorical
import h5py

# Set a seed for reproducibility
tf.random.set_seed(42)

def load_data(filename):
    """Load dataset from an HDF5 file."""
    with h5py.File(filename, "r") as f:
        features = np.array(f['features'])
        labels = np.array(f['labels'])
    return features, labels

def build_model(input_shape, num_classes):
    """Builds a simple neural network model."""
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    # Load data
    features, labels = load_data('your_dataset.h5')

    # Normalize features from 0-1 if they are not already
    features = features / np.max(features)

    # Convert labels to categorical (one-hot encoding)
    labels = to_categorical(labels, num_classes=7)

    # Split data into training and testing sets
    split_index = int(0.8 * len(features))
    train_features, test_features = features[:split_index], features[split_index:]
    train_labels, test_labels = labels[:split_index], labels[split_index:]

    # Create the model
    model = build_model(input_shape=(42,), num_classes=7)
    model.summary()

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

    # Train the model
    history = model.fit(train_features, train_labels, epochs=50, batch_size=32,
                        validation_split=0.2, callbacks=[checkpoint, early_stopping])

    # Load the best saved model
    model = tf.keras.models.load_model('best_model.h5')

    # Evaluate on the test set
    test_loss, test_accuracy = model.evaluate(test_features, test_labels)
    print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))

if __name__ == "__main__":
    main()
