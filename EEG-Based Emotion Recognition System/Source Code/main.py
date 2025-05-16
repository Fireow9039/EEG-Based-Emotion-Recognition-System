from data_utils import load_and_prepare_data, normalize_data
from visualization_utils import (
    visualize_data, visualize_histograms, visualize_boxplot, visualize_correlation,
    visualize_spectral_density, visualize_multi_channel_time_series, visualize_reconstructions
)
from model import RnnSemiSupervisedVAE
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
import tensorflow as tf

def main():
    filepath = 'features_raw.csv'
    data = load_and_prepare_data(filepath)
    scaled_data = normalize_data(data)
    
    visualize_data(scaled_data)
    visualize_histograms(scaled_data)
    visualize_boxplot(scaled_data)
    visualize_correlation(scaled_data)
    visualize_spectral_density(scaled_data[:, 0])
    visualize_multi_channel_time_series(pd.DataFrame(scaled_data), 1000)

    labels = np.random.randint(0, 2, size=(scaled_data.shape[0],))
    data_train, data_test, labels_train, labels_test = train_test_split(scaled_data, labels, test_size=0.2, random_state=40)

    timesteps = 10
    features = data_train.shape[1] // timesteps if data_train.shape[1] % timesteps == 0 else None

    if features is None:
        print("Timesteps do not evenly divide the number of features. Adjusting by padding...")
        extra_features = timesteps - (data_train.shape[1] % timesteps)
        data_train = np.pad(data_train, ((0, 0), (0, extra_features)), 'constant')
        data_test = np.pad(data_test, ((0, 0), (0, extra_features)), 'constant')
        features = data_train.shape[1] // timesteps

    data_train = data_train.reshape(-1, timesteps, features)
    data_test = data_test.reshape(-1, timesteps, features)

    train_dataset = tf.data.Dataset.from_tensor_slices((data_train, labels_train)).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((data_test, labels_test)).batch(32)

    latent_dim = 10
    num_classes = 2
    vae = RnnSemiSupervisedVAE(timesteps, features, latent_dim, num_classes)
    vae.compile(optimizer='adam')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    vae.fit(train_dataset, epochs=50, validation_data=test_dataset, callbacks=[reduce_lr])

    for test_inputs, test_labels in test_dataset.take(1):  # Taking a single batch from the test set
        reconstructed, _, _, _ = vae(test_inputs)
        visualize_reconstructions(test_inputs.numpy().flatten(), reconstructed.numpy().flatten())

    reconstructed, predicted_logits, _, _ = vae(data_test)
    predicted_labels = np.argmax(predicted_logits, axis=1)

    print("Classification Accuracy:", accuracy_score(labels_test, predicted_labels))
    print(classification_report(labels_test, predicted_labels))
    display_confusion_matrix(labels_test, predicted_labels)

if __name__ == "__main__":
    main()
