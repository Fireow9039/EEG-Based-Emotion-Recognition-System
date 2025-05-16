import tensorflow as tf
from tensorflow.keras import Model, layers

class RnnSemiSupervisedVAE(Model):
    def __init__(self, timesteps, features, latent_dim, num_classes):
        super(RnnSemiSupervisedVAE, self).__init__()
        self.timesteps = timesteps
        self.features = features
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(timesteps, features)),
            layers.LSTM(128, return_sequences=True),
            layers.LSTM(64),
            layers.Dense(latent_dim * 2 + num_classes)
        ])
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(64),
            layers.RepeatVector(timesteps),
            layers.LSTM(64, return_sequences=True),
            layers.LSTM(128, return_sequences=True),
            layers.TimeDistributed(layers.Dense(features, activation='sigmoid'))
        ])

    def encode(self, x):
        encoder_output = self.encoder(x)
        z_mean, z_log_var, y_logit = tf.split(encoder_output, num_or_size_splits=[self.latent_dim, self.latent_dim, self.num_classes], axis=1)
        return z_mean, z_log_var, y_logit

    def reparameterize(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], self.latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def decode(self, z):
        return self.decoder(z)

    def call(self, inputs):
        z_mean, z_log_var, y_logit = self.encode(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        reconstruction = self.decode(z)
        return reconstruction, y_logit, z_mean, z_log_var

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            reconstruction, y_logit, z_mean, z_log_var = self(x, training=True)
            reconstruction_loss = tf.reduce_mean(tf.losses.mean_squared_error(x, reconstruction))
            kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
            classification_loss = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y, y_logit))
            total_loss = reconstruction_loss + kl_loss + classification_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss': total_loss}
