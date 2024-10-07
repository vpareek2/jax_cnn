import jax
import jax.numpy as jnp
import pickle
from keras.api.datasets import mnist
from model import cnn
import os

# Enable Metal GPU support
os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'

# Check if Metal is available
print("JAX devices:", jax.devices())
print("Using device:", jax.devices()[0])

def normalize_images(images):
    return jnp.array(images.reshape(-1, 28, 28, 1).astype(jnp.float32) / 255.0)

def load_model(param_file: str):
    with open(param_file, 'rb') as f:
        weights = pickle.load(f)

    @jax.jit
    def predict(params, images):
        return jnp.argmax(cnn(params, images), axis=1)

    return weights, predict

def calculate_accuracy(params, predict_fn, images, labels):
    predictions = predict_fn(params, images)
    return jnp.mean(predictions == labels)

def main():
    _, (test_images, test_labels) = mnist.load_data()
    with jax.default_device(jax.devices()[0]):
        test_images = normalize_images(test_images)
        test_labels = jnp.array(test_labels, dtype=jnp.int32)

    model_file = 'mnist_cnn_params.pkl'
    loaded_weights, predict_fn = load_model(model_file)

    accuracy = calculate_accuracy(loaded_weights, predict_fn, test_images, test_labels)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
