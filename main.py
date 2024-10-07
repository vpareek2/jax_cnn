from typing import Tuple, Iterator, Any, Dict, List, Callable
import jax
import jax.numpy as jnp
import optax
from keras.api.datasets import mnist
import pickle
from model import init_weights, cnn, loss
import os

# Enable Metal GPU support
os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'

# Check if Metal is available
print("JAX devices:", jax.devices())
print("Using device:", jax.devices()[0])

LEARNING_RATE = 0.01
MOMENTUM = 0.9
BATCH_SIZE = 32
NUM_EPOCHS = 10

def create_train_state(rng, learning_rate: float, momentum: float):
    weights = init_weights(rng)
    optimizer = optax.sgd(learning_rate=learning_rate, momentum=momentum)
    opt_state = optimizer.init(weights)

    @jax.jit
    def train_step(weights, opt_state, batch_images, batch_labels):
        def loss_fn(weights):
            logits = cnn(weights, batch_images)
            return loss(logits, batch_labels)

        loss_value, grads = jax.value_and_grad(loss_fn)(weights)
        updates, new_opt_state = optimizer.update(grads, opt_state, weights)
        new_weights = optax.apply_updates(weights, updates)
        return new_weights, new_opt_state, loss_value

    return weights, opt_state, train_step

def train(weights: Any, opt_state: optax.OptState, train_step, train_dataset_fn: Callable[[], Iterator[Tuple[jax.Array, jax.Array]]], num_epochs: int) -> Any:
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        train_dataset = train_dataset_fn()
        for batch_images, batch_labels in train_dataset:
            weights, opt_state, loss_value = train_step(weights, opt_state, batch_images, batch_labels)
            total_loss += loss_value
            num_batches += 1

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, No batches processed")

    return weights

def normalize_images(images):
    return jnp.array(images.reshape(-1, 28, 28, 1).astype(jnp.float32) / 255.0)

def create_data_iterator(images, labels, batch_size, rng):
    num_samples = images.shape[0]
    num_complete_batches, leftover = divmod(num_samples, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def iterator():
        nonlocal rng
        indices = jnp.arange(num_samples)
        rng, shuffle_rng = jax.random.split(rng)
        shuffled_indices = jax.random.permutation(shuffle_rng, indices)

        for i in range(num_batches):
            batch_idx = shuffled_indices[i * batch_size:(i + 1) * batch_size]
            yield images[batch_idx], labels[batch_idx]

    return iterator

def main():
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    # Move initialization to GPU
    with jax.default_device(jax.devices()[0]):
        weights, opt_state, train_step = create_train_state(init_rng, LEARNING_RATE, MOMENTUM)

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Move data to GPU
    with jax.default_device(jax.devices()[0]):
        train_images = normalize_images(train_images)
        test_images = normalize_images(test_images)
        train_labels = jnp.array(train_labels, dtype=jnp.int32)
        test_labels = jnp.array(test_labels, dtype=jnp.int32)

    rng, data_rng = jax.random.split(rng)
    train_dataset = create_data_iterator(train_images, train_labels, BATCH_SIZE, data_rng)

    final_weights = train(weights, opt_state, train_step, train_dataset, NUM_EPOCHS)
    print("Finished Training")

    with open('mnist_cnn_params.pkl', 'wb') as f:
        pickle.dump(final_weights, f)
    print("Model parameters saved to mnist_cnn_params.pkl")

if __name__ == "__main__":
    main()
