import jax
import jax.numpy as jnp
from typing import List, Tuple

# He initialization
def init_weights(key):
    key1, key2, key3, key4 = jax.random.split(key, 4)
    conv1 = jax.random.normal(key1, (3, 3, 1, 32)) * 0.1
    conv2 = jax.random.normal(key2, (3, 3, 32, 64)) * 0.1
    fc1 = jax.random.normal(key3, (7 * 7 * 64, 128)) * 0.1
    fc2 = jax.random.normal(key4, (128, 10)) * 0.1
    return [conv1, conv2, fc1, fc2]

# CNN Architecture
def cnn(layer_weights: List[jax.Array], x: jax.Array) -> jax.Array:
    conv1, conv2, fc1, fc2 = layer_weights

    x = jax.lax.conv_general_dilated(x, conv1, window_strides=(1, 1), padding='SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    x = jax.nn.relu(x)
    x = jax.lax.reduce_window(x, init_value=-jnp.inf, computation=jax.lax.max, window_dimensions=(1, 2, 2, 1), window_strides=(1, 2, 2, 1), padding='VALID')
    x = jax.lax.conv_general_dilated(x, conv2, window_strides=(1, 1), padding='SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    x = jax.nn.relu(x)
    x = jax.lax.reduce_window(x, init_value=-jnp.inf, computation=jax.lax.max, window_dimensions=(1, 2, 2, 1), window_strides=(1, 2, 2, 1),padding='VALID')
    x = x.reshape(x.shape[0], -1)
    x = jnp.dot(x, fc1)
    x = jax.nn.relu(x)
    x = jnp.dot(x, fc2)
    return x

# Categorical Cross-Entropy
def loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    return jnp.mean(-jnp.sum(jax.nn.one_hot(labels, num_classes=logits.shape[-1]) * jax.nn.log_softmax(logits), axis=1))
