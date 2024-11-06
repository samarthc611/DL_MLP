#!/usr/bin/env python
# coding: utf-8


# Import necessary libraries
import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import os

class MlpBlock(nn.Module):
    mlp_dim: int

    @nn.compact
    def __call__(self, x):
        y = nn.Dense(self.mlp_dim)(x)
        y = nn.gelu(y)
        return nn.Dense(x.shape[-1])(y)

class MixerBlock(nn.Module):
    tokens_mlp_dim: int
    channels_mlp_dim: int

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm()(x)
        y = jnp.swapaxes(y, 1, 2)
        y = MlpBlock(self.tokens_mlp_dim)(y)
        y = jnp.swapaxes(y, 1, 2)
        x = x + y
        y = nn.LayerNorm()(x)
        return x + MlpBlock(self.channels_mlp_dim)(y)



# In[6]:


class MlpMixer(nn.Module):
    num_classes: int
    num_blocks: int
    patch_size: int
    hidden_dim: int
    tokens_mlp_dim: int
    channels_mlp_dim: int

    @nn.compact
    def __call__(self, x):
        s = self.patch_size
        x = nn.Conv(self.hidden_dim, (s, s), strides=(s, s))(x)
        x = einops.rearrange(x, 'n h w c -> n (h w) c')

        for _ in range(self.num_blocks):
            x = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)(x)

        x = nn.LayerNorm()(x)
        x = jnp.mean(x, axis=1)
        return nn.Dense(self.num_classes)(x)


# In[8]:


def load_data(batch_size=32, img_size=(224, 224)):
    dataset_path = "Dataset/Public_Medical_Image_Datasets/covid19-pneumonia-dataset"

    train_dir = os.path.join(dataset_path, "train_dir")
    valid_dir = os.path.join(dataset_path, "valid_dir")
    test_dir = os.path.join(dataset_path, "test_dir")

    if not all(os.path.exists(d) for d in [train_dir, valid_dir, test_dir]):
        raise FileNotFoundError("One or more dataset directories not found.")

    train_ds = image_dataset_from_directory(
        train_dir,
        label_mode='int',
        image_size=img_size,
        color_mode='grayscale',
        batch_size=batch_size
    )
    valid_ds = image_dataset_from_directory(
        valid_dir,
        label_mode='int',
        image_size=img_size,
        color_mode='grayscale',
        batch_size=batch_size
    )
    test_ds = image_dataset_from_directory(
        test_dir,
        label_mode='int',
        image_size=img_size,
        color_mode='grayscale',
        batch_size=batch_size
    )
    return train_ds, valid_ds, test_ds


# In[20]:


def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(batch['label'], num_classes=3)).mean()
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    state = state.apply_gradients(grads=grads)

    accuracy = (jnp.argmax(logits, -1) == batch['label']).mean()

    return state, loss, accuracy


def train_model(state, train_ds, num_epochs=10):
    for epoch in range(num_epochs):
        for batch in train_ds:
            batch = {'image': jnp.array(batch[0] / 255.0), 'label': jnp.array(batch[1])}

            state, loss, accuracy = train_step(state, batch)

        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%")


# In[21]:


class TrainState(train_state.TrainState):
    pass

num_classes = 3
model = MlpMixer(num_classes=num_classes, num_blocks=8, patch_size=4, hidden_dim=128, tokens_mlp_dim=256, channels_mlp_dim=512)

train_ds, valid_ds, test_ds = load_data()

rng = jax.random.PRNGKey(0)
dummy_input = jnp.ones((1, 224, 224, 1))
params = model.init(rng, dummy_input)['params']



# In[22]:


state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optax.adam(1e-3),
)

train_model(state, train_ds, num_epochs=10)


# In[ ]:





# In[ ]:




