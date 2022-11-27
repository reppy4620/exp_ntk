#%%
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.example_libraries import stax
import flax.linen as nn
from flax.training.checkpoints import restore_checkpoint
import optax
import tensorflow_datasets as tfds

NUM_CLASSES = 2
NUM_ITER = 1000 # 20000
CHECK_FREQ = 1
SEED = 42
BATCH_SIZE = 10
MID_CHANNEL = 256
EPSILON = 0.1


name = 'mnist'
train_ds, valid_ds = tfds.load(
    name, 
    split=[
        'train[:90%]',
        'train[90%:]'
    ]
)

image_list = []
label_list = []
for i, data in enumerate(tfds.as_numpy(valid_ds)):
    x = data['image'] / 255.
    x = x.reshape(-1)
    y = 1 if data['label'] % 2 == 0 else 0
    image_list.append(x)
    label_list.append(y)
x = np.stack(image_list, axis=0)
y = np.array(label_list)

BATCH_SIZE = len(valid_ds)

print(x.shape, y.shape)

class MLP(nn.Module):
    mid_channel: int
    out_channel: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.mid_channel)(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_channel)(x)
        return x

net = MLP(mid_channel=MID_CHANNEL, out_channel=NUM_CLASSES)
params = net.init(jax.random.PRNGKey(42), jnp.ones((1, 28*28)))

params = restore_checkpoint('ckpt', target=params)

@jax.jit
def loss_fn(params, images, labels):
    logits = net.apply(params, images)
    labels = jax.nn.one_hot(labels, NUM_CLASSES)
    loss = optax.softmax_cross_entropy(logits, labels).mean()
    return loss

@jax.jit
def evaluate(params, images, labels):
    logits = net.apply(params, images)
    y_pred = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(y_pred == labels)
    return y_pred, accuracy

@jax.jit
def update(params, images, labels):
    grad = jax.grad(loss_fn, argnums=1)(params, images, labels)
    grad = jnp.reshape(grad, (BATCH_SIZE, 28*28))
    x_adv = x + EPSILON * jnp.sign(grad)
    return x_adv

pred, acc = evaluate(params, x, y)
before = {'pred': jax.device_get(pred), 'acc': jax.device_get(acc)}


acc_list = list()
for step in tqdm(range(NUM_ITER)):
    x = update(params, x, y)
    _, acc = evaluate(params, x, y)
    acc_list.append(acc)

pred, acc = evaluate(params, x, y)
after = {'pred': jax.device_get(pred), 'acc': jax.device_get(acc)}

Path('img/adv').mkdir(parents=True, exist_ok=True)
plt.plot(acc_list)
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.grid()
plt.savefig('img/adv/adv_acc.png')
plt.close()

x = x[:10]
cnt = 0
ncol, nrow = 5, 2
for i in range(ncol * nrow):
    cnt += 1
    before_pred = before['pred'][i]
    after_pred = after['pred'][i]
    plt.subplot(nrow, ncol, cnt)
    plt.title(f'{before_pred} -> {after_pred}')
    plt.imshow(x[i].reshape(28, 28, 1))
plt.savefig('img/adv/example.png')
plt.close()

print(f'Accuracy: {before["acc"]} -> {after["acc"]}')
