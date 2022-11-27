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
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
import optax
import tensorflow_datasets as tfds

NUM_CLASSES = 10
NUM_ITER = 1000000
CHECK_FREQ = 1000
SEED = 42
BATCH_SIZE = 1024
MID_CHANNEL = 256


name = 'mnist'
train_ds, valid_ds = tfds.load(
    name, 
    split=[
        'train[:90%]',
        'train[90%:]'
    ]
)
train_ds = train_ds.shuffle(1024, seed=SEED).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE).repeat().as_numpy_iterator()
valid_ds = tfds.as_numpy(valid_ds.batch(len(valid_ds)))


@jax.jit
def preprocess(batch):
    images = batch['image'].astype(jnp.float32) / 255.
    images = images.reshape(images.shape[0], -1)
    labels = batch['label']
    return images, labels

init_fn, apply_fn = stax.serial(
    stax.Dense(MID_CHANNEL),
    stax.Relu,
    stax.Dense(NUM_CLASSES)
)

_, params = init_fn(jax.random.PRNGKey(SEED), (-1, 28*28))

optimizer = optax.sgd(0.01)
opt_state = optimizer.init(params)

@jax.jit
def loss_fn(params, batch):
    images, labels = preprocess(batch)
    logits = apply_fn(params, images)
    labels = jax.nn.one_hot(labels, NUM_CLASSES)
    loss = optax.softmax_cross_entropy(logits, labels).mean()
    return loss

@jax.jit
def evaluate(params, batch):
    images, labels = preprocess(batch)
    logits = apply_fn(params, images)
    y_pred = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(y_pred == labels)
    return accuracy

@jax.jit
def update(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss

loss_list = list()
acc_list = list()

best_acc = 0
Path('ckpt/mnist').mkdir(parents=True, exist_ok=True)

bar = tqdm(total=NUM_ITER, dynamic_ncols=True)
for step in range(NUM_ITER):
    bar.set_description_str(f'Iter: {step+1}')
    params, opt_state, loss = update(params, opt_state, next(train_ds))
    loss_list.append(loss)

    if (step + 1) % CHECK_FREQ == 0:
        accs = [jax.device_get(evaluate(params, b)) for b in valid_ds]
        valid_acc = np.mean(accs)
        acc_list.append(valid_acc)
        if valid_acc > best_acc:
            best_acc = valid_acc
            save_checkpoint('ckpt/mnist', params, step=valid_acc, overwrite=True)
    bar.update()
bar.close()

best_params = restore_checkpoint('ckpt', target=params)
acc = np.mean([jax.device_get(evaluate(params, b)) for b in valid_ds])
print(f'Predict from ckpt: {acc}')

Path('img/mnist').mkdir(parents=True, exist_ok=True)

plt.plot(loss_list)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.grid()
plt.savefig('img/mnist/train_loss.png')
plt.close()

plt.plot(acc_list)
plt.xlabel('iteration')
plt.ylabel('acc')
plt.ylim(0, 1)
plt.grid()
plt.savefig('img/mnist/valid_acc.png')
plt.close()

print(f'Best acc: {best_acc} at {(np.argmax(acc_list)+1)*CHECK_FREQ}')
