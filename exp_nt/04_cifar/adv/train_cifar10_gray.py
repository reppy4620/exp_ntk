#%%
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
from jax.example_libraries import stax
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
import optax
import tensorflow_datasets as tfds

IMG_SIZE = 32
NUM_CLASSES = 10
NUM_ITER = 2000000
CHECK_FREQ = 100000
SEED = 42
BATCH_SIZE = 1000
MID_CHANNELS = [256, 512, 1024, 2048, 4096]

name = 'cifar10'
# ds = tfds.load(name, split=tfds.Split.TRAIN).shuffle(1024, seed=42)
train_ds, test_ds = tfds.load('cifar10', split=['train[:80%]', 'train[80%:]'])

def preprocess(x):
    image, label = x['image'], x['label']
    image = tf.image.rgb_to_grayscale(image)
    image = tf.cast(image, tf.float32)
    image = image / 255.
    image = tf.reshape(image, (-1,))
    label = int(label)
    return image, label

train_ds_np = train_ds.map(preprocess).as_numpy_iterator()
test_ds_np = test_ds.map(preprocess).as_numpy_iterator()

def load_x_y(ds):
    x_list, y_list = [], []
    for x, y in ds:
        x = x / np.linalg.norm(x)
        x_list.append(x)
        y_list.append(y)
    return np.stack(x_list, axis=0), np.stack(y_list, axis=0)

train_x, train_y = load_x_y(train_ds_np)
test_x, test_y = load_x_y(test_ds_np)

np.random.seed(SEED)
train_length = 20000
test_length = 10000
inds = np.random.randint(0, train_x.shape[0], train_length)
train_x = train_x[inds]
train_y = train_y[inds]
inds = np.random.randint(0, test_x.shape[0], test_length)
test_x = test_x[inds]
test_y = test_y[inds]

print(train_x.shape, test_x.shape)

def create_ds(x, y):
    xx = tf.data.Dataset.from_tensor_slices(x)
    yy = tf.data.Dataset.from_tensor_slices(y)
    return tf.data.Dataset.zip((xx, yy))

train_ds = create_ds(train_x, train_y).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE).repeat().shuffle(1024).as_numpy_iterator()
valid_ds = tfds.as_numpy(create_ds(test_x, test_y).batch(len(test_ds)))

@jax.jit
def preprocess(batch):
    return batch

def main(mid_channel):

    print(f'Start {mid_channel} channels')

    init_fn, apply_fn = stax.serial(
        stax.Dense(mid_channel),
        stax.Relu,
        stax.Dense(NUM_CLASSES)
    )

    _, params = init_fn(jax.random.PRNGKey(SEED), (-1, IMG_SIZE*IMG_SIZE))

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
    out_dir = Path(f'ckpt/{name}-gray/channel_{mid_channel}')
    out_dir.mkdir(parents=True, exist_ok=True)
    bar = tqdm(total=NUM_ITER, dynamic_ncols=True)
    for step in range(NUM_ITER):
        bar.set_description_str(f'Iter: {step+1}')
        params, opt_state, loss = update(params, opt_state, next(train_ds))
        loss_list.append(loss)

        if (step + 1) % CHECK_FREQ == 0:
            valid_acc = np.mean([jax.device_get(evaluate(params, b)) for b in valid_ds])
            acc_list.append(valid_acc)
            if valid_acc > best_acc:
                best_acc = valid_acc
                save_checkpoint(out_dir, params, step=valid_acc, overwrite=True)
        bar.update()
    bar.close()

    best_params = restore_checkpoint(out_dir, target=params)
    acc = np.mean([jax.device_get(evaluate(best_params, b)) for b in valid_ds])
    print(f'Predict from ckpt: {acc}')

    plt.plot(loss_list)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.grid()
    plt.savefig(out_dir / 'train_loss.png')
    plt.close()

    plt.plot(acc_list)
    plt.xlabel('iteration')
    plt.ylabel('acc')
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig(out_dir / 'valid_acc.png')
    plt.close()

    print(f'Best acc: {best_acc} at {(np.argmax(acc_list)+1)*CHECK_FREQ}')

for c in MID_CHANNELS:
    main(c)
