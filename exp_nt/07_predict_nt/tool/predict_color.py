import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

import jax
import neural_tangents as nt
from neural_tangents import stax
import tensorflow_datasets as tfds
import optax

IMG_SIZE = 32
NUM_CLASSES = 10
SEED = 42

name = sys.argv[1]
train_ds, test_ds = tfds.load(name, split=['train[:80%]', 'train[80%:]'])

def preprocess(x):
    image, label = x['image'], x['label']
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
        y = np.eye(NUM_CLASSES)[y]
        y_list.append(y)
    return np.stack(x_list, axis=0), np.stack(y_list, axis=0)

train_x, train_y = load_x_y(train_ds_np)
test_x, test_y = load_x_y(test_ds_np)

print(train_x.shape, test_x.shape)

np.random.seed(SEED)
train_length = 20000
test_length = 10000
inds = np.random.randint(0, train_x.shape[0], train_length)
train_x = train_x[inds]
train_y = train_y[inds]
inds = np.random.randint(0, test_x.shape[0], test_length)
test_x = test_x[inds]
test_y = test_y[inds]

@jax.jit
def preprocess(batch):
    return batch

def evaluate(x, labels, predict_fn):
    xx1, xx2 = x[:x.shape[0]//2, :], x[x.shape[0]//2:, :]
    preds = [predict_fn(x_test=xx, get='ntk') for xx in [xx1, xx2]]
    pred = np.concatenate(preds, axis=0)
    pred_y = np.argmax(pred, axis=-1)
    gt_y = np.argmax(labels, axis=-1)
    acc = (pred_y == gt_y).mean()
    return acc

def calculate(x, labels, predict_fn, t=None):
    xx1, xx2 = x[:x.shape[0]//2, :], x[x.shape[0]//2:, :]
    preds = [predict_fn(t=t, x_test=xx, get='ntk') for xx in [xx1, xx2]]
    pred = np.concatenate(preds, axis=0)
    mse = (0.5 * (pred - labels) ** 2).mean()
    xentropy = optax.softmax_cross_entropy(pred, labels).mean()
    pred_y = np.argmax(pred, axis=-1)
    gt_y = np.argmax(labels, axis=-1)
    acc = (pred_y == gt_y).mean()
    return mse, xentropy, acc

def main():
    *_, kernel_fn = stax.serial(
        stax.Dense(128),
        stax.Relu(),
        stax.Dense(NUM_CLASSES)
    )

    predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, train_x, train_y, diag_reg=train_x.shape[0] * 1e-8)

    ts = np.logspace(1, 15, 100)

    train_info = [calculate(train_x, train_y, predict_fn, t=t) for t in tqdm(ts)]
    test_info = [calculate(test_x, test_y, predict_fn, t=t) for t in tqdm(ts)]
    
    train_mse, train_xentropy, train_acc = tuple(zip(*train_info))
    test_mse, test_xentropy, test_acc = tuple(zip(*test_info))

    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))
    plt.semilogx(ts, train_mse, label='train')
    plt.semilogx(ts, test_mse, label='test')
    plt.grid()
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('mse')
    plt.savefig(output_dir / 'mse.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.semilogx(ts, train_xentropy, label='train')
    plt.semilogx(ts, test_xentropy, label='test')
    plt.grid()
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('xentropy')
    plt.savefig(output_dir / 'xentropy.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.semilogx(ts, train_acc, label='train')
    plt.semilogx(ts, test_acc, label='test')
    plt.grid()
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('accuracy')
    plt.ylim(0, 1)
    plt.savefig(output_dir / 'accuracy.png')
    plt.close()

    acc = evaluate(test_x, test_y, predict_fn)
    with open(output_dir / 'acc.txt', 'w') as f:
        f.write(f'{acc}')

main()