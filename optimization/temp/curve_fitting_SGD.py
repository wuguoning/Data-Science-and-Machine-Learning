import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
from matplotlib import animation, rc
rc('animation', html='html5')
plt.style.use('seaborn-whitegrid')


def animate_curve_fitting(model,
                             X, y,
                             batch_size=64,
                             epochs=16,
                             lr=0.005,
                             shuffle_buffer=5000,
                             seed=0,
                             verbose=1):
    num_examples = X.shape[0]
    steps_per_epoch = num_examples // batch_size
    total_steps = steps_per_epoch * epochs

    ds = (tf.data.Dataset
          .from_tensor_slices((X, y))
          .repeat()
          .cache()
          .shuffle(shuffle_buffer, seed=seed)
          .batch(batch_size))
    ds_iter = ds.as_numpy_iterator()

    x_min = X.min()
    x_max = X.max()
    X_pop = np.linspace(x_min, x_max, 1000)
    y_min = y.min()
    y_max = y.max()

    # Parameters
    xs = []
    ys = []
    curves = []
    # Callback to save parameters
    def save_params(batch, logs):
        x, y = next(ds_iter)
        xs.append(x.squeeze())
        ys.append(y.squeeze())
        curve = model.predict(X_pop)
        curves.append(curve)

    save_params_cb = keras.callbacks.LambdaCallback(
        on_batch_begin=save_params,
    )

    # Train model to collect parameters
    model.fit(
        ds,
        epochs=epochs,
        callbacks=[save_params_cb],
        steps_per_epoch=steps_per_epoch,
        verbose=verbose,
    )

    # Create Figure
    fig = plt.figure(dpi=150, figsize=(4, 3))
    # Regression Curve
    ax1 = fig.add_subplot(111)
    ax1.set_title("Fitted Curve")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    p10, = ax1.plot(X, y, 'r.', alpha=0.1) # full dataset
    p11, = ax1.plot([], [], 'C3.') # batch
    p12, = ax1.plot([], [], 'k') # fitted line
    # Complete Figure
    fig.tight_layout()

    def init():
        return [p10]

    def update(frame):
        x = xs[frame]
        y = ys[frame]
        p11.set_data(x, y)
        p12.set_data(X_pop, curves[frame])
        return p11, p12

    ani = \
        animation.FuncAnimation(
            fig,
            update,
            frames=range(1, total_steps),
            init_func=init,
            blit=True,
            interval=100,
        )
    plt.close()

    return ani

X = np.random.normal(loc=0.0, scale=1.0, size=256)
err = np.random.normal(loc=0.0, scale=1.0, size=256)
y = 2 * np.square(X) + err

model = keras.Sequential([
    layers.Dense(8),
    layers.Activation('relu'),
    layers.Dense(16),
    layers.Activation('relu'),
    layers.Dense(8),
    layers.Activation('relu'),
    layers.Dense(1)
])
initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=2,
    decay_rate=0.96,
    staircase=False,
)
model.compile(
    optimizer=keras.optimizers.Adam(lr_schedule),
    loss='mse',
)

ani = animate_curve_fitting(model, X, y, batch_size=32, epochs=32, verbose=0)
plt.close()
# ani.save('some/path')
ani
