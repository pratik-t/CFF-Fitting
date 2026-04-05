import tensorflow as tf
import numpy as np

# Your phi points and y_true are fixed for this single theta
# phi_points: shape (100,)
# y_true: shape (100,)

def theory_layer(x, phi):
    return (x**10+10)*tf.cos(phi)

def run_single_restart(x_init, phi_points, y_true, n_epochs=1000, lr=1e-3):
    """
    Run GD from a single starting point x_init.
    Returns the final x value and final loss.
    """
    # x is now a direct trainable variable, not a network output
    # This is the key difference from the DNN approach
    x = tf.Variable([x_init], dtype=tf.float32)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    # Learning rate schedule - same as your DNN setup
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=30,
        decay_rate=0.8,
        staircase=True
    )
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        
        # Loop over phi points one at a time - this is your batch size 1 setup
        # The gradient is computed and applied after EACH phi point
        # This is the per-phi-point MAE you want
        for phi, y in zip(phi_points, y_true):
            with tf.GradientTape() as tape:
                # Theory layer takes current x and this single phi point
                f_pred = theory_layer(x, phi)
                # MAE for this single phi point
                loss = tf.abs(f_pred - y)**2
            
            # Gradient of this single-point loss w.r.t. x directly
            grad = tape.gradient(loss, [x])
            optimizer.apply_gradients(zip(grad, [x]))
            epoch_loss += loss.numpy().item()
        
        epoch_loss /= len(phi_points)
    
    return x.numpy()[0], epoch_loss


def run_all_restarts(phi_points, y_true, x_range, n_restarts=200, **kwargs):
    """
    Run n_restarts independent optimisations from uniformly
    distributed starting points across x_range.
    x_range: tuple (x_min, x_max)
    """
    # Draw starting points uniformly across the full x range
    # This ensures we sample both basins proportionally to their width
    starting_points = np.random.uniform(
        x_range[0], x_range[1], size=n_restarts
    )
    
    results = []
    for i, x_init in enumerate(starting_points):
        x_final, loss_final = run_single_restart(
            x_init, phi_points, y_true, **kwargs
        )
        results.append({
            'x_init': x_init,
            'x_final': x_final,
            'loss_final': loss_final
        })
        
        if i % 20 == 0:
            print(f"Restart {i}/{n_restarts}: "
                  f"x_init={x_init:.2f}, "
                  f"x_final={x_final:.2f}, "
                  f"loss={loss_final:.6f}")
    
    # Sort by final loss - the best restart is the one with lowest loss
    results.sort(key=lambda r: r['loss_final'])
    
    print(f"\nBest result: x* = {results[0]['x_final']:.4f}, "
          f"loss = {results[0]['loss_final']:.6e}")
    print(f"Starting point that found it: x_init = {results[0]['x_init']:.4f}")
    
    return results

phi = np.linspace(10, 20, 10)
phi= np.pi-np.deg2rad(phi)
phi = tf.convert_to_tensor(phi, dtype=tf.float32)

x_true = 2.5
y_true = (x_true**10 + 10) * tf.cos(phi)

x_range = (-10, 0)

results = run_all_restarts(
    phi_points=phi,
    y_true=y_true,
    x_range=x_range,
    n_restarts=50,   # start smaller for testing
    n_epochs=20,
    lr=1e-2
)

import matplotlib.pyplot as plt

xs = [r['x_final'] for r in results]
losses = [r['loss_final'] for r in results]

plt.scatter(xs, losses)
plt.xlabel("Final x")
plt.ylabel("Final loss")
plt.savefig('hi.jpeg')