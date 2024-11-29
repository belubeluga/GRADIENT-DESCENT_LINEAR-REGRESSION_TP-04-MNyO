import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def grad_rosenbrock(x, y, a=1, b=100):
    dfdx = -2 * (a - x) - 4 * b * x * (y - x**2)
    dfdy = 2 * b * (y - x**2)
    return np.array([dfdx, dfdy])

def gradient_descent_trajectory(f, grad_f, start, learning_rate, a=1, b=100, max_iter=10000, tol=1e-6):
    x = np.array(start, dtype=float)
    trajectory = [x.copy()]
    
    for _ in range(max_iter):
        grad = grad_f(x[0], x[1], a, b)
        x -= learning_rate * grad
        trajectory.append(x.copy())
        if np.linalg.norm(grad) < tol:
            break

    return np.array(trajectory)

def gradient_descent(f, grad_f, start, learning_rate, tol=1e-4, max_iter=50000, a=1, b=100):
    x = np.array(start, dtype=float)
    trajectory = [x.copy()]
    eta = learning_rate  # a explorar
    prev_value = f(x[0], x[1], a, b)

    for i in range(max_iter):
        grad = grad_f(x[0], x[1], a, b)
        grad_norm = np.linalg.norm(grad) 

        x -= eta * grad
        trajectory.append(x.copy())

        # CONVERGENCIA
        if grad_norm < tol:
            print(f"Convergencia alcanzada en {i + 1} iteraciones. Tasa de aprendizaje final: {eta:.5f}")
            return x, np.array(trajectory)

    print("No se alcanzó la convergencia en el número máximo de iteraciones.")
    return x, np.array(trajectory)


x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

fig = plt.figure()
plt.style.use('seaborn-v0_8-white') #style :)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='plasma')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X, Y)')
ax.set_title('Visualización de la función')
plt.show()



starting_points = [(-1.5, 1.5), (0.5, 2.0), (-1.5, 2.5), (1.5, 1.0)]
trajectories = [
    gradient_descent_trajectory(rosenbrock, grad_rosenbrock, start, learning_rate=0.001)
    for start in starting_points
]

x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

fig = plt.figure(figsize=(12, 8))
plt.style.use('seaborn-v0_8-white')

ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8, edgecolor='none')

for trajectory in trajectories:
    ax.plot(
        trajectory[:, 0], 
        trajectory[:, 1], 
        rosenbrock(trajectory[:, 0], trajectory[:, 1]), marker="o", markersize=2, linewidth=1
    )

#ax.view_init(elev=40, azim=65) 
ax.view_init(elev=36, azim=75) 
plt.scatter([1], [1], color="red", marker='*', s=400, label="Mínimo")
    
ax.set_xlabel('X', labelpad=10)
ax.set_ylabel('Y', labelpad=10)
ax.set_zlabel('f(X, Y)', labelpad=10)
ax.set_title('Superficie de la función Rosenbrock y trayectorias de descenso', pad=20)

plt.show()








def plot_multiple_trajectories_3d(trajectories, title="Trayectorias del Gradiente Descendente"):
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)

    fig = plt.figure(figsize=(20, 20))

    # First subplot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.contour3D(X, Y, Z, 60, cmap='viridis')
    for i, trajectory in enumerate(trajectories):
        x_vals = trajectory[:, 0]
        y_vals = trajectory[:, 1]
        z_vals = rosenbrock(x_vals, y_vals)
        ax1.plot(x_vals, y_vals, z_vals, color='red', linewidth=3)
        ax1.scatter(x_vals, y_vals, z_vals, color='red', s=10)
    ax1.scatter([1], [1], [rosenbrock(1, 1)], marker='o', color='red', s=100)
    ax1.set_xlabel('$x_{0}$')
    ax1.set_ylabel('$x_{1}$')
    ax1.set_zlabel('$f(x)$')
    ax1.view_init(20, 20)

    # Second subplot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.contour3D(X, Y, Z, 60, cmap='viridis')
    for i, trajectory in enumerate(trajectories):
        x_vals = trajectory[:, 0]
        y_vals = trajectory[:, 1]
        z_vals = rosenbrock(x_vals, y_vals)
        ax2.plot(x_vals, y_vals, z_vals, color='red', linewidth=3)
        ax2.scatter(x_vals, y_vals, z_vals, color='red', s=10)
    ax2.scatter([1], [1], [rosenbrock(1, 1)], marker='o', color='red', s=100)
    ax2.set_xlabel('$x_{0}$')
    ax2.set_ylabel('$x_{1}$')
    ax2.set_zlabel('$f(x)$')
    ax2.axes.zaxis.set_ticklabels([])
    ax2.view_init(90, -90)

    plt.show()


start_points = [(-1.2, 1), (1.5, 2.7), (0, 0), (0, 2.5), (-1.5, -0.4), (2, 2), (0.7, 0.3), (1.5, -0.5)]  # CI
learning_rate = 0.001  # S_{k}

trajectories = []

for s_p in start_points:
    print(f"\nTasa de aprendizaje: {learning_rate}")
    final_point, trajectory = gradient_descent(rosenbrock, grad_rosenbrock, s_p, learning_rate=learning_rate)
    print(f"Punto final estimado: {final_point}")
    print(f"Valor de la función en el punto final: {rosenbrock(final_point[0], final_point[1])}")
    trajectories.append(trajectory)

# Graficar todas las trayectorias en dos subplots 3D
plot_multiple_trajectories_3d(trajectories, title="Trayectorias del Gradiente Descendente en la Función de Rosenbrock")    