import numpy as np
import matplotlib.pyplot as plt

# Función de Rosenbrock
def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

# Gradiente de la función de Rosenbrock
def grad_rosenbrock(x, y, a=1, b=100):
    dfdx = -2 * (a - x) - 4 * b * x * (y - x**2)
    dfdy = 2 * b * (y - x**2)
    return np.array([dfdx, dfdy])

# Algoritmo de Gradiente Descendente
def gradient_descent(f, grad_f, start, learning_rate, tol=1e-6, max_iter=10000, a=1, b=100):
    x = np.array(start, dtype=float)
    trajectory = [x.copy()]
    for i in range(max_iter):
        grad = grad_f(x[0], x[1], a, b)
        x -= learning_rate * grad
        trajectory.append(x.copy())
        if np.linalg.norm(grad) < tol:
            print(f"Convergencia alcanzada en {i + 1} iteraciones.")
            break
    else:
        print("No se alcanzó la convergencia en el número máximo de iteraciones.")
    return x, np.array(trajectory)

# Visualización de la trayectoria
def plot_trajectory(trajectory, a=1, b=100):
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y, a, b)

    plt.figure(figsize=(10, 6))
    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(label="f(x, y)")
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', label="Trayectoria")
    plt.scatter(trajectory[0, 0], trajectory[0, 1], color='blue', label="Inicio")
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', label="Final")
    plt.title("Trayectoria del Gradiente Descendente en la función de Rosenbrock")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

# Parámetros de prueba
start_point = (-1.2, 1)  # Condición inicial
learning_rates = [0.001, 0.01, 0.1, 1]  # Diferentes tasas de aprendizaje

for eta in learning_rates:
    print(f"\nTasa de aprendizaje: {eta}")
    final_point, trajectory = gradient_descent(rosenbrock, grad_rosenbrock, start_point, eta)
    print(f"Punto final: {final_point}")
    plot_trajectory(trajectory)
