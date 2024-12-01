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








def plot_multiple_trajectories_3d(trajectories, title="Trayectorias del Gradiente Descendente", colors = [
                                                                                                '#f8bbd0',  # 200
                                                                                                '#fce4ec',  # 100
                                                                                                '#f48fb1',  # 300
                                                                                                '#f06292',  # 400
                                                                                                '#ec407a',  # 500
                                                                                                '#e91e63',  # 600
                                                                                                '#d81b60',  # 700
                                                                                                '#c2185b',  # 800
                                                                                                '#ad1457',  # 900
                                                                                                '#880e4f'   # 1000
                                                                                            ]):
    
    x = np.linspace(-2.5, 2.5, 400)
    y = np.linspace(-2.5, 3.5, 400)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)

    plt.style.use('seaborn-v0_8-white')
    fig = plt.figure(figsize=(10, 10))

    # First subplot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.contour3D(X, Y, Z, 60, cmap='viridis')
    for i, trajectory in enumerate(trajectories):
        x_vals = trajectory[:, 0]
        y_vals = trajectory[:, 1]
        z_vals = rosenbrock(x_vals, y_vals)
        ax1.plot(x_vals, y_vals, z_vals, color=colors[i % len(colors)], linewidth=3)
        ax1.scatter(x_vals, y_vals, z_vals, color=colors[i % len(colors)], s=10)
    ax1.scatter([1], [1], [rosenbrock(1, 1)], marker='o', color='red', s=100)
    ax1.set_xlabel('$x_{0}$')
    ax1.set_ylabel('$x_{1}$')
    ax1.set_zlabel('$f(x)$')
    
    ax1.view_init(20, 60)

    # Second subplot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.contour3D(X, Y, Z, 60, cmap='viridis')
    for i, trajectory in enumerate(trajectories):
        x_vals = trajectory[:, 0]
        y_vals = trajectory[:, 1]
        z_vals = rosenbrock(x_vals, y_vals)
        ax2.plot(x_vals, y_vals, z_vals, color=colors[i % len(colors)], linewidth=3)
        ax2.scatter(x_vals, y_vals, z_vals, color=colors[i % len(colors)], s=10)
    ax2.scatter([1], [1], [rosenbrock(1, 1)], marker='o', color='red', s=100)
    ax2.set_xlabel('$x_{0}$')
    ax2.set_ylabel('$x_{1}$')
    ax2.set_zlabel('$f(x)$')
    ax2.axes.zaxis.set_ticklabels([])
    #ax2.view_init(0, 45)

    #ax2.set_xlim(-4, 4)
    #ax2.set_ylim(-4, 4)
    #ax2.set_zlim(0, 10000)

    ax2.view_init(20, 20)
    

    plt.show()

start_points = [(np.float64(2.0), np.float64(0.0)), (np.float64(1.4142135623730951), np.float64(1.414213562373095)), (np.float64(1.2246467991473532e-16), np.float64(2.0)), (np.float64(-1.414213562373095), np.float64(1.4142135623730951)), (np.float64(-2.0), np.float64(2.4492935982947064e-16)), (np.float64(-1.4142135623730954), np.float64(-1.414213562373095)), (np.float64(-3.6739403974420594e-16), np.float64(-2.0)), (np.float64(1.4142135623730947), np.float64(-1.4142135623730954))]
learning_rate = 0.001  # S_{k}

trajectories = []

for s_p in start_points:
    print(f"\nTasa de aprendizaje: {learning_rate}")
    final_point, trajectory = gradient_descent(rosenbrock, grad_rosenbrock, s_p, learning_rate=learning_rate)
    print(f"Punto final estimado: {final_point}")
    print(f"Valor de la función en el punto final: {rosenbrock(final_point[0], final_point[1])}")
    trajectories.append(trajectory)

#plot_multiple_trajectories_3d(trajectories, title="Trayectorias del Gradiente Descendente en la Función de Rosenbrock")



start_points = [(-1,3),(5,2)]
learning_rate = 0.001  # S_{k}

trajectories = []

for s_p in start_points:
    print(f"\nTasa de aprendizaje: {learning_rate}")
    final_point, trajectory = gradient_descent(rosenbrock, grad_rosenbrock, s_p, learning_rate=learning_rate)
    print(f"Punto final estimado: {final_point}")
    print(f"Valor de la función en el punto final: {rosenbrock(final_point[0], final_point[1])}")
    trajectories.append(trajectory)

#plot_multiple_trajectories_3d(trajectories, title="Trayectorias del Gradiente Descendente en la Función de Rosenbrock", colors=['red', 'orange'], )






def hessian_rosenbrock(x, y, a=1, b=100):
    d2f_dx2 = 2 - 4 * b * y + 12 * b * x**2
    d2f_dy2 = 2 * b
    d2f_dxdy = -4 * b * x
    return np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])

def newton_method(f, grad_f, hessian_f, start, tol=1e-8, max_iter=100):
    x = np.array(start, dtype=float)
    trajectory = [x.copy()]
    errorsN = [np.linalg.norm(x - np.array([1, 1]))] 

    for i in range(max_iter):
        grad = grad_f(x[0], x[1])
        hessian = hessian_f(x[0], x[1])

        x -= np.linalg.inv(hessian) @ grad
        trajectory.append(x.copy())
        errorsN.append(np.linalg.norm(x - np.array([1, 1])))


        if np.linalg.norm(grad) < tol:
            print(f"Convergencia alcanzada en {i + 1} iteraciones.")
            break
    else:
        print("No se alcanzó la convergencia en el número máximo de iteraciones.")
    
    return x, np.array(trajectory), errorsN

start = [-1.5, 2.0]
learning_rate = 0.001

min_point_gd, trajectory_gd = gradient_descent(rosenbrock, grad_rosenbrock, start, learning_rate, max_iter=10000)
print(f"Punto mínimo encontrado por descenso de gradiente: {min_point_gd}")


min_point_newton, trajectory_newton, errorsN = newton_method(rosenbrock, grad_rosenbrock, hessian_rosenbrock, start)
print(f"Punto mínimo encontrado por método de Newton: {min_point_newton}")
#plot_multiple_trajectories([trajectory_newton, trajectory_gd], title="Trayectoria del método de Newton", colors=['#f8bbd0', '#ad1457'])


plt.style.use('dark_background') #style :) seaborn-v0_8-poster  dark_background
#plt.style.use('seaborn-v0_8-white')
x = np.linspace(0.5, 1.5, 400)
y = np.linspace(0.5, 1.5, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

plt.figure(figsize=(14, 8))
contour = plt.contour(X, Y, Z, levels=np.logspace(-0.5, 3.5, 20), cmap="coolwarm")


plt.plot(trajectory_gd[:, 0], trajectory_gd[:, 1], 'r.-', lw= 5, color='#f8bbd0', markersize=10,  label="Descenso de gradiente")
plt.plot(trajectory_newton[:, 0], trajectory_newton[:, 1], 'b.-', lw= 5, color= '#d81b60', markersize=10, label="Método de Newton")

plt.plot([1], [1], '*', color="yellow", label="Mínimo", markersize=25)

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Comparación de trayectorias", fontsize=20, fontweight='bold', pad=20, fontname='Arial')
plt.grid(False)
cbar = plt.colorbar(contour)
cbar.set_label('Valor de la función')
plt.legend()
#plt.show()









def gradient_descent(f, grad_f, start, learning_rate, tol=1e-4, max_iter=5000, a=1, b=100):
    x = np.array(start, dtype=float)
    trajectory = [x.copy()]
    errors = [np.linalg.norm(x - np.array([1, 1]))]  # Distancia inicial al mínimo (1,1)

    for _ in range(max_iter):
        grad = grad_f(x[0], x[1], a, b)
        x -= learning_rate * grad
        trajectory.append(x.copy())
        errors.append(np.linalg.norm(x - np.array([1, 1])))  # Calcular distancia al mínimo

        if np.linalg.norm(grad) < tol:
            break

    return np.array(trajectory), np.array(errors)

# Parámetros iniciales
start_point = (-1.5, 2.0)
learning_rate = 0.001

# Ejecutar el descenso de gradiente
trajectory, errorsGD = gradient_descent(rosenbrock, grad_rosenbrock, start_point, learning_rate)

plt.style.use('seaborn-v0_8-white')
# Gráfica de los errores
plt.figure(figsize=(10, 5))
plt.plot(errorsGD[:20], label="Error (distancia al mínimo) del Gradient Descent", color='orange', linewidth=2)
plt.plot(errorsN[:20], label="Error (distancia al mínimo) del Método de Newton", color='pink', linewidth=2)
#plt.yscale('log')  # Escala logarítmica para errores
plt.xlabel("Iteraciones")
plt.ylabel("Error log")
plt.title("Convergencia del error hacia el mínimo absoluto")
plt.grid(True)
legend = plt.legend(facecolor='white', edgecolor='black', labelcolor='black')
plt.show()