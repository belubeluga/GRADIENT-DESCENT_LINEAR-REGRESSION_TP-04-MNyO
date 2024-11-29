using Plots
using LinearAlgebra

# Definir la función de Rosenbrock
function rosenbrock(x, y, a=1, b=100)
    return (a - x)^2 + b * (y - x^2)^2
end

# Gradiente de la función de Rosenbrock
function grad_rosenbrock(x, y, a=1, b=100)
    dx = -2 * (a - x) - 4 * b * x * (y - x^2)
    dy = 2 * b * (y - x^2)
    return [dx, dy]
end

# Método de gradiente descendente
function gradient_descent(f, grad_f, start, learning_rate, tol=1e-8, max_iter=10000, a=1, b=100)
    x, y = start  # Asegurarnos de que start es una tupla (x, y)
    trajectory = [(x, y)]  # Almacenar la trayectoria de puntos (x, y)
    eta = learning_rate
    prev_value = f(x, y, a, b)

    for i in 1:max_iter
        grad = grad_f(x, y, a, b)

        grad_norm = norm(grad)
        if grad_norm > 1e4
            grad ./= grad_norm * 1e4
        end

        # Actualizar x y y por separado
        x .= x - eta * grad[1]
        y .= y - eta * grad[2]
        push!(trajectory, (x, y))

        current_value = f(x, y, a, b)

        if current_value > prev_value
            eta *= 0.5  # Reducir la tasa de aprendizaje
            println("Tasa de aprendizaje reducida a $eta en iteración $i.")
        end
        prev_value = current_value

        if grad_norm < tol
            println("Convergencia alcanzada en $i iteraciones. Tasa de aprendizaje final: $eta")
            break
        end
    end

    if grad_norm < tol
        println("Convergencia alcanzada en $i iteraciones. Tasa de aprendizaje final: $eta")
    else
        println("No se alcanzó la convergencia en el número máximo de iteraciones.")
    end

    return (x, y), trajectory
end

# Función para graficar las trayectorias
function plot_multiple_trajectories(trajectories, title="Trayectorias del Gradiente Descendente")
    x_vals = LinRange(-2, 2, 400)
    y_vals = LinRange(-1, 3, 400)
    X, Y = meshgrid(x_vals, y_vals)
    Z = rosenbrock.(X, Y)

    # Crear el gráfico
    plot(contour(X, Y, Z, levels=log10.(LinRange(0.01, 1000, 50))), xlabel="x", ylabel="y", title=title,
         label="Curvas de nivel", color=:twilight)

    colors = [:purple, :blue, :green, :orange, :red, :yellow, :cyan, :magenta]

    # Dibujar las trayectorias
    for (i, trajectory) in enumerate(trajectories)
        x_vals = [point[1] for point in trajectory]
        y_vals = [point[2] for point in trajectory]
        plot!(x_vals, y_vals, label="Trayectoria $i", color=colors[i % length(colors)], linewidth=2)
        scatter!(x_vals[1], y_vals[1], label="Inicio $i", color=colors[i % length(colors)], markersize=5)
        scatter!(x_vals[end], y_vals[end], label="Final $i", color=colors[i % length(colors)], markersize=5)
    end

    # Marcar el mínimo
    scatter!(1, 1, label="Mínimo", color=:black, markersize=8, marker=:star5)

    # Mostrar el gráfico
    display(plot)
end

# Puntos de inicio y tasa de aprendizaje
start_points = [(Float64(-1.2), Float64(1)), (Float64(1.5), Float64(2.7)), (Float64(0.0), Float64(0.0)),
                (Float64(0.0), Float64(2.5)), (Float64(-1.5), Float64(-0.4)), (Float64(2.0), Float64(2.0)),
                (Float64(0.7), Float64(0.3)), (Float64(1.5), Float64(-0.5))]
learning_rate = 0.001

trajectories = []

# Ejecutar el algoritmo de gradiente descendente y almacenar las trayectorias
for s_p in start_points
    println("\nTasa de aprendizaje: $learning_rate")
    final_point, trajectory = gradient_descent(rosenbrock, grad_rosenbrock, s_p, learning_rate)
    println("Punto final estimado: $final_point")
    println("Valor de la función en el punto final: $(rosenbrock(final_point[1], final_point[2]))")
    push!(trajectories, trajectory)
end

# Graficar todas las trayectorias en el mismo gráfico
plot_multiple_trajectories(trajectories, title="Trayectorias del Gradiente Descendente en la Función de Rosenbrock")
