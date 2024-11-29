# Mejorar y completar el código existente con los elementos faltantes
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Función para cargar y preparar datos
def load_and_prepare_data():
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    # Agregar columna de unos para la ordenada al origen
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Estandarizar las características (excepto la columna de 1s)
    scaler = StandardScaler()
    X_train[:, :-1] = scaler.fit_transform(X_train[:, :-1])  
    X_test[:, :-1] = scaler.transform(X_test[:, :-1])  
    
    return X_train, X_test, y_train, y_test

# Solución con pseudoinversa
def pseudoinverse_solution(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

# Solución con gradiente descendente
def gradient_descent_solution(X, y, learning_rate=0.01, tol=1e-6, max_iter=10000):
    n, d = X.shape
    w = np.zeros(d)
    trajectory = []  # Para almacenar los valores de w en cada iteración
    errors = []  # Para almacenar el ECM en cada iteración
    
    for i in range(max_iter):
        # Calcular predicciones y gradiente
        predictions = X @ w
        residuals = y - predictions
        gradient = -2 / n * X.T @ residuals
        
        # Actualizar pesos
        w -= learning_rate * gradient
        
        # Calcular ECM y almacenar
        ecm = np.mean(residuals ** 2)
        errors.append(ecm)
        trajectory.append(w.copy())
        
        # Verificar convergencia
        if np.linalg.norm(gradient) < tol:
            break
    
    return w, errors, trajectory

# Graficar evolución del ECM
def plot_ecm(errors):
    plt.figure(figsize=(10, 6))
    plt.plot(errors, label="ECM por iteración", color="blue")
    plt.xlabel("Iteraciones")
    plt.ylabel("ECM")
    plt.title("Evolución del ECM en Gradiente Descendente")
    plt.legend()
    plt.grid()
    plt.show()

# Comparar soluciones y errores
def compare_solutions(X_train, X_test, y_train, y_test, pseudoinv_w, gd_w):
    # Calcular predicciones y errores
    pseudoinv_train_error = np.mean((y_train - X_train @ pseudoinv_w) ** 2)
    pseudoinv_test_error = np.mean((y_test - X_test @ pseudoinv_w) ** 2)
    
    gd_train_error = np.mean((y_train - X_train @ gd_w) ** 2)
    gd_test_error = np.mean((y_test - X_test @ gd_w) ** 2)
    
    # Graficar comparación
    labels = ["Pseudoinversa", "Gradiente Descendente"]
    train_errors = [pseudoinv_train_error, gd_train_error]
    test_errors = [pseudoinv_test_error, gd_test_error]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, train_errors, width, label="Entrenamiento")
    plt.bar(x + width/2, test_errors, width, label="Prueba")
    
    plt.xlabel("Método")
    plt.ylabel("Error Cuadrático Medio")
    plt.title("Comparación de errores entre métodos")
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis="y")
    plt.show()

# Ejecutar el flujo completo
X_train, X_test, y_train, y_test = load_and_prepare_data()

# Solución con pseudoinversa
pseudoinv_w = pseudoinverse_solution(X_train, y_train)

# Solución con gradiente descendente
gd_w, errors, _ = gradient_descent_solution(X_train, y_train, learning_rate=0.01)

# Graficar evolución del ECM
plot_ecm(errors)

# Comparar las soluciones
compare_solutions(X_train, X_test, y_train, y_test, pseudoinv_w, gd_w)
