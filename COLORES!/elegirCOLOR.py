import matplotlib.pyplot as plt
import numpy as np

# Categorías de colormaps
categories = {
    "Perceptually Uniform Sequential": ['viridis', 'plasma', 'cividis', 'inferno', 'magma', 'turbo'],
    "Sequential": ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'Greys'],
    "Diverging": ['coolwarm', 'bwr', 'PiYG', 'PRGn', 'RdYlBu', 'RdYlGn'],
    "Cyclic": ['twilight', 'twilight_shifted', 'hsv'],
    "Qualitative": ['tab10', 'Set1', 'Set2', 'Set3', 'Pastel1', 'Dark2'],
    "Miscellaneous": ['flag', 'prism', 'jet', 'rainbow', 'nipy_spectral']
}

# Datos para gráficos continuos y discretos
x = np.linspace(0, 10, 500)
y = np.sin(x) * x

x_discrete = np.arange(10)
y_discrete = np.random.randint(1, 10, size=10)

# Mostrar ejemplos de cada categoría
for category, cmaps in categories.items():
    print(f"Categoría: {category}")
    for cmap in cmaps:
        # Gráfico continuo
        plt.figure(figsize=(8, 4))
        plt.scatter(x, y, c=y, cmap=cmap, s=10)
        plt.colorbar(label=f"Colormap: {cmap}")
        plt.title(f"Gráfico Continuo - {category}: {cmap}", fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Gráfico discreto
        plt.figure(figsize=(8, 4))
        plt.bar(x_discrete, y_discrete, color=plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(x_discrete))))
        plt.title(f"Gráfico Discreto - {category}: {cmap}", fontsize=14)
        plt.tight_layout()
        plt.show()
