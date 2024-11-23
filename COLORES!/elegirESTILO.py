import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)
scatter_x = np.linspace(0, 10, 10)
scatter_y = np.sin(scatter_x)

styles = [
    'Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 
    'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 
    'ggplot', 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 
    'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 
    'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 
    'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 
    'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 
    'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10'
]

for style in styles:
    plt.style.use(style)
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label="Línea Senoidal")
    plt.scatter(scatter_x, scatter_y, color='red', label="Puntos Discretos")
    plt.title(f"Estilo: {style}", fontsize=14, fontweight='bold')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()
