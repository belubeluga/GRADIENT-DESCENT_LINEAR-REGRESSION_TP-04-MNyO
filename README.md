# TP04 - Métodos Numéricos y Optimización (MNyO) - Primer Semestre 2024

Este trabajo práctico aborda dos problemas clásicos de optimización numérica utilizando el método de **gradiente descendente**:  
1. La minimización de la **función de Rosenbrock** en 2D.  
2. La **regresión lineal por mínimos cuadrados** aplicada al dataset California Housing.

📄 [Descargar informe TP04 en PDF](TP04_informe_MNyO.pdf.zip)

---

## 🧪 PUNTO 1: Optimización en 2 dimensiones - Función de Rosenbrock

Se implementa el **algoritmo de gradiente descendente** para minimizar la función:


### Objetivos:

- Estudiar la **convergencia** del método en función del **learning rate**.
- Analizar el impacto de diferentes **condiciones iniciales**.
- Visualizar trayectorias de descenso en el plano (x, y).
- (Opcional) Comparar con el **método de Newton**.

---

## 📊 PUNTO 2: Cuadrados Mínimos por Gradiente Descendente

Se realiza una regresión lineal para predecir `MedHouseVal` en el dataset `California Housing` de `sklearn`.

### Métodos implementados:

1. **Pseudoinversa**
2. **Gradiente descendente**
3. **Ridge Regression**

### Análisis:

- Comparación de soluciones obtenidas por ambos métodos
- Evolución del error en entrenamiento y testeo
- Sensibilidad frente al η y regularización

---

## ✅ Requisitos

- Python 3.x
- numpy
- matplotlib
- pandas
- scikit-learn
- Jupyter Notebook

---
