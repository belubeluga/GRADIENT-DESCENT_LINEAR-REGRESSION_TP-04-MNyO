# TP04 - Numerical Methods and Optimization (NM&O) - First Semester 2024

This practical assignment addresses two classical numerical optimization problems using the **gradient descent** method:  
1. Minimization of the **Rosenbrock function** in 2D.  
2. **Linear regression via least squares**, applied to the California Housing dataset.

📄 [Download TP04 Report in PDF](informe_TP04_MNyO.pdf.zip)

---

## 🧪 PART 1: 2D Optimization – Rosenbrock Function

The **gradient descent algorithm** is implemented to minimize the function:

### Objectives:

- Study the **convergence** of the method based on different **learning rates**  
- Analyze the impact of varying **initial conditions**  
- Visualize descent trajectories in the (x, y) plane  
- *(Optional)* Compare with the **Newton's method**

---

## 📊 PART 2: Least Squares via Gradient Descent

A linear regression model is trained to predict `MedHouseVal` using the `California Housing` dataset from `sklearn`.

### Implemented Methods:

1. **Pseudoinverse**  
2. **Gradient Descent**  
3. **Ridge Regression**

### Analysis:

- Comparison of solutions obtained by each method  
- Evolution of training and test error  
- Sensitivity to η (learning rate) and regularization

---

## ✅ Requirements

- Python 3.x  
- numpy  
- matplotlib  
- pandas  
- scikit-learn  
- Jupyter Notebook
