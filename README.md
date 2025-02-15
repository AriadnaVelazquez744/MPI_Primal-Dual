# 🔥 Implementación del Método de Puntos Interiores Primal-Dual en Python  

📌 **Autores:** Ariadna Velázquez Rey y Lía López Rosales
📌 **Repositorio:** [GitHub](https://github.com/AriadnaVelazquez744/MPI_Primal-Dual.git)  

---

## 📖 **Descripción del Proyecto**  
Este proyecto implementa el **método de puntos interiores primal-dual** para resolver **problemas de optimización lineal**. Se compara su rendimiento con el método **Simplex**, utilizando `scipy.optimize.linprog`.  

El algoritmo se basa en resolver **sistemas de ecuaciones de Karush-Kuhn-Tucker (KKT)** mediante el **método de Newton**, en combinación con una **búsqueda de línea** para garantizar convergencia.

---

## 📚 **Fundamentos Teóricos**  

### 🔹 **¿Qué es el Método de Puntos Interiores Primal-Dual?**  
Es una técnica de optimización que resuelve problemas de **programación lineal (PL)** sin moverse por las caras del politopo factible, como hace el método Simplex. En su lugar, sigue una trayectoria en el interior de la región factible mediante **funciones de barrera**.  

A diferencia de los métodos exclusivamente primales o duales, el método **primal-dual** actualiza simultáneamente:
1. **La solución primal** x  
2. **Los multiplicadores de Lagrange** λ  
3. **Las variables de holgura dual** s  

Estos valores se ajustan iterativamente hasta encontrar la solución óptima.

---

### 🔹 **Planteamiento Matemático**  

Dado un problema de optimización lineal en forma estándar:  

**Minimizar:**  
```
c^T * x
```
**Sujeto a:**  
```
A * x = b
x >= 0
```

Se define el **lagrangiano** del problema con multiplicadores de Lagrange λ y variables de holgura s:  
```
L(x, λ, s) = c^T * x - λ^T * (A * x - b) - s^T * x
```
Las **condiciones KKT** (Karush-Kuhn-Tucker) son:  

1. **Estacionariedad:**  
   ```
   c - A^T * λ - s = 0
   ```
2. **Factibilidad primal:**  
   ```
   A * x = b
   ```
3. **Factibilidad dual:**  
   ```
   x >= 0,  s >= 0
   ```
4. **Complementariedad:**  
   ```
   x_i * s_i = 0,  para todo i
   ```

El **método primal-dual** introduce un **parámetro de barrera μ** y reemplaza la condición de complementariedad por:  
```
X * S * e = μ * e
```
donde:  
- `X = diag(x)`  
- `S = diag(s)`  
- `e = (1,1,...,1)^T`

Este parámetro μ disminuye gradualmente hasta converger a la solución óptima.

---

### 🔹 **Resolviendo el Sistema Newtoniano**  

Para cada iteración, resolvemos el siguiente sistema de ecuaciones lineales:  

```
|  0   A^T  I  |   | Δx  |   =   |  A^T * λ + s - c  |
|  A    0   0  | * | Δλ  |   =   |       A * x - b   |
|  S    0   X  |   | Δs  |   =   |      X * S * e - μ * e  |
```

Utilizando **factorización LU**, encontramos los valores de `Δx, Δλ, Δs` para actualizar:  
```
x  <-  x + α * Δx  
λ  <-  λ + α * Δλ  
s  <-  s + α * Δs  
```
donde **α** se obtiene mediante **búsqueda de línea** para garantizar que `x` y `s` sigan siendo positivos.

---

## 💻 **Implementación Computacional**  

### 🔹 **Estructura del Código**
```bash
📂 MPI_Primal-Dual
│── 📄 Main.py                  # Código principal, genera problemas LP y ejecuta el algoritmo
│── 📄 primal_dual.py            # Implementación del método primal-dual
│── 📄 numerical_utils.py        # Funciones auxiliares (Newton, búsqueda de línea, escalado)
│── 📄 transport_problem.py      # Generador de problemas de transporte
│── 📄 network_flow.py           # Generador de problemas de flujo en redes
│── 📄 README.md                 # Este archivo
```

---

## 🚀 **Cómo Ejecutar el Código**  

### 🔹 **Requisitos**  
- **Python 3.8+**  
- Librerías necesarias: `numpy`, `scipy`, `matplotlib`, `cvxpy`

Para instalarlas, ejecuta:
```bash
pip install numpy scipy matplotlib cvxpy
```

### 🔹 **Ejecutar el código principal**
```bash
python Main.py
```
Esto ejecutará el algoritmo Primal-Dual y lo comparará con Simplex.

### 🔹 **Ejemplo de Ejecución**
```
--- Comparación del Algoritmo Primal-Dual con Simplex ---
Usando find_initial_point_cvxpy...
Tiempo de ejecución (Primal-Dual): 0.118 segundos
Tiempo de ejecución (Simplex): 0.023 segundos
```

---

## 📊 **Comparación con Simplex**
| **Método**         | **Tiempo de Ejecución** | **Precisión**  |
|-------------------|-------------------|-------------|
| Primal-Dual      | 0.118s            | Aproximado |
| Simplex (HiGHS)  | 0.023s            | Exacto     |

📌 **El método Primal-Dual es más eficiente en problemas grandes** pero puede ser menos preciso en soluciones con ceros exactos.

---

## 🔧 **Mejoras Futuras**
✅ **Ajuste de la regularización en el sistema KKT**  
✅ **Métodos híbridos entre Puntos Interiores y Simplex**  
✅ **Mejor búsqueda de línea para acelerar la convergencia**  
