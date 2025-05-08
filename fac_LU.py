import numpy as np                 # Importamos NumPy para manejo de arrays y operaciones numéricas
import scipy.linalg as la          # Importamos SciPy (linalg) para utilizar la factorización LU
import sys                         # Importamos sys para terminar la ejecución en caso de error

# 1. Definición de la matriz A y el vector b.
# Para demostración, definimos A y b directamente.
# (En un caso real, podríamos solicitarlos al usuario mediante input() u otra interfaz)
A = np.array([[2,3,1.5],      # Matriz de coeficientes A (4x4 en este ejemplo)
              [1,2,0.8],
              [0.5,0.7,1.2]
              ], dtype=float)
b = np.array([380,200,150], dtype=float)  # Vector de términos independientes b (dimension 4)

# 5. Manejo de errores: verificar que A sea cuadrada y que las dimensiones coincidan con b.
if A.shape[0] != A.shape[1]:
    print("Error: la matriz A debe ser cuadrada.")      # Mensaje de error si A no es cuadrada
    sys.exit(1)                                         # Terminar ejecución si hay error
if b.shape[0] != A.shape[0]:
    print("Error: el vector b debe tener la misma dimensión que A.")  # Error si dimensiones no coinciden
    sys.exit(1)

# Verificar que la matriz no sea singular (determinante = 0 implicaría que no tiene inversa ni LU estándar).
detA = np.linalg.det(A)
if abs(detA) < 1e-9:                                   # Usamos un umbral para considerar el cero por posibles errores de redondeo
    print("Error: la matriz A es singular y no se puede factorizar LU.")  # Mensaje de error si A es singular
    sys.exit(1)

# 2. Factorización LU de la matriz A.
# Utilizamos la función lu de SciPy que devuelve P, L, U tal que P*A = L*U (P es matriz de permutación de filas).
P, L, U = la.lu(A)

# Mostrar las matrices L y U obtenidas de la factorización
print("Matriz L (triangular inferior):")
print(L)
print("Matriz U (triangular superior):")
print(U)

# 3. Sustitución hacia adelante: resolver L * y = P * b.
# Si no hubo pivoteo, P es la identidad y P*b = b. En caso contrario, usamos P*b (permuta las filas de b).
b_permutado = P.dot(b)   # Calculamos el vector b permutado según P
n = A.shape[0]           # Tamaño del sistema (número de ecuaciones o incógnitas)
y = np.zeros(n)          # Inicializamos el vector y de solución intermedia con ceros

# Recorremos cada ecuación de L*y = b_permutado para despejar los componentes de y
for i in range(n):
    suma = 0
    for j in range(i):
        suma += L[i, j] * y[j]                  # Calculamos la suma L[i,j]*y[j] para j < i
    # Despejamos y[i]: (b_permutado[i] - suma) / L[i,i]
    y[i] = (b_permutado[i] - suma) / L[i, i]     # (En teoría L[i,i] es 1 en factorización LU estándar)

print("Vector y (solución de L*y = P*b):")
print(y)

# 4. Sustitución hacia atrás: resolver U * x = y.
x = np.zeros(n)          # Inicializamos el vector solución x con ceros
# Recorremos las ecuaciones de U*x = y comenzando desde la última fila hacia arriba
for i in range(n-1, -1, -1):
    suma = 0
    for j in range(i+1, n):
        suma += U[i, j] * x[j]                  # Calculamos la suma U[i,j]*x[j] para j > i
    # Despejamos x[i]: (y[i] - suma) / U[i,i]
    x[i] = (y[i] - suma) / U[i, i]

print("Vector x (solución de U*x = y, es decir, solución final de A*x = b):")
print(x)
