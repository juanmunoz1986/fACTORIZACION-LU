
import numpy as np                  # Biblioteca para cálculo numérico
import sys                          # Para finalizar el script en caso de error

# ──────────────────────────────────────────────────────────────
# 1. Definimos la matriz de coeficientes A y el vector b
# ──────────────────────────────────────────────────────────────
A = np.array([[2.0, 3.0, 1.5],      # Fila 1 → 2x + 3y + 1.5z
              [1.0, 2.0, 0.8],      # Fila 2 → 1x + 2y + 0.8z
              [0.5, 0.7, 1.2]])     # Fila 3 → 0.5x + 0.7y + 1.2z

b = np.array([380.0,                # Término independiente ecuación 1
              200.0,                # Término independiente ecuación 2
              150.0])               # Término independiente ecuación 3

# ──────────────────────────────────────────────────────────────
# 2. Parámetros de la iteración
# ──────────────────────────────────────────────────────────────
w           = 1.270       # Factor de relajación (0<w≤1). 1 → paso completo.
tol         = 0.0000001     # Tolerancia para la norma del residuo
max_iter    = 50        # Iteraciones máximas permitidas
mostrar_pas = True      # Si es True, imprime cada iteración

# ──────────────────────────────────────────────────────────────
# 3. Comprobamos que la matriz A sea invertible
# ──────────────────────────────────────────────────────────────
detA = np.linalg.det(A)             # Determinante de A
if np.isclose(detA, 0.0):           # ¿Determinante (casi) cero?
    print("❌ Error: La matriz de coeficientes es singular; "
          "no puede aplicarse Newton-Raphson.")
    sys.exit(1)                     # Terminamos con código de error 1

# ──────────────────────────────────────────────────────────────
# 4. Inicializamos la solución con ceros (puedes elegir otro valor)
# ──────────────────────────────────────────────────────────────
x = np.zeros(3)                     # Vector inicial [0, 0, 0]

# ──────────────────────────────────────────────────────────────
# 5. Bucle principal de Newton-Raphson con relajación
#    Para un sistema lineal: J = A (constante)
#    Paso:  Δ = –A⁻¹·(A·x – b)  ,  x_{k+1} = x_k + w·Δ
# ──────────────────────────────────────────────────────────────
for k in range(1, max_iter + 1):        # Contador de iteraciones (1…max_iter)
    residuo = A @ x - b                 # Calculamos F(x) = A·x – b
    norma   = np.linalg.norm(residuo)   # Norma L2 del residuo

    if mostrar_pas:                     # Muestra el progreso si se habilitó
        print(f"Iter {k:02d}  ||r|| = {norma:.4e}   x = {x}")

    if norma < tol:                     # ¿Convergió?
        print("\n✅ Convergencia alcanzada.\n")
        break                           # Salimos del bucle principal

    # Resolvemos A·Δ = –residuo  para hallar el paso de Newton
    try:
        delta = np.linalg.solve(A, -residuo)
    except np.linalg.LinAlgError as e:  # Captura problemas inusuales (pivoteo)
        print(f"❌ Error al resolver el paso de Newton: {e}")
        sys.exit(2)

    # Actualizamos la solución con factor de relajación
    x = x + w * delta

else:
    # Si el bucle for agota todas las iteraciones sin romper, no hubo convergencia
    print("❌ El método no convergió en el número máximo de iteraciones "
          f"({max_iter}). Última ||r|| = {norma:.4e}")
    sys.exit(3)

# ──────────────────────────────────────────────────────────────
# 6. Resultado final
# ──────────────────────────────────────────────────────────────
print("Resultado final:")
print(f"x = {x[0]:10.6f}")
print(f"y = {x[1]:10.6f}")
print(f"z = {x[2]:10.6f}")
