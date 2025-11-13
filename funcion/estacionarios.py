# estacionarios_simb_mejorado.py
from sympy import symbols, Eq, groebner, solve, Poly, factor, sqrt, nroots, real_roots
from sympy import I, expand, simplify
import sympy as sp

x, y = symbols('x y', real=True)

# Definir g y f
g = 2*x**3*y - y**3
f = g**2 + x**2

print("Función f:", f)
print("g =", g)

# Derivadas simbólicas
fx = sp.diff(f, x)    # ∂f/∂x
fy = sp.diff(f, y)    # ∂f/∂y

print("\nDerivadas parciales:")
print("fx =", fx)
print("fy =", fy)

# Expandir y simplificar
P1 = expand(fx)
P2 = expand(fy)

print("\nPolinomios expandidos:")
print("P1 =", P1)
print("P2 =", P2)

# VERIFICAR SI (0,0) ES SOLUCIÓN
print(f"\nVerificando (0,0): P1(0,0) = {P1.subs({x:0, y:0})}, P2(0,0) = {P2.subs({x:0, y:0})}")

# ANÁLISIS ALTERNATIVO - Factorizar los polinomios
print("\n" + "="*50)
print("ANÁLISIS DETALLADO")
print("="*50)

# Factor P2 primero (es más simple)
print("\nFactorizando P2:")
P2_fact = factor(P2)
print("P2 =", P2_fact)

# P2 = y * (4*x^6 - 12*x^3*y^2 + 9*y^4 + 1)
# Esto nos da DOS casos:

# CASO 1: y = 0
print("\n*** CASO 1: y = 0 ***")
# Sustituir y=0 en P1
P1_y0 = P1.subs(y, 0)
print("Con y=0, P1 se reduce a:", P1_y0)

# Resolver P1_y0 = 0
sol_y0 = solve(P1_y0, x)
print("Soluciones con y=0:", sol_y0)

# CASO 2: 4*x^6 - 12*x^3*y^2 + 9*y^4 + 1 = 0
print("\n*** CASO 2: 4*x^6 - 12*x^3*y^2 + 9*y^4 + 1 = 0 ***")

# También podemos analizar P1
print("\nFactorizando P1:")
P1_fact = factor(P1)
print("P1 =", P1_fact)

# P1 = 2*x*(4*x^6*y^2 - 12*x^3*y^4 + 9*y^6 + 1)

# Esto nos da otros dos casos:
# CASO A: x = 0
print("\n*** CASO A: x = 0 ***")
P2_x0 = P2.subs(x, 0)
print("Con x=0, P2 se reduce a:", P2_x0)
sol_x0 = solve(P2_x0, y)
print("Soluciones con x=0:", sol_x0)

# CASO B: 4*x^6*y^2 - 12*x^3*y^4 + 9*y^6 + 1 = 0
print("\n*** CASO B: 4*x^6*y^2 - 12*x^3*y^4 + 9*y^6 + 1 = 0 ***")

# MÉTODO SISTEMÁTICO CON RESULTANTES
print("\n" + "="*50)
print("MÉTODO CON RESULTANTES MEJORADO")
print("="*50)

# Calcular la resultante
R = sp.resultant(P1, P2, y)
R_simp = factor(R)
print("Resultante en x:", R_simp)

# Las raíces de la resultante nos dan las coordenadas x posibles
x_poly = Poly(R_simp, x)
print(f"Grado del polinomio en x: {x_poly.degree()}")

# Buscar raíces reales
print("\nBuscando raíces REALES de la resultante:")
try:
    x_real_roots = real_roots(x_poly)
    print(f"Raíces reales encontradas: {len(x_real_roots)}")
    for i, root in enumerate(x_real_roots):
        print(f"x_{i} = {root.evalf()}")
except:
    print("No se pudieron encontrar raíces reales simbólicas, usando nroots...")
    x_all_roots = nroots(x_poly, n=15)
    x_real_roots = [r for r in x_all_roots if abs(r.as_real_imag()[1]) < 1e-10]
    print(f"Raíces reales encontradas (numéricas): {len(x_real_roots)}")
    for i, root in enumerate(x_real_roots):
        print(f"x_{i} = {root}")

# Para cada x real, encontrar y correspondiente
print("\n" + "="*50)
print("PUNTOS CRÍTICOS ENCONTRADOS")
print("="*50)

puntos_criticos = []

# Agregar soluciones del caso y=0
for x_val in sol_y0:
    if x_val.is_real:
        puntos_criticos.append((x_val, 0))
        print(f"Punto: ({x_val.evalf()}, 0)")

# Agregar soluciones del caso x=0  
for y_val in sol_x0:
    if y_val.is_real:
        puntos_criticos.append((0, y_val))
        print(f"Punto: (0, {y_val.evalf()})")

# Buscar otras soluciones numéricamente
print("\nBuscando otras soluciones numéricamente:")
for x_root in x_real_roots:
    x_val = x_root.evalf()
    if abs(x_val) < 1e-10:  # Ya tenemos x=0
        continue
    
    # Resolver P2=0 para este x
    P2_x = P2.subs(x, x_val)
    try:
        y_sols = nroots(Poly(P2_x, y), n=15)
        for y_sol in y_sols:
            if abs(y_sol.as_real_imag()[1]) < 1e-10:
                y_val = y_sol.as_real_imag()[0]
                # Verificar que también satisface P1=0
                P1_val = P1.subs({x: x_val, y: y_val}).evalf()
                if abs(P1_val) < 1e-8:
                    punto = (float(x_val), float(y_val))
                    if punto not in puntos_criticos:
                        puntos_criticos.append(punto)
                        print(f"Punto: ({x_val}, {y_val})")
    except:
        continue

# Eliminar duplicados y ordenar
puntos_criticos = sorted(list(set(puntos_criticos)))
print(f"\nTotal de puntos críticos encontrados: {len(puntos_criticos)}")
for i, (px, py) in enumerate(puntos_criticos):
    print(f"Punto {i+1}: ({px:.6f}, {py:.6f})")

# VERIFICACIÓN FINAL
print("\n" + "="*50)
print("VERIFICACIÓN FINAL")
print("="*50)
for px, py in puntos_criticos:
    val1 = float(P1.subs({x: px, y: py}).evalf())
    val2 = float(P2.subs({x: px, y: py}).evalf())
    print(f"Punto ({px:.6f}, {py:.6f}): P1 = {val1:.2e}, P2 = {val2:.2e}")
