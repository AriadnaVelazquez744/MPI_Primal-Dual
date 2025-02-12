"""
network_flow.py - Genera un problema de flujo de red para ser utilizado en main.py.
Problema:
    Minimizar el costo total de flujo en una red dirigida.
    Restricciones:
    - El flujo en cada arista está limitado por su capacidad.
    - Conservación de flujo en todos los nodos excepto en el origen y destino.
"""

import numpy as np

def generate_network_flow_problem():
    """
    Genera un problema de flujo de red pequeño para probar el algoritmo primal-dual.
    """
    # Número de nodos y aristas
    num_nodes = 6
    num_arcs = 10

    # Matriz de incidencia nodal (A)
    A = np.array([
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Nodo 1 (fuente)
        [-1, 1, 0, 0, 0, 0, 0, 1, 0, 0], # Nodo 2
        [0, -1, 1, 0, 0, 0, 0, 0, 1, 0], # Nodo 3
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 1], # Nodo 4
        [0, 0, 0, -1, 1, 0, 0, 0, 0, 0], # Nodo 5
        [0, 0, 0, 0, -1, 1, 0, -1, -1, -1]  # Nodo 6 (sumidero)
    ])

    # Vector de demanda/suministro (b)
    b = np.array([10, 0, 0, 0, 0, -10])  # Nodo 1 suministra 10, nodo 6 recibe 10

    # Vector de costos (c) aleatorio para cada arista
    c = np.random.randint(1, 10, size=num_arcs)

    return A, b, c
