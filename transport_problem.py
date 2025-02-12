"""
transport_problem.py - Genera un problema de transporte para ser utilizado en main.py.
Problema:
    Minimizar el costo de transporte de bienes desde varios orígenes a varios destinos.
    Restricciones:
    - La oferta en cada origen no puede ser excedida.
    - La demanda en cada destino debe ser satisfecha.
"""

import numpy as np

def generate_transport_problem():
    """
    Genera un problema de transporte con 3 orígenes y 4 destinos.
    """
    # Número de orígenes y destinos
    num_origins = 3
    num_destinations = 4
    num_variables = num_origins * num_destinations

    # Matriz de restricciones (A)
    A = np.zeros((num_origins + num_destinations, num_variables))
    
    # Restricciones de oferta
    for i in range(num_origins):
        A[i, i*num_destinations:(i+1)*num_destinations] = 1
    
    # Restricciones de demanda
    for j in range(num_destinations):
        A[num_origins + j, j::num_destinations] = 1

    # Vector de oferta y demanda (b)
    supply = np.array([15, 25, 10])
    demand = np.array([5, 15, 20, 10])
    b = np.concatenate((supply, demand))

    # Vector de costos (c) aleatorio
    c = np.random.randint(1, 20, size=num_variables)

    return A, b, c
