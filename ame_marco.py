"""Rigidez lateral de un marco plano con vigas flexibles.

Cada nodo (eje, piso) tiene 3 grados de libertad (ux, uz, θ). Los pisos son diafragmas rígidos en su plano,
por lo que ux es el mismo en todos los nodos de un piso. Se ensambla la rigidez de columnas y vigas
(flexión + axial) y se condensan uz y θ para dejar una matriz K de n x n (un desplazamiento lateral por nivel).
"""
import numpy as np


def _k_barra(E, A, I, L):
    a, b = E * A / L, E * I
    return np.array([
        [a, 0, 0, -a, 0, 0],
        [0, 12 * b / L ** 3, 6 * b / L ** 2, 0, -12 * b / L ** 3, 6 * b / L ** 2],
        [0, 6 * b / L ** 2, 4 * b / L, 0, -6 * b / L ** 2, 2 * b / L],
        [-a, 0, 0, a, 0, 0],
        [0, -12 * b / L ** 3, -6 * b / L ** 2, 0, 12 * b / L ** 3, -6 * b / L ** 2],
        [0, 6 * b / L ** 2, 2 * b / L, 0, -6 * b / L ** 2, 4 * b / L]])


def _rot(c, s):
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    T = np.zeros((6, 6))
    T[:3, :3] = T[3:, 3:] = R
    return T


def rigidez_marco(E, alturas_cm, xs_cm, columnas, vigas):
    """K lateral (n x n) del marco.

    alturas_cm : altura de cada entrepiso (cm), de abajo hacia arriba.
    xs_cm      : posición de cada eje (cm).
    columnas   : por nivel, lista de dicts con eje (1..), A, I (cm², cm4) y artic (articulada en su base).
    vigas      : por piso 1..n, dict con I y A de la viga que une todos los ejes en ese piso.
    """
    n, ne = len(alturas_cm), len(xs_cm)
    # numeración de grados de libertad: laterales primero
    idx = {}
    cont = n
    for k in range(1, n + 1):
        for e in range(1, ne + 1):
            idx[(k, e)] = (k - 1, cont, cont + 1)  # ux, uz, θ
            cont += 2
    for c in columnas[0]:
        idx[(0, c['eje'])] = (-1, -1, -1)  # base empotrada
    ndof = cont
    K = np.zeros((ndof, ndof))
    zs = np.concatenate([[0.0], np.cumsum(alturas_cm)])

    def ensamblar(k_loc, T, dofs, liberar_inicio=False):
        if liberar_inicio:  # articulación en el extremo inicial: se condensa su giro
            k_loc = k_loc - np.outer(k_loc[:, 2], k_loc[2, :]) / k_loc[2, 2]
        kg = T.T @ k_loc @ T
        for i, gi in enumerate(dofs):
            if gi < 0:
                continue
            for j, gj in enumerate(dofs):
                if gj >= 0:
                    K[gi, gj] += kg[i, j]

    for i, cols in enumerate(columnas):  # nivel i+1: del piso i al piso i+1
        L = alturas_cm[i]
        for c in cols:
            ensamblar(_k_barra(E, c['A'], c['I'], L), _rot(0.0, 1.0),
                      list(idx[(i, c['eje'])]) + list(idx[(i + 1, c['eje'])]), c['artic'])
    for k in range(1, n + 1):
        v = vigas[k - 1]
        for e in range(1, ne):
            L = xs_cm[e] - xs_cm[e - 1]
            ensamblar(_k_barra(E, v['A'], v['I'], L), _rot(1.0, 0.0), list(idx[(k, e)]) + list(idx[(k, e + 1)]))

    Kaa, Kab, Kbb = K[:n, :n], K[:n, n:], K[n:, n:]
    try:
        Kred = Kaa - Kab @ np.linalg.solve(Kbb, Kab.T)
    except np.linalg.LinAlgError:
        raise ValueError('El marco es inestable (revisa columnas, articulaciones y vigas).')
    return 0.5 * (Kred + Kred.T)
