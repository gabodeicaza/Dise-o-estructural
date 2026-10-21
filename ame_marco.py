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
    columnas   : por nivel, lista de dicts con eje (1..), A, I (cm², cm4), artic (articulada en su base),
                 artic_sup (articulada arriba) y rodillo (apoyo de rodillos en la base, solo nivel 1). Opcionales:
                 hasta (piso donde termina una columna que cruza pisos sin viga) y omitir (continuación ya incluida).
    vigas      : por piso 1..n, dict con I, A, E (opcional) y sin (tramos sin viga, opcional).
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
        if c.get('rodillo'):  # rodillo: el apoyo se desplaza libremente en horizontal (no transmite cortante)
            idx[(0, c['eje'])] = (cont, -1, cont + 1)  # ux y giro libres: es un pasador sobre rodillos
            cont += 2
        else:
            idx[(0, c['eje'])] = (-1, -1, -1)  # base empotrada
    ndof = cont
    K = np.zeros((ndof, ndof))
    zs = np.concatenate([[0.0], np.cumsum(alturas_cm)])

    def ensamblar(k_loc, T, dofs, liberar_inicio=False, liberar_fin=False):
        if liberar_inicio:  # articulación en el extremo inicial: se condensa su giro
            k_loc = k_loc - np.outer(k_loc[:, 2], k_loc[2, :]) / k_loc[2, 2]
        if liberar_fin:  # articulación en el extremo final (parte superior de la columna)
            k_loc = k_loc - np.outer(k_loc[:, 5], k_loc[5, :]) / k_loc[5, 5]
        kg = T.T @ k_loc @ T
        for i, gi in enumerate(dofs):
            if gi < 0:
                continue
            for j, gj in enumerate(dofs):
                if gj >= 0:
                    K[gi, gj] += kg[i, j]

    for i, cols in enumerate(columnas):  # nivel i+1: del piso i al piso i+1 (o hasta 'hasta' si cruza pisos sin viga)
        for c in cols:
            if c.get('omitir'):  # continuación de una columna que ya se ensambló completa
                continue
            top = c.get('hasta', i + 1)
            L = zs[top] - zs[i]
            ensamblar(_k_barra(c.get('E', E), c['A'], c['I'], L), _rot(0.0, 1.0),
                      list(idx[(i, c['eje'])]) + list(idx[(top, c['eje'])]), c['artic'], c.get('artic_sup', False))
    for k in range(1, n + 1):
        v = vigas[k - 1]
        sin = v.get('sin', ())  # tramos (0, 1, …) sin viga en este piso
        for e in range(1, ne):
            if (e - 1) in sin:
                continue
            L = xs_cm[e] - xs_cm[e - 1]
            ensamblar(_k_barra(v.get('E', E), v['A'], v['I'], L), _rot(1.0, 0.0), list(idx[(k, e)]) + list(idx[(k, e + 1)]))

    usado = np.abs(np.diag(K)) > 0          # los nodos sin ningún elemento (piso omitido por una columna) se descartan
    usado[:n] = True
    ub = usado[n:]
    Kaa, Kab, Kbb = K[:n, :n], K[:n, n:][:, ub], K[n:, n:][np.ix_(ub, ub)]
    try:
        Kred = Kaa - Kab @ np.linalg.solve(Kbb, Kab.T)
    except np.linalg.LinAlgError:
        raise ValueError('El marco es inestable (revisa columnas, articulaciones y vigas).')
    return 0.5 * (Kred + Kred.T)
