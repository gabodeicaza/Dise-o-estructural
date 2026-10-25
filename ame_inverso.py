"""Problema inverso: masas y pesos de un edificio a partir de sus periodos y formas modales (y de K).

Con Φ (cada columna es un modo) y Λ = diag(ω²), la matriz dinámica A = M⁻¹K cumple A = Φ·Λ·Φ⁻¹, y de ahí M⁻¹ = A·K⁻¹.
Es el procedimiento de la solución del examen 2 (estructura existente). La M obtenida debe salir diagonal; si no, los datos
modales no corresponden a un edificio de masas concentradas en los pisos.
"""
import numpy as np

import ame_detalle as det
import ame_sismico as ame

G = ame.G
ENTRADAS = ('Periodos T (s)', 'Frecuencias ω (rad/s)', 'Frecuencias f (Hz)', 'Valores propios λ = ω² (rad/s)²')
UNIDADES_K = ('t/cm', 'kg/cm')
ORIENTACIONES = ('Nivel 1 en la primera fila', 'Nivel superior en la primera fila')
TOL_DIAGONAL = 0.01          # |M_ij| / √(M_ii·M_jj) máximo para considerar diagonal a la M obtenida
TOL_PERIODO = 0.01           # error relativo máximo al recalcular los periodos con la M diagonal


def matriz_k_de_rigideces(k):
    """K tridiagonal de un edificio de cortante a partir de las rigideces de entrepiso (nivel 1 primero)."""
    k = np.asarray(k, float)
    n = k.size
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] += k[i]
        if i + 1 < n:
            K[i, i] += k[i + 1]
            K[i, i + 1] -= k[i + 1]
            K[i + 1, i] -= k[i + 1]
    return K


def _lambda(valores, entrada):
    v = np.asarray(valores, float)
    if (v <= 0).any():
        raise ValueError('Los periodos, frecuencias o valores propios deben ser mayores que cero.')
    if entrada == ENTRADAS[0]:
        return (2 * np.pi / v) ** 2
    if entrada == ENTRADAS[1]:
        return v ** 2
    if entrada == ENTRADAS[2]:
        return (2 * np.pi * v) ** 2
    return v


def masas_desde_modos(K, valores, Phi, entrada=ENTRADAS[0], unidad_K='t/cm', longitudes=None, g=G):
    """Devuelve un dict con A, M⁻¹, M completa, masas, pesos (t), carga distribuida y comprobaciones.

    K: matriz n×n (nivel 1 primero). valores: n periodos, frecuencias o λ (modo 1 = el de mayor periodo, en el orden de las
    columnas de Phi). Phi: n×n, filas = niveles (nivel 1 primero), columnas = modos."""
    K = np.asarray(K, float)
    Phi = np.asarray(Phi, float)
    n = K.shape[0]
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError('K debe ser una matriz cuadrada.')
    if np.abs(K - K.T).max() > 1e-6 * max(np.abs(K).max(), 1.0):
        raise ValueError('K debe ser simétrica.')
    if Phi.shape != (n, n):
        raise ValueError(f'Las formas modales deben ser una matriz de {n}×{n} (una fila por nivel y una columna por modo); '
                         f'recibí {Phi.shape[0]}×{Phi.shape[1]}. Con menos modos que niveles no se puede despejar M.')
    lam = _lambda(valores, entrada)
    if lam.size != n:
        raise ValueError(f'Hacen falta {n} valores (uno por modo) y recibí {lam.size}.')
    if abs(np.linalg.det(Phi)) < 1e-12:
        raise ValueError('La matriz de formas modales es singular: dos modos son iguales o una columna es cero.')
    if unidad_K not in UNIDADES_K:
        raise ValueError('La unidad de K debe ser t/cm o kg/cm.')
    A = Phi @ np.diag(lam) @ np.linalg.inv(Phi)
    Minv = A @ np.linalg.inv(K)
    M_full = np.linalg.inv(Minv)
    masas = np.diag(M_full).copy()
    off = max((abs(M_full[i, j]) / np.sqrt(abs(masas[i] * masas[j])) for i in range(n) for j in range(n) if i != j and masas[i] * masas[j] != 0),
              default=0.0)
    avisos = []
    if (masas <= 0).any():
        avisos.append('Alguna masa sale negativa o cero: revisa los datos (K, periodos o el orden de niveles y modos).')
    if off > TOL_DIAGONAL:
        avisos.append(f'La M obtenida no es diagonal (término fuera de la diagonal hasta {off:.1%} de las masas): los datos '
                      'modales no son consistentes con masas concentradas por nivel. Se toma solo la diagonal.')
    T_obt = lam_check = None
    if (masas > 0).all():
        lam_check = np.sort(np.linalg.eigvals(np.linalg.solve(np.diag(masas), K)).real)
        T_obt = 2 * np.pi / np.sqrt(lam_check)
        err = float(np.max(np.abs(np.sort(lam)[::-1] / np.sort(lam_check)[::-1] - 1))) if (lam_check > 0).all() else np.inf
        if err > TOL_PERIODO:
            avisos.append(f'Con la M diagonal los periodos no se reproducen (error de λ hasta {err:.1%}): revisa los datos.')
    factor_t = 1.0 if unidad_K == 't/cm' else 1e-3                # masa en (t o kg)·s²/cm -> peso en t
    pesos = masas * g * factor_t
    w = None
    if longitudes is not None:
        L = np.asarray(longitudes, float)
        if L.size != n or (L <= 0).any():
            raise ValueError(f'Las longitudes deben ser {n} valores mayores que cero (una por nivel).')
        w = pesos / L
    return dict(n=n, K=K, Phi=Phi, lam=lam, w=np.sqrt(lam), T=2 * np.pi / np.sqrt(lam), A=A, Minv=Minv, M_full=M_full,
                masas=masas, pesos=pesos, w_dist=w, longitudes=None if longitudes is None else np.asarray(longitudes, float),
                T_comprobado=T_obt, unidad_K=unidad_K, off=float(off), avisos=avisos, g=g)


def texto_M(r):
    """M como vector de masas (para el modo de matrices) y los pesos, en texto de captura."""
    return ' '.join(f'{x:.10g}' for x in r['masas']), ' '.join(f'{x:.10g}' for x in r['pesos'])


def texto_K(K):
    return '\n'.join(' '.join(f'{x:.10g}' for x in fila) for fila in np.asarray(K, float))


def bloques(r):
    """Paso a paso del problema inverso, con los mismos bloques que el resto del detalle."""
    n = r['n']
    nv = [f'N{i}' for i in range(1, n + 1)]
    modos = [f'Modo {j + 1}' for j in range(n)]
    unid_masa = 't·s²/cm' if r['unidad_K'] == 't/cm' else 'kg·s²/cm'
    B = [det.titulo('MASAS A PARTIR DE LOS MODOS', 'problema inverso: periodos y formas modales conocidos'),
         det.paso(1, 'Valores propios de cada modo',
                  'Voy a pasar los periodos a frecuencias circulares ω = 2π/T y a valores propios λ = ω².',
                  'Λ es la matriz diagonal con los λ de cada modo, en el mismo orden que las columnas de Φ.'),
         det.formula(r'\omega_j=\dfrac{2\pi}{T_j}\qquad \lambda_j=\omega_j^{2}\qquad \boldsymbol{\Lambda}=\mathrm{diag}(\lambda_j)',
                     'ω = 2π/T ;  λ = ω² ;  Λ = diag(λ)'),
         det.tabla('', ['Modo', 'T (s)', 'ω (rad/s)', 'λ = ω² (rad/s)²'],
                   [[j + 1, float(r['T'][j]), float(r['w'][j]), float(r['lam'][j])] for j in range(n)], 5),
         det.matriz('Φ (formas modales; cada columna es un modo)', r['Phi'], 5, '', nv, modos, latex_nombre=r'\boldsymbol{\Phi}'),
         det.matriz('K', r['K'], 5, 'kg/cm' if r['unidad_K'] == 'kg/cm' else 't/cm', nv, nv, latex_nombre=r'\mathbf{K}'),
         det.paso(2, 'Matriz dinámica A',
                  'Voy a reconstruir A = M⁻¹K con las formas y valores propios, porque cada modo cumple A·φ = λ·φ.',
                  'Escrito para todos los modos a la vez: A·Φ = Φ·Λ, y de ahí A = Φ·Λ·Φ⁻¹.'),
         det.formula(r'\mathbf{A}\,\boldsymbol{\Phi}=\boldsymbol{\Phi}\,\boldsymbol{\Lambda}\ \Rightarrow\ \mathbf{A}=\boldsymbol{\Phi}\,\boldsymbol{\Lambda}\,\boldsymbol{\Phi}^{-1}',
                     'A·Φ = Φ·Λ  →  A = Φ·Λ·Φ⁻¹'),
         det.matriz('A', r['A'], 4, '1/s²', nv, nv, latex_nombre=r'\mathbf{A}'),
         det.paso(3, 'Matriz de masas M',
                  'Voy a despejar M de A = M⁻¹K: multiplico por K⁻¹ para obtener M⁻¹ = A·K⁻¹ y la invierto.',
                  'Con masas concentradas en los pisos, M sale diagonal: su diagonal son las masas de los niveles.'),
         det.formula(r'\mathbf{A}=\mathbf{M}^{-1}\mathbf{K}\ \Rightarrow\ \mathbf{M}^{-1}=\mathbf{A}\,\mathbf{K}^{-1}',
                     'A = M⁻¹K  →  M⁻¹ = A·K⁻¹'),
         det.matriz('M⁻¹', r['Minv'], 5, f'1/({unid_masa})', nv, nv, latex_nombre=r'\mathbf{M}^{-1}'),
         det.matriz('M', r['M_full'], 6, unid_masa, nv, nv, latex_nombre=r'\mathbf{M}')]
    for a in r['avisos']:
        B.append(det.nota(a))
    fil = []
    for i in range(n):
        base = [i + 1, float(r['masas'][i]), f'{r["masas"][i]:.6f} · {r["g"]:g}' + (' / 1000' if r['unidad_K'] == 'kg/cm' else ''),
                float(r['pesos'][i])]
        if r['w_dist'] is not None:
            base += [float(r['longitudes'][i]), float(r['w_dist'][i])]
        fil.append(base)
    enc = ['Nivel', f'm ({unid_masa})', 'W = m·g', 'W (t)'] + (['L (m)', 'w = W/L (t/m)'] if r['w_dist'] is not None else [])
    B += [det.paso(4, 'Pesos de cada nivel',
                   'Voy a convertir cada masa en peso con W = m·g, con g = 981 cm/s².',
                   'Si diste la longitud de cada nivel, el peso se reparte como carga uniforme w = W/L.'),
          det.formula(r'W_i=m_i\,g\qquad w_i=\dfrac{W_i}{L_i}', 'W = m·g ;  w = W/L'),
          det.tabla('', enc, fil, 5),
          det.texto(f'Peso total W = {float(r["pesos"].sum()):.5f} t')]
    if r['T_comprobado'] is not None:
        B.append(det.tabla('Comprobación: periodos recalculados con la M diagonal y K', ['Modo', 'T dado (s)', 'T recalculado (s)'],
                           [[j + 1, float(np.sort(r['T'])[::-1][j]), float(np.sort(r['T_comprobado'])[::-1][j])] for j in range(n)], 5))
    return B
