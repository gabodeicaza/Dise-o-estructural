"""Resolver al revés: ¿qué sección necesitan las columnas para cumplir una distorsión o un periodo?

Trabaja sobre el modelo por secciones (por nivel o por columna). Las columnas de los niveles elegidos se reemplazan por
una sección de la forma pedida (cuadrada, circular o rectangular con una relación h/b) y se busca el tamaño:

* distorsión: el menor lado, en pasos de `paso` cm, con el que todos los entrepisos cumplen el límite de γ;
* periodo: el lado con el que el periodo fundamental es el pedido.
"""
import copy
import math

import numpy as np

import ame_sismico as ame

FORMAS = ('Cuadrada', 'Circular', 'Rectangular')


def seccion_texto(forma, lado, rel=1.0):
    """Texto de la sección (cm) para un lado dado. Rectangular: b = lado y h = rel·lado (h en la dirección del sismo)."""
    if forma == 'Circular':
        return f'Ø{lado:g}'
    if forma == 'Rectangular':
        return f'{lado:g}x{rel * lado:g}'
    return f'{lado:g}x{lado:g}'


def _base(d):
    if str(d.get('unidad_seccion') or 'cm').strip().lower() != 'cm':
        raise ValueError('El dimensionado trabaja con secciones en cm: cambia las unidades de las secciones a cm.')
    if d.get('modo') == 'matrices':
        raise ValueError('El dimensionado necesita la estructura por secciones; con matrices K y M no hay secciones que ajustar.')
    return ame.niveles_desde_columnas(d) if d.get('columnas') else copy.deepcopy(d)


def _con_lados(d, lados, forma, rel):
    d2 = copy.deepcopy(d)
    for nivel, lado in lados.items():
        d2['niveles'][nivel - 1]['seccion'] = seccion_texto(forma, lado, rel)
    return d2


def _validar(d, niveles, forma, rel, paso):
    n = len(d['niveles'])
    if not niveles:
        raise ValueError('Elige al menos un nivel para dimensionar.')
    mal = [i for i in niveles if not 1 <= i <= n]
    if mal:
        raise ValueError(f'El nivel {mal[0]} no existe (la estructura tiene {n} niveles).')
    if forma not in FORMAS:
        raise ValueError(f'Forma no válida: {forma}. Usa {", ".join(FORMAS)}.')
    if rel <= 0:
        raise ValueError('La relación h/b debe ser mayor que cero.')
    if paso <= 0:
        raise ValueError('El paso debe ser mayor que cero.')


def _razones(res, estados, limite):
    """Razón γ / límite de cada entrepiso (de abajo hacia arriba), máxima entre los estados considerados."""
    est = res['revision']['estados']
    claves = [res['clave']] if estados == 'diseno' else [c for c in ('DL', 'SV', 'OI') if est[c]['limite'] is not None]
    r = np.zeros(res['p']['n'])
    usados = []
    for c in claves:
        lim = limite if (limite is not None and c == res['clave']) else est[c]['limite']
        if lim is None:
            continue
        r = np.maximum(r, est[c]['dist'] / lim)
        usados.append((c, lim))
    if not usados:
        raise ValueError('El estado de diseño no tiene límite de distorsión: escribe uno en "Límites de distorsión" '
                         'o en el campo de límite del dimensionado.')
    return r, usados


def dimensionar_distorsion(d, niveles, forma='Cuadrada', rel=1.0, paso=5.0, lado_ini=20.0, lado_max=150.0,
                           estados='diseno', limite=None, max_iter=2000):
    """Menor lado (en pasos) con el que cada entrepiso cumple el límite de distorsión.

    Empieza con `lado_ini` en todos los niveles elegidos y agranda el nivel del entrepiso más crítico hasta cumplir.
    estados: 'diseno' (solo el estado de cálculo) o 'todos' (todos los estados con límite). limite: γ que sustituye
    al del estado de diseño. Devuelve un dict con los lados, el resultado final y el historial."""
    d = _base(d)
    niveles = sorted(set(int(i) for i in niveles))
    _validar(d, niveles, forma, rel, paso)
    lados = {i: float(lado_ini) for i in niveles}
    res = ame.calcular(ame.preparar_modelo(_con_lados(d, lados, forma, rel)))
    antes_r, _ = _razones(res, estados, limite)
    historial, cumple = [], False
    for it in range(max_iter):
        r, usados = _razones(res, estados, limite)
        historial.append(dict(iteracion=it, lados=dict(lados), razon_max=float(r.max())))
        if r.max() <= 1 + 1e-9:
            cumple = True
            break
        peor = int(np.argmax(r)) + 1
        if peor not in lados:
            raise ValueError(f'El entrepiso crítico es el {peor}, que no está entre los niveles a dimensionar '
                             f'({", ".join(map(str, niveles))}). Agrégalo o cambia sus datos.')
        if lados[peor] >= lado_max:
            break
        lados[peor] = min(lados[peor] + paso, float(lado_max))
        res = ame.calcular(ame.preparar_modelo(_con_lados(d, lados, forma, rel)))
    r, usados = _razones(res, estados, limite)
    return dict(meta='distorsion', cumple=cumple, lados=lados, forma=forma, rel=rel, res=res, razones=r, usados=usados,
                textos={i: seccion_texto(forma, l, rel) for i, l in lados.items()}, historial=historial,
                iteraciones=len(historial), razones_inicial=antes_r, niveles=niveles,
                dist=[float(x) for x in res['revision']['estados'][res['clave']]['dist']], T=[float(t) for t in res['T']])


def _T1(d, lado, niveles, forma, rel):
    p = ame.preparar_modelo(_con_lados(d, {i: lado for i in niveles}, forma, rel))
    lam = np.linalg.eigvals(np.linalg.solve(p['M'], p['K'])).real
    return float(2 * np.pi / np.sqrt(lam.min()))


def dimensionar_periodo(d, niveles, T_obj, forma='Cuadrada', rel=1.0, paso=5.0, lado_min=10.0, lado_max=200.0):
    """Lado (igual en todos los niveles elegidos) con el que el periodo fundamental es T_obj.

    Devuelve el lado exacto y el lado redondeado hacia arriba al paso (con el que T₁ ≤ T_obj)."""
    d = _base(d)
    niveles = sorted(set(int(i) for i in niveles))
    _validar(d, niveles, forma, rel, paso)
    if T_obj <= 0:
        raise ValueError('El periodo objetivo debe ser mayor que cero.')
    t_min, t_max = _T1(d, lado_max, niveles, forma, rel), _T1(d, lado_min, niveles, forma, rel)
    if not t_min <= T_obj <= t_max:
        raise ValueError(f'Con lados entre {lado_min:g} y {lado_max:g} cm el periodo fundamental queda entre {t_min:.4f} y '
                         f'{t_max:.4f} s: no se puede llegar a {T_obj:g} s cambiando solo esos niveles.')
    a, b = lado_min, lado_max                               # T₁ baja al aumentar el lado: bisección
    for _ in range(80):
        m = 0.5 * (a + b)
        if _T1(d, m, niveles, forma, rel) > T_obj:
            a = m
        else:
            b = m
    exacto = 0.5 * (a + b)
    redondo = math.ceil(exacto / paso - 1e-9) * paso
    res = ame.calcular(ame.preparar_modelo(_con_lados(d, {i: redondo for i in niveles}, forma, rel)))
    sec = ame.parse_secciones(seccion_texto(forma, exacto, rel), 'dimensionado')[0]
    return dict(meta='periodo', lado_exacto=exacto, lado=redondo, forma=forma, rel=rel, niveles=niveles, T_obj=T_obj,
                T_exacto=_T1(d, exacto, niveles, forma, rel), T=_T1(d, redondo, niveles, forma, rel), res=res,
                texto=seccion_texto(forma, redondo, rel), I_exacta=float(sec['Ix']), A_exacta=float(sec['A']))
