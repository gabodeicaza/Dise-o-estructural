"""Bajada de cargas de un edificio de varios niveles con una planta (cuadrícula) distinta por nivel."""
import numpy as np

import ame_bajada as bj
import ame_planta as pl


def columnas_edificio(niveles, masks, dx, dy, tipos, pp_viga, pp_col, extras=None):
    """niveles: de arriba hacia abajo, lista de dict(nombre, tipo, h) con h = altura del entrepiso que queda debajo de ese nivel (m).
    masks: matriz de códigos de cada nivel (mismo orden). Para cada columna que existe en algún nivel devuelve la bajada
    (lista de filas de ame_bajada.bajada, con una fila final «Base» que suma la columna hasta el desplante).
    El peso de un tramo de columna se suma en el nivel de abajo y solo si la columna existe arriba y abajo."""
    ny, nx = len(dy) + 1, len(dx) + 1
    extras = extras or [None] * len(masks)
    ex = [pl.nodos_existentes(m, x) for m, x in zip(masks, extras)]
    out = {}
    for j in range(ny):
        for i in range(nx):
            if not any(e[j, i] for e in ex):
                continue
            filas = []
            for r, n in enumerate(niveles):
                if ex[r][j, i]:
                    At, Lv = pl.columna(masks[r], dx, dy, i, j)
                else:
                    At = Lv = 0.0
                hc = niveles[r - 1]['h'] if r > 0 and ex[r][j, i] and ex[r - 1][j, i] else 0.0
                filas.append(dict(nombre=n['nombre'], tipo=n['tipo'], At=At, Lv=Lv, hc=hc))
            if ex[-1][j, i]:
                filas.append(dict(nombre='Base', tipo=niveles[-1]['tipo'], At=0.0, Lv=0.0, hc=niveles[-1]['h']))
            out[(j, i)] = bj.bajada(filas, tipos, pp_viga, pp_col)
    return out


def pesos_por_nivel(niveles, masks, dx, dy, tipos, pp_viga, pp_col, extras=None):
    """Peso sísmico de cada nivel con CM + CVacc (kg): losa, vigas y la mitad de las columnas de arriba y de abajo."""
    pesos = []
    extras = extras or [None] * len(masks)
    for r, (n, m) in enumerate(zip(niveles, masks)):
        res = pl.resumen_nivel(m, dx, dy, extras[r])
        cm, cv = tipos[n['tipo']]['CA']
        h_dev = n['h'] / 2
        h_arr = niveles[r - 1]['h'] / 2 if r > 0 else 0.0
        losa = res['area'] * (cm + cv)
        vig = res['vigas'] * pp_viga
        col = res['columnas'] * (h_dev + h_arr) * pp_col
        pesos.append(dict(nombre=n['nombre'], area=res['area'], vigas_m=res['vigas'], columnas=res['columnas'],
                          losa=losa, vigas=vig, cols=col, W=losa + vig + col))
    return pesos


def cargas_tramos(mask, dx, dy, tipo, pp_viga, grupo='B'):
    """Carga uniforme equivalente de cada tramo de viga de un nivel (t/m): CM = w·A/L + p.p., CV = w·A/L, de diseño con gravedad
    y accidental (1.1·(CM + CVacc))."""
    fcm, fcv = bj.FC_GRUPO[grupo]
    L = pl.letras(len(dy) + 1)
    filas = []
    for t in pl.tramos_viga(mask, dx, dy):
        k = t['A'] / t['L']
        cm = (tipo['CG'][0] * k + pp_viga) / 1000
        cv = tipo['CG'][1] * k / 1000
        cva = tipo['CA'][1] * k / 1000
        cma = (tipo['CA'][0] * k + pp_viga) / 1000
        filas.append(dict(tramo=pl.nombre_tramo(t, L), L=t['L'], A=t['A'], CM=cm, CV=cv, wu=fcm * cm + fcv * cv,
                          wacc=1.1 * (cma + cva), tipo=t['tipo'], i=t['i'], j=t['j']))
    return filas


def cargas_eje(niveles, masks, dx, dy, tipos, pp_viga, fila):
    """Pórtico plano del eje de letra `fila` (0 = A, 1 = B…): claros y cargas por tramo (t/m) de cada piso, de abajo hacia arriba.
    CM incluye el peso propio de la viga; CV = CVmáx de gravedad; CVacc = CV accidental. Exige que el eje tenga vigas continuas de
    extremo a extremo y los mismos claros en todos los niveles."""
    spans_ref, cargas, alturas, nombres = None, dict(CM=[], CV=[], CVacc=[]), [], []
    letra = pl.letras(len(dy) + 1)[fila]
    for r in range(len(niveles) - 1, -1, -1):                      # de abajo hacia arriba
        n, m = niveles[r], masks[r]
        tr = sorted((t for t in pl.tramos_viga(m, dx, dy) if t['tipo'] == 'x' and t['j'] == fila), key=lambda t: t['i'])
        if not tr:
            raise ValueError(f"El eje {letra} no tiene vigas en el nivel {n['nombre']}.")
        idx = [t['i'] for t in tr]
        if idx != list(range(idx[0], idx[0] + len(idx))):
            raise ValueError(f"En el nivel {n['nombre']} las vigas del eje {letra} no son continuas.")
        spans = [dx[i] for i in idx]
        if spans_ref is None:
            spans_ref = spans
        elif spans != spans_ref:
            raise ValueError('Los niveles tienen claros distintos en ese eje: este análisis exige el mismo eje en todos los niveles.')
        tp = tipos[n['tipo']]
        cm, cv, cva = [], [], []
        for t in tr:
            k = t['A'] / t['L']
            cm.append((tp['CG'][0] * k + pp_viga) / 1000)
            cv.append(tp['CG'][1] * k / 1000)
            cva.append(tp['CA'][1] * k / 1000)
        cargas['CM'].append(cm)
        cargas['CV'].append(cv)
        cargas['CVacc'].append(cva)
        alturas.append(n['h'])
        nombres.append(n['nombre'])
    return dict(spans=spans_ref, cargas=cargas, alturas=alturas, nombres=nombres)


def fuerzas_triangulares(pesos_kg, alturas_acum, coef):
    """Reparto estático triangular de un cortante basal V = coef·ΣW (t): F_i = V·W_i·h_i/Σ(W·h). Entradas de abajo hacia arriba.
    Solo es un valor de partida; las fuerzas del examen salen del modal espectral."""
    W = [w / 1000 for w in pesos_kg]
    V = coef * sum(W)
    den = sum(w * h for w, h in zip(W, alturas_acum))
    return [V * w * h / den for w, h in zip(W, alturas_acum)]


def figura_elevacion(niveles, masks, dx, dy, eje, resaltar=None, titulo='', extras=None):
    """Vista en elevación (corte por un eje) como en los apuntes: niveles con su nombre a la izquierda, altura de cada entrepiso, columnas
    y vigas del eje, apoyos en la base y E1, E2… a la derecha. eje = ('X', j) corta por el eje de letra j; ('Y', i) por el eje de número i.
    resaltar = (j, i) de una columna para marcarla."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(niveles)
    extras = extras or [None] * len(masks)
    ex = [pl.nodos_existentes(m, x) for m, x in zip(masks, extras)]                # de arriba hacia abajo
    sentido, idx = eje
    if sentido == 'X':
        pos = np.concatenate([[0], np.cumsum(dx)])
        nodo = lambda r, k: ex[r][idx, k]
        tramos = lambda r: {t['i'] for t in pl.tramos_viga(masks[r], dx, dy) if t['tipo'] == 'x' and t['j'] == idx}
        marcas = [str(k + 1) for k in range(len(pos))]
        claros = dx
        def es_sel(k):
            return resaltar is not None and resaltar == (idx, k)
    else:
        pos = np.concatenate([[0], np.cumsum(dy)])
        nodo = lambda r, k: ex[r][k, idx]
        tramos = lambda r: {t['j'] for t in pl.tramos_viga(masks[r], dx, dy) if t['tipo'] == 'y' and t['i'] == idx}
        marcas = pl.letras(len(pos))
        claros = dy
        def es_sel(k):
            return resaltar is not None and resaltar == (k, idx)
    hs = [niveles[r]['h'] for r in range(n - 1, -1, -1)]          # alturas de abajo hacia arriba (entrepiso debajo de cada nivel)
    zs = np.concatenate([[0], np.cumsum(hs)])
    fig, ax = plt.subplots(figsize=(max(7, min(15, pos[-1] * 0.55 + 4)), max(4.5, min(11, zs[-1] * 0.5 + 2))))
    for k in range(1, n + 1):                                   # k = nivel de abajo hacia arriba; r = índice de arriba hacia abajo
        r = n - k
        for i in tramos(r):
            ax.plot([pos[i], pos[i + 1]], [zs[k]] * 2, color='#1d3557', lw=4, solid_capstyle='butt', zorder=3)
        for c in range(len(pos)):
            arriba_ok = nodo(r, c)
            abajo_ok = k == 1 or nodo(r + 1, c)
            if arriba_ok and abajo_ok:
                ax.plot([pos[c]] * 2, [zs[k - 1], zs[k]], color='#e76f51' if es_sel(c) else '#1d3557', lw=5 if es_sel(c) else 3,
                        solid_capstyle='butt', zorder=4 if es_sel(c) else 2)
    for c in range(len(pos)):                                   # apoyos empotrados
        if nodo(n - 1, c):
            ax.plot([pos[c] - .5, pos[c] + .5], [0, 0], color='#222', lw=2)
            for x in np.arange(pos[c] - .5, pos[c] + .5, .2):
                ax.plot([x, x - .18], [0, -.35], color='#222', lw=1)
    for k in range(0, n + 1):
        nombre = 'PB' if k == 0 else niveles[n - k]['nombre'] + (' (Azotea)' if k == n else '')
        ax.text(pos[0] - 1.1, zs[k], nombre, ha='right', va='center', fontsize=11, fontweight='bold', color='#1d3557')
        if k > 0:
            ax.text(pos[0] - 2.6 - 1.4 * (len(nombre) > 10), (zs[k - 1] + zs[k]) / 2, f'{hs[k - 1]:g}', ha='right', va='center', fontsize=11)
            ax.annotate('', xy=(pos[0] - 2.3, zs[k]), xytext=(pos[0] - 2.3, zs[k - 1]),
                        arrowprops=dict(arrowstyle='<->', color='#888', lw=1))
            ax.text(pos[-1] + 1.2, (zs[k - 1] + zs[k]) / 2, f'E{k}', ha='left', va='center', fontsize=11, fontweight='bold')
    for c, m in enumerate(marcas):
        ax.text(pos[c], zs[-1] + .9, m, ha='center', va='bottom', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='circle', fc='white', ec='#1d3557'))
    for i, c in enumerate(claros):
        ax.text((pos[i] + pos[i + 1]) / 2, -.9, f'{c:g}', ha='center', va='top', fontsize=10)
    if titulo:
        ax.set_title(titulo, fontsize=13, fontweight='bold', pad=22)
    ax.set_xlim(pos[0] - 6.5, pos[-1] + 3)
    ax.set_ylim(-2, zs[-1] + 2.6)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig


def figura_3d(niveles, masks, dx, dy, extras=None, resaltar=None, elev=24, azim=-62, titulo='', mostrar=None):
    """Vista 3D del edificio: columnas, vigas y losas por nivel (cada nivel con su propia planta). Losa en azul, voladizo en naranja claro;
    la columna `resaltar` = (j, i) en naranja con su área tributaria sombreada en cada nivel. El nivel de más arriba es el primero."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    n = len(niveles)
    extras = extras or [None] * n
    ex = [pl.nodos_existentes(m, x) for m, x in zip(masks, extras)]          # de arriba hacia abajo
    xs = np.concatenate([[0], np.cumsum(dx)])
    ys = np.concatenate([[0], np.cumsum(dy)])
    Y = lambda y: ys[-1] - y                                                  # la fila A queda al fondo
    hs = [niveles[r]['h'] for r in range(n - 1, -1, -1)]
    zs = np.concatenate([[0], np.cumsum(hs)])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    vis = lambda r: mostrar is None or niveles[r]['nombre'] in mostrar
    for k in range(1, n + 1):
        r = n - k
        if not vis(r):
            continue
        m = masks[r]
        polis, colores = [], []
        for j in range(len(dy)):
            for i in range(len(dx)):
                if m[j, i] in (pl.LOSA, pl.VOL):
                    z = zs[k]
                    polis.append([(xs[i], Y(ys[j]), z), (xs[i + 1], Y(ys[j]), z), (xs[i + 1], Y(ys[j + 1]), z), (xs[i], Y(ys[j + 1]), z)])
                    colores.append('#f4c28d' if m[j, i] == pl.VOL else '#9ec5e8')
        if polis:
            ax.add_collection3d(Poly3DCollection(polis, facecolors=colores, edgecolors='#7a8794', linewidths=.5, alpha=.32))
        for t in pl.tramos_viga(m, dx, dy):
            if t['tipo'] == 'x':
                ax.plot([xs[t['i']], xs[t['i'] + 1]], [Y(ys[t['j']])] * 2, [zs[k]] * 2, color='#1d3557', lw=2.2)
            else:
                ax.plot([xs[t['i']]] * 2, [Y(ys[t['j']]), Y(ys[t['j'] + 1])], [zs[k]] * 2, color='#1d3557', lw=2.2)
        if resaltar is not None and ex[r][resaltar[0], resaltar[1]]:
            rects = pl.rects_tributarios(m, dx, dy, resaltar[1], resaltar[0])
            ax.add_collection3d(Poly3DCollection(
                [[(x0, Y(y0), zs[k]), (x0 + w, Y(y0), zs[k]), (x0 + w, Y(y0 + h), zs[k]), (x0, Y(y0 + h), zs[k])] for x0, y0, w, h in rects],
                facecolors='#e76f51', edgecolors='#e76f51', alpha=.55))
        for j in range(len(dy) + 1):
            for i in range(len(dx) + 1):
                if ex[r][j, i] and (k == 1 or ex[r + 1][j, i]):
                    sel = resaltar == (j, i)
                    ax.plot([xs[i]] * 2, [Y(ys[j])] * 2, [zs[k - 1], zs[k]], color='#e76f51' if sel else '#33415c', lw=4 if sel else 2.6)
    for j in range(len(dy) + 1):
        for i in range(len(dx) + 1):
            if ex[n - 1][j, i] and vis(n - 1):
                ax.scatter([xs[i]], [Y(ys[j])], [0], marker='^', color='#222', s=40)
    for k in range(0, n + 1):
        if k > 0 and not vis(n - k):
            continue
        nombre = 'PB' if k == 0 else niveles[n - k]['nombre']
        ax.text(-0.5, ys[-1] + 0.5, zs[k], nombre, ha='right', va='center', fontsize=9, color='#1d3557')
    for i in range(len(xs)):
        ax.text(xs[i], -0.8, 0, str(i + 1), ha='center', fontsize=10, fontweight='bold')
    for j, L in enumerate(pl.letras(len(ys))):
        ax.text(-0.9, Y(ys[j]), 0, L, ha='center', fontsize=10, fontweight='bold')
    ax.set_xlim(0, xs[-1])
    ax.set_ylim(0, ys[-1])
    ax.set_zlim(0, max(zs[-1], 1))
    ax.set_box_aspect((xs[-1], ys[-1], zs[-1] * 1.0))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if titulo:
        ax.set_title(titulo, fontsize=13, fontweight='bold')
    return fig
