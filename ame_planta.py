"""Tablero de planta para la bajada de cargas: ejes en X (números) y en Y (letras, de arriba hacia abajo), tableros de losa
entre ejes y columnas en las intersecciones. Calcula el área tributaria y los metros de viga de cada columna (mitad de cada
claro adyacente, como en los apuntes) y el área que cada tablero manda a cada viga (reparto a 45°)."""
import numpy as np

import ame_viga as av


def parse_claros(txt):
    try:
        v = [float(x) for x in str(txt).replace(',', ' ').split()]
    except ValueError:
        raise ValueError('Escribe los claros como números separados por espacios, por ejemplo «5 4».')
    if not v or min(v) <= 0:
        raise ValueError('Escribe al menos un claro y que sean positivos.')
    return v


def letras(n):
    return [chr(ord('A') + i) for i in range(n)]


def nodos_existentes(mask):
    """mask[j][i] = hay losa en el tablero de la fila j (de arriba hacia abajo) y columna i. Una columna existe si toca algún tablero."""
    ny, nx = mask.shape
    ex = np.zeros((ny + 1, nx + 1), bool)
    for j in range(ny):
        for i in range(nx):
            if mask[j, i]:
                ex[j:j + 2, i:i + 2] = True
    return ex


def columna(mask, dx, dy, i, j):
    """At (m²) y metros de viga (Lv) de la columna en el eje i (0..nx), fila j (0..ny): mitad de cada claro adyacente."""
    ny, nx = mask.shape
    At = 0.0
    for jj in (j - 1, j):
        for ii in (i - 1, i):
            if 0 <= jj < ny and 0 <= ii < nx and mask[jj, ii]:
                At += dx[ii] / 2 * dy[jj] / 2
    Lv = 0.0
    for (ii, jj, largo) in vigas_en_nodo(mask, dx, dy, i, j):
        Lv += largo / 2
    return At, Lv


def vigas_en_nodo(mask, dx, dy, i, j):
    """Tramos de viga (entre ejes) que llegan al nodo: los lados de tableros existentes."""
    ny, nx = mask.shape
    out = []

    def hay(jj, ii):
        return 0 <= jj < ny and 0 <= ii < nx and mask[jj, ii]
    for ii in (i - 1, i):                       # horizontales: entre (ii,j) y (ii+1,j), arriba o abajo hay tablero
        if 0 <= ii < nx and (hay(j - 1, ii) or hay(j, ii)):
            out.append((ii, j, dx[ii]))
    for jj in (j - 1, j):                       # verticales
        if 0 <= jj < ny and (hay(jj, i - 1) or hay(jj, i)):
            out.append((i, jj, dy[jj]))
    return out


def area_a_viga(corto, largo, lado_largo):
    """Área de un tablero que va a una de sus vigas: dos direcciones (largo ≤ 2·corto) con trapecio/triángulo a 45°; si el
    tablero trabaja en una dirección (largo > 2·corto) las vigas largas reciben corto·largo/2 y las cortas nada."""
    if largo > 2 * corto:
        return corto * largo / 2 if lado_largo else 0.0
    return av.area_tributaria(corto, largo, 'largo' if lado_largo else 'corto')


def tramos_viga(mask, dx, dy):
    """Todos los tramos de viga con su longitud y el área tributaria que les llega de los tableros vecinos.
    Devuelve lista de dict(tipo 'x'|'y', i, j, L, A)."""
    ny, nx = mask.shape
    out = []
    for j in range(ny + 1):
        for i in range(nx):
            A, hay = 0.0, False
            for jj in (j - 1, j):
                if 0 <= jj < ny and mask[jj, i]:
                    hay = True
                    A += area_a_viga(min(dx[i], dy[jj]), max(dx[i], dy[jj]), dx[i] >= dy[jj])
            if hay:
                out.append(dict(tipo='x', i=i, j=j, L=dx[i], A=A))
    for i in range(nx + 1):
        for j in range(ny):
            A, hay = 0.0, False
            for ii in (i - 1, i):
                if 0 <= ii < nx and mask[j, ii]:
                    hay = True
                    A += area_a_viga(min(dx[ii], dy[j]), max(dx[ii], dy[j]), dy[j] >= dx[ii])
            if hay:
                out.append(dict(tipo='y', i=i, j=j, L=dy[j], A=A))
    return out


def nombre_tramo(t, ny_letras):
    if t['tipo'] == 'x':
        return f"{ny_letras[t['j']]}  {t['i'] + 1}–{t['i'] + 2}"
    return f"{t['i'] + 1}  {ny_letras[t['j']]}–{ny_letras[t['j'] + 1]}"


def regiones_tablero(w, h):
    """Reparto a 45° de un tablero w×h (origen arriba a la izquierda, y hacia abajo): lista de (polígono, viga, área) donde viga es
    'arriba', 'abajo', 'izq' o 'der'. Dos direcciones: trapecios hacia los lados largos y triángulos hacia los cortos; si el lado
    largo pasa de 2 veces el corto el tablero trabaja en una dirección (rectángulos hacia los lados largos)."""
    def area(p):
        return abs(sum(p[k][0] * p[(k + 1) % len(p)][1] - p[(k + 1) % len(p)][0] * p[k][1] for k in range(len(p)))) / 2
    TL, TR, BR, BL = (0, 0), (w, 0), (w, h), (0, h)
    if w >= h:
        if w > 2 * h:
            r = [([TL, TR, (w, h / 2), (0, h / 2)], 'arriba'), ([(0, h / 2), (w, h / 2), BR, BL], 'abajo')]
        else:
            a, b = (h / 2, h / 2), (w - h / 2, h / 2)
            r = [([TL, TR, b, a], 'arriba'), ([a, b, BR, BL], 'abajo'), ([TL, a, BL], 'izq'), ([TR, BR, b], 'der')]
    else:
        if h > 2 * w:
            r = [([TL, (w / 2, 0), (w / 2, h), BL], 'izq'), ([(w / 2, 0), TR, BR, (w / 2, h)], 'der')]
        else:
            a, b = (w / 2, w / 2), (w / 2, h - w / 2)
            r = [([TL, a, b, BL], 'izq'), ([a, TR, BR, b], 'der'), ([TL, TR, a], 'arriba'), ([BL, b, BR], 'abajo')]
    return [(p, v, area(p)) for p, v in r]


def figura_planta(mask, dx, dy, sel=None, tramo=None, modo='sel'):
    """Dibuja la planta. sel=(i,j) sombrea el área tributaria de esa columna; tramo=dict sombrea ese tramo de viga."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    ny, nx = mask.shape
    xs = np.concatenate([[0], np.cumsum(dx)])
    ys = np.concatenate([[0], np.cumsum(dy)])          # y hacia abajo
    fig, ax = plt.subplots(figsize=(max(6, min(14, xs[-1] * 0.9)), max(4, min(10, ys[-1] * 0.9))))
    for j in range(ny):
        for i in range(nx):
            if mask[j, i]:
                ax.add_patch(Rectangle((xs[i], ys[j]), dx[i], dy[j], fc='#eef3f8', ec='none'))
                ax.text(xs[i] + dx[i] / 2, ys[j] + dy[j] / 2, f'{dx[i]:g} × {dy[j]:g}', ha='center', va='center', fontsize=9, color='#555')
    paleta = ['#f4a261', '#2a9d8f', '#e9c46a', '#8ab17d', '#b5838d', '#6d9dc5', '#ee9b00', '#9b5de5']
    if modo == 'todas':
        k = 0
        for j in range(ny + 1):
            for i in range(nx + 1):
                a, _ = columna(mask, dx, dy, i, j)
                if a <= 0:
                    continue
                for jj in (j - 1, j):
                    for ii in (i - 1, i):
                        if 0 <= jj < ny and 0 <= ii < nx and mask[jj, ii]:
                            x0 = xs[ii] + (dx[ii] / 2 if ii == i - 1 else 0)
                            y0 = ys[jj] + (dy[jj] / 2 if jj == j - 1 else 0)
                            ax.add_patch(Rectangle((x0, y0), dx[ii] / 2, dy[jj] / 2, fc=paleta[k % len(paleta)], alpha=.55, ec='white', lw=1.2))
                ax.text(xs[i] + (.25 if i < nx else -.25), ys[j] + (.35 if j < ny else -.35), f'{a:g} m²', ha='left' if i < nx else 'right',
                        va='center', fontsize=10, fontweight='bold', color='#222', zorder=6,
                        bbox=dict(boxstyle='round,pad=.15', fc='white', ec='none', alpha=.8))
                k += 1
    if modo == 'vigas':
        col = {'arriba': '#f4a261', 'abajo': '#e9c46a', 'izq': '#2a9d8f', 'der': '#8ab17d'}
        for j in range(ny):
            for i in range(nx):
                if not mask[j, i]:
                    continue
                for pol, v, a in regiones_tablero(dx[i], dy[j]):
                    pts = [(xs[i] + x, ys[j] + y) for x, y in pol]
                    ax.add_patch(plt.Polygon(pts, fc=col[v], alpha=.55, ec='white', lw=1.2))
                    cx_, cy_ = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
                    if a > 0.05:
                        ax.text(cx_, cy_, f'{a:.2f}', ha='center', va='center', fontsize=9, fontweight='bold', color='#222', zorder=6)
    if sel is not None and modo == 'sel':
        i, j = sel
        for jj in (j - 1, j):
            for ii in (i - 1, i):
                if 0 <= jj < ny and 0 <= ii < nx and mask[jj, ii]:
                    x0 = xs[ii] + (dx[ii] / 2 if ii == i - 1 else 0)
                    y0 = ys[jj] + (dy[jj] / 2 if jj == j - 1 else 0)
                    ax.add_patch(Rectangle((x0, y0), dx[ii] / 2, dy[jj] / 2, fc='#f4a261', alpha=.6, ec='#e76f51', hatch='//'))
    for t in ([tramo] if tramo else []):
        if t['tipo'] == 'x':
            ax.plot([xs[t['i']], xs[t['i'] + 1]], [ys[t['j']]] * 2, color='#e76f51', lw=5, alpha=.7, zorder=2)
        else:
            ax.plot([xs[t['i']]] * 2, [ys[t['j']], ys[t['j'] + 1]], color='#e76f51', lw=5, alpha=.7, zorder=2)
    ex = nodos_existentes(mask)
    for j in range(ny + 1):
        for i in range(nx):
            if any(0 <= jj < ny and mask[jj, i] for jj in (j - 1, j)):
                ax.plot([xs[i], xs[i + 1]], [ys[j]] * 2, color='#1d3557', lw=1.6, zorder=3)
    for i in range(nx + 1):
        for j in range(ny):
            if any(0 <= ii < nx and mask[j, ii] for ii in (i - 1, i)):
                ax.plot([xs[i]] * 2, [ys[j], ys[j + 1]], color='#1d3557', lw=1.6, zorder=3)
    L = letras(ny + 1)
    for j in range(ny + 1):
        for i in range(nx + 1):
            if ex[j, i]:
                ax.add_patch(Rectangle((xs[i] - .15, ys[j] - .15), .3, .3, fc='#1d3557', zorder=4))
    for i in range(nx + 1):
        ax.text(xs[i], -0.45, str(i + 1), ha='center', va='bottom', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='circle', fc='white', ec='#1d3557'))
    for j in range(ny + 1):
        ax.text(-0.45, ys[j], L[j], ha='right', va='center', fontsize=12, fontweight='bold', bbox=dict(boxstyle='circle', fc='white', ec='#1d3557'))
    for i in range(nx):
        ax.text((xs[i] + xs[i + 1]) / 2, ys[-1] + .5, f'{dx[i]:g}', ha='center', va='top', fontsize=10)
    for j in range(ny):
        ax.text(xs[-1] + .5, (ys[j] + ys[j + 1]) / 2, f'{dy[j]:g}', ha='left', va='center', fontsize=10)
    ax.set_xlim(-1.4, xs[-1] + 1.4)
    ax.set_ylim(ys[-1] + 1.2, -1.4)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig
