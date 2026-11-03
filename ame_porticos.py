"""Pórtico plano de concreto: análisis de CM, CV y sismo (ida y regreso), combinaciones, envolventes de M y V por viga,
armado a flexión y estribos (apuntes del Dr. Rivero, tercer parcial) y figuras tipo ETABS.

Unidades de entrada: m, cm, kg/cm², t/m, t.  Cada nodo (piso, eje) tiene uz y θ propios; ux es la misma en todos los nodos
de un piso (diafragma rígido), por lo que la fuerza sísmica de cada nivel se aplica a ese grado de libertad.
Convención de momentos: positivo = tensión abajo en vigas (el diagrama se dibuja del lado de la tensión, como en ETABS).
"""
import math

import numpy as np

import ame_marco as marco
import ame_parcial3 as p3

CASOS = ('CM', 'CV', 'CVacc', 'S')
NOMBRE_CASO = {'CM': 'Carga muerta', 'CV': 'Carga viva máxima', 'CVacc': 'Carga viva accidental', 'S': 'Sismo (ida, +)'}


def combinaciones(grupo='B', estabilidad=False, FC_acc=1.1):
    """Combinaciones de diseño como coeficientes sobre (CM, CV, CVacc, S): gravitacional, sismo de ida y de regreso
    (apuntes p. 2: FC = 1.1 con CM + CVacc + sismo; 0.9 en estabilidad)."""
    fcm, fcv = (1.5, 1.7) if grupo == 'A' else (1.3, 1.5)
    c = {'Gravedad': (fcm, fcv, 0.0, 0.0),
         'Sismo ida (+)': (FC_acc, 0.0, FC_acc, FC_acc),
         'Sismo regreso (−)': (FC_acc, 0.0, FC_acc, -FC_acc)}
    if estabilidad:
        c['Estabilidad ida (+)'] = (0.9, 0.0, 0.0, FC_acc)
        c['Estabilidad regreso (−)'] = (0.9, 0.0, 0.0, -FC_acc)
    return c


def _geometria(spans, alturas, col_secc, vig_secc, E, fi_col, fi_vig):
    xs = np.concatenate([[0.0], np.cumsum(spans)]) * 100.0
    n, ne = len(alturas), len(xs)
    cols = []
    for k in range(n):
        b, h = col_secc[k]
        cols.append(dict(b=b, h=h, A=b * h, I=fi_col * b * h ** 3 / 12))
    vigs = []
    for k in range(n):
        b, h = vig_secc[k]
        vigs.append(dict(b=b, h=h, A=b * h, I=fi_vig * b * h ** 3 / 12))
    return xs, n, ne, cols, vigs


def resolver(spans, alturas, col_secc, vig_secc, E, cargas, F_sismo, base='Empotrada', fi_col=1.0, fi_vig=1.0):
    """Resuelve los cuatro casos. cargas: dict de listas por piso (de abajo hacia arriba) con 'CM', 'CV', 'CVacc' (t/m,
    uniforme en todas las crujías del piso, o lista por crujía). F_sismo: fuerza lateral de cada nivel (t), de abajo
    hacia arriba. Devuelve dict con diagramas por caso y desplazamientos."""
    spans = [float(s) for s in spans]
    xs, n, ne, cols, vigs = _geometria(spans, alturas, col_secc, vig_secc, E, fi_col, fi_vig)
    zs = np.concatenate([[0.0], np.cumsum(alturas)]) * 100.0
    idx, cont = {}, n
    for k in range(1, n + 1):
        for e in range(1, ne + 1):
            idx[(k, e)] = (k - 1, cont, cont + 1)
            cont += 2
    for e in range(1, ne + 1):
        idx[(0, e)] = (-1, -1, -1) if base == 'Empotrada' else (-1, -1, cont)
        if base != 'Empotrada':
            cont += 1
    nd = cont
    K = np.zeros((nd, nd))
    elems = []                                    # (tipo, k, i, A, I, L, T, dofs)
    for k in range(1, n + 1):
        for e in range(1, ne + 1):
            c = cols[k - 1]
            L = alturas[k - 1] * 100.0
            elems.append(('c', k, e, c, L, marco._rot(0.0, 1.0), list(idx[(k - 1, e)]) + list(idx[(k, e)])))
        for j in range(1, ne):
            v = vigs[k - 1]
            L = xs[j] - xs[j - 1]
            elems.append(('b', k, j, v, L, marco._rot(1.0, 0.0), list(idx[(k, j)]) + list(idx[(k, j + 1)])))
    for tipo, k, i, s, L, T, dofs in elems:
        kg = T.T @ marco._k_barra(E, s['A'], s['I'], L) @ T
        for a, ga in enumerate(dofs):
            if ga < 0:
                continue
            for b_, gb in enumerate(dofs):
                if gb >= 0:
                    K[ga, gb] += kg[a, b_]

    def q_de(caso, k, j):
        v = cargas.get(caso, [0.0] * n)[k - 1]
        v = v[j - 1] if isinstance(v, (list, tuple, np.ndarray)) else v
        return float(v) * 10.0                    # t/m → kg/cm

    salida = dict(spans=spans, alturas=list(alturas), xs=xs / 100, zs=zs / 100, n=n, ne=ne, caso={}, vigas=vigs, cols=cols,
                  dim_vigas=vig_secc, dim_cols=col_secc)
    for caso in CASOS:
        P = np.zeros(nd)
        if caso == 'S':
            for k in range(1, n + 1):
                P[k - 1] += float(F_sismo[k - 1]) * 1000.0
        f0 = {}
        for tipo, k, i, s, L, T, dofs in elems:
            if tipo != 'b':
                continue
            q = q_de(caso, k, i)
            if q == 0:
                continue
            loc = np.array([0, q * L / 2, q * L ** 2 / 12, 0, q * L / 2, -q * L ** 2 / 12])
            f0[(k, i)] = loc
            eq = -(T.T @ loc)
            for a, ga in enumerate(dofs):
                if ga >= 0:
                    P[ga] += eq[a]
        u = np.linalg.solve(K, P)
        diag, desp = {}, {}
        for tipo, k, i, s, L, T, dofs in elems:
            ue = np.array([u[g] if g >= 0 else 0.0 for g in dofs])
            fl = marco._k_barra(E, s['A'], s['I'], L) @ (T @ ue)
            if tipo == 'b':
                fl = fl + f0.get((k, i), np.zeros(6))
                q = q_de(caso, k, i)
                x = np.linspace(0, L, 41)
                M = -fl[2] + fl[1] * x - q * x ** 2 / 2
                V = fl[1] - q * x
                diag[('b', k, i)] = dict(x=x / 100, M=M / 1e5, V=V / 1000, q=q / 10)
            else:
                z = np.linspace(0, L, 11)
                M = -fl[2] + fl[1] * z
                V = np.full_like(z, fl[1])
                diag[('c', k, i)] = dict(x=z / 100, M=M / 1e5, V=V / 1000, N=-fl[0] / 1000)
        for k in range(0, n + 1):
            for e in range(1, ne + 1):
                g = idx[(k, e)]
                desp[(k, e)] = tuple(u[x] if x >= 0 else 0.0 for x in g)
        salida['caso'][caso] = dict(diag=diag, desp=desp)
    return salida


def combinar(res, coef, nombre=None):
    """Diagramas de una combinación (suma lineal de los casos). coef: (cCM, cCV, cCVacc, cS)."""
    out = {}
    base = res['caso']['CM']['diag']
    for clave, d0 in base.items():
        M = sum(c * res['caso'][caso]['diag'][clave]['M'] for c, caso in zip(coef, CASOS))
        V = sum(c * res['caso'][caso]['diag'][clave]['V'] for c, caso in zip(coef, CASOS))
        out[clave] = dict(x=d0['x'], M=M, V=V)
    return out


def envolvente(res, combos):
    """Envolvente máxima y mínima de M y V (todas las combinaciones) por elemento, y la combinación que rige."""
    mats = {nombre: combinar(res, c) for nombre, c in combos.items()}
    env = {}
    for clave in res['caso']['CM']['diag']:
        Ms = np.array([m[clave]['M'] for m in mats.values()])
        Vs = np.array([m[clave]['V'] for m in mats.values()])
        env[clave] = dict(x=res['caso']['CM']['diag'][clave]['x'], Mmax=Ms.max(0), Mmin=Ms.min(0), Vmax=Vs.max(0),
                          Vmin=Vs.min(0), nombres=list(mats))
    return env, mats


def desplazamientos(res, caso='S'):
    """Desplazamiento lateral y distorsión por entrepiso del caso (cm, adimensional)."""
    n = res['n']
    ux = np.array([res['caso'][caso]['desp'][(k, 1)][0] for k in range(1, n + 1)])
    dif = np.diff(np.concatenate([[0.0], ux]))
    return ux, dif / (np.array(res['alturas']) * 100)


# ---------------------------------------------------------------- diseño de vigas
def desarrollo_flexion(Mu_tm, b, h, r, fc, fy, signo=''):
    """Cálculo de As desarrollado paso a paso para un momento último; devuelve (As_diseño, lineas, ok)."""
    d = h - r
    lim = p3.limites_flexion(b, d, fc, fy)
    fcc = lim['fcc']
    lin = [f'Mu{signo} = {Mu_tm:.3f} t·m = {Mu_tm * 1e5:,.0f} kg·cm; d = h − r = {h:g} − {r:g} = {d:g} cm',
           f"f″c = 0.85·{fc:g} = {fcc:.2f} kg/cm²; β1 = {lim['beta1']:.3f}",
           f"Asb = (f″c/fy)(6000β1/(fy+6000))·b·d = {lim['Asb']:.2f} cm²;  As,máx = 0.75·Asb = {lim['As_max']:.2f} cm²",
           f"As,mín = máx(0.8√f′c/fy·b·d = {0.8 * math.sqrt(fc) / fy * b * d:.2f}; 14/fy·b·d = {14 / fy * b * d:.2f}) = {lim['As_min']:.2f} cm²"]
    if Mu_tm <= 1e-9:
        lin.append(f"No hay momento de este signo → rige As,mín = {lim['As_min']:.2f} cm²")
        return lim['As_min'], lin, True, lim
    a2, a1, a0 = fy ** 2 / (2 * b * fcc), -fy * d, Mu_tm * 1e5 / p3.FR_FLEXION
    disc = a1 ** 2 - 4 * a2 * a0
    lin.append(f'Cuadrática: ({a2:.3f}) As² − ({-a1:,.0f}) As + {a0:,.0f} = 0   (FR = 0.9)')
    if disc < 0:
        lin.append('Discriminante negativo: la sección no alcanza → aumenta b, h o f′c.')
        return None, lin, False, lim
    As = (-a1 - math.sqrt(disc)) / (2 * a2)
    lin.append(f'As = [{-a1:,.0f} − √({disc:,.0f})]/(2·{a2:.3f}) = {As:.2f} cm² (raíz menor)')
    if As > lim['As_max']:
        lin.append(f"As = {As:.2f} > As,máx = {lim['As_max']:.2f} → cambia la sección (o usa doble refuerzo)")
        return As, lin, False, lim
    As_d = max(As, lim['As_min'])
    if As_d > As:
        lin.append(f'Rige As,mín → As = {As_d:.2f} cm²')
    return As_d, lin, True, lim


def _elige_armado(As_req, b, r, fc, fy, d, preferir=None):
    """Mejor armado (menor exceso) que cabe en un lecho, o en dos si hace falta. Devuelve dict con armado, As, MR, lechos."""
    cands = p3.proponer_armados(As_req, top=12)
    rec_lat = 4.0                                 # recubrimiento libre + estribo #3 aprox.
    for c in cands:
        partes = [p.strip() for p in c['armado'].split('+')]
        diam = []
        for p in partes:
            n = int(p.split('#')[0])
            nom = '#' + p.split('#')[1]
            diam += [{'#3': 0.95, '#4': 1.27, '#5': 1.59, '#6': 1.91, '#8': 2.54, '#10': 3.18}[nom]] * n
        nb = len(diam)
        libre = (b - 2 * (rec_lat + 0.95) - sum(diam)) / (nb - 1) if nb > 1 else 99
        sep_min = max(2.5, max(diam))
        c['cabe_1_lecho'] = libre >= sep_min
        c['sep_libre'] = libre
    ok = [c for c in cands if c['cabe_1_lecho']]
    mejor = (ok or cands or [None])[0]
    if mejor is None:
        return None
    mejor['MR'] = p3.momento_resistente(mejor['As'], b, d, fc, fy) / 1e5
    return mejor


def disenar_viga(env, clave, b, h, r, fc, fy, col_h_cm, paño=True, Av=1.42, fy_est=4200.0, lambda_=1.0):
    """Diseña una viga (clave ('b', piso, crujía)) con su envolvente: 6 secciones de flexión y los estribos."""
    e = env[clave]
    x, Mmax, Mmin = e['x'] * 100, e['Mmax'], e['Mmin']
    L = x[-1]
    cara = (col_h_cm / 2 if paño else 0.0)
    d = h - r
    fi = int(np.argmin(np.abs(x - cara)))
    fd = int(np.argmin(np.abs(x - (L - cara))))
    zona = slice(fi, fd + 1)
    medio = np.where((x >= 0.25 * L) & (x <= 0.75 * L))[0]            # el centro del claro se busca en la parte central
    ic = int(medio[np.argmax(Mmax[medio])])
    pos = {'izquierdo': fi, 'centro': ic, 'derecho': fd}
    secciones = []
    for nombre, i in pos.items():
        for lecho, Mu in (('superior', max(0.0, -Mmin[i])), ('inferior', max(0.0, Mmax[i]))):
            As, lin, ok, lim = desarrollo_flexion(Mu, b, h, r, fc, fy, '−' if lecho == 'superior' else '+')
            arm = _elige_armado(As, b, r, fc, fy, d) if As is not None and ok else None
            secciones.append(dict(posicion=nombre, lecho=lecho, x=x[i] / 100, Mu=Mu, As=As, ok=ok, lineas=lin, armado=arm,
                                  lim=lim))
    # resumen de acero: superior en cada apoyo, inferior corrido (el mayor de los tres)
    def get(pn, lc):
        return next(s for s in secciones if s['posicion'] == pn and s['lecho'] == lc)
    inf = max((get(pn, 'inferior') for pn in pos), key=lambda s: s['As'] or 0)
    ok_viga = all(s['ok'] for s in secciones)
    # cortante: máximo en valor absoluto cerca de cada apoyo (a "d" del paño según la práctica; aquí en el paño)
    Vmax = float(max(np.max(np.abs(e['Vmax'][zona])), np.max(np.abs(e['Vmin'][zona]))))
    As_trac = get('izquierdo', 'superior')['armado']['As'] if get('izquierdo', 'superior')['armado'] else get('izquierdo', 'superior')['lim']['As_min']
    cort = p3.cortante_viga(Vmax * 1000, b, d, h, L, fc, fy_est, Av, As_trac, lambda_)
    return dict(clave=clave, L=L / 100, b=b, h=h, d=d, secciones=secciones, inferior_corrido=inf, Vu=Vmax, cortante=cort,
                ok=ok_viga and cort['ok'], x_cara=(cara / 100), fc=fc, fy=fy)


def resumen_vigas(env, dims, fc, fy, col_h, r=5.0, paño=True):
    """Diseña todas las vigas; dims[k] = (b, h) de cada piso."""
    out = {}
    for clave in env:
        if clave[0] == 'b':
            b, h = dims[clave[1] - 1]
            out[clave] = disenar_viga(env, clave, b, h, r, fc, fy, col_h[clave[1] - 1], paño)
    return out


# ---------------------------------------------------------------- figuras tipo ETABS
def _plt():
    import matplotlib
    matplotlib.use('Agg', force=False)
    import matplotlib.pyplot as plt
    return plt


def figura_diagrama(res, diag, tipo='M', titulo='', unidad=None, extra=None, escala=0.38, etiquetas=True):
    """Marco con el diagrama de M o V de todos los elementos. diag: {clave: dict(x, M, V)}.
    extra: segundo diagrama (envolvente mínima) para superponer; el principal es el máximo."""
    plt = _plt()
    xs, zs, n, ne = res['xs'], res['zs'], res['n'], res['ne']
    unidad = unidad or ('t·m' if tipo == 'M' else 't')
    cl = ('#F2D027', '#D63B3B') if tipo == 'M' else ('#69B7E8', '#F29A2E')
    todos = [diag] + ([extra] if extra else [])
    vmax = max(float(np.max(np.abs(d[k][tipo]))) for d in todos for k in d) or 1.0
    sc = escala * min(res['alturas'] + [res['spans'][0]]) / vmax
    fig, ax = plt.subplots(figsize=(9.5, 1.6 + 1.5 * n))
    for j in range(ne):
        ax.plot([xs[j]] * 2, [0, zs[-1]], color='#555', lw=0.8, zorder=1)
    for k in range(1, n + 1):
        ax.plot([xs[0], xs[-1]], [zs[k]] * 2, color='#555', lw=0.8, zorder=1)
    for d in todos:
        for clave, v in d.items():
            val = v[tipo]
            if clave[0] == 'b':
                _, k, j = clave
                px = xs[j - 1] + v['x']
                py0 = np.full_like(px, zs[k])
                py = py0 - val * sc if tipo == 'M' else py0 + val * sc
                ax.fill_between(px, py0, py, where=val >= 0, color=cl[0], alpha=0.85, lw=0, interpolate=True, zorder=2)
                ax.fill_between(px, py0, py, where=val <= 0, color=cl[1], alpha=0.85, lw=0, interpolate=True, zorder=2)
                ax.plot(px, py, color='#333', lw=0.6, zorder=3)
                if etiquetas:
                    for i in {0, len(val) - 1, int(np.argmax(np.abs(val)))}:
                        ax.annotate(f'{val[i]:.2f}', (px[i], py[i]), fontsize=6.5, ha='center', va='top' if val[i] >= 0 else 'bottom',
                                    color='#222', zorder=5)
            else:
                _, k, e = clave
                pz = zs[k - 1] + v['x']
                px0 = np.full_like(pz, xs[e - 1])
                px = px0 + val * sc
                ax.fill_betweenx(pz, px0, px, where=val >= 0, color=cl[0], alpha=0.85, lw=0, interpolate=True, zorder=2)
                ax.fill_betweenx(pz, px0, px, where=val <= 0, color=cl[1], alpha=0.85, lw=0, interpolate=True, zorder=2)
                ax.plot(px, pz, color='#333', lw=0.6, zorder=3)
                if etiquetas:
                    for i in {0, len(val) - 1}:
                        ax.annotate(f'{val[i]:.2f}', (px[i], pz[i]), fontsize=6.5, ha='left' if val[i] >= 0 else 'right',
                                    va='center', color='#222', zorder=5)
    for j in range(ne):
        ax.plot(xs[j], 0, marker='s', color='#222', ms=6)
    ax.set_aspect('equal')
    ax.set_title(f'{titulo}  [{unidad}]', fontsize=11, color='#0B2545')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.margins(0.08)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def figura_cargas(res, F_sismo, cargas):
    plt = _plt()
    xs, zs, n, ne = res['xs'], res['zs'], res['n'], res['ne']
    fig, ax = plt.subplots(figsize=(9.5, 1.6 + 1.5 * n))
    for j in range(ne):
        ax.plot([xs[j]] * 2, [0, zs[-1]], color='#1D4E89', lw=2)
    for k in range(1, n + 1):
        ax.plot([xs[0], xs[-1]], [zs[k]] * 2, color='#1D4E89', lw=2)
        txt = ' · '.join(f'{c} {np.atleast_1d(cargas[c][k - 1])[0]:g}' for c in ('CM', 'CV', 'CVacc') if c in cargas)
        ax.text((xs[0] + xs[-1]) / 2, zs[k] + 0.12, f'{txt} t/m', ha='center', fontsize=8, color='#0B2545')
        ax.annotate('', xy=(xs[0], zs[k]), xytext=(xs[0] - 1.1, zs[k]),
                    arrowprops=dict(arrowstyle='-|>', color='#D63B3B', lw=1.8))
        ax.text(xs[0] - 1.15, zs[k] + 0.1, f'F{k} = {F_sismo[k - 1]:.2f} t', ha='right', fontsize=8, color='#D63B3B')
    for j in range(ne):
        ax.plot(xs[j], 0, marker='s', color='#222', ms=6)
    ax.set_aspect('equal')
    ax.set_title('Modelo del pórtico, cargas y fuerzas sísmicas (sismo de ida)', fontsize=11, color='#0B2545')
    ax.margins(0.12)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def figura_deformada(res, caso='S', signo=1.0, titulo='Deformada'):
    plt = _plt()
    xs, zs, n, ne = res['xs'], res['zs'], res['n'], res['ne']
    desp = res['caso'][caso]['desp']
    mx = max(max(abs(v[0]), abs(v[1])) for v in desp.values()) or 1.0
    f = 0.5 / (mx / 100)              # amplifica para que el desplazamiento máximo mida 0.5 m en el dibujo
    fig, ax = plt.subplots(figsize=(9.5, 1.6 + 1.5 * n))

    def pto(k, e):
        u = desp[(k, e)]
        return xs[e - 1] + signo * u[0] / 100 * f, zs[k] + signo * u[1] / 100 * f
    for k in range(1, n + 1):
        for e in range(1, ne + 1):
            a, b = pto(k - 1, e), pto(k, e)
            ax.plot([xs[e - 1], xs[e - 1]], [zs[k - 1], zs[k]], color='#bbb', lw=1)
            ax.plot([a[0], b[0]], [a[1], b[1]], color='#1D4E89', lw=2)
        for e in range(1, ne):
            a, b = pto(k, e), pto(k, e + 1)
            ax.plot([xs[e - 1], xs[e]], [zs[k], zs[k]], color='#bbb', lw=1)
            ax.plot([a[0], b[0]], [a[1], b[1]], color='#1D4E89', lw=2)
    ax.set_aspect('equal')
    ax.set_title(f'{titulo} (amplificada ×{f:.0f})', fontsize=11, color='#0B2545')
    ax.margins(0.1)
    fig.tight_layout()
    return fig


def figura_viga(env, dis, mats=None):
    """Una viga: envolvente de M y V, y acero requerido vs propuesto a lo largo del claro."""
    plt = _plt()
    clave = dis['clave']
    e = env[clave]
    x = e['x']
    fig, axs = plt.subplots(3, 1, figsize=(9.5, 8.2), sharex=True)
    a = axs[0]
    if mats:
        for nombre, m in mats.items():
            a.plot(x, m[clave]['M'], lw=0.9, ls='--', label=nombre)
    a.fill_between(x, 0, e['Mmax'], color='#F2D027', alpha=0.85, label='Envolvente máx. (+)')
    a.fill_between(x, 0, e['Mmin'], color='#D63B3B', alpha=0.85, label='Envolvente mín. (−)')
    a.invert_yaxis()
    a.axhline(0, color='#333', lw=0.8)
    a.set_ylabel('M (t·m)  [tensión abajo = +]')
    a.legend(fontsize=7, ncol=3)
    a.set_title(f'Viga piso {clave[1]}, crujía {clave[2]}: envolventes y acero', fontsize=11, color='#0B2545')
    a = axs[1]
    if mats:
        for nombre, m in mats.items():
            a.plot(x, m[clave]['V'], lw=0.9, ls='--')
    a.fill_between(x, 0, e['Vmax'], color='#69B7E8', alpha=0.85)
    a.fill_between(x, 0, e['Vmin'], color='#F29A2E', alpha=0.85)
    a.axhline(0, color='#333', lw=0.8)
    a.set_ylabel('V (t)')
    a = axs[2]
    b, h = dis['b'], dis['h']
    Asreq_sup = [max(0.0, -m) for m in e['Mmin']]
    Asreq_inf = [max(0.0, m) for m in e['Mmax']]
    d = dis['d']
    lim = dis['secciones'][0]['lim']

    fc, fy = dis['fc'], dis['fy']
    sup = [max(lim['As_min'], p3.area_requerida(M * 1e5, b, d, fc, fy) or lim['As_max']) if M > 0 else lim['As_min'] for M in Asreq_sup]
    inf = [max(lim['As_min'], p3.area_requerida(M * 1e5, b, d, fc, fy) or lim['As_max']) if M > 0 else lim['As_min'] for M in Asreq_inf]
    a.plot(x, sup, color='#D63B3B', label='As requerida superior')
    a.plot(x, inf, color='#C99A00', label='As requerida inferior')
    a.axhline(lim['As_max'], color='#555', ls=':', label='As,máx')
    a.axhline(lim['As_min'], color='#999', ls=':', label='As,mín')
    sup_prop = max((s['armado']['As'] for s in dis['secciones'] if s['lecho'] == 'superior' and s['armado']), default=None)
    inf_prop = dis['inferior_corrido']['armado']['As'] if dis['inferior_corrido']['armado'] else None
    if inf_prop:
        a.axhline(inf_prop, color='#C99A00', lw=2, alpha=0.5, label=f'inferior propuesto {dis["inferior_corrido"]["armado"]["armado"]}')
    a.set_ylabel('As (cm²)')
    a.set_xlabel('x (m)')
    a.legend(fontsize=7, ncol=3)
    for ax_ in axs:
        for s in ('top', 'right'):
            ax_.spines[s].set_visible(False)
    fig.tight_layout()
    return fig
