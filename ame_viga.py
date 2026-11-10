"""Viga continua sola (sin pórtico): momentos y cortantes por elementos finitos de viga (EI constante), envolvente de diseño,
puntos de diseño, armado longitudinal (corridas y bastones) y zonas de estribos, como en los ejemplos de clase.

Unidades: t, m. M positivo = tensión abajo, V positivo = la parte izquierda sube. Cargas hacia abajo.
La viga se define por nodos (extremos, apoyos y puntos de carga); entre nodos consecutivos hay un segmento con su carga uniforme."""
import math

import numpy as np

import ame_parcial3 as p3

TIPOS = ['—', 'Articulado', 'Empotrado']
APOYOS = ['Articulado', 'Empotrado', 'Libre (volado)']


def _ke(L):
    return np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L ** 2, -6 * L, 2 * L ** 2],
                     [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L ** 2, -6 * L, 4 * L ** 2]]) / L ** 3


def resolver_nodos(xn, tipos, w, P=None, npts=41):
    """Una condición de carga. xn: posiciones de los nodos (m); tipos: '—' (sin apoyo), 'Articulado' o 'Empotrado' por nodo;
    w: carga uniforme de cada segmento (t/m); P: carga puntual en cada nodo (t). EI = 1 (los momentos de una viga prismática no
    dependen de EI). Devuelve dict(x, M, V, R, xn): x, M, V con forma (segmentos × npts); R = reacción vertical (t) de cada nodo."""
    xn = np.asarray(xn, float)
    ne = len(xn) - 1
    nn = ne + 1
    w = np.asarray(w, float)
    P = np.zeros(nn) if P is None else np.asarray(P, float)
    K = np.zeros((2 * nn, 2 * nn))
    F = np.zeros(2 * nn)
    feq = []
    for e in range(ne):
        L = xn[e + 1] - xn[e]
        q = -w[e]
        fe = np.array([q * L / 2, q * L ** 2 / 12, q * L / 2, -q * L ** 2 / 12])
        feq.append(fe)
        ix = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        K[np.ix_(ix, ix)] += _ke(L)
        F[ix] += fe
    F[0::2] -= P
    fijos = []
    for k, t in enumerate(tipos):
        if t == 'Articulado':
            fijos.append(2 * k)
        elif t == 'Empotrado':
            fijos += [2 * k, 2 * k + 1]
    if len(fijos) < 2:
        raise ValueError('La viga no tiene apoyos suficientes: necesita al menos dos apoyos o un empotramiento.')
    libres = [d for d in range(2 * nn) if d not in fijos]
    u = np.zeros(2 * nn)
    if libres:
        sub = K[np.ix_(libres, libres)]
        if np.linalg.cond(sub) > 1e12:
            raise ValueError('La viga es inestable (mecanismo): revisa los apoyos.')
        u[libres] = np.linalg.solve(sub, F[libres])
    X, M, V = [], [], []
    R = P.copy()
    for e in range(ne):
        L = xn[e + 1] - xn[e]
        ix = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        f = _ke(L) @ u[ix] - feq[e]
        R[e] += f[0]
        R[e + 1] += f[2]
        xl = np.linspace(0.0, L, npts)
        q = -w[e]
        V.append(f[0] + q * xl)
        M.append(f[0] * xl + q * xl ** 2 / 2 - f[1])
        X.append(xn[e] + xl)
    R = np.where([t != '—' for t in tipos], R, 0.0)
    return dict(x=np.array(X), M=np.array(M), V=np.array(V), R=R, xn=xn)


def resolver_viga(spans, w, izq='Articulado', der='Articulado', npts=41):
    """Viga de claros iguales a spans con los apoyos intermedios articulados y las condiciones de los extremos de APOYOS."""
    xn = np.concatenate([[0.0], np.cumsum(spans)])
    n = len(spans)
    tipos = ['—' if izq.startswith('Libre') else izq] + ['Articulado'] * (n - 1) + ['—' if der.startswith('Libre') else der]
    return resolver_nodos(xn, tipos, w, None, npts)


def envolvente_viga(xn, tipos, cm, cv, Pcm=None, Pcv=None, grupo='B', pp=0.0, alternar=False, npts=41):
    """Cargas de diseño: wu = fcm·(CM + pp) + fcv·CV en cada segmento y Pu = fcm·Pcm + fcv·Pcv en cada nodo (Grupo B: 1.3/1.5;
    Grupo A: 1.5/1.7). Con alternar, la carga viva se coloca solo donde más desfavorece (patrones por claro y por volado);
    sin alternar, todo cargado (como en clase). Devuelve dict(x, xn, tipos, ap, Mmax, Mmin, Vmax, Vmin, R, wu, Pu, fcm, fcv)."""
    fcm, fcv = (1.5, 1.7) if grupo == 'A' else (1.3, 1.5)
    xn = np.asarray(xn, float)
    ne, nn = len(xn) - 1, len(xn)
    cm, cv = np.asarray(cm, float), np.asarray(cv, float)
    Pcm = np.zeros(nn) if Pcm is None else np.asarray(Pcm, float)
    Pcv = np.zeros(nn) if Pcv is None else np.asarray(Pcv, float)
    ap = [k for k, t in enumerate(tipos) if t != '—']
    base = resolver_nodos(xn, tipos, cm + pp, Pcm, npts)
    if alternar:
        g_el = [sum(1 for a in ap if a <= e) for e in range(ne)]
        g_nd = [sum(1 for a in ap if a < k) for k in range(nn)]
        casos = []
        for g in range(len(ap) + 1):
            casos.append(resolver_nodos(xn, tipos, np.where(np.array(g_el) == g, cv, 0.0), np.where(np.array(g_nd) == g, Pcv, 0.0), npts))
        Mp, Mn = sum(np.maximum(c['M'], 0) for c in casos), sum(np.minimum(c['M'], 0) for c in casos)
        Vp, Vn = sum(np.maximum(c['V'], 0) for c in casos), sum(np.minimum(c['V'], 0) for c in casos)
        Rp = sum(np.maximum(c['R'], 0) for c in casos)
    else:
        c = resolver_nodos(xn, tipos, cv, Pcv, npts)
        Mp = Mn = c['M']
        Vp = Vn = c['V']
        Rp = c['R']
    return dict(x=base['x'], xn=xn, tipos=list(tipos), ap=ap, fcm=fcm, fcv=fcv, pp=pp,
                Mmax=fcm * base['M'] + fcv * Mp, Mmin=fcm * base['M'] + fcv * Mn,
                Vmax=fcm * base['V'] + fcv * Vp, Vmin=fcm * base['V'] + fcv * Vn, R=fcm * base['R'] + fcv * Rp,
                wu=fcm * (cm + pp) + fcv * cv, Pu=fcm * Pcm + fcv * Pcv)


def _en(env, clave, e, xg):
    return float(np.interp(xg, env['x'][e], env[clave][e]))


def puntos_diseno(env, ancho_apoyo_cm=0.0, minimo=0.01):
    """Puntos de flexión: momento negativo en cada apoyo (lecho superior; en el paño si ancho_apoyo_cm > 0, con la cara de cada
    lado y el mayor) y momento positivo máximo de cada claro entre apoyos (lecho inferior). Lista de dict(nombre, x, Mu, lecho,
    tipo, s) con s = número de apoyo (0, 1, …) o de claro."""
    xn, ap = env['xn'], env['ap']
    ne = len(xn) - 1
    c = ancho_apoyo_cm / 200.0
    pts = []
    for s, k in enumerate(ap):
        vals = []
        if k > 0:
            vals.append(_en(env, 'Mmin', k - 1, xn[k] - min(c, (xn[k] - xn[k - 1]) / 2)))
        if k < ne:
            vals.append(_en(env, 'Mmin', k, xn[k] + min(c, (xn[k + 1] - xn[k]) / 2)))
        mu = -min(vals)
        if mu > minimo:
            pts.append(dict(nombre=f'Apoyo {s + 1} (x = {xn[k]:g} m)', x=float(xn[k]), Mu=float(mu), lecho='superior', tipo='apoyo', s=s))
    for s in range(len(ap) - 1):
        es = range(ap[s], ap[s + 1])
        e = max(es, key=lambda i: env['Mmax'][i].max())
        i = int(np.argmax(env['Mmax'][e]))
        mu = float(env['Mmax'][e][i])
        if mu > minimo:
            pts.append(dict(nombre=f'Claro {s + 1} (x = {env["x"][e][i]:.2f} m)', x=float(env['x'][e][i]), Mu=mu, lecho='inferior', tipo='claro', s=s))
    return pts


def cortantes_diseno(env, ancho_apoyo_cm=0.0):
    """Vu en la cara de cada apoyo, a cada lado donde hay viga: lista de dict(nombre, Vu, s, lado)."""
    xn, ap = env['xn'], env['ap']
    ne = len(xn) - 1
    c = ancho_apoyo_cm / 200.0
    out = []
    for s, k in enumerate(ap):
        for lado, e, xg in (('izquierda', k - 1, xn[k] - c), ('derecha', k, xn[k] + c)):
            if e < 0 or e >= ne:
                continue
            xg = min(max(xg, xn[e]), xn[e + 1])
            v = max(abs(_en(env, 'Vmax', e, xg)), abs(_en(env, 'Vmin', e, xg)))
            out.append(dict(nombre=f'Apoyo {s + 1}, {lado}', Vu=v, s=s, lado=lado))
    return out


def zonas_cortante(env):
    """Una zona de estribos por segmento entre nodos: Vu máximo del segmento y de dónde tomar el acero en tensión:
    el apoyo contiguo con mayor cortante en su extremo (acero superior) o, si no toca ningún apoyo, el claro (acero inferior)."""
    xn, ap, tipos = env['xn'], env['ap'], env['tipos']
    zonas = []
    for e in range(len(xn) - 1):
        vmax = np.maximum(np.abs(env['Vmax'][e]), np.abs(env['Vmin'][e]))
        ext = []
        for k, ext_v in ((e, vmax[0]), (e + 1, vmax[-1])):
            if tipos[k] != '—':
                ext.append((ext_v, ap.index(k)))
        if ext:
            ref = ('apoyo', max(ext)[1])
        else:
            ref = ('claro', max(0, sum(1 for a in ap if a <= e) - 1))
        zonas.append(dict(e=e, x0=float(xn[e]), x1=float(xn[e + 1]), Vu=float(vmax.max()), ref=ref))
    return zonas


def _region(xf, cond, i0):
    """Índices [lo, hi] del tramo contiguo de `cond` que contiene i0 (None si cond[i0] es falso)."""
    if not cond[i0]:
        return None
    lo = hi = i0
    while lo > 0 and cond[lo - 1]:
        lo -= 1
    while hi < len(cond) - 1 and cond[hi + 1]:
        hi += 1
    return lo, hi


def armado_longitudinal(env, puntos, fichas, b, fc, fy, var, d, As_min, ab, paso=0.01):
    """Varillas corridas y bastones como en clase: corrida = máx(As,mín; 50 % del mayor As real) en número de varillas (≥ 2),
    arriba y abajo y a todo lo largo; bastones donde el momento de la envolvente excede el MR de las corridas, anclados una
    distancia d más allá. Devuelve dict(n_corr, As_corr, MRc, bastones, lineas)."""
    reales = [f['As_real'] for f in fichas if f['As_real']]
    As_c = max(As_min, 0.5 * max(reales)) if reales else As_min
    n_c = max(2, math.ceil(As_c / ab - 1e-9))
    MRc = p3.momento_resistente(n_c * ab, b, d, fc, fy) / 1e5
    L = float(env['xn'][-1])
    xf = np.arange(0.0, L + paso / 2, paso)
    X = env['x'].ravel()
    orden = np.argsort(X, kind='stable')
    Mmin = np.interp(xf, X[orden], env['Mmin'].ravel()[orden])
    Mmax = np.interp(xf, X[orden], env['Mmax'].ravel()[orden])
    bast = []
    for p, f in zip(puntos, fichas):
        if not f['n'] or f['n'] <= n_c:
            continue
        i0 = int(round(p['x'] / paso))
        cond = (-Mmin > MRc) if p['lecho'] == 'superior' else (Mmax > MRc)
        reg = _region(xf, cond, min(i0, len(xf) - 1))
        if reg is None:
            continue
        x0, x1 = max(0.0, xf[reg[0]] - d / 100), min(L, xf[reg[1]] + d / 100)
        bast.append(dict(lecho=p['lecho'], punto=p['nombre'], n=f['n'] - n_c, x0=float(x0), x1=float(x1),
                         req0=float(xf[reg[0]]), req1=float(xf[reg[1]])))
    lin = [f'Mayor As real = {max(reales):.2f} cm²; 50 % = {0.5 * max(reales):.2f} cm²; As,mín = {As_min:.2f} cm²' if reales else '',
           f'Corrida = máx(As,mín; 50 % del mayor As real) = {As_c:.2f} cm² → n = {As_c:.2f}/{ab} = {As_c / ab:.2f} → '
           f'{n_c}{var} ({n_c * ab:.2f} cm², MR = {MRc:.2f} t·m) arriba y abajo, a todo lo largo, con gancho en los extremos']
    return dict(n_corr=n_c, As_corr=As_c, MRc=MRc, bastones=bast, lineas=[t for t in lin if t])


def figura_viga_sola(env, titulo='', ancho=13.0, alto=7.5, fuente=9):
    """Envolvente de M (positivo hacia abajo, tensión abajo) y de V, con valores en apoyos, claros y extremos de segmento."""
    import ame_porticos as pt
    plt = pt._plt()
    xn, ne = env['xn'], len(env['xn']) - 1
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(ancho, alto), sharex=True)
    for ax in (a1, a2):
        ax.axhline(0, color='#0B2545', lw=2.4)
        for k, t in enumerate(env['tipos']):
            if t != '—':
                ax.plot([xn[k]], [0], marker='^' if t == 'Articulado' else 's', color='#0B2545', ms=8, clip_on=False, zorder=5)
        ax.grid(alpha=0.25)
    for e in range(ne):
        x = env['x'][e]
        a1.fill_between(x, 0, np.maximum(env['Mmax'][e], 0), color='#F2C12E', alpha=0.75, lw=0)
        a1.fill_between(x, 0, np.minimum(env['Mmin'][e], 0), color='#D9534F', alpha=0.75, lw=0)
        a1.fill_between(x, 0, np.maximum(env['Mmin'][e], 0), color='#F2C12E', alpha=0.9, lw=0)
        a1.fill_between(x, 0, np.minimum(env['Mmax'][e], 0), color='#D9534F', alpha=0.9, lw=0)
        a1.plot(x, env['Mmax'][e], color='#6b5300', lw=0.8)
        a1.plot(x, env['Mmin'][e], color='#7a1f1f', lw=0.8)
        a2.fill_between(x, 0, env['Vmax'][e], color='#4C9BE8', alpha=0.55, lw=0)
        a2.fill_between(x, 0, env['Vmin'][e], color='#4C9BE8', alpha=0.55, lw=0)
        a2.plot(x, env['Vmax'][e], color='#0B3D75', lw=0.8)
        a2.plot(x, env['Vmin'][e], color='#0B3D75', lw=0.8)
        k = int(np.argmax(env['Mmax'][e]))
        if env['Mmax'][e][k] > 0.05 and 0 < k < len(x) - 1:
            a1.annotate(f'{env["Mmax"][e][k]:.2f}', (x[k], env['Mmax'][e][k]), textcoords='offset points', xytext=(0, 6),
                        ha='center', fontsize=fuente, fontweight='bold')
        for xe, i, lado in ((x[0], 0, 1), (x[-1], -1, -1)):
            v = env['Vmax'][e][i] if abs(env['Vmax'][e][i]) >= abs(env['Vmin'][e][i]) else env['Vmin'][e][i]
            if abs(v) > 0.05:
                a2.annotate(f'{v:.2f}', (xe, v), textcoords='offset points', xytext=(4 * lado, 5 if v > 0 else -13),
                            ha='left' if lado > 0 else 'right', fontsize=fuente)
    for k in range(ne + 1):
        vals = ([float(env['Mmin'][k - 1][-1])] if k > 0 else []) + ([float(env['Mmin'][k][0])] if k < ne else [])
        mn = min(vals)
        if mn < -0.05:
            a1.annotate(f'{mn:.2f}', (xn[k], mn), textcoords='offset points', xytext=(0, -13), ha='center', fontsize=fuente,
                        fontweight='bold', color='#7a1f1f')
    a1.invert_yaxis()
    a1.set_ylabel('M (t·m)  tensión abajo ↓')
    a2.set_ylabel('V (t)')
    a2.set_xlabel('x (m)')
    a1.set_title(titulo or 'Envolvente de momento y cortante', fontsize=fuente + 2)
    fig.tight_layout()
    return fig


def figura_alzado_viga(env, arm, zonas, estribos, h, r, nombre_var, nombre_est, ancho=13.0, alto=4.2, fuente=9):
    """Alzado: corridas arriba y abajo (con ganchos), bastones con su longitud y zonas de estribos con su separación.
    estribos: lista de separaciones (cm) por zona, en el mismo orden que zonas."""
    import ame_porticos as pt
    plt = pt._plt()
    xn = env['xn']
    L = float(xn[-1])
    H = h / 100.0
    rr = r / 100.0
    fig, ax = plt.subplots(figsize=(ancho, alto))
    ax.add_patch(plt.Rectangle((0, 0), L, H, fill=False, ec='#0B2545', lw=1.8))
    for k, t in enumerate(env['tipos']):
        if t != '—':
            ax.plot([xn[k]], [-0.02], marker='^' if t == 'Articulado' else 's', color='#0B2545', ms=9, clip_on=False, zorder=5)
    g = 0.08
    for y in (rr, H - rr):
        ax.plot([0, L], [y, y], color='#13315C', lw=2.6, solid_capstyle='butt')
        sgn = 1 if y < H / 2 else -1
        ax.plot([0, 0], [y, y + sgn * 0.22], color='#13315C', lw=2.6)
        ax.plot([L, L], [y, y + sgn * 0.22], color='#13315C', lw=2.6)
    caja = dict(fc='white', ec='none', pad=1.5)
    ax.text(L * 0.3, H - rr - 0.1, f'{arm["n_corr"]}{nombre_var} corridas (arriba)', ha='center', va='top', fontsize=fuente, color='#13315C', bbox=caja)
    ax.text(L * 0.5, rr + 0.1, f'{arm["n_corr"]}{nombre_var} corridas (abajo)', ha='center', va='bottom', fontsize=fuente, color='#13315C', bbox=caja)
    for i, bs in enumerate(arm['bastones']):
        y = H - rr - g * (1 + i % 2) if bs['lecho'] == 'superior' else rr + g * (1 + i % 2)
        ax.plot([bs['x0'], bs['x1']], [y, y], color='#D9534F', lw=3.4, solid_capstyle='butt')
        ax.text((bs['x0'] + bs['x1']) / 2, y + (-0.1 if bs['lecho'] == 'superior' else 0.05), f'+{bs["n"]}{nombre_var}  ({bs["x1"] - bs["x0"]:.2f} m)',
                ha='center', va='top' if bs['lecho'] == 'superior' else 'bottom', fontsize=fuente - 0.5, color='#B00020')
    for z, s in zip(zonas, estribos):
        if not s:
            continue
        xs = np.arange(z['x0'] + 0.05, z['x1'] - 0.01, s / 100.0)
        for xv in xs:
            ax.plot([xv, xv], [0.03, H - 0.03], color='#6c757d', lw=0.7)
        ax.text((z['x0'] + z['x1']) / 2, H + 0.08, f'E{nombre_est} @ {s} cm', ha='center', fontsize=fuente, color='#0B2545')
        ax.plot([z['x0']] * 2, [H + 0.02, H + 0.2], color='#0B2545', lw=0.8)
    ax.plot([L] * 2, [H + 0.02, H + 0.2], color='#0B2545', lw=0.8)
    ax.set_xlim(-0.3, L + 0.3)
    ax.set_ylim(-0.5 - H * 0.2, H + 0.45)
    ax.set_aspect('auto')
    ax.set_yticks([])
    ax.set_xlabel('x (m)')
    ax.set_title('Alzado del armado: corridas, bastones y estribos', fontsize=fuente + 2)
    fig.tight_layout()
    return fig
