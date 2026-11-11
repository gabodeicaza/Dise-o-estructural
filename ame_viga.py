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
    return dict(x=np.array(X), M=np.array(M), V=np.array(V), R=R, xn=xn, u=u, w=w)


def flechas_unitarias(xn, tipos, w, P=None, npts=41):
    """Desplazamiento vertical v(x) con EI = 1 (t·m³): funciones de Hermite con los giros y desplazamientos nodales, más la
    solución particular de la carga uniforme del segmento (−w·ξ²(L − ξ)²/24). Negativo = hacia abajo. Forma (segmentos × npts)."""
    r = resolver_nodos(xn, tipos, w, P, npts)
    xn = np.asarray(xn, float)
    u, w = r['u'], r['w']
    V = []
    for e in range(len(xn) - 1):
        L = xn[e + 1] - xn[e]
        xi = np.linspace(0.0, L, npts)
        t = xi / L
        N = [1 - 3 * t ** 2 + 2 * t ** 3, L * (t - 2 * t ** 2 + t ** 3), 3 * t ** 2 - 2 * t ** 3, L * (t ** 3 - t ** 2)]
        ue = u[2 * e:2 * e + 4]
        V.append(sum(Ni * ui for Ni, ui in zip(N, ue)) - w[e] * xi ** 2 * (L - xi) ** 2 / 24)
    return np.array(V)


LIMITES_FLECHA = {'L/480 + 0.3 cm (hay elementos no estructurales que pueden dañarse)': (480, 0.3),
                  'L/240 + 0.5 cm (sin elementos que puedan dañarse)': (240, 0.5)}


def revision_flechas(xn, tipos, w_serv, P_serv, b, h, fc, factor_I=0.35, factor_dif=2.0, limite=None, npts=81):
    """Estado límite de servicio (hoja del examen 3, «ELS»): flecha inmediata con las cargas de servicio sin factorizar,
    E = 14000√f′c e I = factor_I·b·h³/12; flecha total = (1 + factor_dif)·δinst (examen: 3·δinst); se compara con L/480 + 0.3 cm
    (o L/240 + 0.5 cm) en cada claro entre apoyos y en cada volado (L del volado). Devuelve dict(E, Ig, I, EI, filas, v, x, ok)."""
    div, extra = LIMITES_FLECHA[limite or next(iter(LIMITES_FLECHA))]
    E = 14000 * math.sqrt(fc)
    Ig = b * h ** 3 / 12
    I = factor_I * Ig
    EI = E * 10 * I * 1e-8                      # t·m²
    v = flechas_unitarias(xn, tipos, w_serv, P_serv, npts) / EI * 100   # cm, negativo hacia abajo
    xn = np.asarray(xn, float)
    ne = len(xn) - 1
    X = np.array([np.linspace(xn[e], xn[e + 1], npts) for e in range(ne)])
    ap = [k for k, t in enumerate(tipos) if t != '—']
    tramos = []
    if ap and ap[0] > 0:
        tramos.append(('Volado izquierdo', 0, ap[0]))
    tramos += [(f'Claro {i + 1}', ap[i], ap[i + 1]) for i in range(len(ap) - 1)]
    if ap and ap[-1] < ne:
        tramos.append(('Volado derecho', ap[-1], ne))
    filas = []
    for nombre, k0, k1 in tramos:
        seg = np.abs(v[k0:k1])
        i = np.unravel_index(int(np.argmax(seg)), seg.shape)
        Lc = (xn[k1] - xn[k0]) * 100
        d_in = float(seg[i])
        d_tot = (1 + factor_dif) * d_in
        lim = Lc / div + extra
        filas.append(dict(tramo=nombre, L=Lc, x=float(X[k0:k1][i]), d_inst=d_in, d_total=d_tot, d_perm=lim, ok=d_tot <= lim))
    return dict(E=E, Ig=Ig, I=I, EI=EI, v=v, x=X, filas=filas, ok=all(f['ok'] for f in filas), div=div, extra=extra,
                factor_I=factor_I, factor_dif=factor_dif)


def resolver_viga(spans, w, izq='Articulado', der='Articulado', npts=41):
    """Viga de claros iguales a spans con los apoyos intermedios articulados y las condiciones de los extremos de APOYOS."""
    xn = np.concatenate([[0.0], np.cumsum(spans)])
    n = len(spans)
    tipos = ['—' if izq.startswith('Libre') else izq] + ['Articulado'] * (n - 1) + ['—' if der.startswith('Libre') else der]
    return resolver_nodos(xn, tipos, w, None, npts)


FACTORES = {'B': (1.3, 1.5), 'A': (1.5, 1.7), 'Accidental': (1.1, 1.1)}


def envolvente_viga(xn, tipos, cm, cv, Pcm=None, Pcv=None, grupo='B', pp=0.0, alternar=False, npts=41, cos=None):
    """Cargas de diseño: wu = fcm·(CM + pp) + fcv·CV en cada segmento y Pu = fcm·Pcm + fcv·Pcv en cada nodo (Grupo B: 1.3/1.5;
    Grupo A: 1.5/1.7; Accidental: 1.1(CM + CVa), con CV = carga viva instantánea). pp puede ser un valor por segmento.
    cos: cos θ de cada segmento (vigas inclinadas, cargas por metro horizontal); el cortante de diseño se toma V·cos θ. Con alternar, la carga viva se coloca solo donde más desfavorece (patrones por claro y por volado);
    sin alternar, todo cargado (como en clase). Devuelve dict(x, xn, tipos, ap, Mmax, Mmin, Vmax, Vmin, R, wu, Pu, fcm, fcv)."""
    fcm, fcv = FACTORES[grupo]
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
                cos=np.ones(ne) if cos is None else np.asarray(cos, float),
                Mmax=fcm * base['M'] + fcv * Mp, Mmin=fcm * base['M'] + fcv * Mn,
                Vmax=fcm * base['V'] + fcv * Vp, Vmin=fcm * base['V'] + fcv * Vn, R=fcm * base['R'] + fcv * Rp,
                wu=fcm * (cm + pp) + fcv * cv, Pu=fcm * Pcm + fcv * Pcv)


def _cos(env, e):
    c = env.get('cos')
    return 1.0 if c is None else float(c[e])


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
            v = max(abs(_en(env, 'Vmax', e, xg)), abs(_en(env, 'Vmin', e, xg))) * _cos(env, e)
            out.append(dict(nombre=f'Apoyo {s + 1}, {lado}', Vu=v, s=s, lado=lado))
    return out


def zonas_cortante(env):
    """Una zona de estribos por segmento entre nodos: Vu máximo del segmento y de dónde tomar el acero en tensión:
    el apoyo contiguo con mayor cortante en su extremo (acero superior) o, si no toca ningún apoyo, el claro (acero inferior)."""
    xn, ap, tipos = env['xn'], env['ap'], env['tipos']
    zonas = []
    for e in range(len(xn) - 1):
        vmax = np.maximum(np.abs(env['Vmax'][e]), np.abs(env['Vmin'][e])) * _cos(env, e)
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


def tramos_estribos(env, zonas, seps, Lconf_m=0.0, s_conf=None):
    """Separación final a lo largo de la viga: la de cada zona por cortante y, si Lconf_m > 0, s = mín(s_zona; s_conf) dentro
    de Lconf (= 2h) a cada lado de cada apoyo (integridad estructural, como en el examen 3). Devuelve lista de
    dict(x0, x1, s, confinado), uniendo tramos contiguos iguales; s = None si la zona no tiene solución."""
    xn = env['xn']
    L = float(xn[-1])
    conf = [] if not Lconf_m or not s_conf else [(max(0.0, xn[k] - Lconf_m), min(L, xn[k] + Lconf_m)) for k in env['ap']]
    cortes = sorted({0.0, L} | {z['x0'] for z in zonas} | {z['x1'] for z in zonas} | {c for par in conf for c in par})
    out = []
    for a, b in zip(cortes[:-1], cortes[1:]):
        if b - a < 1e-9:
            continue
        m = (a + b) / 2
        i = next(j for j, z in enumerate(zonas) if z['x0'] - 1e-9 <= m <= z['x1'] + 1e-9)
        s = seps[i]
        dentro = any(c0 - 1e-9 <= m <= c1 + 1e-9 for c0, c1 in conf)
        if dentro and s is not None:
            s = min(s, s_conf)
        if out and out[-1]['s'] == s and out[-1]['confinado'] == dentro and abs(out[-1]['x1'] - a) < 1e-9:
            out[-1]['x1'] = float(b)
        else:
            out.append(dict(x0=float(a), x1=float(b), s=s, confinado=dentro))
    return out


def area_tributaria(corto, largo, lado='largo'):
    """Área de losa (m²) que un tablero corto × largo, con reparto a 45°, manda a una de sus vigas: lado largo → trapecio
    (2·largo − corto)·corto/4; lado corto → triángulo corto²/4 (hoja «CORTANTE» de DISEÑO DE TRABES)."""
    corto, largo = min(corto, largo), max(corto, largo)
    return (2 * largo - corto) * corto / 4 if lado == 'largo' else corto ** 2 / 4


def carga_losas_a_viga(tableros, L):
    """tableros: lista de dict(corto, largo, lado, w) con w en t/m² (o kg/m²); devuelve la carga uniforme equivalente sobre la
    viga de longitud L: w_viga = Σ(w·A_trib)/L, y el detalle por tablero."""
    det = []
    for t in tableros:
        A = area_tributaria(t['corto'], t['largo'], t.get('lado', 'largo'))
        det.append(dict(**t, A=A, wA=t['w'] * A))
    return dict(w=sum(d['wA'] for d in det) / L if L > 0 else 0.0, detalle=det)


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


def figura_alzado_viga(env, arm, zonas, estribos, h, r, nombre_var, nombre_est, ancho=13.0, alto=4.2, fuente=9, tramos=None, piel=None):
    """Alzado: corridas arriba y abajo (con ganchos), bastones con su longitud y zonas de estribos con su separación.
    estribos: lista de separaciones (cm) por zona, en el mismo orden que zonas. tramos: salida de tramos_estribos (con la zona
    de confinamiento sombreada); si se da, sustituye a zonas/estribos. piel: salida de ame_parcial3.acero_piel."""
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
    if tramos is None:
        tramos = [dict(x0=z['x0'], x1=z['x1'], s=s, confinado=False) for z, s in zip(zonas, estribos)]
    if piel and piel.get('aplica'):
        for j in range(1, piel['n_cara'] + 1):
            y = rr + j * (H - 2 * rr) / (piel['n_cara'] + 1)
            ax.plot([0, L], [y, y], color='#2a9d8f', lw=1.1, ls=(0, (6, 3)))
        ax.text(L * 0.75, rr + (H - 2 * rr) / (piel['n_cara'] + 1), f'piel: {piel["n_cara"]}{piel["var"]} por cara @ {piel["sep"]:.1f} cm',
                ha='center', va='bottom', fontsize=fuente - 1, color='#1d6f65', bbox=caja)
    for i, t in enumerate(tramos):
        s = t['s']
        if not s:
            continue
        if t.get('confinado'):
            ax.add_patch(plt.Rectangle((t['x0'], 0), t['x1'] - t['x0'], H, fc='#F2C12E', alpha=0.18, ec='none'))
        xs = np.arange(t['x0'] + 0.05, t['x1'] - 0.01, s / 100.0)
        for xv in xs:
            ax.plot([xv, xv], [0.03, H - 0.03], color='#6c757d', lw=0.7)
        ax.text((t['x0'] + t['x1']) / 2, H + (0.08 if i % 2 == 0 else 0.24), f'E{nombre_est} @ {s}' + (' (conf.)' if t.get('confinado') else ''),
                ha='center', fontsize=fuente - (1 if len(tramos) > 6 else 0), color='#0B2545')
        ax.plot([t['x0']] * 2, [H + 0.02, H + 0.2], color='#0B2545', lw=0.8)
    ax.plot([L] * 2, [H + 0.02, H + 0.2], color='#0B2545', lw=0.8)
    ax.set_xlim(-0.3, L + 0.3)
    ax.set_ylim(-0.5 - H * 0.2, H + 0.55)
    ax.set_aspect('auto')
    ax.set_yticks([])
    ax.set_xlabel('x (m)')
    ax.set_title('Alzado del armado: corridas, bastones y estribos (amarillo: zona de confinamiento 2h)', fontsize=fuente + 2)
    fig.tight_layout()
    return fig
