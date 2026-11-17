"""Examen 3 · viga de concreto reforzado por secciones: varios puntos de flexión (cada uno con su Mu) y varios tramos de cortante
(cada uno con su Vu), y al final el armado de TODA la viga: varillas corridas, bastones, estribos por tramo, confinamiento y piel.

`calcular(d, puntos, tramos)` usa los mismos motores de las demás pantallas (ame_parcial3 y ame_porticos) y `bloques(res)` lo pasa al
formato de ame_detalle: la pantalla, el PDF, el Excel y el texto muestran lo mismo. Una sección sola es el caso de un punto y un tramo.
"""
import io
import math

import ame_detalle as det
import ame_parcial3 as p3
import ame_porticos as pt

DEFECTO = dict(b=35.0, h=70.0, r=5.0, fc=250.0, fy=4200.0, var='#8', est='#4', ramas=2, fy_est=4200.0, s_tope=25.0)


def _limpio(txt):
    return str(txt).replace('**', '')


def calcular(d, puntos, tramos):
    """d: b, h, r (cm); fc, fy (kg/cm²); var, est (#n); ramas; fy_est; s_tope (cm).
    puntos: [dict(punto, Mu (t·m), lecho 'superior'|'inferior', prop '4#8' o '', c_prop cm o 0)].
    tramos: [dict(tramo, Vu (t), As (cm², None = el mayor As real superior), s_prop cm o 0)]."""
    d = {**DEFECTO, **d}
    b, h, r, fc, fy = (float(d[k]) for k in ('b', 'h', 'r', 'fc', 'fy'))
    var, est, ramas = d['var'], d['est'], int(d['ramas'])
    fy_est, s_tope = float(d['fy_est']), float(d['s_tope'])
    if min(b, h, fc, fy, fy_est) <= 0:
        raise ValueError('Las dimensiones, f′c y fy deben ser positivos.')
    if r >= h / 2:
        raise ValueError('r debe ser menor que la mitad de h.')
    if not puntos:
        raise ValueError('Captura al menos un punto de flexión (Mu).')
    if not tramos:
        raise ValueError('Captura al menos un tramo de cortante (Vu).')
    if any(float(p['Mu']) < 0 for p in puntos) or any(float(t['Vu']) < 0 for t in tramos):
        raise ValueError('Mu y Vu no pueden ser negativos: captura sus valores absolutos.')
    db, est_db, a_est, ab = pt.DIAMETROS[var], pt.DIAMETROS[est], p3.VARILLAS[est], p3.VARILLAS[var]
    Av = a_est * ramas
    rec = max(r - est_db - db / 2, 1.5)
    Mu_max = max(float(p['Mu']) for p in puntos)
    Vu_max = max(float(t['Vu']) for t in tramos)
    viga = pt.ficha_viga(b, h, fc, fy, var, est, ramas, rec, r_fija=r, Mu_max=Mu_max, Vu_max=Vu_max)
    dd = h - r
    lim = p3.limites_flexion(b, dd, fc, fy)
    res = dict(d=d, b=b, h=h, r=r, fc=fc, fy=fy, var=var, est=est, ramas=ramas, fy_est=fy_est, s_tope=s_tope, Av=Av, db=db, ab=ab,
               est_db=est_db, rec=rec, viga=viga, d_ef=dd, lim=lim, Mu_max=Mu_max, Vu_max=Vu_max, error=None, avisos=[])

    pts = []
    for p in puntos:
        Mu = float(p['Mu'])
        ficha = pt.ficha_punto(Mu, b, h, fc, fy, var, rec=rec, est_db=est_db, r_fija=r, agrupar=True, lecho=p['lecho'],
                               propuesta=pt.parse_propuesta(p.get('prop', '')))
        e = dict(punto=str(p['punto']), lecho=p['lecho'], Mu=Mu, ficha=ficha, c_prop=float(p.get('c_prop') or 0), error=None,
                 alternativas=[], tanteo=None, tanteo_manual=None)
        if not ficha['As_real']:
            e['error'] = 'La sección no alcanza con acero subreforzado para este Mu: aumenta b, h o f′c.'
        else:
            e['alternativas'] = [dict(a, MR=p3.momento_resistente(a['As'], b, ficha['d'], fc, fy) / 1e5)
                                 for a in p3.proponer_armados(ficha['As_usar'])] if ficha['As_usar'] else []
            e['tanteo'] = p3.tanteo_eje_neutro(ficha['As_real'], b, ficha['d'], fc, fy)
            if e['c_prop'] > 0:
                e['tanteo_manual'] = p3.tanteo_eje_neutro(ficha['As_real'], b, ficha['d'], fc, fy, c_prop=e['c_prop'])
        pts.append(e)
    res['puntos'] = pts
    buenos = [e for e in pts if e['ficha']['As_real']]
    if not buenos:
        res.update(error='Ninguna sección alcanza con acero subreforzado: aumenta b, h o f′c.', ok=False, ok_flex=False, ok_cort=False,
                   tramos=[], corrido=None, integ=None, piel=None)
        return res

    # varillas corridas: máx(As,mín; 50 % del mayor As real), arriba y abajo; el resto son bastones
    As_max_real = max(e['ficha']['As_real'] for e in buenos)
    As_corr = max(lim['As_min'], 0.5 * As_max_real)
    n_corr = max(2, int(math.ceil(As_corr / ab - 1e-9)))
    bastones = [(e['punto'], e['ficha']['n'] - n_corr) for e in buenos if e['ficha']['n'] > n_corr]
    res['corrido'] = dict(As_max_real=As_max_real, As_corr=As_corr, n=n_corr, As=n_corr * ab, bastones=bastones)

    sup = [e['ficha']['As_real'] for e in buenos if e['lecho'] == 'superior']
    As_auto = max(sup) if sup else As_max_real
    res['As_auto'] = As_auto
    res['lim_cort'] = p3.limites_cortante_tex(b, dd, fc, Av, ramas, a_est)
    trs = []
    for t in tramos:
        Vu = float(t['Vu'])
        As_t = t.get('As')
        As_t = As_auto if As_t in (None, '') or float(As_t) <= 0 else float(As_t)
        cort = p3.cortante_ficha(Vu * 1000, b, dd, fc, fy_est, Av, As_t, s_tope)
        s_prop = float(t.get('s_prop') or 0)
        trs.append(dict(tramo=str(t['tramo']), Vu=Vu, As=As_t, cort=cort, s_prop=s_prop,
                        alt=(p3.alternativas_estribo(Vu * 1000, b, dd, fc, fy_est, As_t, s_tope) if cort['ok'] and cort['s_usar'] < 10 else []),
                        rev=(p3.revision_cortante(Vu * 1000, b, dd, fc, fy_est, Av, As_t, s_prop, s_tope) if s_prop > 0 else None)))
    res['tramos'] = trs
    res['integ'] = p3.integridad_estructural(h, dd, db, est_db, s_tope)
    res['piel'] = p3.acero_piel(b, h, r, est if est in ('#3', '#4') else '#3')
    res['ok_flex'] = bool(viga['ok_flexion'] and all(e['ficha']['ok'] for e in pts))
    res['ok_cort'] = bool(viga['ok_cortante'] and all(t['cort']['ok'] for t in trs) and all(t['rev']['ok'] for t in trs if t['rev']))
    res['ok'] = res['ok_flex'] and res['ok_cort']
    return res


# ------------------------------------------------------------------ la viga completa a partir de las secciones que se van añadiendo
TIPOS = ('Apoyo', 'Centro de claro')
LETRAS = 'abcdefghijklmnopqrstuvwxyz'


def armar(lista):
    """lista: secciones en el orden de izquierda a derecha, cada una dict(tipo 'Apoyo'|'Centro de claro', Mu (t·m, positivo),
    L (m, claro libre entre caras desde el apoyo anterior; solo en apoyos después del primero), Vu (t, del claro que cierra),
    prop (armado propuesto, opcional)). Devuelve (puntos, tramos, geom) para `calcular`."""
    if not lista:
        raise ValueError('Añade al menos dos apoyos y el centro del claro entre ellos.')
    if lista[0]['tipo'] != 'Apoyo':
        raise ValueError('La primera sección debe ser el apoyo del extremo izquierdo.')
    xn, spans, puntos, tramos = [0.0], [], [], []
    pend = None                                   # momento positivo del claro en curso
    Mapoyo = [float(lista[0]['Mu'])]
    for i, s in enumerate(lista):
        nombre = LETRAS[i] if i < len(LETRAS) else f'p{i + 1}'
        Mu = float(s['Mu'])
        if Mu < 0:
            raise ValueError('Captura los momentos como valores positivos (el programa sabe que en los apoyos es negativo).')
        if s['tipo'] == 'Apoyo':
            if i > 0:
                L, Vu = float(s.get('L') or 0), float(s.get('Vu') or 0)
                if L <= 0 or Vu <= 0:
                    raise ValueError(f'El apoyo «{nombre}» cierra un claro: captura su longitud L y su Vu.')
                if pend is None:
                    raise ValueError(f'Falta el momento positivo (centro de claro) entre los apoyos anteriores y «{nombre}».')
                x0 = xn[-1]
                xn.append(x0 + L)
                spans.append(dict(x0=x0, x1=x0 + L, L=L, Vu=Vu, Ml=Mapoyo[-1], Mr=Mu, Mc=pend))
                Mapoyo.append(Mu)
                tramos.append(dict(tramo=str(len(tramos) + 1), Vu=Vu))
                pend = None
            puntos.append(dict(punto=nombre, Mu=Mu, lecho='superior', prop=str(s.get('prop') or '').strip(), x=xn[-1], tipo='apoyo'))
        else:
            if pend is not None or len(xn) - 1 != len(spans):
                raise ValueError(f'Entre dos apoyos solo puede haber un centro de claro («{nombre}» sobra).')
            if i == 0:
                raise ValueError('Antes del centro de claro va el apoyo izquierdo.')
            pend = Mu
            puntos.append(dict(punto=nombre, Mu=Mu, lecho='inferior', prop=str(s.get('prop') or '').strip(), x=None, tipo='claro'))
    if pend is not None:
        raise ValueError('El último claro no está cerrado: añade el apoyo de su extremo derecho (con L y Vu).')
    if not spans:
        raise ValueError('Añade al menos dos apoyos y el centro del claro entre ellos.')
    # el centro de cada claro queda a la mitad del claro
    k = 0
    for p in puntos:
        if p['tipo'] == 'claro':
            p['x'] = (spans[k]['x0'] + spans[k]['x1']) / 2
            k += 1
    return puntos, tramos, dict(xn=xn, spans=spans)


def _envolvente(geom, npts=41):
    """Diagramas de cada claro como en el pizarrón: momento parabólico por tres puntos (cara izquierda, centro, cara derecha) y
    cortante lineal de +Vu a −Vu (carga uniforme)."""
    import numpy as np
    xs, M, V = [], [], []
    for s in geom['spans']:
        L, Ml, Mr, Mc = s['L'], s['Ml'], s['Mr'], s['Mc']
        a = -2 * (Ml + Mr + 2 * Mc) / L ** 2
        b = (Ml - Mr) / L - a * L
        t = np.linspace(0, L, npts)
        xs.append(s['x0'] + t)
        M.append(a * t ** 2 + b * t - Ml)
        V.append(s['Vu'] * (1 - 2 * t / L))
    M, V = np.array(M), np.array(V)
    n = len(geom['xn'])
    return dict(x=np.array(xs), xn=np.array(geom['xn']), tipos=['Articulado'] * n, ap=list(range(n)), Mmax=M, Mmin=M, Vmax=V, Vmin=V,
                cos=np.ones(len(M)))


def _bastones(env, puntos, n_corr, d_cm, paso=0.01):
    """Bastón de cada punto cuya varilla excede las corridas: va desde donde el momento de ese signo se anula (punto de inflexión)
    hasta una distancia d más allá, a cada lado. En los extremos de la viga se ancla con gancho en el apoyo."""
    import numpy as np
    from ame_viga import _region
    L = float(env['xn'][-1])
    xf = np.arange(0.0, L + paso / 2, paso)
    X = env['x'].ravel()
    o = np.argsort(X, kind='stable')
    Mm = np.interp(xf, X[o], env['Mmax'].ravel()[o])
    out = []
    for p in puntos:
        f = p['ficha']
        if not f['As_real'] or f['n'] <= n_corr:
            continue
        i0 = min(int(round(p['x'] / paso)), len(xf) - 1)
        reg = _region(xf, (Mm < -1e-9) if p['lecho'] == 'superior' else (Mm > 1e-9), i0)
        if reg is None:
            continue
        x0, x1 = max(0.0, xf[reg[0]] - d_cm / 100), min(L, xf[reg[1]] + d_cm / 100)
        out.append(dict(punto=p['punto'], lecho=p['lecho'], n=f['n'] - n_corr, x=float(p['x']), x0=float(x0), x1=float(x1),
                        izq=float(p['x'] - x0), der=float(x1 - p['x']), gancho_izq=x0 <= 1e-9, gancho_der=x1 >= L - 1e-9))
    return out


def calcular_viga(d, lista):
    """Calcula la viga completa a partir de la lista de secciones añadidas (ver `armar`)."""
    puntos, tramos, geom = armar(lista)
    res = calcular(d, [dict(punto=p['punto'], Mu=p['Mu'], lecho=p['lecho'], prop=p['prop']) for p in puntos], tramos)
    res['geom'] = geom
    res['lista'] = lista
    if res['error']:
        return res
    for e, p in zip(res['puntos'], puntos):
        e['x'], e['tipo'] = p['x'], p['tipo']
    env = _envolvente(geom)
    res['env'] = env
    cor = res['corrido']
    base = [dict(punto=e['punto'], lecho=e['lecho'], x=e['x'], ficha=e['ficha']) for e in res['puntos']]
    bast = _bastones(env, base, cor['n'], res['d_ef'])
    cor['bastones'] = [(b['punto'], b['n']) for b in bast]
    res['bastones'] = bast
    # secciones distintas a lo largo de la viga (para los cortes 1-1, 2-2, …)
    cortes, vistos = [], {}
    for e in res['puntos']:
        nt = cor['n'] + sum(b['n'] for b in bast if b['lecho'] == 'superior' and b['x0'] - 1e-9 <= e['x'] <= b['x1'] + 1e-9)
        nb = cor['n'] + sum(b['n'] for b in bast if b['lecho'] == 'inferior' and b['x0'] - 1e-9 <= e['x'] <= b['x1'] + 1e-9)
        if (nt, nb) not in vistos:
            vistos[(nt, nb)] = len(cortes) + 1
            cortes.append(dict(num=len(cortes) + 1, x=e['x'], punto=e['punto'], n_top=nt, n_bot=nb))
    res['cortes'] = cortes
    # separación de estribos a lo largo de la viga (confinamiento 2h a cada lado de cada apoyo)
    import ame_viga as av
    seps = [t['cort']['s_usar'] if t['cort']['ok'] else None for t in res['tramos']]
    zonas = [dict(e=i, x0=s['x0'], x1=s['x1'], Vu=s['Vu'], ref=('apoyo', i)) for i, s in enumerate(geom['spans'])]
    res['estribos'] = av.tramos_estribos(env, zonas, seps, Lconf_m=2 * res['h'] / 100, s_conf=res['integ']['s_usar'])
    return res


def _dibujar_seccion(ax, b, h, rec, est_db, db, n_top, n_bot, titulo):
    from matplotlib.patches import Circle, Rectangle
    ax.add_patch(Rectangle((0, 0), b, h, fc='#F1F6FC', ec='#13315C', lw=1.2))
    e0 = rec
    ax.add_patch(Rectangle((e0, e0), b - 2 * e0, h - 2 * e0, fc='none', ec='#1D4E89', lw=2.2))
    r0 = rec + est_db + db / 2
    cap = max(1, int((b - 2 * r0) // (db + 3.0)) + 1)

    def fila(n, y0, sentido):
        k = 0
        while n > 0:
            m = min(n, cap)
            xs = [b / 2] if m == 1 else [r0 + i * (b - 2 * r0) / (m - 1) for i in range(m)]
            for x in xs:
                ax.add_patch(Circle((x, y0 + sentido * k * (db + 2.5)), db / 2, fc='#0B2545', ec='none'))
            n -= m
            k += 1
    fila(n_top, h - r0, -1)
    fila(n_bot, r0, 1)
    ax.set_xlim(-2, b + 2)
    ax.set_ylim(-2, h + 2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(titulo, fontsize=10, color='#0B2545', fontweight='bold')


def figura_hoja(res):
    """Lo que se dibuja en el pizarrón: cortante, momento, alzado de la viga con corridas, bastones (con sus longitudes), cortes y
    estribos, y debajo las secciones 1-1, 2-2, …"""
    import numpy as np
    plt = pt._plt()
    from matplotlib.patches import Rectangle
    geom, env = res['geom'], res['env']
    xn = np.array(geom['xn'])
    L = float(xn[-1])
    cor = res['corrido']
    var, est = res['var'], res['est']
    n_s = max(1, len(res['cortes']))
    fig = plt.figure(figsize=(12.5, 10.6), layout='constrained')
    sf = fig.subfigures(2, 1, height_ratios=[3.5, 1.15])
    aV, aM, aE = sf[0].subplots(3, 1, sharex=True, gridspec_kw=dict(height_ratios=[1.15, 1.35, 2.1]))
    azul, oscuro = '#1D4E89', '#0B2545'
    for ax in (aV, aM, aE):
        for x in xn:
            ax.axvline(x, color='0.55', lw=0.9, ls=(0, (5, 4)), zorder=0)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    # ---- Vu
    aV.axhline(0, color='0.35', lw=0.9)
    for s in geom['spans']:
        aV.fill([s['x0'], s['x0'], s['x1'], s['x1']], [0, s['Vu'], -s['Vu'], 0], color='#4C9BE8', alpha=0.35, lw=0)
        aV.plot([s['x0'], s['x1']], [s['Vu'], -s['Vu']], color='#0B3D75', lw=1.4)
        aV.plot([s['x0'], s['x0']], [0, s['Vu']], color='#0B3D75', lw=1.0)
        aV.plot([s['x1'], s['x1']], [0, -s['Vu']], color='#0B3D75', lw=1.0)
        aV.annotate(f'{s["Vu"]:g}', (s['x0'], s['Vu']), xytext=(3, 3), textcoords='offset points', fontsize=9, color=oscuro)
        aV.annotate(f'{s["Vu"]:g}', (s['x1'], -s['Vu']), xytext=(-3, -3), textcoords='offset points', fontsize=9, color=oscuro, ha='right', va='top')
    vm = max(s['Vu'] for s in geom['spans'])
    aV.set_ylim(-1.35 * vm, 1.35 * vm)
    aV.set_ylabel('Vu (t)')
    # ---- Mu (positivo hacia arriba, como en el pizarrón)
    aM.axhline(0, color='0.35', lw=0.9)
    for e, s in enumerate(geom['spans']):
        aM.fill_between(env['x'][e], 0, env['Mmax'][e], color='#F2C12E', alpha=0.35, lw=0)
        aM.plot(env['x'][e], env['Mmax'][e], color='#6b5300', lw=1.4)
        aM.annotate(f'{s["Mc"]:g}', ((s['x0'] + s['x1']) / 2, s['Mc']), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9, color=oscuro)
    for k, x in enumerate(xn):
        v = res['puntos']
        apo = [p for p in v if p['tipo'] == 'apoyo']
        aM.annotate(f'{apo[k]["Mu"]:g}', (x, -apo[k]['Mu']), xytext=(0, -5), textcoords='offset points', ha='center', va='top', fontsize=9, color=oscuro)
    mm = max(res['Mu_max'], 1.0)
    aM.set_ylim(-1.3 * mm, 0.95 * max(s['Mc'] for s in geom['spans']) + 0.25 * mm)
    aM.set_ylabel('Mu (t·m)')
    yy = 0.28 * mm
    aM.annotate('', xy=(0, yy), xytext=(res['d_ef'] / 100, yy), arrowprops=dict(arrowstyle='<->', lw=0.9, color=azul))
    aM.text(res['d_ef'] / 100 + 0.15, yy, f'd = {res["d_ef"]:g} cm', ha='left', va='center', fontsize=8.5, color=azul)
    # ---- alzado
    aE.set_ylim(-2.45, 2.0)
    aE.set_yticks([])
    aE.spines['left'].set_visible(False)
    aE.add_patch(Rectangle((0, 0), L, 1, fc='#F7FAFD', ec='0.45', lw=1.0, zorder=1))
    aE.add_patch(Rectangle((0.03, 0.05), L - 0.06, 0.9, fc='none', ec=azul, lw=1.4, zorder=2))
    yt, yb = 0.82, 0.18
    aE.plot([0.03, L - 0.03], [yt, yt], color=azul, lw=3.0, zorder=4, solid_capstyle='butt')
    aE.plot([0.03, L - 0.03], [yb, yb], color=azul, lw=3.0, zorder=4, solid_capstyle='butt')
    for xg, sg in ((0.03, 1), (L - 0.03, -1)):               # ganchos de las corridas
        aE.plot([xg, xg], [yt, 0.36], color=azul, lw=3.0, zorder=4)
        aE.plot([xg, xg], [yb, 0.64], color=azul, lw=3.0, zorder=4)
    bast = res['bastones']
    filas = []
    for i, b in enumerate(bast):
        ocup = [filas[j] for j in range(i) if bast[j]['lecho'] == b['lecho'] and bast[j]['x0'] < b['x1'] and b['x0'] < bast[j]['x1']]
        filas.append(next(f for f in range(8) if f not in ocup))
    for b, fi in zip(bast, filas):
        y = (yt - 0.11 - 0.09 * fi) if b['lecho'] == 'superior' else (yb + 0.11 + 0.09 * fi)
        aE.plot([b['x0'], b['x1']], [y, y], color=oscuro, lw=3.0, zorder=5, solid_capstyle='butt')
        if b['gancho_izq']:
            aE.plot([b['x0'], b['x0']], [y, y - 0.28 if b['lecho'] == 'superior' else y + 0.28], color=oscuro, lw=3.0, zorder=5)
        if b['gancho_der']:
            aE.plot([b['x1'], b['x1']], [y, y - 0.28 if b['lecho'] == 'superior' else y + 0.28], color=oscuro, lw=3.0, zorder=5)
        xm = b['x']
        arriba = b['lecho'] == 'superior'
        aE.annotate(f'{b["n"]}{var}', xy=(xm, y), xytext=(xm, 1.45 if arriba else -0.15), ha='center', va='bottom' if arriba else 'top',
                    fontsize=9.5, fontweight='bold', color=oscuro, zorder=6,
                    arrowprops=dict(arrowstyle='-', lw=0.8, color='0.4'), bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none'))
        yd = -0.28 - (0.0 if arriba else 0.0)
        for lado, a, c in (('izq', b['x0'], b['x']), ('der', b['x'], b['x1'])):
            ln = b[lado]
            if ln < 1e-6:
                continue
            aE.annotate('', xy=(a, yd), xytext=(c, yd), arrowprops=dict(arrowstyle='<->', lw=0.7, color='0.35'))
            aE.text((a + c) / 2, yd - 0.06, f'{ln:.2f}', ha='center', va='top', fontsize=8.5, color=oscuro)
    for c in res['cortes']:
        xc = min(c['x'] + 0.012 * L, L)
        aE.plot([xc, xc], [-0.02, 1.02], color='#C0392B', lw=1.1, ls='-.', zorder=7)
        aE.annotate(str(c['num']), xy=(xc, 0.0), xytext=(xc, -0.95), ha='center', va='top', fontsize=10, fontweight='bold', color='#C0392B',
                    arrowprops=dict(arrowstyle='->', lw=0.9, color='#C0392B'),
                    bbox=dict(boxstyle='circle,pad=0.2', fc='white', ec='#C0392B', lw=0.9), zorder=8)
    for t in res['estribos']:
        if t['s'] is None:
            continue
        aE.add_patch(Rectangle((t['x0'], -2.25), t['x1'] - t['x0'], 0.45, fc='#FFF4D6' if t['confinado'] else '#EAF2FB', ec=azul, lw=0.7))
        if t['x1'] - t['x0'] > 0.035 * L:
            aE.text((t['x0'] + t['x1']) / 2, -2.02, f'{t["s"]:g}', ha='center', va='center', fontsize=8, color=oscuro)
    aE.text(0, -1.65, f'Estribos E{est}, {res["ramas"]} ramas, separación en cm (amarillo = confinamiento 2h a cada lado del apoyo)',
            fontsize=8.5, color=oscuro, va='center')
    aE.set_xlim(-0.02 * L, 1.02 * L)
    aE.set_xlabel('x (m)')
    aE.set_xticks(xn)
    aE.set_xticklabels([f'{v:g}' for v in xn], fontsize=8)
    aV.set_title(f'Viga {res["b"]:g} × {res["h"]:g} cm · corridas {cor["n"]}{var} arriba y abajo con gancho · bastones y longitudes en m',
                 fontsize=11, color=oscuro)
    # ---- secciones
    axs = sf[1].subplots(1, n_s, squeeze=False)[0]
    for ax, c in zip(axs, res['cortes']):
        _dibujar_seccion(ax, res['b'], res['h'], res['rec'], res['est_db'], res['db'], c['n_top'], c['n_bot'], f'Secc. {c["num"]}-{c["num"]}')
        ax.text(res['b'] / 2, -6, f'arriba {c["n_top"]}{var}\nabajo {c["n_bot"]}{var}', ha='center', va='top', fontsize=8.5, color=oscuro)
    return fig


def figura(res):
    """Cuadrícula con la sección de cada punto (lecho en tensión)."""
    buenos = [e for e in res['puntos'] if e['ficha']['As_real']]
    return pt.figura_puntos([e['ficha'] for e in buenos], res['b'], res['h'], res['rec'], res['est_db'], [f'{e["punto"]})' for e in buenos])


def _lineas_tanteo(e):
    return [f'c = {e["c"]:.3f} cm → a = β1·c = {e["a"]:.3f} cm; C = a·b·f″c = {e["C"]:,.1f} kg',
            f'εs = 0.003(d − c)/c = {e["es"]:.5f} → {"el acero fluye" if e["fluye"] else "el acero NO fluye"}; fs = {e["fs"]:,.0f} kg/cm²',
            f'T = As·fs = {e["T"]:,.1f} kg → {"C > T: baja c" if e["C"] > e["T"] + 1 else "C < T: sube c" if e["C"] < e["T"] - 1 else "C = T"}',
            f'MR = FR·T·(d − a/2) = {e["MR"] / 1e5:.2f} t·m']


def tabla_flexion(res):
    """(encabezados, filas) de la flexión punto por punto."""
    enc = ['Punto', 'Lecho', 'Mu (t·m)', 'As req (cm²)', 'Armado', 'As real (cm²)', 'MR (t·m)', 'Sep. libre (cm)', 'OK']
    filas = []
    for e in res['puntos']:
        f = e['ficha']
        if f['As_real']:
            filas.append([e['punto'], e['lecho'], e['Mu'], f['As_usar'], f['armado'], f['As_real'], f['MR'], f['sep_libre'],
                          '✔' if f['ok'] else '✘'])
        else:
            filas.append([e['punto'], e['lecho'], e['Mu'], '—', '—', '—', '—', '—', '✘'])
    return enc, filas


def tabla_cortante(res):
    enc = ['Tramo', 'Vu (t)', 'As tens. (cm²)', 'ρ', 'VcR (kg)', 'Vu−VcR (kg)', 's calc (cm)', 'smáx (cm)', 'Estribos']
    filas = []
    for t in res['tramos']:
        c = t['cort']
        filas.append([t['tramo'], t['Vu'], t['As'], c['rho'], c['VcR'], c['Vs'], c['s_calc'] if c['s_calc'] is not None else '—',
                      c['s_max'], f'E{res["est"]}, {res["ramas"]}r @ {c["s_usar"]} cm' if c['ok'] else 'NO'])
    return enc, filas


def resumen(res):
    """[(etiqueta, valor, detalle)] de la viga completa."""
    cor, piel, integ = res['corrido'], res['piel'], res['integ']
    items = [('Varillas corridas', f'{cor["n"]}{res["var"]} arriba y abajo',
              f'As = {cor["As"]:.2f} cm² (máx(As,mín; 50 % de {cor["As_max_real"]:.2f}) = {cor["As_corr"]:.2f})')]
    if res.get('bastones'):
        items.append(('Bastones', ' · '.join(f'{b["punto"]}) +{b["n"]}{res["var"]}' for b in res['bastones']),
                      'longitudes en m (izquierda | derecha): ' + ' · '.join(f'{b["punto"]}) {b["izq"]:.2f} | {b["der"]:.2f}' for b in res['bastones'])))
    elif cor['bastones']:
        items.append(('Bastones', ' · '.join(f'{p}) +{n}{res["var"]}' for p, n in cor['bastones']), 'se suman a las corridas en cada punto'))
    ests = {}
    for t in res['tramos']:
        ests.setdefault(f'E{res["est"]} {res["ramas"]}r @ {t["cort"]["s_usar"]} cm' if t['cort']['ok'] else 'NO PASA', []).append(t['tramo'])
    items.append(('Estribos', ' · '.join(f'{k} (tramo {", ".join(v)})' for k, v in ests.items()), f'Vu máx = {res["Vu_max"]:g} t'))
    items.append(('Confinamiento', f'2h = {integ["Lconf"]:g} cm', f'estribos @ {integ["s_usar"]} cm en cada extremo'))
    items.append(('Acero de piel', f'{piel["n_cara"]}{piel["var"]} por cara @ {piel["sep"]:.1f} cm' if piel['aplica'] else 'No se requiere',
                  f'As colocado {piel["As_col"]:.2f} cm²' if piel['aplica'] else 'h ≤ 60 cm'))
    return items


def bloques(res, con_figura=True):
    """Bloques de ame_detalle para el PDF, el Excel y el texto."""
    sub = (f'b × h = {res["b"]:g} × {res["h"]:g} cm · r = {res["r"]:g} cm · f′c = {res["fc"]:g} · fy = {res["fy"]:g} kg/cm² · '
           f'{len(res["puntos"])} punto(s) de flexión · {len(res.get("tramos") or [])} tramo(s) de cortante')
    B = [det.titulo('Viga de concreto reforzado', sub)]
    if res['error']:
        return B + [det.paso(1, 'Revisión de la sección', res['error'], 'Cambia las dimensiones o el concreto y vuelve a calcular.')]
    B.append(det.destacado('Armado de toda la viga', resumen(res)))
    malos = [t for t, ok in (('flexión', res['ok_flex']), ('cortante', res['ok_cort'])) if not ok]
    B.append(det.nota('La viga cumple: subreforzada en todos los puntos, el cortante no excede el límite y el armado alcanza.' if res['ok']
                      else 'La viga NO cumple una de las revisiones: ' + ', '.join(malos) + '.'))
    if con_figura and res.get('geom'):
        B.append(det.figura('Alzado de la viga: diagramas, armado y secciones', figura_hoja(res), 17))
    B.append(det.paso(1, 'Datos y revisiones de la viga', 'Materiales, peralte, límites de acero y cortante máximo.',
                      f'MR con As,máx contra el Mu mayor ({res["Mu_max"]:g} t·m) y V límite contra el Vu mayor ({res["Vu_max"]:g} t).'))
    B += [det.texto(_limpio(t)) for t in res['viga']['lineas']]
    B.append(det.paso(2, 'Flexión en cada punto', 'Cuadrática de equilibrio con FR = 0.9, As de diseño y número de varillas.',
                      'MR con el área real del armado y separación libre entre barras.'))
    enc, filas = tabla_flexion(res)
    B.append(det.tabla('Resumen de flexión', enc, filas, 2))
    for e in res['puntos']:
        B.append(det.texto(f'Punto {e["punto"]}) · Mu = {e["Mu"]:g} t·m · tensión {e["lecho"]}'))
        B += [det.texto(_limpio(t)) for t in (e['ficha']['lineas'] if not e['error'] else [e['error']])]
        if e['alternativas']:
            B.append(det.tabla('Otros armados posibles', ['Armado', 'As (cm²)', 'Exceso (cm²)', 'MR (t·m)'],
                               [[a['armado'], a['As'], a['exceso'], a['MR']] for a in e['alternativas']], 2))
    if con_figura:
        B.append(det.figura('Secciones con el armado a tensión', figura(res), 17))
    B.append(det.paso(3, 'Tanteo del eje neutro', 'Con el As real de cada punto se busca c hasta que C = T.',
                      'Se revisa que el acero fluya (subreforzada) y se obtiene MR.'))
    filas = [[e['punto'], e['ficha']['As_real'], e['tanteo']['c'], e['tanteo']['a'], e['tanteo']['es'], 'sí' if e['tanteo']['fluye'] else 'NO',
              e['tanteo']['MR'] / 1e5] for e in res['puntos'] if e['tanteo']]
    B.append(det.tabla('Equilibrio C = T', ['Punto', 'As real (cm²)', 'c (cm)', 'a (cm)', 'εs', 'Fluye', 'MR (t·m)'], filas, 4))
    for e in res['puntos']:
        if e['tanteo_manual']:
            B.append(det.texto(f'Punto {e["punto"]}) con el tanteo c = {e["c_prop"]:g} cm:'))
            B += [det.texto(t) for t in _lineas_tanteo(e['tanteo_manual'])]
    cor = res['corrido']
    B.append(det.paso(4, 'Varillas corridas y bastones', 'Corrido a todo lo largo, arriba y abajo, con ganchos al inicio y al final.',
                      'Los bastones completan el As real de cada punto y se anclan una distancia d más allá de donde ya no se requieren.'))
    B.append(det.texto(f'Mayor As real = {cor["As_max_real"]:.2f} cm²; 50 % = {0.5 * cor["As_max_real"]:.2f} cm²; As,mín = {res["lim"]["As_min"]:.2f} cm²'))
    B.append(det.texto(f'Corrido = máx(As,mín; 50 % As mayor) = {cor["As_corr"]:.2f} cm² → n = {cor["As_corr"]:.2f}/{res["ab"]} = '
                       f'{cor["As_corr"] / res["ab"]:.2f} → {cor["n"]}{res["var"]} ({cor["As"]:.2f} cm²) arriba y abajo'))
    if res.get('bastones'):
        B.append(det.tabla('Bastones: del punto de inflexión del momento más d a cada lado (m)',
                           ['Punto', 'Lecho', 'Varillas', 'Izquierda (m)', 'Derecha (m)', 'Total (m)'],
                           [[b['punto'], b['lecho'], f'{b["n"]}{res["var"]}', b['izq'], b['der'], b['izq'] + b['der']] for b in res['bastones']], 2))
    else:
        B.append(det.texto('Bastones por punto: ' + (' · '.join(f'{p}) +{n}{res["var"]}' for p, n in cor['bastones']) if cor['bastones']
                                                      else 'ninguno, las corridas alcanzan en todos los puntos')))
    B.append(det.paso(5, 'Cortante: estribos por tramo', f'Estribo E{res["est"]} de {res["ramas"]} ramas (Av = {res["Av"]:.2f} cm²), '
                      f'fy = {res["fy_est"]:g} kg/cm², tope de separación {res["s_tope"]:g} cm.',
                      f'ρ con el As en tensión de cada tramo (si no lo das: el mayor As real superior, {res["As_auto"]:.2f} cm²).'))
    lc = p3.limites_cortante(res['b'], res['d_ef'], res['fc'])
    B.append(det.tabla('Límites (kg)', ['1.1·FR·b·d·√f′c', '1.25·FR·b·d·√f′c', '2.2·FR·b·d·√f′c', '0.25·FR·b·d·√f′c', '2.65·FR·b·d·√f′c'],
                       [[lc['L11'], lc['L125'], lc['L22'], lc['Vmin'], lc['Vmax']]], 0))
    enc, filas = tabla_cortante(res)
    B.append(det.tabla('Resumen de cortante', enc, filas, 4))
    for t in res['tramos']:
        B.append(det.texto(f'Tramo {t["tramo"]} · Vu = {t["Vu"]:g} t'))
        B += [det.texto(_limpio(x)) for x in t['cort']['lineas']]
        if not t['cort']['ok']:
            B.append(det.nota('Vu excede el límite de la sección: agranda b o h.'))
        if t['alt']:
            B.append(det.nota(f'@ {t["cort"]["s_usar"]} cm es muy cerrado para colar. Alternativas con s ≥ 10 cm: '
                              + ' · '.join(o['texto'] for o in t['alt'][:4])))
        if t['rev']:
            rv = t['rev']
            B.append(det.texto(f'Revisión de s = {t["s_prop"]:g} cm: VsR = {rv["VsR"] / 1000:.2f} t; VR = VcR + VsR = {rv["VR"] / 1000:.2f} t '
                               f'{"≥" if rv["ok_resistencia"] else "<"} Vu = {t["Vu"]:.2f} t; s {"≤" if rv["ok_separacion"] else ">"} '
                               f'smáx = {rv["s_max"]:.2f} cm → {"PASA" if rv["ok"] else "NO PASA"}'))
    B.append(det.paso(6, 'Zona de confinamiento y acero de piel', 'Integridad estructural: 2h desde la cara de cada apoyo.',
                      'Acero por temperatura en las caras laterales si h > 60 cm.'))
    B += [det.texto(t) for t in res['integ']['lineas']]
    B += [det.texto(t) for t in res['piel']['lineas']]
    return B


def pdf(res, proyecto='', autor=''):
    """PDF de la viga (bytes)."""
    import datetime

    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import cm
    from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle

    import ame_pdf as ap
    ap._registrar_fuentes()
    st = ap._estilos()
    Ancho = letter[0] - 3.6 * cm
    buf = io.BytesIO()

    def pie(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(ap.AZUL_CLARO)
        canvas.line(1.8 * cm, 1.5 * cm, letter[0] - 1.8 * cm, 1.5 * cm)
        canvas.setFont('DejaVu', 7.5)
        canvas.setFillColor(ap.AZUL_MED)
        canvas.drawString(1.8 * cm, 1.05 * cm, 'Viga de concreto reforzado' + (f' · {proyecto}' if proyecto else ''))
        canvas.drawRightString(letter[0] - 1.8 * cm, 1.05 * cm, f'Página {doc.page}')
        canvas.restoreState()

    doc = BaseDocTemplate(buf, pagesize=letter, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=2 * cm,
                          title='Viga de concreto reforzado', author=autor or 'AME')
    doc.addPageTemplates([PageTemplate(id='p', frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='f')],
                                       onPage=pie)])
    bl = bloques(res)
    banda = Table([[Paragraph('<font color="white" size="19"><b>Viga de concreto reforzado</b></font><br/>'
                              f'<font color="#81A4CD" size="9">{ap._escapar(bl[0]["sub"])}</font>', st['base']),
                    Paragraph(f'<font color="#81A4CD" size="8.5">Proyecto: {ap._escapar(proyecto or "—")}<br/>'
                              f'Elaboró: {ap._escapar(autor or "—")}<br/>Fecha: {datetime.date.today().strftime("%d/%m/%Y")}</font>',
                              st['base'])]], colWidths=[Ancho * 0.7, Ancho * 0.3])
    banda.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), ap.NAVY), ('LEFTPADDING', (0, 0), (-1, -1), 14),
                               ('TOPPADDING', (0, 0), (-1, -1), 12), ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                               ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
    E = [banda, Spacer(1, 8)] + ap._bloques(bl[1:], st, Ancho)
    try:
        doc.build(E)
    finally:
        for x in bl:
            if x['t'] == 'figura':
                import matplotlib.pyplot as plt
                plt.close(x['fig'])
    return buf.getvalue()


def excel(res):
    """Libro de Excel con el mismo procedimiento (bytes)."""
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = 'Viga'
    det.bloques_a_hoja(ws, bloques(res, con_figura=False))
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def texto(res):
    """Procedimiento en texto plano (UTF-8)."""
    return det.bloques_a_texto(bloques(res, con_figura=False))
