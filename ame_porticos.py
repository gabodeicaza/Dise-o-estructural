"""Pórtico plano de concreto: análisis de CM, CV y sismo (ida y regreso), combinaciones, envolventes de M y V por viga,
armado a flexión por lechos, estribos (apuntes del Dr. Rivero, tercer parcial) y figuras tipo ETABS.

El pórtico se describe con una «geometría» (geo) que se arma de dos maneras:
  · geo_simple: pórtico regular de captura rápida (mismas columnas y vigas en cada piso);
  · geo_modelo: el MISMO modelo de la pestaña «1 · Estructura» (columnas por eje, secciones distintas por nivel,
    tramos sin losa, columnas que cruzan pisos sin viga, apoyos articulados y volados).
Unidades de entrada: m, cm, kg/cm², t/m, t. Cada nodo (piso, eje) tiene uz y θ propios; ux es la misma en todos los nodos
de un piso (diafragma rígido), por lo que la fuerza sísmica de cada nivel se aplica a ese grado de libertad.
Convención de momentos: positivo = tensión abajo en vigas (el diagrama se dibuja del lado de la tensión, como ETABS).
"""
import math

import numpy as np

import ame_marco as marco
import ame_parcial3 as p3

CASOS = ('CM', 'CV', 'CVacc', 'S')
DIAMETROS = {'#3': 0.95, '#4': 1.27, '#5': 1.59, '#6': 1.91, '#8': 2.54, '#10': 3.18}     # cm
NUM_VARILLA = {'#3': 3, '#4': 4, '#5': 5, '#6': 6, '#8': 8, '#10': 10}


def combinaciones(grupo='B', estabilidad=False, FC_acc=1.1, gravedad=True):
    """Combinaciones de diseño como coeficientes sobre (CM, CV, CVacc, S): gravitacional, sismo de ida y de regreso
    (apuntes p. 2: FC = 1.1 con CM + CVacc + sismo; 0.9 en estabilidad)."""
    fcm, fcv = (1.5, 1.7) if grupo == 'A' else (1.3, 1.5)
    c = {'Gravedad': (fcm, fcv, 0.0, 0.0)} if gravedad else {}
    c.update({'Sismo ida (+)': (FC_acc, 0.0, FC_acc, FC_acc),
              'Sismo regreso (−)': (FC_acc, 0.0, FC_acc, -FC_acc)})
    if estabilidad:
        c['Estabilidad ida (+)'] = (0.9, 0.0, 0.0, FC_acc)
        c['Estabilidad regreso (−)'] = (0.9, 0.0, 0.0, -FC_acc)
    return c


def varilla_sugerida(h):
    """Regla del profesor: el número de la varilla ≈ h(cm)/10; si ese número no existe (no hay #7) se toma la siguiente
    disponible hacia arriba (h = 70 → #8). Devuelve (varilla, texto)."""
    ref = h / 10.0
    disp = [v for v, n in NUM_VARILLA.items() if n >= ref - 1e-9]
    elegida = disp[0] if disp else list(NUM_VARILLA)[-1]
    if f'#{ref:g}' in NUM_VARILLA:
        txt = f'h/10 = {ref:g} → varilla #{ref:g}.'
    else:
        txt = f'h/10 = {ref:g} → no existe la #{ref:g}: se toma la siguiente disponible, {elegida}.'
    return elegida, txt


# ================================================================ geometrías
def _tramos_nivel(vi, vd, nb):
    """Etiquetas de los tramos con carga de un nivel: volado izq. (si hay) + crujías + volado der. (si hay)."""
    return (['vi'] if vi > 0 else []) + list(range(1, nb + 1)) + (['vd'] if vd > 0 else [])


def _etiquetas(nv, ne):
    return nv['etiquetas'] if 'etiquetas' in nv else _tramos_nivel(nv['vi'], nv['vd'], ne - 1)


def geo_simple(spans, alturas, col_secc, vig_secc, E, cargas, base='Empotrada', fi_col=1.0, fi_vig=1.0, pp=0.0):
    """Pórtico regular: col_secc/vig_secc = [(b, h)] por nivel (h en el plano del pórtico); cargas = dict de listas por
    piso (valor único por piso o lista por crujía)."""
    xs = np.concatenate([[0.0], np.cumsum(spans)])
    ne, n = len(xs), len(alturas)
    niveles = []
    for k in range(n):
        cb, ch = col_secc[k]
        vb, vh = vig_secc[k]
        cols = [dict(eje=e, A=cb * ch, I=fi_col * cb * ch ** 3 / 12, E=E, artic=(k == 0 and base != 'Empotrada'),
                     artic_sup=False, rodillo=False, b=cb, h=ch) for e in range(1, ne + 1)]
        q = {}
        for caso in ('CM', 'CV', 'CVacc'):
            v = cargas.get(caso, [0.0] * n)[k]
            q[caso] = list(v) if isinstance(v, (list, tuple, np.ndarray)) else [float(v)] * (ne - 1)
            if caso == 'CM' and pp:
                q[caso] = [x + pp * vb * vh / 1e4 for x in q[caso]]
        niveles.append(dict(cols=cols, viga=dict(A=vb * vh, I=fi_vig * vb * vh ** 3 / 12, E=E, b=vb, h=vh), sin=set(),
                            vi=0.0, vd=0.0, q=q))
    return dict(xs=xs, alturas=list(alturas), niveles=niveles, E=E)


def geo_modelo(modelo, datos, cargas_txt, pp=0.0):
    """Geometría del modelo de la pestaña 1 (vigas flexibles). cargas_txt[k] = dict(CM, CV, CVacc) con texto «un valor
    para todos los tramos o uno por tramo» (volado izq. + crujías + volado der.), t/m."""
    import ame_sismico as ame
    if modelo.get('vigas') != 'flexibles':
        raise ValueError('Para usar el modelo de la pestaña 1 elige «Flexibles (marco plano)» en el tipo de vigas y captura la '
                         'viga de cada piso.')
    g = modelo['geom']
    xs = np.asarray(g['xs'], float)
    ne = len(xs)
    nb = ne - 1
    vi0 = ame._num_opc(datos, 'volado_izq', None) or 0.0
    vd0 = ame._num_opc(datos, 'volado_der', None) or 0.0
    niveles = []
    for k, (nv, r) in enumerate(zip(g['niveles'], datos['niveles'])):
        v_i, v_d = ame._num_opc(r, 'volado_izq', None), ame._num_opc(r, 'volado_der', None)
        vi, vd = (vi0 if v_i is None else v_i), (vd0 if v_d is None else v_d)
        etiquetas = _tramos_nivel(vi, vd, nb)
        q = {}
        for caso in ('CM', 'CV', 'CVacc'):
            txt = str(cargas_txt[k].get(caso, '') or '').strip()
            v = ame.parse_vector(txt, f'Nivel {k + 1} - {caso}') if txt else np.zeros(1)
            if len(v) == 1:
                v = np.repeat(v, len(etiquetas))
            elif len(v) != len(etiquetas):
                raise ValueError(f'Nivel {k + 1} - {caso}: hay {len(etiquetas)} tramo(s) con carga (volado izq. + crujías + volado '
                                 f'der.) y escribiste {len(v)}.')
            q[caso] = list(map(float, v))
        vg = nv['viga']
        h_v = float(vg['alto'])
        if pp:                              # peso propio de la viga (t/m); un volado sin carga no tiene viga
            for i_, et in enumerate(etiquetas):
                if et not in ('vi', 'vd') or any(q[c][i_] for c in q):
                    q['CM'][i_] += pp * vg['A'] / 1e4
        cols = []
        for c in nv['cols']:
            eq = float(c['eq'])
            cols.append(dict(eje=c['eje'], A=c['A'], I=c['I'], E=c['E'], artic=c['artic'], artic_sup=c['artic_sup'],
                             rodillo=c['rodillo'], hasta=c.get('hasta', k + 1), omitir=bool(c.get('k_cont')),
                             b=c['A'] / eq if eq else 0.0, h=eq))
        niveles.append(dict(cols=cols, viga=dict(A=vg['A'], I=vg['I'], E=vg['E'], b=vg['A'] / h_v, h=h_v),
                            sin=set(nv['sin']), vi=float(vi), vd=float(vd), q=q, etiquetas=etiquetas))
    return dict(xs=xs, alturas=list(map(float, modelo['alturas'])), niveles=niveles, E=modelo['E'])


# ================================================================ análisis
def _condensa(k, i):
    return k - np.outer(k[:, i], k[i, :]) / k[i, i]


def resolver_geo(geo, F_sismo):
    """Resuelve CM, CV, CVacc y sismo. F_sismo: fuerza lateral de cada nivel (t), de abajo hacia arriba."""
    xs_m, alturas, niveles = geo['xs'], geo['alturas'], geo['niveles']
    n, ne = len(alturas), len(xs_m)
    xs = xs_m * 100.0
    zs = np.concatenate([[0.0], np.cumsum(alturas)]) * 100.0
    idx, cont = {}, n
    for k in range(1, n + 1):
        for e in range(0, ne + 2):
            if e == 0 and niveles[k - 1]['vi'] <= 0:
                continue
            if e == ne + 1 and niveles[k - 1]['vd'] <= 0:
                continue
            idx[(k, e)] = (k - 1, cont, cont + 1)
            cont += 2
    for c in niveles[0]['cols']:
        e = c['eje']
        if c.get('rodillo'):
            idx[(0, e)] = (cont, -1, cont + 1)
            cont += 2
        else:
            idx[(0, e)] = (-1, -1, -1)
    nd = cont
    K = np.zeros((nd, nd))
    elems = []            # tipo, clave, sec, L, T, dofs, lib_ini, lib_fin, x0 (cm)
    for i, nv in enumerate(niveles):
        for c in nv['cols']:
            if c.get('omitir'):
                continue
            top = c.get('hasta', i + 1)
            e = c['eje']
            L = zs[top] - zs[i]
            elems.append(('c', ('c', i + 1, e), c, L, marco._rot(0.0, 1.0), list(idx[(i, e)]) + list(idx[(top, e)]),
                          bool(c['artic']), bool(c['artic_sup']), zs[i]))
    for k in range(1, n + 1):
        nv = niveles[k - 1]
        v = nv['viga']
        tramos = _etiquetas(nv, ne)
        for j in range(1, ne):
            if (j - 1) in nv['sin']:
                continue
            L = xs[j] - xs[j - 1]
            elems.append(('b', ('b', k, j), v, L, marco._rot(1.0, 0.0), list(idx[(k, j)]) + list(idx[(k, j + 1)]),
                          False, False, xs[j - 1]))
        for lado in ('vi', 'vd'):
            vol = nv[lado]
            if vol <= 0 or all(nv['q'][c][tramos.index(lado)] == 0 for c in ('CM', 'CV', 'CVacc')):
                continue
            L = vol * 100.0
            if lado == 'vi':
                elems.append(('b', ('b', k, 0), v, L, marco._rot(1.0, 0.0), list(idx[(k, 0)]) + list(idx[(k, 1)]), False, False,
                              xs[0] - L))
            else:
                elems.append(('b', ('b', k, ne), v, L, marco._rot(1.0, 0.0), list(idx[(k, ne)]) + list(idx[(k, ne + 1)]),
                              False, False, xs[-1]))

    def klocal(sec, L, li, lf):
        kk = marco._k_barra(sec['E'], sec['A'], sec['I'], L)
        if li:
            kk = _condensa(kk, 2)
        if lf:
            kk = _condensa(kk, 5)
        return kk
    for tipo, clave, sec, L, T, dofs, li, lf, x0 in elems:
        kg = T.T @ klocal(sec, L, li, lf) @ T
        for a, ga in enumerate(dofs):
            if ga < 0:
                continue
            for b_, gb in enumerate(dofs):
                if gb >= 0:
                    K[ga, gb] += kg[a, b_]
    usado = np.abs(np.diag(K)) > 0
    usado[:n] = True
    sel = np.where(usado)[0]
    Ks = K[np.ix_(sel, sel)]
    try:
        if np.linalg.cond(Ks) > 1e14:
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        raise ValueError('El marco es inestable (revisa apoyos, articulaciones y vigas).')

    def q_de(caso, clave):
        if caso == 'S':
            return 0.0
        nv = niveles[clave[1] - 1]
        j = clave[2]
        lab = 'vi' if j == 0 else ('vd' if j == ne else j)
        return float(nv['q'][caso][_etiquetas(nv, ne).index(lab)]) * 10.0

    salida = dict(spans=list(np.diff(xs_m)), alturas=list(alturas), xs=xs_m, zs=zs / 100, n=n, ne=ne, caso={}, geo=geo)
    for caso in CASOS:
        P = np.zeros(nd)
        if caso == 'S':
            for k in range(1, n + 1):
                P[k - 1] += float(F_sismo[k - 1]) * 1000.0
        f0 = {}
        for tipo, clave, sec, L, T, dofs, li, lf, x0 in elems:
            if tipo != 'b':
                continue
            q = q_de(caso, clave)
            if q == 0:
                continue
            loc = np.array([0, q * L / 2, q * L ** 2 / 12, 0, q * L / 2, -q * L ** 2 / 12])
            f0[clave] = loc
            eq = -(T.T @ loc)
            for a, ga in enumerate(dofs):
                if ga >= 0:
                    P[ga] += eq[a]
        u = np.zeros(nd)
        u[sel] = np.linalg.solve(Ks, P[sel])
        diag = {}
        for tipo, clave, sec, L, T, dofs, li, lf, x0 in elems:
            ue = np.array([u[g] if g >= 0 else 0.0 for g in dofs])
            fl = klocal(sec, L, li, lf) @ (T @ ue)
            if tipo == 'b':
                fl = fl + f0.get(clave, np.zeros(6))
                q = q_de(caso, clave)
                x = np.linspace(0, L, 41)
                M = -fl[2] + fl[1] * x - q * x ** 2 / 2
                V = fl[1] - q * x
                diag[clave] = dict(x=x / 100, p=(x0 + x) / 100, M=M / 1e5, V=V / 1000, q=q / 10, b=sec.get('b'), h=sec.get('h'))
            else:
                z = np.linspace(0, L, 11)
                M = -fl[2] + fl[1] * z
                V = np.full_like(z, fl[1])
                diag[clave] = dict(x=z / 100, p=(x0 + z) / 100, M=M / 1e5, V=V / 1000, N=-fl[0] / 1000)
        desp = {kk: tuple(u[x] if x >= 0 else 0.0 for x in g) for kk, g in idx.items()}
        salida['caso'][caso] = dict(diag=diag, desp=desp)
    return salida


def resolver(spans, alturas, col_secc, vig_secc, E, cargas, F_sismo, base='Empotrada', fi_col=1.0, fi_vig=1.0, pp=0.0):
    """Pórtico regular de captura rápida (ver geo_simple)."""
    return resolver_geo(geo_simple(spans, alturas, col_secc, vig_secc, E, cargas, base, fi_col, fi_vig, pp), F_sismo)


def combinar(res, coef):
    """Diagramas de una combinación (suma lineal de los casos). coef: (cCM, cCV, cCVacc, cS)."""
    out = {}
    for clave, d0 in res['caso']['CM']['diag'].items():
        M = sum(c * res['caso'][caso]['diag'][clave]['M'] for c, caso in zip(coef, CASOS))
        V = sum(c * res['caso'][caso]['diag'][clave]['V'] for c, caso in zip(coef, CASOS))
        out[clave] = dict(x=d0['x'], p=d0['p'], M=M, V=V)
    return out


def envolvente(res, combos):
    """Envolvente máxima y mínima de M y V (todas las combinaciones) por elemento."""
    mats = {nombre: combinar(res, c) for nombre, c in combos.items()}
    env = {}
    for clave, d0 in res['caso']['CM']['diag'].items():
        Ms = np.array([m[clave]['M'] for m in mats.values()])
        Vs = np.array([m[clave]['V'] for m in mats.values()])
        env[clave] = dict(x=d0['x'], p=d0['p'], Mmax=Ms.max(0), Mmin=Ms.min(0), Vmax=Vs.max(0), Vmin=Vs.min(0),
                          nombres=list(mats), b=d0.get('b'), h=d0.get('h'))
    return env, mats


def desplazamientos(res, caso='S'):
    """Desplazamiento lateral (cm) y distorsión por entrepiso del caso."""
    n = res['n']
    d = res['caso'][caso]['desp']
    ux = np.array([d[next(k for k in d if k[0] == i)][0] for i in range(1, n + 1)])
    dif = np.diff(np.concatenate([[0.0], ux]))
    return ux, dif / (np.array(res['alturas']) * 100)


# ================================================================ armado por lechos
def _capa(m, db, util, x0, sep_min, nmax):
    """Posiciones (cm, desde la cara izquierda de la sección) de m barras en un lecho. Si caben sueltas se reparten parejo;
    si no, se agrupan de 2 en 2 (pareja de barras pegadas, la impar al centro) y se reparten los grupos. Devuelve
    (xs, tamaños de grupo, separación libre mínima entre grupos)."""
    if m <= nmax:
        sizes = [1] * m
    else:
        pares = m // 2
        sizes = [2] * (pares // 2) + ([1] if m % 2 else []) + [2] * (pares - pares // 2)
    ancho = sum(sizes) * db
    gap = (util - ancho) / (len(sizes) - 1) if len(sizes) > 1 else util - ancho
    x, xs = x0 - db / 2 + (0 if len(sizes) > 1 else gap / 2), []
    for s in sizes:
        xs += [x + db / 2 + i * db for i in range(s)]
        x += s * db + (gap if len(sizes) > 1 else 0)
    return xs, sizes, gap


def distribuir_lechos(n, db, b, rec, est_db, agrupar=True):
    """Reparte n barras de diámetro db (cm) en lechos dentro del ancho b. Separación libre mínima = máx(db, 2.5 cm) entre
    barras (o entre grupos) y entre lechos. Con agrupar=True, cuando no caben sueltas en un lecho se ponen de 2 en 2 pegadas
    antes de pasar a un segundo lecho. Devuelve capas (barras por lecho), xs (posición de cada barra por lecho), r al
    centroide (cm), separación libre mínima, y si cabe en 3 lechos como máximo."""
    sep_min = max(db, 2.5)
    util = b - 2 * (rec + est_db)
    nmax = max(1, int((util + sep_min) // (db + sep_min)))
    gmax = max(1, int((util + sep_min) // (2 * db + sep_min)))
    cap = max(nmax, 2 * gmax) if agrupar else nmax
    x0 = rec + est_db + db / 2
    x1 = b - rec - est_db - db / 2
    capas, xs, grupos, gaps, resto = [], [], [], [], n
    while resto > 0:
        m = min(cap, resto)
        resto -= m
        px, sizes, gap = _capa(m, db, x1 - x0 + db, x0, sep_min, nmax)
        capas.append(m)
        xs.append(px)
        grupos.append(sizes)
        gaps.append(gap if len(sizes) > 1 else util)
    ys = [rec + est_db + db / 2 + i * (db + sep_min) for i in range(len(capas))]
    r = sum(c * y for c, y in zip(capas, ys)) / n
    return dict(capas=capas, xs=xs, grupos=grupos, r=r, sep_libre=min(gaps), sep_min=sep_min, nmax=nmax, cap=cap,
                agrupado=any(s == 2 for g in grupos for s in g), cabe=len(capas) <= 3, ys=ys)


def desarrollo_flexion(Mu_tm, b, h, r, fc, fy, signo=''):
    """Cálculo de As desarrollado para un momento último; devuelve (As_diseño, lineas, ok, límites)."""
    d = h - r
    lim = p3.limites_flexion(b, d, fc, fy)
    fcc = lim['fcc']
    lin = [f'Mu{signo} = {Mu_tm:.3f} t·m = {Mu_tm * 1e5:,.0f} kg·cm; d = h − r = {h:g} − {r:.2f} = {d:.2f} cm',
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


def armar_seccion(Mu_tm, b, h, fc, fy, varilla, rec=4.0, est_db=0.95, signo='', agrupar=True):
    """Dimensiona una sección con la varilla elegida: calcula As con la d real del acero en lechos (se itera) y propone el
    número de barras. Devuelve dict con As, lineas, armado (n, varilla, capas, As real, MR, cabe, r, d)."""
    db, a_bar = DIAMETROS[varilla], p3.VARILLAS[varilla]
    r = rec + est_db + db / 2
    for _ in range(8):
        As, lin, ok, lim = desarrollo_flexion(Mu_tm, b, h, r, fc, fy, signo)
        if As is None:
            return dict(As=None, lineas=lin, ok=False, lim=lim, armado=None, Mu=Mu_tm)
        n = max(2, math.ceil(As / a_bar - 1e-9))
        lech = distribuir_lechos(n, db, b, rec, est_db, agrupar)
        if abs(lech['r'] - r) < 0.02:
            break
        r = lech['r']
    n_ini = n
    for _ in range(30):                           # sube barras mientras el MR con la d real no alcance
        lech = distribuir_lechos(n, db, b, rec, est_db, agrupar)
        As_real = n * a_bar
        MR = p3.momento_resistente(As_real, b, h - lech['r'], fc, fy) / 1e5
        if MR + 1e-9 >= Mu_tm or not lech['cabe']:
            break
        n += 1
    lim2 = p3.limites_flexion(b, h - lech['r'], fc, fy)
    ok_f = ok and As_real <= lim2['As_max'] and As_real >= lim2['As_min'] and lech['cabe'] and MR + 1e-9 >= Mu_tm
    lin = list(lin)
    lin.append(f'Varilla {varilla} (Ø {db:.2f} cm, a = {a_bar} cm²): n = As/a = {As / a_bar:.2f} → {n_ini} barras'
               + (f'; con la d real suben a {n}' if n != n_ini else ''))
    forma = ('agrupadas de 2 en 2 (pegadas) ' if lech['agrupado'] else '')
    lin.append(f'Lechos: {" + ".join(map(str, lech["capas"]))} barras {forma}(sueltas caben {lech["nmax"]} por lecho; separación libre '
               f'mínima {lech["sep_min"]:.2f} cm; entre barras/grupos queda {lech["sep_libre"]:.2f} cm) → r al centroide = '
               f'{lech["r"]:.2f} cm, d = {h - lech["r"]:.2f} cm')
    lin.append(f'As real = {n}{varilla} = {As_real:.2f} cm²; MR = {MR:.2f} t·m {"≥" if MR + 1e-9 >= Mu_tm else "<"} Mu = {Mu_tm:.2f} t·m')
    if not lech['cabe']:
        lin.append('No caben en 3 lechos: aumenta el ancho, el peralte o usa una varilla mayor.')
    if As_real > lim2['As_max']:
        lin.append(f'As real > As,máx = {lim2["As_max"]:.2f} cm²: cambia la sección.')
    return dict(As=As, lineas=lin, ok=ok_f, lim=lim2, Mu=Mu_tm,
                armado=dict(armado=f'{n}{varilla}', n=n, varilla=varilla, capas=lech['capas'], As=As_real, MR=MR, cabe=lech['cabe'],
                            sep_libre=lech['sep_libre'], sep_min=lech['sep_min'], r=lech['r'], d=h - lech['r'], lech=lech))


def disenar_viga(env, clave, b, h, fc, fy, cara_i, cara_d, sup_var, inf_var, est_var='#3', rec=4.0, fy_est=4200.0,
                 lambda_=1.0, agrupar=True):
    """Diseña una viga con su envolvente: seis secciones (izq., centro, der. × superior, inferior) con las varillas elegidas
    y los estribos. cara_i/cara_d = distancia del eje al paño de la columna (cm; 0 en un extremo libre)."""
    e = env[clave]
    x, Mmax, Mmin = e['x'] * 100, e['Mmax'], e['Mmin']
    L = x[-1]
    est_db = DIAMETROS[est_var]
    fi = int(np.argmin(np.abs(x - cara_i)))
    fd = int(np.argmin(np.abs(x - (L - cara_d))))
    if fd <= fi:
        fi, fd = 0, len(x) - 1
    zona = slice(fi, fd + 1)
    medio = np.where((x >= 0.25 * L) & (x <= 0.75 * L))[0]
    ic = int(medio[np.argmax(Mmax[medio])])
    pos = {'izquierdo': fi, 'centro': ic, 'derecho': fd}
    secciones = []
    for nombre, i in pos.items():
        for lecho, Mu in (('superior', max(0.0, -Mmin[i])), ('inferior', max(0.0, Mmax[i]))):
            var = sup_var if lecho == 'superior' else inf_var
            s = armar_seccion(Mu, b, h, fc, fy, var, rec, est_db, '−' if lecho == 'superior' else '+', agrupar)
            s.update(posicion=nombre, lecho=lecho, x=x[i] / 100)
            secciones.append(s)

    def get(pn, lc):
        return next(s for s in secciones if s['posicion'] == pn and s['lecho'] == lc)
    Vmax = float(max(np.max(np.abs(e['Vmax'][zona])), np.max(np.abs(e['Vmin'][zona]))))
    ramas = 4 if b >= 40 else 2
    Av = p3.VARILLAS[est_var] * ramas
    As_trac = max((get(pn, 'superior')['armado']['As'] if get(pn, 'superior')['armado'] else 0.0) for pn in ('izquierdo', 'derecho'))
    cort = p3.cortante_viga(Vmax * 1000, b, h - 5.0, h, L, fc, fy_est, Av, As_trac or 1.0, lambda_)
    cort['ramas'], cort['estribo'] = ramas, est_var
    return dict(clave=clave, L=L / 100, b=b, h=h, secciones=secciones, Vu=Vmax, cortante=cort, fc=fc, fy=fy, sup_var=sup_var,
                inf_var=inf_var, est_var=est_var, rec=rec, cara_i=cara_i, cara_d=cara_d,
                ok=all(s['ok'] for s in secciones) and cort['ok'],
                inferior_corrido=max((get(pn, 'inferior') for pn in pos), key=lambda s: s['armado']['As'] if s['armado'] else 0))


def caras_columna(geo, clave):
    """Distancia (cm) del eje al paño de las columnas de los dos extremos de la viga (0 en un volado libre)."""
    _, k, j = clave
    ne = len(geo['xs'])

    def hcol(eje):
        h = 0.0
        for nivel in (k - 1, k):                 # columna de abajo y de arriba del piso k
            if nivel < len(geo['niveles']):
                for c in geo['niveles'][nivel]['cols']:
                    if c['eje'] == eje:
                        h = max(h, c['h'])
        return h / 2
    if j == 0:
        return 0.0, hcol(1)
    if j == ne:
        return hcol(ne), 0.0
    return hcol(j), hcol(j + 1)


def resumen_vigas(env, res, fc, fy, elecciones, est_var='#3', rec=4.0, agrupar=True, paño=True):
    """Diseña todas las vigas. elecciones[clave] = (varilla superior, varilla inferior); si falta, la sugerida por h/10."""
    out = {}
    geo = res['geo']
    for clave, e in env.items():
        if clave[0] != 'b':
            continue
        b, h = e['b'], e['h']
        sup, inf = elecciones.get(clave, (varilla_sugerida(h)[0],) * 2)
        ci, cd = caras_columna(geo, clave) if paño else (0.0, 0.0)
        out[clave] = disenar_viga(env, clave, b, h, fc, fy, ci, cd, sup, inf, est_var, rec, agrupar=agrupar)
    return out


def etiqueta_viga(clave, ne):
    _, k, j = clave
    return f'Piso {k} · ' + ('volado izquierdo' if j == 0 else ('volado derecho' if j == ne else f'crujía {j}'))


# ================================================================ figuras
def _plt():
    import matplotlib
    matplotlib.use('Agg', force=False)
    import matplotlib.pyplot as plt
    return plt


def _tam(res, ancho=None, alto=None):
    xs, zs = res['xs'], res['zs']
    w = ancho or float(np.clip(1.0 * (xs[-1] - xs[0]) + 8, 12, 24))
    h = alto or float(np.clip(1.6 * (zs[-1]) + 3.5, 6, 16))
    return w, h


def figura_diagrama(res, diag, tipo='M', titulo='', unidad=None, extra=None, escala=0.4, etiquetas='extremos', piso=None,
                    ancho=None, alto=None, columnas=True, fuente=9, umbral=0.12):
    """Marco con el diagrama de M o V de todos los elementos (o de un piso). extra: diagrama mínimo para superponer.
    etiquetas: 'todas', 'extremos' (extremos y valor máximo; se omiten los valores menores que umbral·máximo para no amontonar)
    o 'ninguna'. columnas=False oculta los valores de las columnas; fuente = tamaño de las etiquetas."""
    plt = _plt()
    xs, zs, n, ne = res['xs'], res['zs'], res['n'], res['ne']
    unidad = unidad or ('t·m' if tipo == 'M' else 't')
    cl = ('#F2D027', '#D63B3B') if tipo == 'M' else ('#69B7E8', '#F29A2E')

    def visible(clave):
        if piso is None:
            return True
        return clave[1] == piso or (clave[0] == 'c' and clave[1] == piso + 1)
    todos = [{k: v for k, v in d.items() if visible(k)} for d in ([diag] + ([extra] if extra else []))]
    vmax = max([float(np.max(np.abs(d[k][tipo]))) for d in todos for k in d] + [1e-9])
    ref = min(min(res['alturas']), min(res['spans']) if res['spans'] else 99)
    sc = escala * ref / vmax
    fw, fh = _tam(res, ancho, alto)
    if piso is not None:
        fh = max(6.0, fh * 0.55)
    fig, ax = plt.subplots(figsize=(fw, fh))
    z_lo, z_hi = (0, zs[-1]) if piso is None else (zs[piso - 1], zs[min(piso + 1, n)])
    for j in range(ne):
        ax.plot([xs[j]] * 2, [z_lo, z_hi], color='#777', lw=0.8, zorder=1)
    for k in range(1, n + 1):
        if z_lo <= zs[k] <= z_hi:
            ax.plot([xs[0], xs[-1]], [zs[k]] * 2, color='#777', lw=0.8, zorder=1)
    box = dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.78)
    for idx_d, d in enumerate(todos):
        signo = (1 if idx_d == 0 else -1) if extra else 0         # envolvente: el máximo rotula (+) y el mínimo (−)
        for clave, v in d.items():
            val = v[tipo]
            if clave[0] == 'b':
                k = clave[1]
                px = v['p']
                py0 = np.full_like(px, zs[k])
                py = py0 - val * sc if tipo == 'M' else py0 + val * sc
                ax.fill_between(px, py0, py, where=val >= 0, color=cl[0], alpha=0.85, lw=0, interpolate=True, zorder=2)
                ax.fill_between(px, py0, py, where=val <= 0, color=cl[1], alpha=0.85, lw=0, interpolate=True, zorder=2)
                ax.plot(px, py, color='#333', lw=0.7, zorder=3)
                if etiquetas != 'ninguna':
                    ids = {0: 'left', len(val) - 1: 'right'}
                    lo, hi = int(0.2 * len(val)), int(0.8 * len(val))          # el pico se busca en el tramo central
                    im = lo + int(np.argmax(np.abs(val[lo:hi])))
                    if etiquetas == 'todas':
                        ids.update({i: 'center' for i in range(5, len(val) - 5, 5)})
                    ids[im] = 'center'
                    for i, ha in ids.items():
                        if (etiquetas == 'extremos' and abs(val[i]) < umbral * vmax) or (signo and val[i] * signo < 0):
                            continue
                        ax.annotate(f'{val[i]:.2f}', (px[i], py[i]),
                                    xytext=(5 if ha == 'left' else (-5 if ha == 'right' else 0), -10 if val[i] >= 0 else 10),
                                    textcoords='offset points', fontsize=fuente, ha=ha, va='center', bbox=box, zorder=6)
            else:
                e = clave[2]
                pz = v['p']
                px0 = np.full_like(pz, xs[e - 1])
                px = px0 + val * sc
                ax.fill_betweenx(pz, px0, px, where=val >= 0, color=cl[0], alpha=0.85, lw=0, interpolate=True, zorder=2)
                ax.fill_betweenx(pz, px0, px, where=val <= 0, color=cl[1], alpha=0.85, lw=0, interpolate=True, zorder=2)
                ax.plot(px, pz, color='#333', lw=0.7, zorder=3)
                if etiquetas != 'ninguna' and columnas:
                    for i, dy in ((0, 10), (len(val) - 1, -10)):
                        if (etiquetas == 'extremos' and abs(val[i]) < umbral * vmax) or (signo and val[i] * signo < 0):
                            continue
                        lado = 1 if val[i] >= 0 else -1
                        ax.annotate(f'{val[i]:.2f}', (px[i], pz[i]), xytext=(10 * lado, dy), textcoords='offset points',
                                    fontsize=fuente, ha='left' if lado > 0 else 'right', va='center', bbox=box, zorder=6)
    for j in range(ne):
        ax.plot(xs[j], 0, marker='s', color='#222', ms=7, zorder=7)
    ax.set_aspect('equal')
    ax.set_title(f'{titulo}  [{unidad}]', fontsize=13, color='#0B2545', pad=10)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.margins(0.07)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def figura_cargas(res, F_sismo):
    """Esquema del pórtico (vigas, columnas, tramos sin losa, columnas continuas) y las fuerzas de sismo de ida."""
    plt = _plt()
    xs, zs, n, ne = res['xs'], res['zs'], res['n'], res['ne']
    geo = res['geo']
    fw, fh = _tam(res)
    fig, ax = plt.subplots(figsize=(fw, fh))
    for i, nv in enumerate(geo['niveles']):
        for c in nv['cols']:
            if c.get('omitir'):
                continue
            top = c.get('hasta', i + 1)
            ax.plot([xs[c['eje'] - 1]] * 2, [zs[i], zs[top]], color='#1D4E89', lw=3.5 if c['h'] >= 40 else 2.5, zorder=2)
            ax.text(xs[c['eje'] - 1] + 0.12, (zs[i] + zs[top]) / 2, f'{c["b"]:.0f}×{c["h"]:.0f}', fontsize=7.5, color='#13315C',
                    rotation=90, va='center')
        k = i + 1
        tr = _etiquetas(nv, ne)
        for j in range(1, ne):
            if (j - 1) in nv['sin']:
                ax.text((xs[j - 1] + xs[j]) / 2, zs[k] + 0.1, 'sin losa', ha='center', fontsize=7.5, color='#999', style='italic')
                continue
            ax.plot([xs[j - 1], xs[j]], [zs[k]] * 2, color='#1D4E89', lw=3, zorder=2)
            q = [nv['q'][c][tr.index(j)] for c in ('CM', 'CV', 'CVacc')]
            ax.text((xs[j - 1] + xs[j]) / 2, zs[k] + 0.12, f'CM {q[0]:g} · CV {q[1]:g} · CVacc {q[2]:g}', ha='center', fontsize=7.5,
                    color='#0B2545')
        for lado, vol, x0, x1 in (('vi', nv['vi'], xs[0] - nv['vi'], xs[0]), ('vd', nv['vd'], xs[-1], xs[-1] + nv['vd'])):
            if vol > 0 and any(nv['q'][c][tr.index(lado)] for c in ('CM', 'CV', 'CVacc')):
                ax.plot([x0, x1], [zs[k]] * 2, color='#3E7CB1', lw=3, zorder=2)
                ax.text((x0 + x1) / 2, zs[k] + 0.12, f'vol. {vol:g} m', ha='center', fontsize=7.5, color='#0B2545')
        ax.text(xs[-1] + 0.4, zs[k] + 0.35, f'viga {nv["viga"]["b"]:.0f}×{nv["viga"]["h"]:.0f}', ha='left', fontsize=7.5, color='#13315C')
        x_ini = xs[0] - max(nv['vi'], 0)
        ax.annotate('', xy=(x_ini, zs[k]), xytext=(x_ini - 1.2, zs[k]), arrowprops=dict(arrowstyle='-|>', color='#D63B3B', lw=2))
        ax.text(x_ini - 1.25, zs[k] + 0.15, f'F{k} = {F_sismo[k - 1]:.2f} t', ha='right', fontsize=8.5, color='#D63B3B')
    for j in range(ne):
        ax.plot(xs[j], 0, marker='s', color='#222', ms=7)
    ax.set_aspect('equal')
    ax.set_title('Modelo del pórtico, cargas (t/m) y fuerzas sísmicas del sismo de ida', fontsize=13, color='#0B2545')
    ax.margins(0.12)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.set_yticks(zs)
    fig.tight_layout()
    return fig


def figura_deformada(res, caso='S', signo=1.0, titulo='Deformada'):
    plt = _plt()
    xs, zs, n, ne = res['xs'], res['zs'], res['n'], res['ne']
    desp = res['caso'][caso]['desp']
    mx = max(max(abs(v[0]), abs(v[1])) for v in desp.values()) or 1.0
    f = 0.6 / (mx / 100)
    fw, fh = _tam(res)
    fig, ax = plt.subplots(figsize=(fw / 1.5, fh / 1.3))
    geo = res['geo']

    def pto(k, e):
        u = desp.get((k, e), (0.0, 0.0, 0.0))
        return xs[e - 1] + signo * u[0] / 100 * f, zs[k] + signo * u[1] / 100 * f
    for i, nv in enumerate(geo['niveles']):
        for c in nv['cols']:
            if c.get('omitir'):
                continue
            top = c.get('hasta', i + 1)
            a, b = pto(i, c['eje']), pto(top, c['eje'])
            ax.plot([xs[c['eje'] - 1]] * 2, [zs[i], zs[top]], color='#bbb', lw=1)
            ax.plot([a[0], b[0]], [a[1], b[1]], color='#1D4E89', lw=2)
        k = i + 1
        for j in range(1, ne):
            if (j - 1) in nv['sin']:
                continue
            a, b = pto(k, j), pto(k, j + 1)
            ax.plot([xs[j - 1], xs[j]], [zs[k], zs[k]], color='#bbb', lw=1)
            ax.plot([a[0], b[0]], [a[1], b[1]], color='#1D4E89', lw=2)
    ax.set_aspect('equal')
    ax.set_title(f'{titulo} (amplificada ×{f:.0f})', fontsize=12, color='#0B2545')
    ax.margins(0.1)
    fig.tight_layout()
    return fig


def figura_viga(env, dis, mats=None):
    """Una viga: envolventes de M y V, y acero requerido vs propuesto a lo largo del claro."""
    plt = _plt()
    clave = dis['clave']
    e = env[clave]
    x = e['x']
    fig, axs = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    a = axs[0]
    if mats:
        for nombre, m in mats.items():
            a.plot(x, m[clave]['M'], lw=1.1, ls='--', label=nombre)
    a.fill_between(x, 0, e['Mmax'], color='#F2D027', alpha=0.85, label='Envolvente máx. (+)')
    a.fill_between(x, 0, e['Mmin'], color='#D63B3B', alpha=0.85, label='Envolvente mín. (−)')
    for pn in ('izquierdo', 'centro', 'derecho'):
        s = next(s for s in dis['secciones'] if s['posicion'] == pn)
        a.axvline(s['x'], color='#555', lw=0.7, ls=':')
    a.invert_yaxis()
    a.axhline(0, color='#333', lw=0.8)
    a.set_ylabel('M (t·m)   tensión abajo = +', fontsize=10)
    a.legend(fontsize=8, ncol=3)
    a.set_title(f'{dis["b"]:g}×{dis["h"]:g} cm · envolventes de momento, cortante y acero', fontsize=12, color='#0B2545')
    a = axs[1]
    if mats:
        for nombre, m in mats.items():
            a.plot(x, m[clave]['V'], lw=1.1, ls='--')
    a.fill_between(x, 0, e['Vmax'], color='#69B7E8', alpha=0.85)
    a.fill_between(x, 0, e['Vmin'], color='#F29A2E', alpha=0.85)
    a.axhline(0, color='#333', lw=0.8)
    a.set_ylabel('V (t)', fontsize=10)
    a = axs[2]
    b, h, fc, fy = dis['b'], dis['h'], dis['fc'], dis['fy']
    d = h - 5.0
    lim = p3.limites_flexion(b, d, fc, fy)

    def req(M):
        if M <= 0:
            return lim['As_min']
        r = p3.area_requerida(M * 1e5, b, d, fc, fy)
        return max(lim['As_min'], r if r is not None else lim['As_max'])
    a.plot(x, [req(max(0, -m)) for m in e['Mmin']], color='#D63B3B', lw=2, label='As requerida superior')
    a.plot(x, [req(max(0, m)) for m in e['Mmax']], color='#C99A00', lw=2, label='As requerida inferior')
    a.axhline(lim['As_max'], color='#555', ls=':', label='As,máx')
    a.axhline(lim['As_min'], color='#999', ls=':', label='As,mín')
    inf = dis['inferior_corrido']['armado']
    if inf:
        a.axhline(inf['As'], color='#C99A00', lw=3, alpha=0.45, label=f'inferior corrido {inf["armado"]} = {inf["As"]:.2f}')
    for pn in ('izquierdo', 'derecho'):
        s = next(s for s in dis['secciones'] if s['posicion'] == pn and s['lecho'] == 'superior')
        if s['armado']:
            izq = pn == 'izquierdo'
            a.hlines(s['armado']['As'], x[0] if izq else x[-1] - dis['L'] / 3, x[0] + dis['L'] / 3 if izq else x[-1],
                     color='#D63B3B', lw=3, alpha=0.45, label=f'superior {pn} {s["armado"]["armado"]}')
    a.set_ylabel('As (cm²)', fontsize=10)
    a.set_xlabel('x (m)')
    a.legend(fontsize=8, ncol=3)
    for ax_ in axs:
        for s in ('top', 'right'):
            ax_.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _dibuja_seccion(ax, b, h, rec, est_db, sup, inf):
    """Dibuja una sección con estribo y barras. sup/inf = dict(lech=distribuir_lechos(...), varilla=...) o None.
    Devuelve True si alguna barra no cumple la separación libre o no cabe en 3 lechos (se pintan en rojo)."""
    from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
    ax.add_patch(Rectangle((0, 0), b, h, fc='#E8EEF6', ec='#13315C', lw=2))
    ax.add_patch(FancyBboxPatch((rec, rec), b - 2 * rec, h - 2 * rec, boxstyle='round,pad=0,rounding_size=1.5', fc='none',
                                ec='#3E7CB1', lw=2.2))
    malo = False
    for lecho, a in (('superior', sup), ('inferior', inf)):
        if not a:
            continue
        lech, db = a['lech'], DIAMETROS[a['varilla']]
        rojo = lech['sep_libre'] + 1e-6 < lech['sep_min'] or not lech['cabe']
        malo = malo or rojo
        for xs_, y in zip(lech['xs'], lech['ys']):
            yy = y if lecho == 'inferior' else h - y
            for xx in xs_:
                ax.add_patch(Circle((xx, yy), db / 2, fc='#D63B3B' if rojo else '#13315C', ec='black', lw=0.8, zorder=5))
    ax.set_xlim(-6, b + 6)
    ax.set_ylim(-4, h + 6)
    ax.set_aspect('equal')
    ax.axis('off')
    return malo


def figura_secciones(dis):
    """Cortes de la viga en apoyo izquierdo, centro y apoyo derecho con las barras elegidas (sueltas o agrupadas de 2 en 2),
    los estribos y las separaciones libres; las barras que no cumplen la separación o que no caben se marcan en rojo."""
    plt = _plt()
    b, h, rec = dis['b'], dis['h'], dis['rec']
    est_db = DIAMETROS[dis['est_var']]
    fig, axs = plt.subplots(1, 3, figsize=(15, max(6.5, 15 / 3 * (h + 20) / (b + 12) + 1.5)))
    for ax, pn in zip(axs, ('izquierdo', 'centro', 'derecho')):
        ss = {lc: next(s for s in dis['secciones'] if s['posicion'] == pn and s['lecho'] == lc) for lc in ('superior', 'inferior')}
        pars = {lc: (dict(lech=s['armado']['lech'], varilla=s['armado']['varilla']) if s['armado'] else None) for lc, s in ss.items()}
        malo = _dibuja_seccion(ax, b, h, rec, est_db, pars['superior'], pars['inferior'])
        textos = []
        for lecho, s in ss.items():
            a = s['armado']
            if not a:
                textos.append((f'{lecho}: sin solución', '#B00020'))
                continue
            forma = ' agrupadas 2 en 2' if a['lech']['agrupado'] else ''
            col = '#B00020' if (not a['cabe'] or not s['ok']) else '#13315C'
            textos.append((f'{"Sup." if lecho == "superior" else "Inf."}: {a["armado"]} ({" + ".join(map(str, a["capas"]))} por lecho{forma})\n'
                           f'As = {a["As"]:.2f} cm² · MR = {a["MR"]:.2f} / Mu = {s["Mu"]:.2f} t·m', col))
        ax.text(b / 2, h + 5.5, f'{pn.capitalize()} · x = {ss["superior"]["x"]:.2f} m', ha='center', fontsize=11, fontweight='bold',
                color='#0B2545')
        for i, (tx, c) in enumerate(textos):
            ax.text(b / 2, -6 - 6.5 * i, tx, ha='center', va='top', fontsize=8.2, color=c)
        ax.text(b / 2, h + 1, f'b = {b:g} cm', ha='center', fontsize=8, color='#555')
        ax.text(-1.5, h / 2, f'h = {h:g} cm', rotation=90, va='center', ha='right', fontsize=8, color='#555')
        ax.set_ylim(-6 - 6.5 * 2 - 8, h + 9)
        if malo:
            ax.text(b / 2, h / 2, 'SEPARACIÓN\nINSUFICIENTE', ha='center', va='center', fontsize=12, color='#B00020', alpha=0.4,
                    fontweight='bold')
    fig.suptitle(f'Secciones de la viga {b:g}×{h:g} cm · rec. {rec:g} cm · estribo {dis["est_var"]} @ '
                 f'{dis["cortante"].get("s_usar", "—")} cm ({dis["cortante"]["ramas"]} ramas)', fontsize=12, color='#0B2545')
    fig.tight_layout()
    return fig


# ================================================================ viga continua del nivel: corrido, bastones y anclaje
def armado_nivel(env, res, k, fc, fy, var_sup, var_inf, est_var='#3', rec=4.0, agrupar=True, paño=True, apoyo_unico=False):
    """Armado de la viga del nivel k como viga continua (procedimiento de clase):
    1) puntos de diseño a), b), c)… (caras de apoyo y centros de cada crujía);
    2) varillas corridas arriba y abajo con área ≥ As,mín y ≥ 50 % de la mayor As real;
    3) bastones donde la As requerida (envolvente) supera lo corrido;
    4) cada bastón se ancla una distancia d más allá del punto donde ya no se requiere en el diagrama Mu.
    apoyo_unico=True: en cada apoyo interior se usa un solo momento negativo, el mayor de los dos lados (como en clase).
    Devuelve None si el nivel no tiene vigas con carga."""
    claves = sorted([c for c in env if c[0] == 'b' and c[1] == k], key=lambda c: env[c]['p'][0])
    if not claves:
        return None
    geo = res['geo']
    b, h = env[claves[0]]['b'], env[claves[0]]['h']
    env = {c: dict(env[c], Mmax=env[c]['Mmax'].copy(), Mmin=env[c]['Mmin'].copy()) for c in claves}      # copia local
    caras = {c: (caras_columna(geo, c) if paño else (0.0, 0.0)) for c in claves}
    if apoyo_unico:
        pos = {}
        for c in claves:
            x_, L_ = env[c]['x'] * 100, env[c]['x'][-1] * 100
            pos[c] = (int(np.argmin(np.abs(x_ - caras[c][0]))), int(np.argmin(np.abs(x_ - (L_ - caras[c][1])))))
        for a_, b_ in zip(claves, claves[1:]):
            if env[b_]['p'][0] - env[a_]['p'][-1] > 1e-6:          # tramos no contiguos (sin losa)
                continue
            ia, ib = pos[a_][1], pos[b_][0]
            env[a_]['Mmin'][ia] = env[b_]['Mmin'][ib] = min(env[a_]['Mmin'][ia], env[b_]['Mmin'][ib])
    dis, P, Mx, Mn, max_paso = {}, [], [], [], 0.0
    for c in claves:
        e = env[c]
        ci, cd = caras[c]
        dis[c] = disenar_viga(env, c, e['b'], e['h'], fc, fy, ci, cd, var_sup, var_inf, est_var, rec, agrupar=agrupar)
        x, L = e['x'] * 100, e['x'][-1] * 100
        mx, mn = e['Mmax'].copy(), e['Mmin'].copy()
        fi, fd = int(np.argmin(np.abs(x - ci))), int(np.argmin(np.abs(x - (L - cd))))
        if fd > fi:                                  # dentro de la columna se usa el valor del paño
            mx[:fi], mn[:fi] = mx[fi], mn[fi]
            mx[fd + 1:], mn[fd + 1:] = mx[fd], mn[fd]
        P.append(e['p'])
        Mx.append(mx)
        Mn.append(mn)
        max_paso = max(max_paso, float(np.max(np.diff(e['p']))))
    P, Mx, Mn = np.concatenate(P), np.concatenate(Mx), np.concatenate(Mn)
    seg = np.concatenate([[0], np.cumsum(np.diff(P) > 1.5 * max_paso)])        # tramos sin losa separan segmentos

    secs = [s for d in dis.values() for s in d['secciones'] if s['armado']]
    d_sup = min([s['armado']['d'] for s in secs if s['lecho'] == 'superior'] or [h - 5.0])
    d_inf = min([s['armado']['d'] for s in secs if s['lecho'] == 'inferior'] or [h - 5.0])
    a_sup, a_inf = p3.VARILLAS[var_sup], p3.VARILLAS[var_inf]
    lim = p3.limites_flexion(b, min(d_sup, d_inf), fc, fy)
    As_real_max = max([s['armado']['As'] for s in secs] or [0.0])
    As_corr = max(lim['As_min'], 0.5 * As_real_max)
    n_corr = {'superior': max(2, math.ceil(As_corr / a_sup - 1e-9)), 'inferior': max(2, math.ceil(As_corr / a_inf - 1e-9))}

    def req(M, d):
        l_ = p3.limites_flexion(b, d, fc, fy)
        if M <= 0:
            return l_['As_min']
        r_ = p3.area_requerida(M * 1e5, b, d, fc, fy)
        return max(l_['As_min'], r_ if r_ is not None else l_['As_max'])
    req_sup = np.array([req(max(0.0, -m), d_sup) for m in Mn])
    req_inf = np.array([req(max(0.0, m), d_inf) for m in Mx])
    bastones = []
    for lecho, rq, a_, var, d in (('superior', req_sup, a_sup, var_sup, d_sup), ('inferior', req_inf, a_inf, var_inf, d_inf)):
        need = np.ceil(rq / a_ - 1e-9)
        for j in range(1, int(need.max() - n_corr[lecho]) + 1):
            mask = need >= n_corr[lecho] + j
            i = 0
            while i < len(P):
                if not mask[i]:
                    i += 1
                    continue
                i1 = i
                while i1 + 1 < len(P) and mask[i1 + 1] and seg[i1 + 1] == seg[i]:
                    i1 += 1
                en_seg = np.where(seg == seg[i])[0]
                xa, xb = float(P[i]), float(P[i1])
                x_ini = max(float(P[en_seg[0]]), xa - d / 100)
                x_fin = min(float(P[en_seg[-1]]), xb + d / 100)
                bastones.append(dict(lecho=lecho, nivel=j, varilla=var, req_ini=xa, req_fin=xb, x_ini=x_ini, x_fin=x_fin,
                                     L=x_fin - x_ini, d=d))
                i = i1 + 1
    puntos, letras = [], iter('abcdefghijklmnopqrstuvwxyz')
    for c in claves:
        x0 = float(env[c]['p'][0])
        for pn in ('izquierdo', 'centro', 'derecho'):
            s = next(s for s in dis[c]['secciones'] if s['posicion'] == pn)
            puntos.append(dict(letra=next(letras, '·'), clave=c, posicion=pn, x=x0 + s['x']))

    def n_final(lecho, x):
        return n_corr[lecho] + sum(1 for q in bastones if q['lecho'] == lecho and q['x_ini'] - 1e-9 <= x <= q['x_fin'] + 1e-9)
    for q in puntos:
        q['n_sup'], q['n_inf'] = n_final('superior', q['x']), n_final('inferior', q['x'])
    Mu_max = float(max(Mx.max(), -Mn.min()))
    d_v = min(d_sup, d_inf)
    MR_max = p3.momento_resistente(lim['As_max'], b, d_v, fc, fy) / 1e5
    Vu_max = max(d['Vu'] for d in dis.values())
    V_lim = 2.65 * p3.FR_CORTANTE * b * d_v * math.sqrt(fc) / 1000
    return dict(nivel=k, claves=claves, dis=dis, P=P, req_sup=req_sup, req_inf=req_inf, n_corr=n_corr, bastones=bastones,
                puntos=puntos, var_sup=var_sup, var_inf=var_inf, b=b, h=h, rec=rec, est_var=est_var, agrupar=agrupar,
                As_real_max=As_real_max, As_corr=As_corr, d_sup=d_sup, d_inf=d_inf, lim=lim, n_final=n_final,
                Mu_max=Mu_max, MR_max=MR_max, Vu_max=Vu_max, V_lim=V_lim, max_paso=max_paso,
                pasos=[f'Subreforzada: MR(As,máx) = 0.9·{lim["As_max"]:.2f}·{fy:g}·(d − As,máx·fy/(2·b·f″c)) = {MR_max:.2f} t·m '
                       f'{"≥" if MR_max >= Mu_max else "<"} Mu máx = {Mu_max:.2f} t·m → {"OK" if MR_max >= Mu_max else "cambia la sección"}',
                       f'Cortante: V límite = 2.65·FR·b·d·√f′c = {V_lim:.1f} t {"≥" if V_lim >= Vu_max else "<"} Vu máx = {Vu_max:.1f} t → '
                       f'{"OK" if V_lim >= Vu_max else "cambia la sección"}',
                       f'Corrido: As,mín = {lim["As_min"]:.2f} cm²; 50 % de la mayor As real ({As_real_max:.2f}) = {0.5 * As_real_max:.2f} cm² → '
                       f'{n_corr["superior"]}{var_sup} arriba y {n_corr["inferior"]}{var_inf} abajo, a todo lo largo y con ganchos en los extremos',
                       'Bastones donde la As requerida (envolvente) supera lo corrido; cada uno se ancla una distancia d = '
                       f'{d_sup:.1f} cm (sup.) / {d_inf:.1f} cm (inf.) más allá de donde ya no se requiere en el diagrama Mu'])


def figura_alzado(niv, res):
    """Alzado de la viga continua del nivel: corrido con ganchos, bastones con su longitud, puntos a), b)… y, abajo, la
    sección en cada punto con las barras finales (corrido + bastones que pasan por ahí)."""
    plt = _plt()
    from matplotlib.gridspec import GridSpec
    geo = res['geo']
    k, b, h, rec = niv['nivel'], niv['b'], niv['h'], niv['rec']
    est_db = DIAMETROS[niv['est_var']]
    P = niv['P']
    xs_ej = res['xs']
    hv = h / 100.0
    pts = niv['puntos']
    ncol = min(len(pts), 8)
    nfil = math.ceil(len(pts) / ncol)
    ancho = float(np.clip(0.9 * (P.max() - P.min()) + 6, 14, 30))
    fig = plt.figure(figsize=(ancho, 6.0 + 3.6 * nfil))
    gs = GridSpec(1 + nfil, ncol, height_ratios=[2.4] + [1.6] * nfil, figure=fig)
    ax = fig.add_subplot(gs[0, :])
    corte = np.diff(P) > 1.5 * niv['max_paso']
    inicios = [P[0]] + [P[i + 1] for i in range(len(P) - 1) if corte[i]]
    fines = [P[i] for i in range(len(P) - 1) if corte[i]] + [P[-1]]
    for a_, b_ in zip(inicios, fines):
        ax.add_patch(plt.Rectangle((a_, 0), b_ - a_, hv, fc='#E8EEF6', ec='#13315C', lw=1.6, zorder=1))
    for c in geo['niveles'][k - 1]['cols']:
        if c.get('omitir'):
            continue
        xe, w = xs_ej[c['eje'] - 1], c['h'] / 100.0
        ax.add_patch(plt.Rectangle((xe - w / 2, -0.45), w, 0.45, fc='#CFCFCF', ec='#555', hatch='///', lw=1, zorder=0))
        ax.text(xe, -0.6, f'eje {c["eje"]}', ha='center', va='top', fontsize=8, color='#555')
    yt, yb = hv - 0.07, 0.07
    for lecho, y, n, var, col, sg in (('superior', yt, niv['n_corr']['superior'], niv['var_sup'], '#D63B3B', -1),
                                      ('inferior', yb, niv['n_corr']['inferior'], niv['var_inf'], '#13315C', 1)):
        for a_, b_ in zip(inicios, fines):
            ax.plot([a_ + 0.05, b_ - 0.05], [y, y], color=col, lw=3.2, zorder=4, solid_capstyle='butt')
            for xg in (a_ + 0.05, b_ - 0.05):          # ganchos al inicio y al final
                ax.plot([xg, xg], [y, y + sg * 0.22 * hv], color=col, lw=3.2, zorder=4)
        ax.annotate(f'{n}{var} corridas', (inicios[0], y), xytext=(inicios[0] - 0.3, y + (0.5 if lecho == 'superior' else -0.5)),
                    fontsize=9, color=col, fontweight='bold', ha='right', arrowprops=dict(arrowstyle='-', color=col, lw=0.8))
    grupos = {}                                       # bastones con el mismo tramo se rotulan juntos («2#8»)
    for q in niv['bastones']:
        grupos.setdefault((q['lecho'], round(q['x_ini'], 2), round(q['x_fin'], 2), q['varilla']), []).append(q)
    cont = {'superior': 0, 'inferior': 0}
    for (lecho, xi, xf, var), qs in sorted(grupos.items(), key=lambda kv: kv[0][1]):
        sup = lecho == 'superior'
        col = '#D63B3B' if sup else '#13315C'
        for q in qs:
            y = (yt - 0.06 * q['nivel']) if sup else (yb + 0.06 * q['nivel'])
            ax.plot([xi, xf], [y, y], color=col, lw=2.2, zorder=5, solid_capstyle='butt')
        i = cont[lecho]
        cont[lecho] += 1
        yy = hv + 0.4 + 0.5 * (i % 4) if sup else -0.9 - 0.5 * (i % 4)
        ax.annotate(f'{len(qs)}{var}  L = {xf - xi:.2f} m', ((xi + xf) / 2, y), xytext=((xi + xf) / 2, yy),
                    fontsize=8, ha='center', va='bottom' if sup else 'top', color=col, zorder=8,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=col, lw=0.6, alpha=0.95),
                    arrowprops=dict(arrowstyle='-', color=col, lw=0.6, ls=':'))
    for q in pts:
        ax.plot([q['x'], q['x']], [-0.45, hv + 0.25], color='#777', lw=0.8, ls='--', zorder=2)
        ax.text(q['x'], hv + 0.27, f'{q["letra"]})', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333')
    ax.set_xlim(P.min() - 2.6, P.max() + 1.2)
    ax.set_ylim(-3.1, hv + 2.9)
    ax.set_yticks([])
    ax.set_xlabel('x (m)')
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.set_title(f'Nivel {k} · viga {b:g}×{h:g} cm · corrido, bastones y anclaje (d = {niv["d_sup"]:.0f} cm) · vista no a escala vertical',
                 fontsize=12, color='#0B2545')
    for i, q in enumerate(pts):
        a2 = fig.add_subplot(gs[1 + i // ncol, i % ncol])
        ls_ = distribuir_lechos(q['n_sup'], DIAMETROS[niv['var_sup']], b, rec, est_db, niv['agrupar'])
        li_ = distribuir_lechos(q['n_inf'], DIAMETROS[niv['var_inf']], b, rec, est_db, niv['agrupar'])
        malo = _dibuja_seccion(a2, b, h, rec, est_db, dict(lech=ls_, varilla=niv['var_sup']), dict(lech=li_, varilla=niv['var_inf']))
        a2.set_title(f'{q["letra"]}) x = {q["x"]:.2f} m', fontsize=9, color='#B00020' if malo else '#0B2545')
        a2.text(b / 2, -6, f'sup {q["n_sup"]}{niv["var_sup"]} · inf {q["n_inf"]}{niv["var_inf"]}', ha='center', va='top', fontsize=7.5,
                color='#B00020' if malo else '#13315C')
    fig.tight_layout()
    return fig


# ================================================================ cálculo desarrollado paso a paso (procedimiento de clase)
def ficha_viga(b, h, fc, fy, varilla, est_var, ramas, rec=4.0, r_fija=None, Mu_max=None, Vu_max=None):
    """Datos y revisiones de la viga como en el pizarrón: Ec, f″c, β1, Asb, As,máx, As,mín, d, varilla y estribo, MR con
    As,máx contra Mu máx (subreforzada) y V límite contra Vu máx. Devuelve dict(lineas, d, lim, ok_flexion, ok_cortante)."""
    db, ab = DIAMETROS[varilla], p3.VARILLAS[varilla]
    est_db, a_est = DIAMETROS[est_var], p3.VARILLAS[est_var]
    r = r_fija if r_fija else rec + est_db + db / 2
    d = h - r
    lim = p3.limites_flexion(b, d, fc, fy)
    fcc = lim['fcc']
    lin = [f'**Materiales.** Ec = 14000·√f′c = 14000·√{fc:g} = {14000 * math.sqrt(fc):,.0f} kg/cm²; Es = 2,000,000 kg/cm²; '
           f'fy = {fy:g} kg/cm²',
           f"f″c = 0.85·f′c = 0.85·{fc:g} = {fcc:.2f} kg/cm²; β1 = {lim['beta1']:.2f} (f′c ≤ 280)",
           f'**Sección.** b = {b:g} cm, h = {h:g} cm; ' + (f'r = {r:g} cm (fija)' if r_fija else
                                                           f'r supuesta = rec + Øest + Ø/2 = {rec:g} + {est_db:.2f} + {db / 2:.2f} = {r:.2f} cm')
           + f'; d = h − r = {d:.2f} cm',
           f'**Varilla {varilla}:** Ø = {db:.2f} cm, Ab = {ab} cm².  **Estribo {est_var}:** Ø = {est_db:.2f} cm, {ramas} ramas → '
           f'Av = {ramas}·{a_est} = {ramas * a_est:.2f} cm²',
           f"**Límites de acero.** Asb = (f″c/fy)·(6000·β1/(fy+6000))·b·d = ({fcc:.1f}/{fy:g})·(6000·{lim['beta1']:.2f}/{fy + 6000:g})·{b:g}·{d:.2f} "
           f"= {lim['Asb']:.2f} cm²",
           f"As,máx = 0.75·Asb = {lim['As_max']:.2f} cm²",
           f"As,mín = máx(0.8·√f′c/fy·b·d = {0.8 * math.sqrt(fc) / fy * b * d:.2f}; 14/fy·b·d = {14 / fy * b * d:.2f}) = {lim['As_min']:.2f} cm²"]
    MR_max = p3.momento_resistente(lim['As_max'], b, d, fc, fy) / 1e5
    lin.append(f"**Subreforzada.** MR(As,máx) = FR·As,máx·fy·(d − As,máx·fy/(2·b·f″c)) = 0.9·{lim['As_max']:.2f}·{fy:g}·"
               f"({d:.2f} − {lim['As_max'] * fy / (2 * b * fcc):.2f}) = {MR_max:.2f} t·m")
    ok_f = True
    if Mu_max is not None:
        ok_f = MR_max >= Mu_max
        lin.append(f'MR(As,máx) = {MR_max:.2f} {"≥" if ok_f else "<"} Mu máx = {Mu_max:.2f} t·m → '
                   f'{"OK: todas las secciones pueden ser subreforzadas" if ok_f else "NO: cambia la sección (o usa doble refuerzo)"}')
    V_lim = 2.65 * p3.FR_CORTANTE * b * d * math.sqrt(fc) / 1000
    lin.append(f'**Cortante máximo.** V límite = 2.65·FR·b·d·√f′c = 2.65·0.75·{b:g}·{d:.2f}·√{fc:g} = {V_lim:.1f} t')
    ok_c = True
    if Vu_max is not None:
        ok_c = V_lim >= Vu_max
        lin.append(f'V límite = {V_lim:.1f} {"≥" if ok_c else "<"} Vu máx = {Vu_max:.1f} t → '
                   f'{"OK" if ok_c else "NO: agranda la sección"}')
    tex = [('Módulo de elasticidad', rf"E_c = 14000\sqrt{{f'_c}} = 14000\sqrt{{{fc:g}}} = {_n(14000 * math.sqrt(fc))}\ \text{{kg/cm}}^2"),
           ('Esfuerzo del bloque', rf"f''_c = 0.85\,f'_c = 0.85({fc:g}) = {fcc:.2f}\ \text{{kg/cm}}^2,\qquad \beta_1 = {lim['beta1']:.2f}"),
           ('Peralte efectivo', rf"d = h - r = {h:g} - {r:.2f} = {d:.2f}\ \text{{cm}}"),
           ('Acero balanceado', rf"A_{{sb}} = \frac{{f''_c}}{{f_y}}\,\frac{{6000\,\beta_1}}{{f_y+6000}}\,b\,d = \frac{{{fcc:.1f}}}{{{fy:g}}}\cdot\frac{{6000({lim['beta1']:.2f})}}{{{fy + 6000:g}}}({b:g})({d:.2f}) = {lim['Asb']:.2f}\ \text{{cm}}^2"),
           ('Acero máximo', rf"A_{{s,max}} = 0.75\,A_{{sb}} = {lim['As_max']:.2f}\ \text{{cm}}^2"),
           ('Acero mínimo', rf"A_{{s,min}} = \max\left(\frac{{0.8\sqrt{{f'_c}}}}{{f_y}}\,b\,d;\ \frac{{14}}{{f_y}}\,b\,d\right) = \max({0.8 * math.sqrt(fc) / fy * b * d:.2f};\ {14 / fy * b * d:.2f}) = {lim['As_min']:.2f}\ \text{{cm}}^2"),
           ('Momento con As,máx (subreforzada)', rf"M_R = F_R\,A_{{s,max}}\,f_y\left(d - \frac{{A_{{s,max}}f_y}}{{2\,b\,f''_c}}\right) = 0.9({lim['As_max']:.2f})({fy:g})\left({d:.2f} - {lim['As_max'] * fy / (2 * b * fcc):.2f}\right) = {MR_max:.2f}\ \text{{t}}\cdot\text{{m}}"
                                                          + (rf"\ {'\\ge' if ok_f else '<'}\ M_{{u,max}} = {Mu_max:.2f}" if Mu_max is not None else '')),
           ('Cortante máximo de la sección', rf"V_{{lim}} = 2.65\,F_R\,b\,d\sqrt{{f'_c}} = 2.65(0.75)({b:g})({d:.2f})\sqrt{{{fc:g}}} = {V_lim:.1f}\ \text{{t}}"
                                             + (rf"\ {'\\ge' if ok_c else '<'}\ V_{{u,max}} = {Vu_max:.1f}" if Vu_max is not None else ''))]
    return dict(lineas=lin, tex=tex, d=d, r=r, lim=lim, MR_max=MR_max, V_lim=V_lim, ok_flexion=ok_f, ok_cortante=ok_c)


def _n(x, dec=0):
    """Número con separador de miles para LaTeX: 273{,}000."""
    return f'{x:,.{dec}f}'.replace(',', '{,}')


def parse_propuesta(txt):
    """'4#8', '4 #8', '4x8' o '4 8' → (4, '#8'); vacío → None; texto inválido → ValueError."""
    import re
    t = str(txt or '').strip().lower().replace('ø', '#').replace('×', 'x')
    if not t or t in ('nan', 'none', '-', '—'):
        return None
    m = re.fullmatch(r'(\d+)\s*(?:#|x|\s)\s*#?\s*(\d+)', t)
    if not m:
        raise ValueError(f'No entiendo el armado «{txt}». Escríbelo como 4#8 (número de varillas y número de varilla).')
    n, v = int(m.group(1)), f'#{m.group(2)}'
    if v not in DIAMETROS:
        raise ValueError(f'La varilla {v} no existe aquí: usa {", ".join(DIAMETROS)}.')
    if n < 1:
        raise ValueError('El número de varillas debe ser al menos 1.')
    return n, v


def ficha_punto(Mu_tm, b, h, fc, fy, varilla, rec=4.0, est_db=0.95, r_fija=None, agrupar=True, lecho='superior', propuesta=None):
    """Diseño a flexión de UN punto con todo el desarrollo (texto y fórmulas LaTeX).
    · propuesta=None: la app calcula As y propone el número de barras de `varilla`.
    · propuesta=(n, '#v'): tú propones n barras de esa varilla; la app calcula As real = n·Ab, MR, separación y revisa contra As calculada.
    r_fija = r (cm) al centroide como en clase; si no, se usa la posición real de las barras (iterando).
    Devuelve dict(As_req, As_usar, n, As_real, MR, ok, cabe, lineas, tex, lech, d, r, ...)."""
    if propuesta:
        n, varilla = propuesta
    db, ab = DIAMETROS[varilla], p3.VARILLAS[varilla]
    if propuesta:
        lech = distribuir_lechos(n, db, b, rec, est_db, agrupar)
        r = float(r_fija) if r_fija else lech['r']
    elif r_fija:
        r = float(r_fija)
        As0 = _as_req(Mu_tm, b, h - r, fc, fy)
        n = max(2, math.ceil((As0[1] if As0 else 0) / ab - 1e-9))
        lech = distribuir_lechos(n, db, b, rec, est_db, agrupar)
    else:
        r = rec + est_db + db / 2
        n, lech = 2, distribuir_lechos(2, db, b, rec, est_db, agrupar)
        for _ in range(10):
            As0 = _as_req(Mu_tm, b, h - r, fc, fy)
            if As0 is None:
                break
            n = max(2, math.ceil(As0[1] / ab - 1e-9))
            lech = distribuir_lechos(n, db, b, rec, est_db, agrupar)
            if abs(lech['r'] - r) < 0.02:
                break
            r = lech['r']
    d = h - r
    lim = p3.limites_flexion(b, d, fc, fy)
    fcc = lim['fcc']
    base = dict(Mu=Mu_tm, varilla=varilla, lecho=lecho, lim=lim, d=d, r=r)
    lin = [f'Mu = {Mu_tm:.2f} t·m = {Mu_tm * 1e5:,.0f} kg·cm  (tensión {lecho})',
           f'd = h − r = {h:g} − {r:.2f} = {d:.2f} cm' + ('  (r fija)' if r_fija else '  (r al centroide de las barras colocadas)')]
    tex = [('Peralte efectivo', rf'd = h - r = {h:g} - {r:.2f} = {d:.2f}\ \text{{cm}}'),
           ('Momento último', rf'M_u = {Mu_tm:.2f}\ \text{{t}}\cdot\text{{m}} = {_n(Mu_tm * 1e5)}\ \text{{kg}}\cdot\text{{cm}}')]
    if Mu_tm <= 1e-9:
        lin.append(f"Sin momento de este signo → rige As,mín = {lim['As_min']:.2f} cm²")
        tex.append(('Acero mínimo', rf"A_{{s,min}} = {lim['As_min']:.2f}\ \text{{cm}}^2"))
        As_req, As_usar = 0.0, lim['As_min']
        if not propuesta:
            n = max(2, math.ceil(As_usar / ab - 1e-9))
            lech = distribuir_lechos(n, db, b, rec, est_db, agrupar)
    else:
        A_ = fy ** 2 / (2 * b * fcc)
        B_ = fy * d
        C_ = Mu_tm * 1e5 / p3.FR_FLEXION
        disc = B_ ** 2 - 4 * A_ * C_
        lin += ['Cuadrática (FR = 0.9):  (fy²/(2·b·f″c))·As² − (fy·d)·As + Mu/FR = 0',
                f'A = {fy:g}²/(2·{b:g}·{fcc:.1f}) = {A_:.3f};  B = {fy:g}·{d:.2f} = {B_:,.0f};  C = {Mu_tm * 1e5:,.0f}/0.9 = {C_:,.0f}',
                f'Discriminante = B² − 4AC = {B_ ** 2:,.0f} − {4 * A_ * C_:,.0f} = {disc:,.0f}']
        tex += [("Ecuación de equilibrio de momentos", r"\frac{f_y^2}{2\,b\,f''_c}\,A_s^2 - (f_y\,d)\,A_s + \frac{M_u}{F_R} = 0"),
                ('Coeficientes', rf"A = \frac{{f_y^2}}{{2\,b\,f''_c}} = \frac{{{fy:g}^2}}{{2({b:g})({fcc:.1f})}} = {A_:.3f}"
                                 rf",\quad B = f_y\,d = {fy:g}({d:.2f}) = {_n(B_)}"
                                 rf",\quad C = \frac{{M_u}}{{F_R}} = \frac{{{_n(Mu_tm * 1e5)}}}{{0.9}} = {_n(C_)}"),
                ('Discriminante', rf"B^2 - 4AC = {_n(B_ ** 2)} - {_n(4 * A_ * C_)} = {_n(disc)}")]
        if disc < 0:
            lin.append('El discriminante es negativo: la sección no alcanza → aumenta b, h o f′c.')
            return dict(base, As_req=None, As_usar=None, n=n, As_real=None, MR=None, ok=False, cabe=False, lineas=lin, tex=tex, lech=lech,
                        armado='—', sep_libre=None)
        raiz = math.sqrt(disc)
        As1, As2 = (B_ - raiz) / (2 * A_), (B_ + raiz) / (2 * A_)
        lin.append(f'As = (B ∓ √disc)/(2A):  As₁ = {As1:.2f} cm² ✓ (raíz menor);  As₂ = {As2:.2f} cm² ✗ (descartada: excede As,máx = '
                   f"{lim['As_max']:.2f} y el bloque a = As·fy/(b·f″c) = {As2 * fy / (b * fcc):.1f} cm > d)")
        tex.append(('Raíces de As', rf"A_s = \frac{{B \mp \sqrt{{B^2-4AC}}}}{{2A}} \;\Rightarrow\; A_{{s1}} = \frac{{{_n(B_)} - {_n(raiz)}}}{{2({A_:.3f})}}"
                                    rf" = \mathbf{{{As1:.2f}}}\ \text{{cm}}^2\ \checkmark,\quad A_{{s2}} = {As2:.2f}\ \text{{cm}}^2\ \times"))
        As_req = As1
        As_usar = max(As1, lim['As_min'])
        lin.append(f"As,mín = {lim['As_min']:.2f} cm² → As usar = máx({As1:.2f}; {lim['As_min']:.2f}) = {As_usar:.2f} cm²"
                   + ('  (rige As,mín)' if As_usar > As1 else ''))
        tex.append(('Acero a usar', rf"A_{{s,usar}} = \max(A_{{s1}};\,A_{{s,min}}) = \max({As1:.2f};\,{lim['As_min']:.2f}) = {As_usar:.2f}\ \text{{cm}}^2"))
    ok_max = As_usar <= lim['As_max']
    lin.append(f"As usar {'≤' if ok_max else '>'} As,máx = {lim['As_max']:.2f} cm² → {'subreforzada ✓' if ok_max else 'NO: cambia la sección'}")
    tex.append(('Revisión As,máx', rf"A_{{s,usar}} = {As_usar:.2f} {'\\le' if ok_max else '>'} A_{{s,max}} = {lim['As_max']:.2f}\ \text{{cm}}^2"))
    if propuesta:
        lin.append(f'**Armado propuesto por ti:** {n}{varilla} (Ab = {ab} cm² cada una)')
        tex.append(('Armado propuesto', rf"A_{{s,real}} = n\,A_b = {n}({ab}) = {n * ab:.2f}\ \text{{cm}}^2"))
    else:
        lin.append(f'Barras {varilla} (Ab = {ab} cm²): n = As usar / Ab = {As_usar:.2f}/{ab} = {As_usar / ab:.2f} → {n} barras')
        tex.append(('Número de barras', rf"n = \frac{{A_{{s,usar}}}}{{A_b}} = \frac{{{As_usar:.2f}}}{{{ab}}} = {As_usar / ab:.2f}\;\rightarrow\;{n}\ \text{{varillas {varilla.replace('#', chr(92) + '#')}}}"))
        for _ in range(30):                       # sube barras mientras el MR con la posición real de las barras no alcance
            lech = distribuir_lechos(n, db, b, rec, est_db, agrupar) if not r_fija else lech
            r_eff = r_fija if r_fija else lech['r']
            if p3.momento_resistente(n * ab, b, h - r_eff, fc, fy) / 1e5 + 1e-9 >= Mu_tm or not lech['cabe']:
                break
            n += 1
            lin.append(f'Con la d real no alcanza el MR → se sube a {n} barras')
            if r_fija:
                lech = distribuir_lechos(n, db, b, rec, est_db, agrupar)
    r_eff = r_fija if r_fija else lech['r']
    d_eff = h - r_eff
    As_real = n * ab
    MR = p3.momento_resistente(As_real, b, d_eff, fc, fy) / 1e5
    a_blq = As_real * fy / (b * fcc)
    if propuesta and not r_fija and abs(d_eff - d) > 0.01:
        lin.append(f'(La d con tus {n} barras colocadas es {d_eff:.2f} cm; el MR se calcula con esa d.)')
    ok_as = As_real >= As_usar - 1e-9
    lin.append(f'As real = {n}{varilla} = {n}·{ab} = {As_real:.2f} cm² {"≥" if ok_as else "<"} As usar = {As_usar:.2f} cm²'
               + ('' if ok_as else '  ✗ no alcanza el área requerida'))
    lin.append(f'MR = FR·As real·fy·(d − a/2) con a = As real·fy/(b·f″c) = {As_real:.2f}·{fy:g}/({b:g}·{fcc:.1f}) = {a_blq:.2f} cm → '
               f'MR = 0.9·{As_real:.2f}·{fy:g}·({d_eff:.2f} − {a_blq / 2:.2f}) = {MR:.2f} t·m {"≥" if MR + 1e-9 >= Mu_tm else "<"} Mu = {Mu_tm:.2f} t·m')
    tex += [('Revisión de área', rf"A_{{s,real}} = {As_real:.2f} {'\\ge' if ok_as else '<'} A_{{s,usar}} = {As_usar:.2f}\ \text{{cm}}^2"),
            ('Profundidad del bloque', rf"a = \frac{{A_{{s,real}}\,f_y}}{{b\,f''_c}} = \frac{{{As_real:.2f}({fy:g})}}{{{b:g}({fcc:.1f})}} = {a_blq:.2f}\ \text{{cm}}"),
            ('Momento resistente', rf"M_R = F_R\,A_{{s,real}}\,f_y\left(d - \frac{{a}}{{2}}\right) = 0.9({As_real:.2f})({fy:g})\left({d_eff:.2f} - {a_blq / 2:.2f}\right)"
                                   rf" = {MR:.2f}\ \text{{t}}\cdot\text{{m}}\ {'\\ge' if MR + 1e-9 >= Mu_tm else '<'}\ M_u = {Mu_tm:.2f}")]
    _separacion(lin, n, db, b, rec, est_db, lech)
    util = b - 2 * (rec + est_db)
    if len(lech['capas']) == 1 and not lech['agrupado'] and n > 1:
        tex.append(('Separación libre', rf"s = \frac{{b - 2(r_{{ec}} + \phi_{{est}}) - n\,\phi}}{{n-1}} = \frac{{{b:g} - 2({rec:g} + {est_db:.2f}) - {n}({db:.2f})}}{{{n - 1}}} = {lech['sep_libre']:.2f}"
                                        rf"\ \text{{cm}}\ {'\\ge' if lech['sep_libre'] >= lech['sep_min'] else '<'}\ {lech['sep_min']:.2f}"))
    else:
        tex.append(('Separación libre', rf"\text{{ancho útil}} = {util:.2f}\ \text{{cm}};\quad s_{{libre}} = {lech['sep_libre']:.2f}\ \text{{cm}}"
                                        rf"\ {'\\ge' if lech['sep_libre'] >= lech['sep_min'] else '<'}\ {lech['sep_min']:.2f}\ \text{{cm}}"
                                        + (r"\quad(\text{barras agrupadas de 2 en 2})" if lech['agrupado'] else '')))
    ok = ok_max and ok_as and MR + 1e-9 >= Mu_tm and lech['cabe'] and lech['sep_libre'] + 1e-6 >= lech['sep_min'] and As_real <= lim['As_max']
    return dict(base, As_req=As_req, As_usar=As_usar, n=n, As_real=As_real, MR=MR, ok=ok, cabe=lech['cabe'], lineas=lin, tex=tex, lech=lech,
                d=d_eff, r=r_eff, armado=f'{n}{varilla}', sep_libre=lech['sep_libre'], propuesto=bool(propuesta))


def _as_req(Mu_tm, b, d, fc, fy):
    """(As calculada, As de diseño) o None si no hay solución real."""
    lim = p3.limites_flexion(b, d, fc, fy)
    if Mu_tm <= 1e-9:
        return 0.0, lim['As_min']
    As = p3.area_requerida(Mu_tm * 1e5, b, d, fc, fy)
    return None if As is None else (As, max(As, lim['As_min']))


def _separacion(lin, n, db, b, rec, est_db, lech):
    util = b - 2 * (rec + est_db)
    forma = ' (agrupadas de 2 en 2, pegadas)' if lech['agrupado'] else ''
    lin.append(f'**Separación.** Ancho útil = b − 2(rec + Øest) = {b:g} − 2({rec:g} + {est_db:.2f}) = {util:.2f} cm; '
               f'{n} barras de Ø {db:.2f}: {" + ".join(map(str, lech["capas"]))} por lecho{forma}')
    if len(lech['capas']) == 1 and not lech['agrupado'] and n > 1:
        lin.append(f'Separación libre = (ancho útil − n·Ø)/(n − 1) = ({util:.2f} − {n}·{db:.2f})/{n - 1} = {lech["sep_libre"]:.2f} cm '
                   f'{"≥" if lech["sep_libre"] >= lech["sep_min"] else "<"} máx(Ø; 2.5) = {lech["sep_min"]:.2f} cm → '
                   f'{"cabe ✓" if lech["sep_libre"] >= lech["sep_min"] else "NO cabe ✗"}')
    else:
        lin.append(f'Separación libre entre barras/grupos = {lech["sep_libre"]:.2f} cm {"≥" if lech["sep_libre"] >= lech["sep_min"] else "<"} '
                   f'máx(Ø; 2.5) = {lech["sep_min"]:.2f} cm → {"cabe ✓" if lech["cabe"] and lech["sep_libre"] >= lech["sep_min"] else "NO cabe ✗"}')


def figura_puntos(fichas, b, h, rec, est_db, titulos):
    """Cuadrícula con la sección de cada punto (solo el lecho en tensión) y sus datos."""
    plt = _plt()
    n = len(fichas)
    ncol = min(n, 4)
    nfil = math.ceil(n / ncol)
    fig, axs = plt.subplots(nfil, ncol, figsize=(4.6 * ncol, (h / b * 3.2 + 2.2) * nfil), squeeze=False)
    for ax in axs.ravel():
        ax.axis('off')
    for ax, f, t in zip(axs.ravel(), fichas, titulos):
        if not f.get('lech'):
            ax.set_title(f'{t}: sin solución', color='#B00020')
            continue
        par = dict(lech=f['lech'], varilla=f['varilla'])
        sup = par if f.get('lecho', 'superior') == 'superior' else None
        inf = par if f.get('lecho', 'superior') != 'superior' else None
        malo = _dibuja_seccion(ax, b, h, rec, est_db, sup, inf)
        ax.set_title(f'{t}  Mu = {f["Mu"]:.2f} t·m', fontsize=10, color='#B00020' if (malo or not f['ok']) else '#0B2545')
        ax.text(b / 2, -6, f'{f["armado"]} = {f["As_real"]:.2f} cm² · MR = {f["MR"]:.2f}\n'
                           f'sep. libre = {f["sep_libre"]:.2f} cm', ha='center', va='top', fontsize=8.5,
                color='#B00020' if (malo or not f['ok']) else '#13315C')
        ax.set_ylim(-14, h + 8)
    fig.tight_layout()
    return fig
