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


def varilla_sugerida(h):
    """Regla del profesor: el número de la varilla ≈ h(cm)/10. Devuelve (varilla disponible ≤ h/10, texto)."""
    ref = h / 10.0
    disp = [v for v, n in NUM_VARILLA.items() if n <= ref + 1e-9] or ['#3']
    elegida = disp[-1]
    if abs(ref - round(ref)) < 1e-9 and f'#{int(round(ref))}' in NUM_VARILLA:
        txt = f'h/10 = {ref:g} → varilla #{int(round(ref))}.'
    elif abs(ref - round(ref)) < 1e-9:
        txt = f'h/10 = {ref:g} → no existe la #{int(round(ref))}: elige la {elegida} o la siguiente disponible.'
    else:
        txt = f'h/10 = {ref:g} → como máximo la {elegida} (no hay varillas intermedias).'
    return elegida, txt


# ================================================================ geometrías
def _tramos_nivel(vi, vd, nb):
    """Etiquetas de los tramos con carga de un nivel: volado izq. (si hay) + crujías + volado der. (si hay)."""
    return (['vi'] if vi > 0 else []) + list(range(1, nb + 1)) + (['vd'] if vd > 0 else [])


def _etiquetas(nv, ne):
    return nv['etiquetas'] if 'etiquetas' in nv else _tramos_nivel(nv['vi'], nv['vd'], ne - 1)


def geo_simple(spans, alturas, col_secc, vig_secc, E, cargas, base='Empotrada', fi_col=1.0, fi_vig=1.0):
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
        niveles.append(dict(cols=cols, viga=dict(A=vb * vh, I=fi_vig * vb * vh ** 3 / 12, E=E, b=vb, h=vh), sin=set(),
                            vi=0.0, vd=0.0, q=q))
    return dict(xs=xs, alturas=list(alturas), niveles=niveles, E=E)


def geo_modelo(modelo, datos, cargas_txt):
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


def resolver(spans, alturas, col_secc, vig_secc, E, cargas, F_sismo, base='Empotrada', fi_col=1.0, fi_vig=1.0):
    """Pórtico regular de captura rápida (ver geo_simple)."""
    return resolver_geo(geo_simple(spans, alturas, col_secc, vig_secc, E, cargas, base, fi_col, fi_vig), F_sismo)


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
def distribuir_lechos(n, db, b, rec, est_db):
    """Reparte n barras de diámetro db (cm) en lechos dentro del ancho b. Separación libre mínima = máx(db, 2.5 cm) entre
    barras y entre lechos. Devuelve capas (barras por lecho), r al centroide (cm), separación libre del primer lecho,
    y si cabe en 3 lechos como máximo."""
    sep_min = max(db, 2.5)
    util = b - 2 * (rec + est_db)
    nmax = max(1, int((util + sep_min) // (db + sep_min)))
    capas, resto = [], n
    while resto > 0:
        capas.append(min(nmax, resto))
        resto -= capas[-1]
    ys = [rec + est_db + db / 2 + i * (db + sep_min) for i in range(len(capas))]
    r = sum(c * y for c, y in zip(capas, ys)) / n
    sep1 = (util - capas[0] * db) / (capas[0] - 1) if capas[0] > 1 else util
    return dict(capas=capas, r=r, sep_libre=sep1, sep_min=sep_min, nmax=nmax, cabe=len(capas) <= 3)


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


def armar_seccion(Mu_tm, b, h, fc, fy, varilla, rec=4.0, est_db=0.95, signo=''):
    """Dimensiona una sección con la varilla elegida: calcula As con la d real del acero en lechos (se itera) y propone el
    número de barras. Devuelve dict con As, lineas, armado (n, varilla, capas, As real, MR, cabe, r, d)."""
    db, a_bar = DIAMETROS[varilla], p3.VARILLAS[varilla]
    r = rec + est_db + db / 2
    for _ in range(8):
        As, lin, ok, lim = desarrollo_flexion(Mu_tm, b, h, r, fc, fy, signo)
        if As is None:
            return dict(As=None, lineas=lin, ok=False, lim=lim, armado=None, Mu=Mu_tm)
        n = max(2, math.ceil(As / a_bar - 1e-9))
        lech = distribuir_lechos(n, db, b, rec, est_db)
        if abs(lech['r'] - r) < 0.02:
            break
        r = lech['r']
    n_ini = n
    for _ in range(30):                           # sube barras mientras el MR con la d real no alcance
        lech = distribuir_lechos(n, db, b, rec, est_db)
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
    lin.append(f'Lechos: {" + ".join(map(str, lech["capas"]))} barras (caben {lech["nmax"]} por lecho; separación libre '
               f'mínima {lech["sep_min"]:.2f} cm; en el primer lecho queda {lech["sep_libre"]:.2f} cm) → r al centroide = {lech["r"]:.2f} cm, '
               f'd = {h - lech["r"]:.2f} cm')
    lin.append(f'As real = {n}{varilla} = {As_real:.2f} cm²; MR = {MR:.2f} t·m {"≥" if MR + 1e-9 >= Mu_tm else "<"} Mu = {Mu_tm:.2f} t·m')
    if not lech['cabe']:
        lin.append('No caben en 3 lechos: aumenta el ancho, el peralte o usa una varilla mayor.')
    if As_real > lim2['As_max']:
        lin.append(f'As real > As,máx = {lim2["As_max"]:.2f} cm²: cambia la sección.')
    return dict(As=As, lineas=lin, ok=ok_f, lim=lim2, Mu=Mu_tm,
                armado=dict(armado=f'{n}{varilla}', n=n, varilla=varilla, capas=lech['capas'], As=As_real, MR=MR, cabe=lech['cabe'],
                            sep_libre=lech['sep_libre'], sep_min=lech['sep_min'], r=lech['r'], d=h - lech['r']))


def disenar_viga(env, clave, b, h, fc, fy, cara_i, cara_d, sup_var, inf_var, est_var='#3', rec=4.0, fy_est=4200.0,
                 lambda_=1.0):
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
            s = armar_seccion(Mu, b, h, fc, fy, var, rec, est_db, '−' if lecho == 'superior' else '+')
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


def resumen_vigas(env, res, fc, fy, elecciones, est_var='#3', rec=4.0):
    """Diseña todas las vigas. elecciones[clave] = (varilla superior, varilla inferior); si falta, la sugerida por h/10."""
    out = {}
    geo = res['geo']
    for clave, e in env.items():
        if clave[0] != 'b':
            continue
        b, h = e['b'], e['h']
        sup, inf = elecciones.get(clave, (varilla_sugerida(h)[0],) * 2)
        ci, cd = caras_columna(geo, clave)
        out[clave] = disenar_viga(env, clave, b, h, fc, fy, ci, cd, sup, inf, est_var, rec)
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
                    ancho=None, alto=None):
    """Marco con el diagrama de M o V de todos los elementos (o de un piso). extra: diagrama mínimo para superponer.
    etiquetas: 'todas', 'extremos' (extremos y valor máximo) o 'ninguna'."""
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
    for d in todos:
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
                    im = int(np.argmax(np.abs(val[3:-3]))) + 3
                    if etiquetas == 'todas':
                        ids.update({i: 'center' for i in range(5, len(val) - 5, 5)})
                    ids[im] = 'center'
                    for i, ha in ids.items():
                        ax.annotate(f'{val[i]:.2f}', (px[i], py[i]),
                                    xytext=(5 if ha == 'left' else (-5 if ha == 'right' else 0), -10 if val[i] >= 0 else 10),
                                    textcoords='offset points', fontsize=8, ha=ha, va='center', bbox=box, zorder=6)
            else:
                e = clave[2]
                pz = v['p']
                px0 = np.full_like(pz, xs[e - 1])
                px = px0 + val * sc
                ax.fill_betweenx(pz, px0, px, where=val >= 0, color=cl[0], alpha=0.85, lw=0, interpolate=True, zorder=2)
                ax.fill_betweenx(pz, px0, px, where=val <= 0, color=cl[1], alpha=0.85, lw=0, interpolate=True, zorder=2)
                ax.plot(px, pz, color='#333', lw=0.7, zorder=3)
                if etiquetas != 'ninguna':
                    for i, dy in ((0, 10), (len(val) - 1, -10)):
                        lado = 1 if val[i] >= 0 else -1
                        ax.annotate(f'{val[i]:.2f}', (px[i], pz[i]), xytext=(10 * lado, dy), textcoords='offset points',
                                    fontsize=8, ha='left' if lado > 0 else 'right', va='center', bbox=box, zorder=6)
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


def figura_secciones(dis):
    """Cortes de la viga en apoyo izquierdo, centro y apoyo derecho con las barras elegidas, los estribos y las
    separaciones libres; las barras que no cumplen la separación o que no caben se marcan en rojo."""
    plt = _plt()
    from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
    b, h, rec = dis['b'], dis['h'], dis['rec']
    est_db = DIAMETROS[dis['est_var']]
    fig, axs = plt.subplots(1, 3, figsize=(15, max(6.5, 15 / 3 * (h + 20) / (b + 12) + 1.5)))
    for ax, pn in zip(axs, ('izquierdo', 'centro', 'derecho')):
        ax.add_patch(Rectangle((0, 0), b, h, fc='#E8EEF6', ec='#13315C', lw=2))
        ax.add_patch(FancyBboxPatch((rec, rec), b - 2 * rec, h - 2 * rec, boxstyle='round,pad=0,rounding_size=1.5', fc='none',
                                    ec='#3E7CB1', lw=2.2))
        malo = False
        textos = []
        for lecho in ('superior', 'inferior'):
            s = next(s for s in dis['secciones'] if s['posicion'] == pn and s['lecho'] == lecho)
            a = s['armado']
            if not a:
                textos.append((f'{lecho}: sin solución', '#B00020'))
                continue
            db = DIAMETROS[a['varilla']]
            sep_min = max(db, 2.5)
            util = b - 2 * (rec + est_db)
            y = rec + est_db + db / 2
            restan = a['n']
            for capa in a['capas']:
                m_c = min(capa, restan)
                restan -= m_c
                xs_ = [b / 2] if m_c == 1 else list(np.linspace(rec + est_db + db / 2, b - rec - est_db - db / 2, m_c))
                sep = (xs_[1] - xs_[0] - db) if m_c > 1 else util
                if sep + 1e-6 < sep_min:
                    malo = True
                yy = y if lecho == 'inferior' else h - y
                for xx in xs_:
                    ax.add_patch(Circle((xx, yy), db / 2, fc='#D63B3B' if sep + 1e-6 < sep_min else '#13315C', ec='black', lw=0.8,
                                        zorder=5))
                y += db + sep_min
            col = '#B00020' if (not a['cabe'] or not s['ok']) else '#13315C'
            textos.append((f'{"Sup." if lecho == "superior" else "Inf."}: {a["armado"]} ({" + ".join(map(str, a["capas"]))} por lecho)\n'
                           f'As = {a["As"]:.2f} cm² · MR = {a["MR"]:.2f} / Mu = {s["Mu"]:.2f} t·m', col))
        xpos = next(s for s in dis['secciones'] if s['posicion'] == pn)['x']
        ax.text(b / 2, h + 4.5, f'{pn.capitalize()} · x = {xpos:.2f} m', ha='center', fontsize=11, fontweight='bold', color='#0B2545')
        for i, (tx, c) in enumerate(textos):
            ax.text(b / 2, -3 - 5.5 * i, tx, ha='center', va='top', fontsize=8.2, color=c)
        ax.text(b / 2, h + 1, f'b = {b:g} cm', ha='center', fontsize=8, color='#555')
        ax.text(-1.5, h / 2, f'h = {h:g} cm', rotation=90, va='center', ha='right', fontsize=8, color='#555')
        ax.set_xlim(-8, b + 8)
        ax.set_ylim(-3 - 5.5 * 2 - 6, h + 8)
        ax.set_aspect('equal')
        ax.axis('off')
        if malo:
            ax.text(b / 2, h / 2, 'SEPARACIÓN\nINSUFICIENTE', ha='center', va='center', fontsize=12, color='#B00020', alpha=0.4,
                    fontweight='bold')
    fig.suptitle(f'Secciones de la viga {b:g}×{h:g} cm · rec. {rec:g} cm · estribo {dis["est_var"]} @ '
                 f'{dis["cortante"].get("s_usar", "—")} cm ({dis["cortante"]["ramas"]} ramas)', fontsize=12, color='#0B2545')
    fig.tight_layout()
    return fig
