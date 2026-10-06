"""
Análisis Modal Espectral (AME) - marcos de cortante
====================================================
Equivalente al script de MATLAB, con captura de datos flexible y exportación a Excel.

Uso:
    python ame_app.py                           -> abre la interfaz gráfica (recomendado)
    python ame_sismico.py proyecto.json         -> calcula sin interfaz y exporta a Excel
    python ame_sismico.py proyecto.json -o salida.xlsx

A partir de secciones, alturas, cargas, Q, k1 y tipo de edificación arma K y M y hace el AME.
(El proyecto también puede traer K y M escritas a mano: ver preparar_datos.)
Unidades: cm, kg, s  (a0, c y Sa en cm/s^2; cargas en t/m, longitudes y alturas en m).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

G = 981.0  # cm/s^2
AZULES = ['#0B2545', '#3E7CB1', '#1D4E89', '#81A4CD', '#13315C']

# ----------------------------------------------------------------------------
# Lectura flexible de datos
# ----------------------------------------------------------------------------
_NUM = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')
_MILES = re.compile(r'(?<![\d.])\d{1,3}(?:,\d{3})+(?=\.\d|\s|$|;|\])')


def parse_filas(texto):
    """Devuelve lista de filas (listas de float). Filas separadas por salto de línea o ';'."""
    t = texto.replace('−', '-').replace('[', ' ').replace(']', ' ')
    t = _MILES.sub(lambda m: m.group().replace(',', ''), t)
    filas = []
    for linea in re.split(r'[\n;]', t):
        nums = [float(x) for x in _NUM.findall(linea)]
        if nums:
            filas.append(nums)
    return filas


def parse_matriz(texto, nombre):
    filas = parse_filas(texto)
    if not filas:
        raise ValueError(f'La matriz {nombre} está vacía.')
    if len({len(f) for f in filas}) != 1:
        raise ValueError(f'La matriz {nombre} no es rectangular (filas con distinto número de datos).')
    return np.array(filas, dtype=float)


def parse_vector(texto, nombre):
    filas = parse_filas(texto)
    v = [x for f in filas for x in f]
    if not v:
        raise ValueError(f'Falta el dato: {nombre}.')
    return np.array(v, dtype=float)


def parse_num(texto, nombre):
    try:
        return float(str(texto).strip().replace(',', '.'))
    except ValueError:
        raise ValueError(f'Valor numérico no válido en "{nombre}": {texto!r}')


def preparar_datos(d):
    """Convierte el diccionario de textos (GUI/JSON) a datos numéricos validados."""
    K = parse_matriz(d['K'], 'K')
    if K.shape[0] != K.shape[1]:
        raise ValueError(f'K debe ser cuadrada (es {K.shape[0]}x{K.shape[1]}).')
    n = K.shape[0]
    M = parse_matriz(d['M'], 'M')
    if M.shape == (n, n):
        pass
    elif M.size == n and 1 in M.shape:
        M = np.diag(M.ravel())
    else:
        raise ValueError(f'M debe ser {n}x{n} o un vector de {n} masas (es {M.shape[0]}x{M.shape[1]}).')

    niv = parse_filas(d['niveles'])
    if len(niv) != n:
        raise ValueError(f'En "Niveles" hay {len(niv)} filas y K es de {n} grados de libertad.')
    alturas = np.array([f[0] for f in niv])
    if all(len(f) >= 3 for f in niv):
        cargas = np.array([f[1] for f in niv])
        dist_x = np.array([f[2] for f in niv])
    else:
        cargas = dist_x = None  # sin estático

    grupo = d['grupo'].strip().upper()[:1]
    sismo = d['sismo'].strip().upper()[:1]
    if grupo not in ('A', 'B'):
        raise ValueError('El grupo de estructura debe ser A o B.')
    if grupo == 'A' and sismo not in ('B', 'I'):
        raise ValueError('Para el grupo A indica el sismo: B (base) o I (infrecuente).')

    p = dict(M=M, K=K, n=n, alturas=alturas, cargas=cargas, dist_x=dist_x,
             grupo=grupo, sismo=sismo, **_params_espectro(d))
    if cargas is not None:  # el estático usa el peso Wi = carga * longitud
        p['cargas'], p['dist_x'] = cargas * dist_x, np.ones(n)
    return p


def _params_espectro(d):
    return dict(Q=parse_vector(d['Q'], 'Q'),
                k1=parse_num(d['k1'], 'k1'), a0=parse_num(d['a0'], 'a0'), c=parse_num(d['c'], 'c'),
                Ta=parse_num(d['Ta'], 'Ta'), Tb=parse_num(d['Tb'], 'Tb'), k=parse_num(d['k'], 'k'),
                Ts=parse_num(d['Ts'], 'Ts'),
                factor_Fu=parse_num(d.get('factor_Fu', '1.1') or '1.1', 'Factor Fu'),
                mult_V=parse_num(d.get('mult_V', '1') or '1', 'Multiplicador V estático'))


def grupo_sismo(tipo):
    """'Grupo B' | 'Grupo A - sismo base' | 'Grupo A - sismo infrecuente' -> (grupo, sismo)."""
    t = tipo.lower()
    if 'grupo a' in t:
        return 'A', ('I' if 'infrec' in t else 'B')
    return 'B', 'B'


def preparar_estructura(d):
    """Arma K, M y alturas a partir de secciones de columnas, niveles y cargas (marco de cortante).

    d['niveles'] = lista (de abajo hacia arriba) de dicts con: h (m), ncol, b (cm), hs (cm, en la
    dirección del sismo), nart (columnas articuladas en la base), w (t/m), L (m), W (t, opcional).
    """
    niv = d['niveles']
    n = len(niv)
    if n < 1:
        raise ValueError('Agrega al menos un nivel.')
    if str(d.get('E', '')).strip():
        E = parse_num(d['E'], 'E')
    elif str(d.get('fc', '')).strip():
        E = 14000 * np.sqrt(parse_num(d['fc'], "f'c"))
    else:
        raise ValueError("Indica E o f'c del concreto.")

    filas, ks, m, W, alturas = [], [], [], [], []
    for i, r in enumerate(niv, 1):
        def num(clave, et, defecto=None):
            txt = str(r.get(clave, '')).strip()
            if not txt and defecto is not None:
                return defecto
            return parse_num(txt, f'Nivel {i} - {et}')
        h = num('h', 'h entrepiso')
        ncol, b, hs = int(num('ncol', 'Nº columnas')), num('b', 'b'), num('hs', 'h sección')
        nart = int(num('nart', 'Nº articuladas', 0))
        if not 0 <= nart <= ncol:
            raise ValueError(f'Nivel {i}: las columnas articuladas deben estar entre 0 y {ncol}.')
        if str(r.get('W', '')).strip():
            Wi = num('W', 'W')
        else:
            Wi = num('w', 'carga') * num('L', 'longitud')
        I = b * hs ** 3 / 12
        hcm = h * 100
        k_emp, k_art = 12 * E * I / hcm ** 3, 3 * E * I / hcm ** 3
        k = (ncol - nart) * k_emp + nart * k_art
        ks.append(k); alturas.append(h); W.append(Wi); m.append(Wi * 1000 / G)
        filas.append(dict(nivel=i, h=h, ncol=ncol, nart=nart, b=b, hs=hs, I=I, k_emp=k_emp, k_art=k_art,
                          k=k, W=Wi, m=Wi * 1000 / G))

    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = ks[i] + (ks[i + 1] if i + 1 < n else 0)
        if i + 1 < n:
            K[i, i + 1] = K[i + 1, i] = -ks[i + 1]
    grupo, sismo = grupo_sismo(d.get('tipo', 'Grupo B'))
    return dict(M=np.diag(m), K=K, n=n, alturas=np.array(alturas), cargas=np.array(W), dist_x=np.ones(n),
                grupo=grupo, sismo=sismo, E=E, info=filas, **_params_espectro(d))


# ----------------------------------------------------------------------------
# Cálculo
# ----------------------------------------------------------------------------
def _r0(Q):
    return 1.75 if Q < 3 else 2.0


def _fgrupo(p):
    return 0.75 if (p['grupo'] == 'A' and p['sismo'] == 'B') else 1.0


def _k2(T, Ta):
    return np.maximum(0.5 * (1 - np.sqrt(np.asarray(T, float) / Ta)), 0)


def _a_minima(p, Rprima_prom, esc):
    """a_min del espectro; esc=1 -> cm/s^2, esc=1/981 -> g."""
    Ts = p['Ts']
    a1, a2 = 0.04 / Rprima_prom, 0.06 / Rprima_prom
    if Ts < 0.5:
        a = a1
    elif Ts < 1:
        a = a1 + (a2 - a1) * (Ts - 0.5) / 0.5
    else:
        a = a2
    return a * G * esc


def espectro(p, Q, esc, serie):
    """Sa/(Q'R') en la malla 'serie' (esc=1 cm/s^2; esc=1/981 g). Devuelve (curva, a_minima)."""
    a0, c, Ta, Tb, k = p['a0'] * esc, p['c'] * esc, p['Ta'], p['Tb'], p['k']
    Sa = np.zeros_like(serie)
    q = np.zeros_like(serie)
    i1 = serie < Ta
    i2 = (serie >= Ta) & (serie < Tb)
    i3 = serie >= Tb
    Sa[i1] = a0 + (c - a0) * serie[i1] / Ta
    q[i1] = 1 + (Q - 1) * np.sqrt(1 / k) * serie[i1] / Ta
    Sa[i2] = c
    q[i2] = 1 + (Q - 1) * np.sqrt(1 / k)
    pp = k + (1 - k) * (Tb / serie[i3]) ** 2
    Sa[i3] = c * pp * (Tb / serie[i3]) ** 2
    q[i3] = 1 + (Q - 1) * np.sqrt(pp / k)

    k2 = _k2(serie, Ta)
    f = _fgrupo(p)
    # R' promedio (igual que el script original: promedio sobre toda la malla, con r0 de Q(1))
    Rprom = f * np.mean(p['k1'] * _r0(p['Q'][0]) + k2)
    a_min = _a_minima(p, Rprom, esc)
    r = f * (p['k1'] * _r0(Q) + k2)

    curva = Sa / (q * r)
    idx = np.flatnonzero(curva <= a_min)
    if idx.size:
        curva[idx[0]:] = a_min
    return curva, a_min


def _interp(x, xp, fp):
    """Interpolación lineal con extrapolación lineal."""
    x = np.atleast_1d(x)
    y = np.interp(x, xp, fp)
    lo, hi = x < xp[0], x > xp[-1]
    if lo.any():
        y[lo] = fp[0] + (fp[1] - fp[0]) / (xp[1] - xp[0]) * (x[lo] - xp[0])
    if hi.any():
        y[hi] = fp[-1] + (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]) * (x[hi] - xp[-1])
    return y


def calcular(p):
    M, K, n = p['M'], p['K'], p['n']
    res = dict(p=p)

    # --- Modal ---
    A = np.linalg.solve(M, K)
    lam, fi = np.linalg.eig(A)
    lam, fi = lam.real, fi.real
    o = np.argsort(lam)
    lam, fi = lam[o], fi[:, o]
    fi = fi / fi[0, :]
    w = np.sqrt(lam)
    f = w / (2 * np.pi)
    T = 1 / f
    uno = np.ones(n)
    gam = np.array([(fi[:, j] @ M @ uno) / (fi[:, j] @ M @ fi[:, j]) for j in range(n)])
    res.update(A=A, lam=lam, fi=fi, w=w, f=f, T=T, gam=gam, alturas_acum=np.cumsum(p['alturas']))

    # --- Espectro de diseño (dinámico: cm/s^2) ---
    serie = np.arange(0, 8.0005, 0.001)
    Q = p['Q']
    curvas = np.zeros((len(Q), serie.size))
    curvas_g = np.zeros_like(curvas)
    Sa_modal = np.zeros((n, len(Q)))
    a_min = a_min_g = None
    for i, q in enumerate(Q):
        curvas[i], a_min = espectro(p, q, 1.0, serie)
        curvas_g[i], a_min_g = espectro(p, q, 1 / G, serie)
        Sa_modal[:, i] = _interp(T, serie, curvas[i])
    res.update(serie=serie, curvas=curvas, curvas_g=curvas_g, Sa_modal=Sa_modal,
               a_min=a_min, a_min_g=a_min_g)

    # --- Desplazamientos y fuerzas modales (con la primera Q) ---
    d = np.zeros((n, n))
    F = np.zeros((n, n))
    for j in range(n):
        d[:, j] = fi[:, j] * (Sa_modal[j, 0] * gam[j] / lam[j])
        F[:, j] = K @ d[:, j]
    F_final = np.sqrt((F ** 2).sum(axis=1))
    res.update(d=d, F=F, F_final=F_final)

    # --- Tabla de fuerzas por nivel (dinámico) ---
    h_desc = res['alturas_acum'][::-1]
    F_desc = F_final[::-1] / 1000
    Fu = F_desc * p['factor_Fu']
    Vu = np.cumsum(Fu)
    Mvu = h_desc * Fu
    res['tabla_din'] = dict(nivel=np.arange(n, 0, -1), h=h_desc, F=F_desc, Fu=Fu, Vu=Vu, Mvu=Mvu)
    res['V_din'] = Vu[-1]

    # --- R' y QR' por modo ---
    filas = []
    for j in range(n):
        k2 = float(_k2(T[j], p['Ta']))
        for q in Q:
            Rp = _fgrupo(p) * (p['k1'] * _r0(q) + k2)
            filas.append((j + 1, T[j], q, k2, Rp, q * Rp))
    res['QR'] = filas
    res['modo_Tmax'] = int(np.argmax(T)) + 1

    # --- Estático ---
    if p['cargas'] is not None:
        h_inv = res['alturas_acum'][::-1]
        Wi = (p['cargas'] * p['dist_x'])[::-1]
        hWi = h_inv * Wi
        Fmax = float(curvas_g[0].max())
        Fi = Fmax * (h_inv * Wi / hWi.sum()) * Wi.sum()
        Fi11 = Fi * p['factor_Fu']
        Vacum = np.cumsum(Fi11)
        Mvu_e = Fi11 * h_inv
        res['tabla_est'] = dict(nivel=np.arange(n, 0, -1), h=h_inv, Wi=Wi, hWi=hWi, Fi=Fi,
                                Fu=Fi11, Vu=Vacum, Mvu=Mvu_e)
        res['Fmax_est'] = Fmax
        res['V_est'] = Vacum[-1]
        res['factor_escala'] = Vacum[-1] * p['mult_V'] / res['V_din']
    return res


# ----------------------------------------------------------------------------
# Reporte en texto
# ----------------------------------------------------------------------------
def _tabla_txt(headers, filas, fmt='{:>14.6f}'):
    w = max(14, max(len(h) for h in headers) + 2)
    out = ''.join(f'{h:>{w}}' for h in headers) + '\n'
    for fila in filas:
        out += ''.join(f'{v:>{w}d}' if isinstance(v, (int, np.integer)) else f'{v:>{w}.5f}' for v in fila) + '\n'
    return out


def reporte_texto(res):
    p, n = res['p'], res['p']['n']
    np.set_printoptions(linewidth=200, suppress=True, precision=5)
    s = []
    if 'info' in p:
        s.append('=== MODELO ESTRUCTURAL ===')
        s.append(f'E = {p["E"]:,.0f} kg/cm²\n')
        s.append(_tabla_txt(['Nivel', 'h(m)', 'Nº col', 'Nº art', 'b(cm)', 'h(cm)', 'I(cm4)', 'K nivel', 'W(t)', 'm'],
                            [[r['nivel'], r['h'], r['ncol'], r['nart'], r['b'], r['hs'], r['I'], r['k'], r['W'], r['m']]
                             for r in p['info']]))
        s.append('Matriz K [kg/cm]:')
        for i in range(n):
            s.append('  ' + ''.join(f'{x:>14.2f}' for x in p['K'][i]))
        s.append('Matriz M [kg·s²/cm]:')
        for i in range(n):
            s.append('  ' + ''.join(f'{x:>14.4f}' for x in p['M'][i]))
        s.append('')
    s.append('=== RESULTADOS MODALES ===\n')
    s.append('Modo            ' + ''.join(f'{j + 1:>14d}' for j in range(n)))
    for nombre, v in (('λ [rad/s]²', res['lam']), ('ω [rad/s]', res['w']), ('f [Hz]', res['f']),
                      ('T [s]', res['T']), ('γ (partic.)', res['gam'])):
        s.append(f'{nombre:<16}' + ''.join(f'{x:>14.5f}' for x in v))
    s.append(f'Σγ = {res["gam"].sum():.5f}')
    s.append('\nEigenvectores normalizados (columnas = modos):')
    for i in range(n):
        s.append('  ' + ''.join(f'{x:>12.5f}' for x in res['fi'][i]))
    s.append('\nAceleraciones espectrales Sa/(Q\'R\') [cm/s²]:')
    s.append(_tabla_txt(['Modo', 'T(s)'] + [f'Q={q:g}' for q in p['Q']],
                        [[j + 1, res['T'][j]] + list(res['Sa_modal'][j]) for j in range(n)]))
    s.append(f'a_mínima = {res["a_min"]:.6f} cm/s²  ({res["a_min_g"]:.6f} g)   Ts = {p["Ts"]:g}')
    s.append('\nDesplazamientos modales δ [cm] (columnas = modos):')
    for i in range(n):
        s.append('  ' + ''.join(f'{x:>14.6f}' for x in res['d'][i]))
    s.append('\nFuerzas modales F [kg] (columnas = modos):')
    for i in range(n):
        s.append('  ' + ''.join(f'{x:>14.4f}' for x in res['F'][i]))
    s.append('\nRaíz de la suma de cuadrados de F (SRSS) [kg]:')
    s.append('  ' + '  '.join(f'{x:.4f}' for x in res['F_final']))
    s.append('\n=== R\' y Q·R\' POR MODO ===')
    s.append(_tabla_txt(['Modo', 'T(s)', 'Q', 'k2', "R'", "Q·R'"], res['QR']))
    t = res['tabla_din']
    s.append('=== TABLA DE FUERZAS POR NIVEL (DINÁMICO) ===')
    s.append(_tabla_txt(['Nivel', 'h(m)', 'F(t)', 'Fu(t)', 'Vu(t)', 'Mvu(t-m)'],
                        zip(t['nivel'], t['h'], t['F'], t['Fu'], t['Vu'], t['Mvu'])))
    s.append(f'Σ Mvu = {t["Mvu"].sum():.6f} t-m')
    if 'tabla_est' in res:
        e = res['tabla_est']
        s.append('\n=== ESPECTRO DE DISEÑO ESTÁTICO ===')
        s.append(_tabla_txt(['Nivel', 'h(m)', 'Wi(t)', 'Wihi(t-m)', 'Fi(t)', 'Fu(t)', 'Vu(t)', 'Mvu(t-m)'],
                            zip(e['nivel'], e['h'], e['Wi'], e['hWi'], e['Fi'], e['Fu'], e['Vu'], e['Mvu'])))
        s.append(f'ΣWi = {e["Wi"].sum():.6f}   ΣWihi = {e["hWi"].sum():.6f}   ΣMvu = {e["Mvu"].sum():.6f}')
        s.append(f'\nValor máximo del espectro estático = {res["Fmax_est"]:.8f} g')
        s.append(f'V estático  = {res["V_est"]:.6f} t' + (f'  (× {p["mult_V"]:g} = {res["V_est"] * p["mult_V"]:.6f})'
                                                       if p['mult_V'] != 1 else ''))
        s.append(f'V dinámico  = {res["V_din"]:.6f} t')
        s.append(f'FACTOR DE ESCALA = {res["factor_escala"]:.6f}'
                 + ('   (V dinámico ya cumple, factor < 1)' if res['factor_escala'] < 1 else ''))
    else:
        s.append('\n(Sin cargas/longitudes: no se calculó el espectro estático ni el factor de escala.)')
    return '\n'.join(s)


# ----------------------------------------------------------------------------
# Gráficas (Figure puro, sirve para GUI y para exportar)
# ----------------------------------------------------------------------------
def fig_modal(res):
    n = res['p']['n']
    fig = Figure(figsize=(max(6, 2.6 * n), 5.2), dpi=100)
    y = np.concatenate([[0], res['alturas_acum']])
    lim = 1.2 * np.abs(res['fi']).max()
    for j in range(n):
        ax = fig.add_subplot(1, n, j + 1)
        x = np.concatenate([[0], res['fi'][:, j]])
        ax.axvline(0, color='0.7', lw=1)
        ax.plot(x, y, '-o', color='#1f4e79', lw=1.8, ms=5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(0, y.max() * 1.05)
        ax.set_title(f'Modo {j + 1}\nT = {res["T"][j]:.4f} s', fontsize=10)
        ax.set_xlabel('Φ')
        if j == 0:
            ax.set_ylabel('Altura (m)')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_espectro(res):
    p = res['p']
    fig = Figure(figsize=(8, 7), dpi=100)
    for k, (clave, unidad, Tcol) in enumerate((('curvas', 'cm/s²', 1.0), ('curvas_g', 'g', 1 / G))):
        ax = fig.add_subplot(2, 1, k + 1)
        for i, q in enumerate(p['Q']):
            ln, = ax.plot(res['serie'], res[clave][i], lw=2, color=AZULES[i % len(AZULES)], label=f'Q = {q:g}')
            ax.plot(res['T'], _interp(res['T'], res['serie'], res[clave][i]), 'o', color=ln.get_color())
        ax.set_xlabel('T (s)')
        ax.set_ylabel(f"Sa / (Q'R')  [{unidad}]")
        ax.set_title('Espectro de diseño ' + ('(dinámico)' if k == 0 else '(estático)'))
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right')
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Exportar a Excel
# ----------------------------------------------------------------------------
def exportar_excel(res, ruta):
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Alignment, Font, PatternFill
    import io

    p, n = res['p'], res['p']['n']
    wb = Workbook()
    negrita = Font(bold=True)
    relleno = PatternFill('solid', fgColor='DDEBF7')

    def hoja(nombre):
        ws = wb.create_sheet(nombre)
        ws._fila = 1
        return ws

    def titulo(ws, txt):
        ws.cell(ws._fila, 1, txt).font = Font(bold=True, size=12)
        ws._fila += 1

    def tabla(ws, headers, filas, fmt='0.0000'):
        for j, h in enumerate(headers):
            c = ws.cell(ws._fila, 1 + j, h)
            c.font, c.fill = negrita, relleno
            c.alignment = Alignment(horizontal='center')
        ws._fila += 1
        for fila in filas:
            for j, v in enumerate(fila):
                v = v.item() if isinstance(v, np.generic) else v
                c = ws.cell(ws._fila, 1 + j, v)
                if isinstance(v, float):
                    c.number_format = fmt
            ws._fila += 1
        ws._fila += 1

    def ancho(ws, w=16):
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = w

    wb.remove(wb.active)

    # Datos
    ws = hoja('Datos')
    titulo(ws, 'Parámetros')
    tabla(ws, ['Parámetro', 'Valor'], [
        ['Grupo', p['grupo']], ['Sismo', p['sismo'] if p['grupo'] == 'A' else '-'],
        ['k1', p['k1']], ['Q', ', '.join(f'{q:g}' for q in p['Q'])],
        ['a0 [cm/s²]', p['a0']], ['c [cm/s²]', p['c']], ['Ta [s]', p['Ta']], ['Tb [s]', p['Tb']],
        ['k', p['k']], ['Ts [s]', p['Ts']], ['Factor Fu', p['factor_Fu']],
        ['Multiplicador V estático', p['mult_V']]], '0.######')
    titulo(ws, 'Matriz K [kg/cm]')
    tabla(ws, [f'{j + 1}' for j in range(n)], p['K'].tolist(), '#,##0.0000')
    titulo(ws, 'Matriz M [kg s²/cm]')
    tabla(ws, [f'{j + 1}' for j in range(n)], p['M'].tolist(), '0.0000')
    if 'info' in p:
        titulo(ws, f'Modelo estructural (E = {p["E"]:,.0f} kg/cm²)')
        tabla(ws, ['Nivel', 'h (m)', 'Nº col.', 'Nº art.', 'b (cm)', 'h sec. (cm)', 'I (cm4)', 'K nivel (kg/cm)',
                   'W (t)', 'm (kg s²/cm)'],
              [[r['nivel'], r['h'], r['ncol'], r['nart'], r['b'], r['hs'], r['I'], r['k'], r['W'], r['m']]
               for r in p['info']], '#,##0.0000')
    else:
        titulo(ws, 'Niveles (1 = planta baja)')
        tabla(ws, ['Nivel', 'h entrepiso (m)'] + (['Peso Wi (t)'] if p['cargas'] is not None else []),
              [[i + 1, p['alturas'][i]] + ([p['cargas'][i]] if p['cargas'] is not None else []) for i in range(n)])
    ancho(ws, 20)

    # Modal
    ws = hoja('Modal')
    titulo(ws, 'Propiedades modales')
    tabla(ws, ['Modo', 'λ (rad/s)²', 'ω (rad/s)', 'f (Hz)', 'T (s)', 'γ'],
          [[j + 1, res['lam'][j], res['w'][j], res['f'][j], res['T'][j], res['gam'][j]] for j in range(n)],
          '0.00000')
    titulo(ws, 'Eigenvectores normalizados (columnas = modos)')
    tabla(ws, [f'Modo {j + 1}' for j in range(n)], res['fi'].tolist(), '0.00000')
    ancho(ws)

    # Espectro
    ws = hoja('Espectro')
    titulo(ws, "Sa/(Q'R') por modo [cm/s²]")
    tabla(ws, ['Modo', 'T (s)'] + [f'Q = {q:g}' for q in p['Q']],
          [[j + 1, res['T'][j]] + list(res['Sa_modal'][j]) for j in range(n)], '0.0000')
    tabla(ws, ['a_mínima [cm/s²]', 'a_mínima [g]'], [[res['a_min'], res['a_min_g']]], '0.000000')
    titulo(ws, "R' y Q·R' por modo")
    tabla(ws, ['Modo', 'T (s)', 'Q', 'k2', "R'", "Q·R'"], res['QR'], '0.00000')
    ancho(ws)

    # Fuerzas dinámicas
    ws = hoja('Dinámico')
    titulo(ws, 'Desplazamientos modales δ [cm] (columnas = modos)')
    tabla(ws, [f'Modo {j + 1}' for j in range(n)], res['d'].tolist(), '0.000000')
    titulo(ws, 'Fuerzas modales F [kg] (columnas = modos)')
    tabla(ws, [f'Modo {j + 1}' for j in range(n)], res['F'].tolist(), '#,##0.00')
    titulo(ws, 'Fuerzas por nivel')
    t = res['tabla_din']
    tabla(ws, ['Nivel', 'h (m)', 'F (t)', 'Fu (t)', 'Vu (t)', 'Mvu (t-m)'],
          zip(t['nivel'], t['h'], t['F'], t['Fu'], t['Vu'], t['Mvu']), '0.0000')
    tabla(ws, ['Σ Mvu (t-m)'], [[float(t['Mvu'].sum())]], '0.0000')
    ancho(ws)

    # Estático y factor de escala
    if 'tabla_est' in res:
        ws = hoja('Estático')
        e = res['tabla_est']
        titulo(ws, 'Espectro de diseño estático')
        tabla(ws, ['Nivel', 'h (m)', 'Wi (t)', 'Wihi (t-m)', 'Fi (t)', 'Fu (t)', 'Vu (t)', 'Mvu (t-m)'],
              zip(e['nivel'], e['h'], e['Wi'], e['hWi'], e['Fi'], e['Fu'], e['Vu'], e['Mvu']), '0.0000')
        titulo(ws, 'Factor de escala')
        tabla(ws, ['Concepto', 'Valor'], [
            ['Sa/(Q\'R\') máximo estático [g]', res['Fmax_est']],
            ['ΣWi (t)', float(e['Wi'].sum())], ['ΣWihi (t-m)', float(e['hWi'].sum())],
            ['ΣMvu (t-m)', float(e['Mvu'].sum())],
            ['V estático (t)', res['V_est']], ['Multiplicador', p['mult_V']],
            ['V dinámico (t)', res['V_din']], ['FACTOR DE ESCALA', res['factor_escala']]], '0.000000')
        ancho(ws, 30)

    # Gráficas
    ws = hoja('Gráficas')
    fila = 1
    for fig in (fig_modal(res), fig_espectro(res)):
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = XLImage(buf)
        img.width, img.height = img.width * 0.7, img.height * 0.7
        ws.add_image(img, f'A{fila}')
        fila += int(img.height / 20) + 3

    wb.save(ruta)


# ----------------------------------------------------------------------------
# Ejemplo (el de las fotos)
# ----------------------------------------------------------------------------
# Ejemplo (el de las fotos)
# ----------------------------------------------------------------------------
EJEMPLO = dict(
    niveles=[dict(h='4.5', ncol='2', b='25', hs='50', nart='1', w='2.0', L='6', W=''),
             dict(h='3.7', ncol='2', b='25', hs='45', nart='0', w='2.0', L='6', W=''),
             dict(h='5.2', ncol='2', b='20', hs='40', nart='0', w='1.5', L='6', W='')],
    E='158000', fc='', tipo='Grupo B', Q='4', k1='1.0',
    a0='153', c='636', Ta='0.6', Tb='1.8', k='0.505', Ts='0.9', factor_Fu='1.1', mult_V='1')


def main():
    args = sys.argv[1:]
    if not args:
        from ame_app import gui
        gui()
        return
    entrada = Path(args[0])
    salida = Path(args[args.index('-o') + 1]) if '-o' in args else entrada.with_suffix('.xlsx')
    datos = json.loads(entrada.read_text(encoding='utf-8'))
    res = calcular(preparar_estructura(datos) if isinstance(datos.get('niveles'), list) else preparar_datos(datos))
    print(reporte_texto(res))
    exportar_excel(res, salida)
    print(f'\nExportado: {salida}')


if __name__ == '__main__':
    main()
