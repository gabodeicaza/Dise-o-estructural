"""Combinación modal, revisiones posteriores (distorsiones, P-Δ, irregularidades) y avisos de coherencia."""
import numpy as np

import ame_ntc as ntc

COMBINACIONES = ('SRSS', 'CQC', 'Suma absoluta')

# Criterios generales (ASCE 7) cuando la norma de cada país no los fija en este programa
THETA_IGNORAR = 0.10
THETA_MAXIMO = 0.25
IRREG_MASA = 1.5
IRREG_RIGIDEZ_1 = 0.70
IRREG_RIGIDEZ_2 = 0.80


def rho_cqc(w, zeta):
    """Matriz de correlación CQC (Der Kiureghian) para frecuencias w y amortiguamiento zeta."""
    n = len(w)
    rho = np.eye(n)
    for j in range(n):
        for k in range(n):
            if j != k:
                r = w[k] / w[j]
                rho[j, k] = (8 * zeta ** 2 * (1 + r) * r ** 1.5) / ((1 - r ** 2) ** 2 + 4 * zeta ** 2 * r * (1 + r) ** 2)
    return rho


def combinar(valores, w, zeta, metodo):
    """Combina respuestas modales. valores: (filas, modos). Devuelve un valor combinado por fila."""
    v = np.asarray(valores, float)
    if metodo == 'Suma absoluta':
        return np.abs(v).sum(axis=1)
    if metodo == 'CQC':
        rho = rho_cqc(np.asarray(w, float), zeta)
        return np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', v, rho, v), 0.0))
    return np.sqrt((v ** 2).sum(axis=1))  # SRSS


def derivas_modales(d):
    """Distorsión de entrepiso por modo (cm): d es (niveles, modos) de abajo hacia arriba."""
    return np.diff(np.vstack([np.zeros(d.shape[1]), d]), axis=0)


def theta_pdelta(W_desde_abajo, deriva_cm, V_t, h_cm):
    """Coeficiente de estabilidad θ = P·δ / (V·h) por entrepiso (abajo hacia arriba).

    P = peso acumulado sobre el entrepiso (t), δ = deriva elástica con las fuerzas de diseño (cm),
    V = cortante de entrepiso con las fuerzas de diseño (t), h = altura del entrepiso (cm).
    """
    P = np.cumsum(np.asarray(W_desde_abajo)[::-1])[::-1]
    return P * deriva_cm / (V_t * h_cm), P


def irregularidades(W, V_t, deriva_cm):
    """Irregularidades de masa y de rigidez (criterios generales). Devuelve listas de niveles (1 = base)."""
    n = len(W)
    masa = []
    for i in range(n):
        # una azotea más ligera que el piso de abajo no cuenta como irregularidad (ni para ella ni para ese piso)
        vecinos = [W[j] for j in (i - 1, i + 1) if 0 <= j < n and not (j == n - 1 and W[j] < W[i])]
        if vecinos and any(W[i] > IRREG_MASA * v for v in vecinos):
            masa.append(i + 1)
    k = V_t / np.maximum(deriva_cm, 1e-12)
    rigidez = []
    for i in range(n - 1):
        arriba = k[i + 1:i + 4]
        if k[i] < IRREG_RIGIDEZ_1 * k[i + 1] or (len(arriba) == 3 and k[i] < IRREG_RIGIDEZ_2 * arriba.mean()):
            rigidez.append(i + 1)
    return masa, rigidez, k


def _estim_periodo(res):
    p = res['p']
    H = float(np.sum(p['alturas']))
    n = p['n']
    return min(n / 10, 0.073 * H ** 0.75), max(n / 10, 0.073 * H ** 0.75)


def avisos(res):
    """Lista de (severidad, texto). Severidad: 'alerta' (probable error o incumplimiento), 'aviso', 'info'."""
    p, n = res['p'], res['p']['n']
    out = []

    def add(sev, txt):
        out.append((sev, txt))

    # --- Espectro y parámetros
    if p['Ta'] >= p['Tb']:
        add('alerta', f'Ta ({p["Ta"]:g} s) debe ser menor que Tb ({p["Tb"]:g} s).')
    if not 0 < p['k'] <= 1:
        add('alerta', f'k = {p["k"]:g} está fuera del rango esperado (0 < k ≤ 1).')
    if p['a0'] > p['c']:
        add('aviso', f'a0 ({p["a0"]:g}) es mayor que c ({p["c"]:g}): normalmente a0 < c. ¿Está en cm/s² (no en g)?')
    if p['c'] < 5:
        add('alerta', f'c = {p["c"]:g}: parece estar en g. El programa espera cm/s² (por ejemplo 975, no 0.99).')
    if p['k1'] <= 0:
        add('alerta', 'k1 debe ser positivo.')
    for q in p['Q']:
        if q not in (1, 1.5, 2, 3, 4):
            add('aviso', f'Q = {q:g} no es uno de los valores usuales (1, 1.5, 2, 3, 4).')
    z = p.get('zeta', 0.05)
    if not 0.005 <= z <= 0.25:
        add('aviso', f'El amortiguamiento ζ = {z:g} es poco usual (normalmente 0.02 a 0.10).')

    # --- Periodo y modos
    T1 = res['T'][0]
    t_min, t_max = _estim_periodo(res)
    if T1 > 2.5 * t_max or T1 < 0.4 * t_min:
        add('aviso', f'T₁ = {T1:.3f} s difiere mucho del estimado empírico ({t_min:.2f}–{t_max:.2f} s). '
                     'Revisa rigideces, masas y unidades.')
    M, fi = p['M'], res['fi']
    m_ef = (fi[:, 0] @ M @ np.ones(n)) ** 2 / (fi[:, 0] @ M @ fi[:, 0]) / np.trace(M)
    if m_ef < 0.6:
        add('aviso', f'El primer modo solo aporta {100 * m_ef:.0f}% de la masa efectiva: la estructura es irregular '
                     'o los modos superiores importan.')

    # --- Matrices dadas a mano
    if p.get('manual'):
        K = p['K']
        fuera = lambda A: np.abs(A - np.diag(np.diag(A))).max() > 0
        if not fuera(K) and fuera(M):
            add('alerta', 'K es diagonal y M no: parece que pegaste las matrices al revés (K en el cuadro de M y M en el '
                          'de K). Los periodos saldrán enormes.')
        if np.abs(K - K.T).max() > 1e-6 * np.abs(K).max():
            add('alerta', 'K no es simétrica: revisa la captura de la matriz de rigidez.')
        if np.linalg.eigvalsh(0.5 * (K + K.T)).min() <= 0:
            add('alerta', 'K no es definida positiva: la estructura sería inestable. Revisa signos y valores.')
        if np.abs(M - np.diag(np.diag(M))).max() > 0:
            add('info', 'M no es diagonal: se usa tal cual (masas acopladas).')

    # --- Geometría y secciones
    h = p['alturas']
    for i, hi in enumerate(h, 1):
        if hi < 2 or hi > 6:
            add('aviso', f'Nivel {i}: la altura de entrepiso ({hi:g} m) es poco usual.')
    geom = p.get('geom')
    if geom:
        for i, l in enumerate(geom['niveles'], 1):
            cols = l['cols']
            if len(cols) == 1:
                add('aviso', f'Nivel {i}: solo tiene una columna; el marco sería muy flexible o inestable.')
            if i == 1 and cols and sum(c['artic'] for c in cols) > len(cols) / 2:
                add('aviso', 'Más de la mitad de las columnas de la base están articuladas.')
            for c in cols:
                lado = min(c['dx'], c['dy'])
                if lado > 0 and l['h'] * 100 / lado > 20:
                    add('aviso', f'Nivel {i}, eje {c["eje"]}: columna esbelta (h/lado = {l["h"] * 100 / lado:.0f} > 20).')
                    break
        if p.get('vigas') == 'flexibles':
            for i, l in enumerate(geom['niveles'], 1):
                Ic = np.mean([c['I'] for c in l['cols']]) / (l['h'] * 100)  # columna promedio
                Ib = l['viga']['I'] / (geom['xs'][-1] * 100 / len(geom['bays']))  # viga con la crujía promedio
                if Ib < 0.3 * Ic:
                    add('aviso', f'Nivel {i}: las vigas son mucho más flexibles que las columnas '
                                 f'(EI/L viga ≈ {100 * Ib / Ic:.0f}% de la columna); el marco trabaja casi como voladizos.')
        else:
            add('info', 'Vigas rígidas (modelo de cortante): las vigas reales flexibles bajan la rigidez lateral '
                        'y alargan los periodos. Usa vigas flexibles para un resultado más realista.')
    E, mat = p.get('E'), p.get('material')
    if E and mat == 'Concreto' and not 100000 <= E <= 300000:
        add('aviso', f'E = {E:,.0f} kg/cm² es poco usual para concreto.')
    if E and mat == 'Acero' and not 1.9e6 <= E <= 2.2e6:
        add('aviso', f'E = {E:,.0f} kg/cm² es poco usual para acero (≈ 2,040,000).')

    # --- Revisiones
    rev = res.get('revision')
    if rev:
        for clave, e in rev['estados'].items():
            if e['limite'] is None:
                continue
            mal = [i + 1 for i in range(n) if e['dist'][i] > e['limite']]
            if mal:
                gobierna = clave == res['clave']
                sev = 'alerta' if gobierna else ('aviso' if clave == 'DL' else 'info')
                lim = f'{e["limite"]:g}' + (f' (γmáx {e["limite_base"]:g} × γc {e["gamma_c"]:g})' if e['gamma_c'] != 1 else '')
                add(sev, ('' if gobierna else 'Referencia (no es tu estado de diseño) · ')
                    + f'{e["nombre"]}: la distorsión supera {lim} en el nivel(es) '
                      f'{", ".join(map(str, sorted(mal)))} (máx. {e["dist"].max():.4f}).')
        th = rev['theta']
        for i in range(n):
            if th[i] > THETA_MAXIMO:
                add('alerta', f'Nivel {i + 1}: θ = {th[i]:.3f} > {THETA_MAXIMO:g}: posible inestabilidad por P-Δ.')
            elif th[i] > THETA_IGNORAR:
                add('aviso', f'Nivel {i + 1}: θ = {th[i]:.3f} > {THETA_IGNORAR:g}: considera los efectos P-Δ.')
        if rev['irreg_masa']:
            add('aviso', f'Irregularidad de masa en el nivel(es) {", ".join(map(str, rev["irreg_masa"]))} '
                         f'(peso > {IRREG_MASA:g} veces el de un piso adyacente).')
        if rev['irreg_rigidez'] and '5.3.2' not in p.get('irreg', []):
            add('aviso', f'Posible piso blando en el nivel(es) {", ".join(map(str, rev["irreg_rigidez"]))}. '
                         'Si aplica la irregularidad 5.3.2, márcala en "Irregularidades" para corregir γmáx.')
        k = rev['k_ef']
        fuertes = []
        for i in range(n - 1):
            vecinos = [k[j] for j in (i - 1, i + 1) if 0 <= j < n]
            if k[i] < 0.4 * k[i + 1] or k[i] < 0.4 * np.mean(vecinos):
                fuertes.append(i + 1)
        if fuertes and not p.get('fuerte_elev'):
            add('aviso', f'Rigidez del nivel(es) {", ".join(map(str, fuertes))} menor que 40% de la del nivel superior '
                         'o del promedio de los adyacentes: posible irregularidad fuerte en elevación (5.3.3). '
                         "Si aplica: γc = 0.33, Q′ = 1 en el entrepiso débil y análisis no lineal.")

    # --- Marco normativo (apuntes NTC-S 2023)
    nm = res.get('normativa')
    if nm:
        obj = nm.get('objetivo')
        if obj and obj['nota']:
            add('aviso', obj['nota'])
        if p.get('intensidad', 'Base de diseño') != 'Base de diseño':
            add('info', f'Intensidad {p["intensidad"].lower()}: verifica que a0, c, Ta, Tb, k y Ts sean los del espectro '
                        'SASID de esa intensidad.')
        geom_ = p.get('geom')
        if geom_ and p['k1'] != 1.25:
            sug = ntc.k1_sugerido(len(geom_['bays']))
            if abs(p['k1'] - sug) > 1e-9:
                add('info', f'Tu marco tiene {len(geom_["bays"])} crujía(s): según los apuntes k1 = 0.8 con menos de 3 '
                            f'crujías resistentes, 1.0 con 3 o más y 1.25 en sistemas duales. Usaste k1 = {p["k1"]:g}.')
        if p.get('R_unitaria'):
            add('info', f'Material {p.get("material", "").lower()}: R = 1 (los apuntes indican R = 1 para materiales '
                        'distintos del concreto). Puedes desactivarlo en "Otros datos normativos".')
        zdef = ntc.ZETA_MATERIAL.get(p.get('material'))
        if zdef is not None and abs(z - zdef) > 1e-9:
            add('info', f'ζ = {z:g}: los apuntes usan {zdef:g} para {p["material"].lower()}. El espectro de SASID debe '
                        'pedirse con el mismo amortiguamiento.')
        v = nm.get('vmin')
        if v and not v['cumple']:
            add('aviso', f'Cortante basal mínimo: V dinámico = {v["V_din"]:.2f} t < FC·a_min·W = {v["V_min"]:.2f} t. '
                         f'Escala las fuerzas × {v["factor"]:.3f}.')
        e_ = nm['estatico']
        if not e_['aplica']:
            razon = 'es del grupo A' if p['grupo'] == 'A' else f'supera {e_["limite"]:g} m (H = {e_["H"]:.1f} m)'
            add('aviso', f'Según los apuntes el método estático no aplica porque la estructura {razon}: '
                         'el factor de escala se muestra solo como referencia.')
        if p.get('gamma_c', 1.0) != 1.0:
            add('info', p['gamma_c_txt'] + ' Se aplica a los límites de ocupación inmediata y seguridad de vida.')

    # --- Cortante basal
    fe = res.get('factor_escala')
    if fe is not None:
        if fe > 1.5:
            add('aviso', f'El factor de escala ({fe:.2f}) es alto: el cortante dinámico es muy menor que el estático.')
        elif fe > 1:
            add('info', f'Debes escalar las fuerzas dinámicas por {fe:.3f} para alcanzar el cortante estático.')
        elif fe < 0.5:
            add('aviso', f'El factor de escala ({fe:.2f}) es muy bajo: el cortante dinámico supera mucho al estático.')
    if p.get('estado') == 'Seguridad de vida' and len(p['Q']) and p['Q'][0] == 1:
        add('info', 'Q = 1 en seguridad de vida: comportamiento elástico, sin reducción por ductilidad.')
    return out
