"""Tercer parcial (apuntes del Dr. Rivero, pp. 22-35): mampostería, viento y concreto reforzado.

Convenciones: kg, cm, kg/cm² salvo que se indique; cada función devuelve un dict con los resultados y una lista
'pasos' (texto) con el procedimiento. Lo que viene de normas que no están adjuntas (NTC-Mampostería, -Viento y
-Concreto) se toma de los apuntes tal cual; α de R1 está ilegible en los apuntes y se deja como dato editable.
"""
import math

import numpy as np

# ====================================================================== mampostería
K_ALTURA = {'Barda': 2.0, 'Muro interior': 0.8, 'Muro de lindero': 1.0}
TIPOS_MURO = list(K_ALTURA)

# NTC-S tabla 4.3.3 (guía): (Q, γSV, γOI)
TABLA_433 = {
    'Confinada, piezas macizas, con refuerzo horizontal': (2.0, 0.010, 0.0015),
    'Confinada, piezas macizas (tabique rojo recocido)': (2.0, 0.005, 0.0015),
    'Confinada, piezas huecas, con refuerzo horizontal': (2.0, 0.008, 0.0015),
    'Confinada, piezas huecas': (1.5, 0.004, 0.0015),
    'Piezas huecas reforzadas interiormente': (1.5, 0.006, None),
    'No confinada ni reforzada (solo existentes)': (1.0, 0.0015, None),
}


def factor_forma(H, L):
    """f del cortante (apuntes p. 23): 1.5 si H/L ≤ 0.2, −0.625 H/L + 1.625 entre 0.2 y 1.0, 1.0 si H/L ≥ 1."""
    r = H / L
    return 1.5 if r <= 0.2 else (1.0 if r >= 1.0 else -0.625 * r + 1.625)


def muro_compresion(t, H, L, Pu, fm=15.0, tipo='Muro de lindero', L2L1=1.0, cv_cm=0.8, mov_restringido=True,
                    sin_volados=True, b=None, FR=0.6, As_fy=0.0):
    """PR = FR·FE·(f'm·AT + ΣAs·fy) ≥ Pu.  FE: 0.7/0.6 si se cumplen las 3 condiciones; si no, el menor entre eso y la
    fórmula de excentricidad y esbeltez (apuntes p. 23, inciso 6 NTC-Mamp)."""
    b = t if b is None else b
    AT = L * t
    esb = H / t
    cumple = mov_restringido and sin_volados and esb <= 20
    base = 0.7 if (tipo == 'Muro interior' and L2L1 <= 1.5 and cv_cm <= 1.0) else 0.6
    k = K_ALTURA[tipo]
    ec = t / 2 - b / 3
    ep = ec + t / 24
    fe_formula = (1 - 2 * ep / t) * (1 - (k * H / (30 * t)) ** 2)
    fe = base if cumple else max(0.0, min(base, fe_formula))
    PR = FR * fe * (fm * AT + As_fy)
    pasos = [f'AT = L·t = {L:g}×{t:g} = {AT:,.0f} cm²; H/t = {esb:.2f}',
             f'Condiciones: movimiento perpendicular restringido {"sí" if mov_restringido else "NO"}, sin volados '
             f'{"sí" if sin_volados else "NO"}, H/t ≤ 20 {"sí" if esb <= 20 else "NO"} → '
             + ('se cumplen: FE = {:.1f}'.format(fe) if cumple else 'no se cumplen: se usa la fórmula')]
    if not cumple:
        pasos += [f'ec = t/2 − b/3 = {ec:.3f} cm; e′ = ec + t/24 = {ep:.3f} cm',
                  f'k = {k:g} ({tipo}); kH/30t = {k * H / (30 * t):.4f}',
                  f'FE = (1 − 2e′/t)[1 − (kH/30t)²] = {fe_formula:.4f}; el menor con {base} → FE = {fe:.4f}']
    pasos.append(f'PR = {FR}·{fe:.4f}·({fm:g}·{AT:,.0f}) = {PR:,.0f} kg  {"≥" if PR >= Pu else "<"} Pu = {Pu:,.0f} kg → '
                 + ('PASA' if PR >= Pu else 'NO PASA'))
    return dict(AT=AT, esbeltez=esb, cumple=cumple, FE=fe, FE_formula=fe_formula, PR=PR, Pu=Pu, pasa=PR >= Pu, pasos=pasos)


def muro_cortante(t, L, H, Pn, vm=2.0, FR=0.7, Vu=None, VsR=0.0):
    """VmR = FR[(0.5 v'm AT + 0.3 Pn) f] ≤ 1.5 FR v'm AT f  (Pn = CM + CVacc sin factor de carga)."""
    AT = L * t
    f = factor_forma(H, L)
    formula = FR * (0.5 * vm * AT + 0.3 * Pn) * f
    tope = 1.5 * FR * vm * AT * f
    VmR = min(formula, tope)
    VR = VmR + VsR
    pasos = [f'AT = {AT:,.0f} cm²; H/L = {H / L:.4f} → f = {f:.4f}',
             f'VmR = {FR}·(0.5·{vm:g}·{AT:,.0f} + 0.3·{Pn:,.0f})·{f:.4f} = {formula:,.1f} kg',
             f'Tope 1.5·FR·v′m·AT·f = {tope:,.1f} kg → VmR = {VmR:,.1f} kg ' + ('(rige el tope)' if tope < formula else '(rige la fórmula)')]
    pasa = None
    if Vu is not None:
        pasa = VR >= Vu
        pasos.append(f'VR = {VR:,.1f} {"≥" if pasa else "<"} Vu = {Vu:,.1f} kg → ' + ('PASA' if pasa else 'NO PASA'))
    return dict(AT=AT, f=f, formula=formula, tope=tope, VmR=VmR, VR=VR, Vu=Vu, pasa=pasa, pasos=pasos)


def reducciones(Q, k, T, Ta, Tb, R0=2.0, k1=1.0, estado='SV'):
    """Q′ (ec. 3.2.1), R = k1·R0 + k2 y R′ (SV: R; OI: 0.75R con Q = 1)."""
    k2 = max(0.0, 0.5 * (1 - math.sqrt(T / Ta))) if Ta > 0 else 0.0
    R = k1 * R0 + k2
    Qp_Q = Q
    if estado == 'OI':
        Qp_Q = 1.0
    p = k + (1 - k) * (Tb / T) ** 2 if T > 0 else k
    if T <= Ta:
        Qp = 1 + (Qp_Q - 1) * math.sqrt(1 / k) * (T / Ta)
    elif T <= Tb:
        Qp = 1 + (Qp_Q - 1) * math.sqrt(1 / k)
    else:
        Qp = 1 + (Qp_Q - 1) * math.sqrt(p / k)
    Rp = 0.75 * R if estado == 'OI' else R
    return dict(Qp=Qp, R=R, Rp=Rp, k2=k2)


def estatico(W, h, c, k, Q, R0=2.0, k1=1.0, estado='SV', FC=1.1):
    """Método estático en la meseta (apuntes p. 20, NTC-S cap. 6): Fi = c/(Q′R′)·Wi·hi·ΣW/Σ(Wi·hi)."""
    W = np.asarray(W, float)
    h = np.asarray(h, float)
    n = W.size
    Qp = 1.0 if estado == 'OI' else 1 + (Q - 1) * math.sqrt(1 / k)
    R = k1 * R0
    Rp = 0.75 * R if estado == 'OI' else R
    C = c / (Qp * Rp)
    Wh = W * h
    F = C * Wh * W.sum() / Wh.sum()
    Fu = FC * F
    Vu = np.cumsum(Fu[::-1])[::-1]
    Mvu = Fu * h
    pasos = [f'Q′ = {Qp:.4f}; R = k1·R0 = {R:g}; R′ = {Rp:g}; c/(Q′R′) = {c:g}/({Qp:.4f}·{Rp:g}) = {C:.5f}',
             f'ΣW = {W.sum():g}; Σ(W·h) = {Wh.sum():g}; Fi = C·Wi·hi·ΣW/ΣWh; Fu = {FC}·F',
             f'ΣF = {F.sum():.4f} (comprobación C·ΣW = {C * W.sum():.4f}); VBu = {Vu[0]:.4f}; Mvu = {Mvu.sum():.4f}']
    return dict(Qp=Qp, R=R, Rp=Rp, C=C, Wh=Wh, F=F, Fu=Fu, Vu=Vu, Mvu=Mvu, VBu=Vu[0], MVU=Mvu.sum(), pasos=pasos,
                niveles=n)


def distorsion_limite(estructuracion, n_niveles=1, condiciones_irreg=0, fuerte_torsion=False, fuerte_elevacion=False):
    """γSV y γOI de la tabla 4.3.3 con la corrección por más de 6 niveles (Q − 0.5, γSV − 0.004 ≥ 0.002) y por
    irregularidad (γc). Devuelve Q, γSV, γOI y γc."""
    Q, gsv, goi = TABLA_433[estructuracion]
    if n_niveles > 6:
        Q = max(1.0, Q - 0.5)
        gsv = max(0.002, gsv - 0.004)
    if fuerte_elevacion:
        gc = 0.33
    elif fuerte_torsion:
        gc = 0.5 if condiciones_irreg else 0.6
    else:
        gc = {0: 1.0, 1: 0.8, 2: 0.7}.get(condiciones_irreg, 0.6)
    return dict(Q=Q, gamma_SV=gsv, gamma_OI=goi, gamma_c=gc, gamma_SV_corregida=gsv * gc,
                gamma_OI_corregida=None if goi is None else goi * gc)


def a_min(Ts, Rp):
    """a_min del cortante basal mínimo (§7.5): 0.04/R′ (Ts ≤ 0.5), 0.06/R′ (Ts ≥ 1.0), lineal entre ambos."""
    if Ts <= 0.5:
        return 0.04 / Rp
    if Ts >= 1.0:
        return 0.06 / Rp
    return (0.04 + 0.02 * (Ts - 0.5) / 0.5) / Rp


def excentricidades_accidentales(n, b):
    """e_a de cada entrepiso i = [0.05 + 0.05(i−1)/(n−1)]·b (ec. 2.3.2)."""
    return [(0.05 + (0.05 * (i - 1) / (n - 1) if n > 1 else 0.0)) * b for i in range(1, n + 1)]


def revision_muros(muros, direccion=''):
    """muros: lista de dicts con nombre, t, L, H, Pn, Vu. Revisa cada muro y la suma del entrepiso (VR ≥ Vu y ΣVR ≥ ΣVu)."""
    filas, sVR, sVu = [], 0.0, 0.0
    for m in muros:
        r = muro_cortante(m['t'], m['L'], m['H'], m['Pn'], m.get('vm', 2.0), Vu=m['Vu'])
        filas.append(dict(muro=m['nombre'], H_L=m['H'] / m['L'], f=r['f'], Pn=m['Pn'], VmR=r['VmR'], Vu=m['Vu'], pasa=r['pasa']))
        sVR += r['VmR']
        sVu += m['Vu']
    total = sum(f['VmR'] for f in filas)
    cmax = max(filas, key=lambda f: f['VmR']) if filas else None
    concentracion = [(f['muro'], f['VmR'] / total) for f in filas if total and f['VmR'] / total > 0.35]
    return dict(filas=filas, sum_VmR=sVR, sum_Vu=sVu, entrepiso_pasa=sVR >= sVu, todos_pasan=all(f['pasa'] for f in filas),
                concentracion=concentracion)


# ====================================================================== viento
TERRENOS = {  # α, δ (m), c   (apuntes p. 28, tabla 3.1.3.2.1). α de R1 ilegible: 0.061 es una lectura
    'R1 · campo abierto': (0.061, 280.0, 1.142),
    'R2 · pocas obstrucciones': (0.095, 350.0, 1.000),
    'R3 · urbano y suburbano': (0.140, 410.0, 0.832),
    'R4 · mitad de construcciones con h > 20 m': (0.192, 470.0, 0.668),
}
TOPOGRAFIA = {'T1 · faldas de serranías': 0.8, 'T2 · valle cerrado': 0.9, 'T3 · terreno plano': 1.0,
              'T4 · montículo': 1.1, 'T5 · cima, pendiente 10–15 %': 1.2, 'T5 · cima, pendiente > 15 %': 1.5}
VR_ZONAS = {  # m/s (apuntes p. 27, aproximado): (A 200 años, B 50 años, temporal 10 años)
    'ZI · Azcapotzalco, GAM': (28, 25, 19),
    'ZII · M. Hidalgo, Cuauhtémoc, V. Carranza, B. Juárez, Coyoacán, Iztacalco, Iztapalapa': (33, 30, 23),
    'ZIII · Cuajimalpa, Á. Obregón, M. Contreras, Tlalpan, Milpa Alta, Tláhuac, Xochimilco': (39, 35, 27)}
CPE = {'Barlovento': 0.8, 'Sotavento': -0.4, 'Paredes laterales': -0.8, 'Techo plano': -0.8,
       'Techo inclinado en sotavento': -0.7}
TIPOS_ESTRUCTURA = ['Tipo 1 · rígida (H/A ≤ 5 y T ≤ 1 s): estático', 'Tipo 2 · esbelta (H/A > 5 y T > 1 s): estático + turbulencia',
                    'Tipo 3 · cilíndrica (tuberías, chimeneas, silos): vórtices alternantes', 'Tipo 4 · especial: estudios especiales']


def tipo_estructura(H, A, T=None, cilindrica=False, especial=False):
    if especial:
        return TIPOS_ESTRUCTURA[3]
    if cilindrica:
        return TIPOS_ESTRUCTURA[2]
    if H / A > 5 and (T is None or T > 1.0):
        return TIPOS_ESTRUCTURA[1]
    return TIPOS_ESTRUCTURA[0] if (T is None or T <= 1.0) else TIPOS_ESTRUCTURA[1]


def frz(z, alfa, delta, c):
    if z <= 10:
        return c
    if z < delta:
        return c * (z / 10) ** alfa
    return c * (delta / 10) ** alfa


def alturas_tributarias(z):
    """Altura que aporta cada nivel: del punto medio con el nivel inferior al punto medio con el superior; el primero toma
    desde 0 y el último hasta su propia z (apuntes p. 29: 4.5 m abajo y 1.5 m en la azotea con 3 m de entrepiso)."""
    z = list(map(float, z))
    medios = [(a + b) / 2 for a, b in zip(z[:-1], z[1:])]
    bajo = [0.0] + medios
    alto = medios + [z[-1]]
    return [a - b for a, b in zip(alto, bajo)]


def viento_cara(z, ancho, VR, alfa, delta, c, FT=1.0, cp=0.8, constante_en_h=False, FC=1.1):
    """Fuerzas del viento sobre una cara por nivel. cp es el coeficiente (se usa su valor absoluto en la presión).
    constante_en_h=True para sotavento, paredes laterales y techo (Frz con z = h)."""
    z = np.asarray(z, float)
    h = z.max()
    zeval = np.full_like(z, h) if constante_en_h else z
    Frz = np.array([frz(zz, alfa, delta, c) for zz in zeval])
    VD = FT * Frz * VR
    pz = 0.053 * abs(cp) * VD ** 2
    ht = np.array(alturas_tributarias(z))
    AT = ht * ancho
    F = pz * AT / 1000.0
    Fu = FC * F
    Vu = np.cumsum(Fu[::-1])[::-1]
    Mvu = Fu * z
    return dict(z=z, Frz=Frz, VD=VD, pz=pz, AT=AT, F=F, Fu=Fu, Vu=Vu, Mvu=Mvu, VBu=Vu[0], MVU=Mvu.sum(),
                ht=ht)


def techo_plano(largo, ancho, VR, alfa, delta, c, h, FT=1.0, cp=0.8, FC=1.1):
    """Succión sobre el techo plano: p = 0.053·Cp·VD² con Frz(z = h)."""
    VD = FT * frz(h, alfa, delta, c) * VR
    pz = 0.053 * abs(cp) * VD ** 2
    F = largo * ancho * pz
    return dict(VD=VD, pz=pz, F=F, Fu=FC * F)


# ====================================================================== concreto reforzado
VARILLAS = {'#3': 0.71, '#4': 1.27, '#5': 1.98, '#6': 2.85, '#8': 5.07, '#10': 7.92}
FR_FLEXION, FR_CORTANTE = 0.9, 0.75


def beta1(fc):
    return 0.85 if fc <= 280 else max(0.65, 1.05 - fc / 1400)


def limites_flexion(b, d, fc, fy):
    fcc = 0.85 * fc
    b1 = beta1(fc)
    Asb = fcc / fy * (6000 * b1 / (fy + 6000)) * b * d
    return dict(fcc=fcc, beta1=b1, Asb=Asb, As_max=0.75 * Asb,
                As_min=max(0.8 * math.sqrt(fc) / fy * b * d, 14 / fy * b * d))


def momento_resistente(As, b, d, fc, fy, FR=FR_FLEXION):
    """MR = FR·As·fy·(d − As·fy/(2·b·f″c)) en kg·cm."""
    return FR * As * fy * (d - As * fy / (2 * b * 0.85 * fc))


def area_requerida(Mu, b, d, fc, fy, FR=FR_FLEXION):
    """Raíz menor de (fy²/(2 b f″c)) As² − (fy d) As + Mu/FR = 0 (Mu en kg·cm). None si no hay solución real."""
    fcc = 0.85 * fc
    a2, a1, a0 = fy ** 2 / (2 * b * fcc), -fy * d, Mu / FR
    disc = a1 ** 2 - 4 * a2 * a0
    if disc < 0:
        return None
    return (-a1 - math.sqrt(disc)) / (2 * a2)


def diseno_flexion(Mu_tm, b, h, r, fc, fy):
    """Diseño a flexión de una sección rectangular (Mu en t·m). Devuelve As calculada, de diseño y los límites."""
    d = h - r
    lim = limites_flexion(b, d, fc, fy)
    As = area_requerida(Mu_tm * 1e5, b, d, fc, fy)
    pasos = [f"d = h − r = {d:g} cm; f″c = 0.85·f′c = {lim['fcc']:.2f}; β1 = {lim['beta1']:.3f}",
             f"Asb = {lim['Asb']:.2f} cm²; As,máx = 0.75·Asb = {lim['As_max']:.2f}; As,mín = {lim['As_min']:.2f} cm²"]
    if As is None or As > lim['As_max']:
        pasos.append(('No hay solución real' if As is None else f'As = {As:.2f} cm² > As,máx')
                     + f" → cambia la sección o el concreto. MR con As,máx = "
                       f"{momento_resistente(lim['As_max'], b, d, fc, fy) / 1e5:.2f} t·m")
        return dict(d=d, **lim, As_calc=As, As_dis=None, ok=False, pasos=pasos)
    As_dis = max(As, lim['As_min'])
    pasos.append(f'As calculada = {As:.2f} cm²' + (f'; rige As,mín → {As_dis:.2f} cm²' if As_dis > As else ''))
    return dict(d=d, **lim, As_calc=As, As_dis=As_dis, ok=True, pasos=pasos)


def proponer_armados(As_req, ancho_max=None, b=None, top=6):
    """Combinaciones de 2–8 barras de un diámetro, o de dos diámetros contiguos, con área ≥ As_req, ordenadas por exceso."""
    nombres = list(VARILLAS)
    cand = []
    for nom in nombres:
        for n in range(2, 9):
            a = n * VARILLAS[nom]
            if a >= As_req:
                cand.append((a - As_req, f'{n}{nom}', a))
    for i, n1 in enumerate(nombres):
        for n2 in nombres[:i]:
            for c1 in range(2, 5):
                for c2 in range(1, 5):
                    a = c1 * VARILLAS[n1] + c2 * VARILLAS[n2]
                    if a >= As_req:
                        cand.append((a - As_req, f'{c1}{n1} + {c2}{n2}', a))
    cand.sort(key=lambda x: (x[0] + (1.5 if '+' in x[1] else 0.0), len(x[1])))
    vistos, salida = set(), []
    for exceso, texto, a in cand:
        if texto not in vistos:
            vistos.add(texto)
            salida.append(dict(armado=texto, As=a, exceso=exceso))
        if len(salida) >= top:
            break
    return salida


def tanteo_eje_neutro(As, b, d, fc, fy, Es=2.0e6, c_prop=None):
    """Equilibrio C = T por tanteos (como en el Examen 3): da c, a, εs, fs, T, C y MR. c_prop solo evalúa un tanteo."""
    fcc = 0.85 * fc
    b1 = beta1(fc)

    def evalua(c):
        a = b1 * c
        C = a * b * fcc
        es = 0.003 * (d - c) / c
        fs = min(Es * es, fy)
        return dict(c=c, a=a, C=C, es=es, fs=fs, T=As * fs, fluye=es >= fy / Es)

    if c_prop is not None:
        e = evalua(c_prop)
        e['MR'] = FR_FLEXION * e['T'] * (d - e['a'] / 2)
        return e
    lo, hi = 1e-6, d
    for _ in range(200):
        mid = (lo + hi) / 2
        e = evalua(mid)
        if e['C'] > e['T']:
            hi = mid
        else:
            lo = mid
    e = evalua((lo + hi) / 2)
    e['MR'] = FR_FLEXION * e['T'] * (d - e['a'] / 2)
    return e


def cortante_viga(Vu_kg, b, d, h, L, fc, fy_est, Av, ρ_As, lambda_=1.0):
    """Estribos de una viga de ductilidad baja (apuntes p. 34). ρ_As = As longitudinal (cm²) para ρ = As/(b·d).
    Devuelve VcR, límites, s calculada, smáx y la separación a usar (múltiplo de 5 cm hacia abajo)."""
    FR = FR_CORTANTE
    rho = ρ_As / (b * d)
    raiz = math.sqrt(fc)
    if rho <= 0.015:
        Vc = FR * 2 * lambda_ * rho ** (1 / 3) * raiz * b * d
    else:
        Vc = lambda_ * FR * 0.5 * raiz * b * d
    Vc = min(Vc, 1.25 * FR * b * d * raiz * lambda_)
    Vc = max(Vc, FR * 0.25 * lambda_ * raiz * b * d)
    Vmax = 2.65 * FR * b * d * raiz
    pasos = [f'ρ = As/(b·d) = {ρ_As:g}/({b:g}·{d:g}) = {rho:.5f}; L/h = {L / h:.2f}' + (' (< 5: usa puntal y tensor)' if L / h < 5 else ''),
             f'VcR = {Vc:,.0f} kg; límite 2.65·FR·b·d·√f′c = {Vmax:,.0f} kg']
    if Vu_kg > Vmax:
        pasos.append('Vu excede el límite máximo → agranda la sección')
        return dict(rho=rho, VcR=Vc, Vmax=Vmax, ok=False, pasos=pasos, s_usar=None)
    lim_av = [0.5 * d if (Vu_kg - Vc) <= 1.1 * FR * b * d * raiz else 0.25 * d,
              Av * fy_est / (0.2 * raiz * b), Av * fy_est / (3.5 * b), 30.0]
    s_max = min(lim_av)
    if Vu_kg <= Vc:
        s_calc = None
        pasos.append(f'Vu = {Vu_kg:,.0f} ≤ VcR → estribos mínimos')
        s_max = min(0.5 * d, lim_av[1], lim_av[2], 30.0)
    else:
        s_calc = FR * Av * fy_est * d / (Vu_kg - Vc)
        pasos.append(f'Vu − VcR = {Vu_kg - Vc:,.0f} kg; s = FR·Av·fy·d/(Vu − VcR) = {s_calc:.2f} cm')
    pasos.append(f'smáx = min(0.5d o 0.25d = {lim_av[0]:.2f}; Avfy/(0.2√f′c b) = {lim_av[1]:.1f}; Avfy/(3.5b) = {lim_av[2]:.1f}; 30) = {s_max:.2f} cm')
    s = min(x for x in (s_calc, s_max) if x is not None)
    s_usar = max(5, int(s // 5 * 5))
    pasos.append(f'Se usa s = {s_usar} cm (múltiplo de 5 hacia abajo)')
    return dict(rho=rho, VcR=Vc, Vmax=Vmax, s_calc=s_calc, s_max=s_max, s_usar=s_usar, ok=True, pasos=pasos)


def viga_simple(w, L, P=None):
    """Viga simplemente apoyada con carga uniforme w (t/m) sobre L (m): Vu máx y Mu máx."""
    return dict(Vu=w * L / 2, Mu=w * L ** 2 / 8)


def viga_doble_empotrada(w, L):
    return dict(Vu=w * L / 2, Mu_neg=w * L ** 2 / 12, Mu_pos=w * L ** 2 / 24)
