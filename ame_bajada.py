"""Examen 1 · Bajada de cargas de una columna, como en los apuntes del Dr. Rivero (pp. 5 y 8).

De arriba hacia abajo, cada nivel suma al acumulado: carga de la losa en el área tributaria (At·w), peso propio de las vigas que
bajan a la columna (L·pp viga) y peso propio de la columna del entrepiso (h·pp col). Se lleva por separado la combinación de
gravedad (CG: CM y CVmáx), la accidental (CA: CM y CVacc) y la de servicio (CS: CM y CVmed). Unidades: kg, m, kg/m²."""
import math

import numpy as np

CASOS = ['CG', 'CA', 'CS']
NOMBRE_CASO = {'CG': 'Gravedad (CM + CVmáx)', 'CA': 'Accidental (CM + CVacc)', 'CS': 'Servicio (CM + CVmed)'}
FC_GRUPO = {'A': (1.5, 1.7), 'B': (1.3, 1.5)}          # (CM, CVmáx); accidental 1.1 y servicio 1.0 en cualquier grupo
FC_CASO = {'CA': 1.1, 'CS': 1.0}


def pp_lineal(b_cm, h_cm, gamma_tm3=2.4):
    """Peso propio por metro (kg/m) de un elemento de sección b×h (cm) y peso volumétrico γ (t/m³)."""
    return b_cm / 100 * h_cm / 100 * gamma_tm3 * 1000


def total_tipo(componentes):
    """Suma de una lista de componentes de carga (kg/m²)."""
    return float(sum(componentes))


def bajada(niveles, tipos, pp_viga, pp_col):
    """niveles: de arriba hacia abajo, lista de dict(nombre, At (m²), tipo (clave de `tipos`), Lv (m de viga), hc (m de columna)).
    tipos: {clave: {'CG': (CM, CV), 'CA': (CM, CV), 'CS': (CM, CV)}} en kg/m². pp_viga, pp_col en kg/m.
    Devuelve una lista (un dict por nivel) con el incremento y el acumulado de CM y CV de cada caso."""
    acum = {c: [0.0, 0.0] for c in CASOS}
    filas = []
    for n in niveles:
        if n['tipo'] not in tipos:
            raise ValueError(f"El tipo de piso «{n['tipo']}» no está en la tabla de cargas.")
        fila = dict(nombre=n['nombre'], At=n['At'], Lv=n['Lv'], hc=n['hc'], tipo=n['tipo'],
                    pp_vig=n['Lv'] * pp_viga, pp_col=n['hc'] * pp_col)
        for c in CASOS:
            cm_l, cv_l = tipos[n['tipo']][c]
            d_cm = n['At'] * cm_l + fila['pp_vig'] + fila['pp_col']
            d_cv = n['At'] * cv_l
            ant = tuple(acum[c])
            acum[c][0] += d_cm
            acum[c][1] += d_cv
            fila[c] = dict(cm_losa=n['At'] * cm_l, d_cm=d_cm, d_cv=d_cv, cm_ant=ant[0], cv_ant=ant[1], cm=acum[c][0], cv=acum[c][1])
        filas.append(fila)
    return filas


def cargas_base(filas, grupo='B', fc=250.0):
    """Cargas en la base (último nivel acumulado): nominal Pn y de diseño Pu de cada combinación, y el área de la columna
    A = Pu/(0.25 f′c) de los apuntes para la combinación que rige."""
    fcm, fcv = FC_GRUPO[grupo]
    ult = filas[-1]
    out = {}
    for c in CASOS:
        cm, cv = ult[c]['cm'], ult[c]['cv']
        f1, f2 = (fcm, fcv) if c == 'CG' else (FC_CASO[c], FC_CASO[c])
        out[c] = dict(CM=cm, CV=cv, Pn=cm + cv, Pu=f1 * cm + f2 * cv, FC_CM=f1, FC_CV=f2)
    rige = max(('CG', 'CA'), key=lambda c: out[c]['Pu'])
    Pu = out[rige]['Pu']
    A = Pu / (0.25 * fc)
    return dict(casos=out, rige=rige, Pu=Pu, A_col=A, lado=math.sqrt(A))


def carga_viga_desde_losa(w_cm, w_cv, At, L, pp_viga, fc_cm, fc_cv):
    """Carga repartida de diseño (kg/m) sobre una viga que recibe At m² de losa en L m: CM = w·At/L + pp, CV = w·At/L."""
    cm = w_cm * At / L + pp_viga
    cv = w_cv * At / L
    return dict(cm=cm, cv=cv, wu=fc_cm * cm + fc_cv * cv)


def bajada_rapida(filas, w_tm2=1.3, incluye_pp=True, pp_viga=0.0, pp_col=0.0):
    """Predimensionado de los apuntes (p. 4): w = 1.3 t/m² incluyendo el peso propio; P_k = P_{k+1} + A_T·w (+ p.p. vigas + p.p.
    columna si w no los incluye). filas: dict(nombre, At, Lv, hc) de arriba hacia abajo (la última puede ser la base, At = 0).
    Devuelve dict(nombre, At, losa, pv, pc, P) con las cargas en kg."""
    P, out = 0.0, []
    for f in filas:
        losa = f['At'] * w_tm2 * 1000
        pv = 0.0 if incluye_pp else f['Lv'] * pp_viga
        pc = 0.0 if incluye_pp else f['hc'] * pp_col
        P += losa + pv + pc
        out.append(dict(nombre=f['nombre'], At=f['At'], losa=losa, pv=pv, pc=pc, P=P))
    return out


def figura_columna(items, titulo='', cierre=''):
    """Vista de la columna como en los apuntes: losas y vigas por nivel, flecha de carga y P_k = P_{k+1} + … en cada nivel.
    items: lista de (nombre, texto de la fórmula, P en t) de arriba hacia abajo; el último es P0 (base). cierre: texto de Pu y A."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(items)
    fig, ax = plt.subplots(figsize=(11, max(4.5, 1.25 * n + 1.5)))
    ys = [-(k * 1.2) for k in range(n)]
    ax.plot([0, 0], [ys[0] + .5, ys[-1] - .2], color='#1d3557', lw=7, solid_capstyle='butt', zorder=2)
    for k, (nom, txt, P) in enumerate(items):
        y = ys[k]
        es_base = k == n - 1
        if not es_base:
            ax.plot([-1.6, 1.6], [y, y], color='#444', lw=2)
            ax.add_patch(plt.Rectangle((-.3, y - .25), .6, .25, fc='#cfd8e3', ec='#1d3557', zorder=3))
        else:
            ax.plot([-1.6, 1.6], [y - .2, y - .2], color='#444', lw=2)
            for x in np.arange(-1.6, 1.6, .25):
                ax.plot([x, x - .2], [y - .2, y - .45], color='#444', lw=1)
        ax.annotate('', xy=(0, y - .2 if es_base else y + .05), xytext=(0, y + .5),
                    arrowprops=dict(arrowstyle='-|>', color='#e76f51', lw=2), zorder=4)
        ax.text(-1.8, y + .25, nom, ha='right', va='center', fontsize=12, fontweight='bold', color='#e76f51')
        ax.text(1.8, y + .25, f'{txt}  =  {P:,.2f} t', ha='left', va='center', fontsize=10.5)
    if cierre:
        ax.text(0, ys[-1] - .9, cierre, ha='center', va='top', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=.4', fc='#fff3e0', ec='#e76f51'))
    if titulo:
        ax.set_title(titulo, fontsize=13, fontweight='bold')
    ax.set_xlim(-3.2, 12)
    ax.set_ylim(ys[-1] - 1.8, ys[0] + 1)
    ax.axis('off')
    return fig


def tipo_desde_componentes(comp, cv):
    """Tipo de piso en el formato de los apuntes: comp = lista de (concepto, CG, CA, CS) de la carga muerta (kg/m²; CA o CS vacíos =
    igual que CG) y cv = (CVmáx, CVacc, CVmed). Devuelve {'CG': (CM, CV), 'CA': (CM, CV), 'CS': (CM, CV)}."""
    cm = [0.0, 0.0, 0.0]
    for _, cg, ca, cs in comp:
        vals = [cg, cg if ca in (None, '') else ca, cg if cs in (None, '') else cs]
        for k in range(3):
            cm[k] += float(vals[k])
    return dict(CG=(cm[0], float(cv[0])), CA=(cm[1], float(cv[1])), CS=(cm[2], float(cv[2])))
