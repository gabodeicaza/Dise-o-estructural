"""Examen 1 · Bajada de cargas de una columna, como en los apuntes del Dr. Rivero (pp. 5 y 8).

De arriba hacia abajo, cada nivel suma al acumulado: carga de la losa en el área tributaria (At·w), peso propio de las vigas que
bajan a la columna (L·pp viga) y peso propio de la columna del entrepiso (h·pp col). Se lleva por separado la combinación de
gravedad (CG: CM y CVmáx), la accidental (CA: CM y CVacc) y la de servicio (CS: CM y CVmed). Unidades: kg, m, kg/m²."""
import math

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
