"""Bajada de cargas de un edificio de varios niveles con una planta (cuadrícula) distinta por nivel."""
import ame_bajada as bj
import ame_planta as pl


def columnas_edificio(niveles, masks, dx, dy, tipos, pp_viga, pp_col):
    """niveles: de arriba hacia abajo, lista de dict(nombre, tipo, h) con h = altura del entrepiso que queda debajo de ese nivel (m).
    masks: matriz de códigos de cada nivel (mismo orden). Para cada columna que existe en algún nivel devuelve la bajada
    (lista de filas de ame_bajada.bajada, con una fila final «Base» que suma la columna hasta el desplante).
    El peso de un tramo de columna se suma en el nivel de abajo y solo si la columna existe arriba y abajo."""
    ny, nx = len(dy) + 1, len(dx) + 1
    ex = [pl.nodos_existentes(m) for m in masks]
    out = {}
    for j in range(ny):
        for i in range(nx):
            if not any(e[j, i] for e in ex):
                continue
            filas = []
            for r, n in enumerate(niveles):
                if ex[r][j, i]:
                    At, Lv = pl.columna(masks[r], dx, dy, i, j)
                else:
                    At = Lv = 0.0
                hc = niveles[r - 1]['h'] if r > 0 and ex[r][j, i] and ex[r - 1][j, i] else 0.0
                filas.append(dict(nombre=n['nombre'], tipo=n['tipo'], At=At, Lv=Lv, hc=hc))
            if ex[-1][j, i]:
                filas.append(dict(nombre='Base', tipo=niveles[-1]['tipo'], At=0.0, Lv=0.0, hc=niveles[-1]['h']))
            out[(j, i)] = bj.bajada(filas, tipos, pp_viga, pp_col)
    return out


def pesos_por_nivel(niveles, masks, dx, dy, tipos, pp_viga, pp_col):
    """Peso sísmico de cada nivel con CM + CVacc (kg): losa, vigas y la mitad de las columnas de arriba y de abajo."""
    pesos = []
    for r, (n, m) in enumerate(zip(niveles, masks)):
        res = pl.resumen_nivel(m, dx, dy)
        cm, cv = tipos[n['tipo']]['CA']
        h_dev = n['h'] / 2
        h_arr = niveles[r - 1]['h'] / 2 if r > 0 else 0.0
        losa = res['area'] * (cm + cv)
        vig = res['vigas'] * pp_viga
        col = res['columnas'] * (h_dev + h_arr) * pp_col
        pesos.append(dict(nombre=n['nombre'], area=res['area'], vigas_m=res['vigas'], columnas=res['columnas'],
                          losa=losa, vigas=vig, cols=col, W=losa + vig + col))
    return pesos


def cargas_tramos(mask, dx, dy, tipo, pp_viga, grupo='B'):
    """Carga uniforme equivalente de cada tramo de viga de un nivel (t/m): CM = w·A/L + p.p., CV = w·A/L, de diseño con gravedad
    y accidental (1.1·(CM + CVacc))."""
    fcm, fcv = bj.FC_GRUPO[grupo]
    L = pl.letras(len(dy) + 1)
    filas = []
    for t in pl.tramos_viga(mask, dx, dy):
        k = t['A'] / t['L']
        cm = (tipo['CG'][0] * k + pp_viga) / 1000
        cv = tipo['CG'][1] * k / 1000
        cva = tipo['CA'][1] * k / 1000
        cma = (tipo['CA'][0] * k + pp_viga) / 1000
        filas.append(dict(tramo=pl.nombre_tramo(t, L), L=t['L'], A=t['A'], CM=cm, CV=cv, wu=fcm * cm + fcv * cv,
                          wacc=1.1 * (cma + cva), tipo=t['tipo'], i=t['i'], j=t['j']))
    return filas
