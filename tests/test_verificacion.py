"""Verificación independiente de los resultados: cada referencia sale de una fórmula de libro escrita aquí (sin usar el código
que se revisa) o se copia tal cual del Examen 3 y de las hojas de Excel (DISEÑO DE TRABES, NTCC-2023)."""
import math

import numpy as np
import pytest

import ame_parcial3 as p3
import ame_porticos as pt
import ame_viga as av

B, D, FC, FY = 40, 75, 250, 4200          # sección del Examen 3 (h = 80, r = 5)
FCC = 0.85 * FC


def as_cuantia_q(Mu_tm, b=B, d=D):
    """As por el método de la cuantía q (hoja «CÁLCULO DE ACERO»), distinto de la cuadrática del programa."""
    q = 1 - math.sqrt(1 - 2 * Mu_tm * 1e5 / (0.9 * b * d ** 2 * FCC))
    return q * FCC / FY * b * d


# ---------------------------------------------------------------- flexión
@pytest.mark.parametrize('Mu, examen', [(28.67, 10.47), (50.21, 18.89), (89.31, 35.70), (53.05, 20.03)])
def test_as_requerida_coincide_con_metodo_q_y_con_el_examen(Mu, examen):
    As = p3.area_requerida(Mu * 1e5, B, D, FC, FY)
    assert As == pytest.approx(as_cuantia_q(Mu), rel=1e-9)
    assert As == pytest.approx(examen, abs=0.006)


def test_limites_de_acero_del_examen():
    lim = p3.limites_flexion(B, D, FC, FY)
    assert lim['Asb'] == pytest.approx(75.89, abs=0.005)
    assert lim['As_max'] == pytest.approx(56.92, abs=0.005)
    assert lim['As_min'] == pytest.approx(10.00, abs=0.005)


@pytest.mark.parametrize('As, MR_excel', [(56.919642857, 131.1108398), (20.28, 53.6529436), (40.56, 99.6241744),
                                          (10.14, 27.7866859), (15.21, 40.9598683)])
def test_momento_resistente_coincide_con_el_excel(As, MR_excel):
    assert p3.momento_resistente(As, B, D, FC, FY) / 1e5 == pytest.approx(MR_excel, rel=1e-6)


def test_tanteo_del_eje_neutro_cumple_c_igual_a_t():
    e = p3.tanteo_eje_neutro(20.28, B, D, FC, FY)
    assert e['C'] == pytest.approx(e['T'], rel=1e-6)
    assert e['c'] == pytest.approx(20.28 * FY / (FCC * B) / 0.85, rel=1e-6)


# ---------------------------------------------------------------- cortante
@pytest.mark.parametrize('Vu, As, VcR, s', [(20660, 20.28, 13453.3802, 46.5509505), (48400, 10.14, 10677.955, 8.8933407),
                                            (28104.1, 40.56, 16950.197, 30.076916)])
def test_cortante_coincide_con_el_excel_del_examen(Vu, As, VcR, s):
    c = p3.cortante_ficha(Vu, B, D, FC, 4200, 1.42, As, 20)
    assert c['VcR'] == pytest.approx(VcR, rel=1e-6) and c['s_calc'] == pytest.approx(s, rel=1e-6)


def test_cortante_coincide_con_la_hoja_ntcc_2023():
    Av, As = 3 * 0.7125573924808561, 30.40244874584986
    c = p3.cortante_ficha(35184, 30, 65, 250, 4200, Av, As, 25)
    assert c['VcR'] / 1000 == pytest.approx(11.553685, rel=1e-6) and c['s_calc'] == pytest.approx(18.522325, rel=1e-6)
    assert p3.revision_cortante(35184, 30, 65, 250, 4200, Av, As, 20)['VR'] / 1000 == pytest.approx(33.438104, rel=1e-6)


# ---------------------------------------------------------------- análisis de vigas contra fórmulas cerradas
L, W = 6.0, 2.0


def _r(xn, tipos, w, P=None):
    return av.resolver_nodos(xn, tipos, w, P, 201)


@pytest.mark.parametrize('nombre, xn, tipos, w, P, que, ref', [
    ('simple M', [0, L], ['Articulado'] * 2, [W], None, 'Mmax', W * L ** 2 / 8),
    ('simple R', [0, L], ['Articulado'] * 2, [W], None, 'R0', W * L / 2),
    ('doble empotrada apoyo', [0, L], ['Empotrado'] * 2, [W], None, 'Mneg', W * L ** 2 / 12),
    ('doble empotrada centro', [0, L], ['Empotrado'] * 2, [W], None, 'Mmax', W * L ** 2 / 24),
    ('apoyada-empotrada', [0, L], ['Empotrado', 'Articulado'], [W], None, 'Mneg', W * L ** 2 / 8),
    ('apoyada-empotrada R', [0, L], ['Empotrado', 'Articulado'], [W], None, 'R1', 3 * W * L / 8),
    ('dos claros apoyo', [0, L, 2 * L], ['Articulado'] * 3, [W, W], None, 'Mneg', W * L ** 2 / 8),
    ('dos claros R central', [0, L, 2 * L], ['Articulado'] * 3, [W, W], None, 'R1', 5 * W * L / 4),
    ('tres claros apoyo', [0, L, 2 * L, 3 * L], ['Articulado'] * 4, [W] * 3, None, 'Mneg', W * L ** 2 / 10),
    ('volado', [0, 3], ['Empotrado', '—'], [W], [0, 5.0], 'Mneg', W * 9 / 2 + 5 * 3),
    ('carga puntual Pab/L', [0, 2, 6], ['Articulado', '—', 'Articulado'], [0, 0], [0, 10.0, 0], 'Mmax', 10 * 2 * 4 / 6)])
def test_analisis_de_vigas_contra_formulas_cerradas(nombre, xn, tipos, w, P, que, ref):
    r = _r(xn, tipos, w, P)
    val = {'Mmax': r['M'].max(), 'Mneg': -r['M'].min(), 'R0': r['R'][0], 'R1': r['R'][1]}[que]
    assert val == pytest.approx(ref, rel=1e-6), nombre


@pytest.mark.parametrize('xn, tipos, ref', [([0, 6], ['Articulado'] * 2, 5 * 2 * 6 ** 4 / 384), ([0, 3], ['Empotrado', '—'], 2 * 3 ** 4 / 8),
                                            ([0, 6], ['Empotrado'] * 2, 2 * 6 ** 4 / 384)])
def test_flechas_contra_formulas_cerradas(xn, tipos, ref):
    assert -av.flechas_unitarias(xn, tipos, [2.0], None, 201).min() == pytest.approx(ref, rel=1e-6)


def test_comprobaciones_automaticas_del_ejemplo_de_clase():
    xn, t = [0, 2, 6, 10, 13], ['—', 'Articulado', '—', 'Articulado', '—']
    env = av.envolvente_viga(xn, t, [1] * 4, [0.5] * 4, [1, 0, 3, 0, 2], [1, 0, 4, 0, 3], 'B', 0.768)
    pts = av.puntos_diseno(env)
    f = [pt.ficha_punto(p['Mu'], 40, 80, 250, 4200, '#8', 3, 1.27, 5.0, True, lecho=p['lecho']) for p in pts]
    cs = av.comprobaciones(env, pts, f, 40, 250, 4200)
    assert len(cs) >= 9 and all(c['ok'] for c in cs)
    # y detecta un error: un MR que no alcanza sale en rojo
    f[0] = dict(f[0], MR=pts[0]['Mu'] * 0.9)
    assert not all(c['ok'] for c in av.comprobaciones(env, pts, f, 40, 250, 4200))


# ---------------------------------------------------------------- pórtico: equilibrio
def test_portico_equilibrio_horizontal_y_vertical():
    res = pt.resolver([4, 5], [3, 3, 3], [(40, 40)] * 3, [(25, 50)] * 3, 14000 * 250 ** .5,
                      dict(CM=[2] * 3, CV=[.5] * 3, CVacc=[.3] * 3), [3, 6, 9])
    S, CM = res['caso']['S']['diag'], res['caso']['CM']['diag']
    assert abs(sum(S[('c', 1, e)]['V'][0] for e in (1, 2, 3))) == pytest.approx(18.0, rel=1e-9)
    assert sum(abs(float(np.ravel(CM[('c', 1, e)]['N'])[0])) for e in (1, 2, 3)) == pytest.approx(2 * 9 * 3, rel=1e-9)
