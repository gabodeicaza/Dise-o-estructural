"""Tercer parcial: valida contra los ejemplos de los apuntes del Dr. Rivero (pp. 24-35) y la guía recalculada."""
import numpy as np
import pytest

import ame_parcial3 as p


def test_muro_lindero_compresion_apuntes_p24():
    r = p.muro_compresion(14, 350, 400, 8163, tipo='Muro de lindero')
    assert not r['cumple']
    assert r['FE'] == pytest.approx(0.1782, abs=2e-4)
    assert r['PR'] == pytest.approx(8983, abs=10) and r['pasa']


def test_muro_tres_condiciones_y_interior():
    r = p.muro_compresion(14, 250, 350, 30000, tipo='Muro interior', L2L1=1.2, cv_cm=0.7)
    assert r['FE'] == 0.7 and r['PR'] == pytest.approx(30870) and r['pasa']
    assert p.muro_compresion(12, 240, 300, 18000)['PR'] == pytest.approx(19440)
    assert p.muro_compresion(12, 280, 300, 18000)['PR'] == pytest.approx(7467, abs=5)


def test_muro_cortante_apuntes():
    r = p.muro_cortante(14, 400, 350, 7520, Vu=5270)
    assert r['f'] == pytest.approx(1.0781, abs=1e-4) and r['VmR'] == pytest.approx(5928.8, abs=0.5) and r['pasa']
    assert p.muro_cortante(12, 200, 240, 25000)['VmR'] == pytest.approx(5040)          # rige el tope
    assert p.muro_cortante(14, 700, 250, 40500 * 0 + 20250)['f'] == pytest.approx(1.3125 * 0 + p.factor_forma(250, 700))


def test_estatico_apuntes_p25():
    e = p.estatico([70, 80], [5.0, 2.5], 0.6483, 0.505, 2)
    assert e['C'] == pytest.approx(0.1347, abs=1e-4)
    assert e['VBu'] == pytest.approx(22.22, abs=0.01) and e['MVU'] == pytest.approx(90.9, abs=0.1)


def test_tabla_433_y_mas_de_6_niveles():
    r = p.distorsion_limite('Confinada, piezas macizas (tabique rojo recocido)', 8)
    assert (r['Q'], r['gamma_SV']) == (1.5, 0.002)
    r = p.distorsion_limite('Confinada, piezas macizas (tabique rojo recocido)', 3, condiciones_irreg=2)
    assert r['gamma_SV_corregida'] == pytest.approx(0.0035)


def test_amin_y_excentricidad():
    assert p.a_min(0.4, 2) == pytest.approx(0.02) and p.a_min(0.75, 2) == pytest.approx(0.025) and p.a_min(1.3, 2) == pytest.approx(0.03)
    assert p.excentricidades_accidentales(4, 12) == pytest.approx([0.6, 0.8, 1.0, 1.2])


def test_viento_apuntes_p29():
    z = [3, 6, 9, 12, 15]
    v = p.viento_cara(z, 25, 30.08, 0.14, 410, 0.832, cp=0.8)
    assert list(v['AT']) == pytest.approx([112.5, 75, 75, 75, 37.5])
    assert v['VBu'] == pytest.approx(11.20, abs=0.01) and v['MVU'] == pytest.approx(88.80, abs=0.01)
    s = p.viento_cara(z, 25, 30.08, 0.14, 410, 0.832, cp=-0.4, constante_en_h=True)
    assert s['VBu'] == pytest.approx(6.14, abs=0.01) and s['MVU'] == pytest.approx(47.86, abs=0.02)
    assert p.techo_plano(35, 25, 30.08, 0.14, 410, 0.832, 15, cp=-0.8)['Fu'] == pytest.approx(28.63e3, rel=2e-3)


def test_flexion_apuntes_p32_y_p33():
    r = p.diseno_flexion(55, 35, 75, 5, 250, 4200)
    assert r['Asb'] == pytest.approx(61.98, abs=0.01) and r['As_calc'] == pytest.approx(22.90, abs=0.01)
    assert p.momento_resistente(25.35, 35, 70, 250, 4200) / 1e5 == pytest.approx(60.22, abs=0.01)
    r = p.diseno_flexion(35.03, 40, 80, 5, 250, 4200)
    assert r['As_max'] == pytest.approx(56.92, abs=0.01) and r['As_calc'] == pytest.approx(12.90, abs=0.02)
    assert not p.diseno_flexion(35, 25, 50, 5, 250, 4200)['ok']                       # C5 de la guía


def test_tanteo_examen3():
    e = p.tanteo_eje_neutro(11.94, 35, 55, 250, 4200)
    assert e['c'] == pytest.approx(7.93, abs=0.01) and e['fluye'] and e['MR'] / 1e5 == pytest.approx(23.30, abs=0.01)
    assert p.tanteo_eje_neutro(11.94, 35, 55, 250, 4200, c_prop=25)['C'] == pytest.approx(21.25 * 35 * 212.5)


def test_cortante_viga_apuntes_p35():
    r = p.cortante_viga(14830, 25, 45, 50, 500, 250, 4200, 1.42, 11.4)
    assert r['VcR'] == pytest.approx(5774, abs=2) and r['s_usar'] == 20
    assert p.cortante_viga(22620, 25, 45, 50, 500, 250, 4200, 1.42, 15.83)['s_usar'] == 10
    r = p.cortante_viga(20000, 30, 55, 60, 600, 250, 4200, 1.42, 15.21)
    assert r['VcR'] == pytest.approx(8205, abs=3) and r['s_usar'] == 20


def test_armados():
    a = p.proponer_armados(12.91)
    assert a and all(x['As'] >= 12.91 for x in a)


def test_pestana_tercer_parcial_sin_errores():
    pytest.importorskip('streamlit')
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / 'app_web.py'), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.label == 'FE' for m in at.metric)          # mampostería
    assert any(m.label == 'MR (t·m)' for m in at.metric)    # concreto
    assert any(m.label == 'VBu (t)' for m in at.metric)     # viento / estático


# ----------------------------------------------------------------- pórtico, envolvente y armado
def _portico():
    import ame_porticos as pt
    E = 14000 * 250 ** 0.5
    return pt, pt.resolver([4, 5], [3, 3, 3], [(40, 40)] * 3, [(25, 50)] * 3, E,
                           dict(CM=[2, 2, 2], CV=[.5] * 3, CVacc=[.3] * 3), [3, 6, 9])


def test_portico_equilibrio_y_empotramiento_perfecto():
    pt, res = _portico()
    assert sum(res['caso']['S']['diag'][('c', 1, e)]['V'][0] for e in (1, 2, 3)) == pytest.approx(18.0)
    E = 14000 * 250 ** 0.5
    r1 = pt.resolver([5], [3], [(300, 300)], [(25, 50)], E, dict(CM=[1.0], CV=[0.0], CVacc=[0.0]), [0.0])
    d = r1['caso']['CM']['diag'][('b', 1, 1)]
    assert d['M'][0] == pytest.approx(-25 / 12, rel=0.01) and d['M'][20] == pytest.approx(25 / 24, rel=0.01)   # wL²/12 y wL²/24


def test_envolvente_ida_regreso_simetria_y_diseno():
    pt, res = _portico()
    env, mats = pt.envolvente(res, pt.combinaciones('B'))
    ida, reg, grav = mats['Sismo ida (+)'], mats['Sismo regreso (−)'], mats['Gravedad']
    k = ('b', 2, 1)
    # el sismo suma a un lado y resta al otro: la envolvente cubre ambos y la gravedad
    assert (env[k]['Mmax'] >= np.maximum(ida[k]['M'], reg[k]['M']) - 1e-9).all()
    assert (env[k]['Mmin'] <= np.minimum(ida[k]['M'], reg[k]['M']) + 1e-9).all()
    sim = 0.5 * (ida[k]['M'] + reg[k]['M'])                   # la parte sin sismo es la misma en ida y regreso
    assert sim == pytest.approx(pt.combinar(res, (1.1, 0, 1.1, 0))[k]['M'])
    d = pt.disenar_viga(env, k, 25, 50, 5, 250, 4200, 40)
    sup = [s for s in d['secciones'] if s['lecho'] == 'superior']
    assert all(s['armado']['As'] >= s['As'] for s in d['secciones'] if s['armado'])
    assert d['cortante']['s_usar'] in (5, 10, 15, 20)
    assert [s['posicion'] for s in sup] == ['izquierdo', 'centro', 'derecho']


def test_pestana_porticos_sin_errores():
    pytest.importorskip('streamlit')
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / 'app_web.py'), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) >= 8
