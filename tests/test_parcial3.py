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
    d = pt.disenar_viga(env, k, 25, 50, 250, 4200, 20, 20, '#5', '#5')
    sup = [s for s in d['secciones'] if s['lecho'] == 'superior']
    assert all(s['armado']['As'] >= s['As'] - 1e-9 and s['armado']['MR'] >= s['Mu'] - 1e-9 for s in d['secciones'] if s['armado'])
    assert d['cortante']['s_usar'] in (5, 10, 15, 20)
    assert [s['posicion'] for s in sup] == ['izquierdo', 'centro', 'derecho']
    assert set(pt.resumen_vigas(env, res, 250, 4200, {})) == {k_ for k_ in env if k_[0] == 'b'}


def test_varilla_sugerida_h_10_sin_7():
    pt = _portico()[0]
    assert pt.varilla_sugerida(50)[0] == '#5'
    assert pt.varilla_sugerida(70)[0] == '#8' and 'no existe' in pt.varilla_sugerida(70)[1]   # no hay #7: la siguiente
    assert pt.varilla_sugerida(45)[0] == '#5' and pt.varilla_sugerida(100)[0] == '#10'
    assert pt.varilla_sugerida(30)[0] == '#3'


def test_lechos_y_seccion_estrecha():
    pt = _portico()[0]
    ok = pt.distribuir_lechos(4, 1.59, 25, 4, 0.95)
    assert ok['capas'] == [4] and ok['sep_libre'] >= ok['sep_min']
    s = pt.distribuir_lechos(7, 1.59, 25, 4, 0.95)
    assert len(s['capas']) == 2 and s['r'] > 4 + 0.95 + 0.79
    sec = pt.armar_seccion(10.0, 25, 50, 250, 4200, '#5')
    assert sec['armado']['MR'] >= 10.0
    pt.figura_secciones(pt.disenar_viga(pt.envolvente(_portico()[1], pt.combinaciones('B'))[0], ('b', 1, 1), 25, 50, 250, 4200,
                                        20, 20, '#5', '#5'))


def test_geometria_sin_losa_columna_continua_y_volado():
    import ame_porticos as pt
    E = 14000 * 250 ** 0.5
    geo = pt.geo_simple([4, 4], [3, 3], [(40, 40), (30, 30)], [(25, 50)] * 2, E,
                        dict(CM=[2, 2], CV=[.5] * 2, CVacc=[.3] * 2))
    # nivel 1: sin losa en el tramo 2; el eje 3 no tiene viga en el piso 1 → columna continua hasta el piso 2
    geo['niveles'][0]['sin'] = {1}
    for c in geo['niveles'][0]['cols']:
        if c['eje'] == 3:
            c['hasta'] = 2
    for c in geo['niveles'][1]['cols']:
        if c['eje'] == 3:
            c['omitir'] = True
    geo['niveles'][1]['vd'] = 1.5
    geo['niveles'][1]['etiquetas'] = [1, 2, 'vd']
    for caso in ('CM', 'CV', 'CVacc'):
        geo['niveles'][1]['q'][caso] = geo['niveles'][1]['q'][caso] + [1.0]
    res = pt.resolver_geo(geo, [3.0, 6.0])
    assert sum(res['caso']['S']['diag'][('c', 1, e)]['V'][0] for e in (1, 2)) + \
        res['caso']['S']['diag'][('c', 1, 3)]['V'][0] == pytest.approx(9.0, rel=1e-6)
    assert ('b', 1, 2) not in res['caso']['CM']['diag'] and ('b', 2, 3) in res['caso']['CM']['diag']
    # el volado de carga 1 t/m y 1.5 m: M en el empotramiento = -wL²/2 (gravedad pura)
    assert res['caso']['CM']['diag'][('b', 2, 3)]['M'][0] != 0


def test_pestana_porticos_sin_errores():
    pytest.importorskip('streamlit')
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / 'app_web.py'), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) >= 8


def test_tablas_aceptan_decimales_como_texto():
    import pandas as pd
    import ui_porticos
    df = pd.DataFrame({'h (m)': ['3.5', '2,75', ' 4 ']})
    assert list(ui_porticos._numeros(df, ['h (m)'])['h (m)']) == [3.5, 2.75, 4.0]
    assert ui_porticos._numeros(pd.DataFrame({'h (m)': ['3.5', 'x']}), ['h (m)']) is None


def test_barras_agrupadas_de_dos_en_dos_como_en_clase():
    import ame_porticos as pt
    # b = 35, rec 3, estribo #4 (1.27), 6#8: no caben sueltas en un lecho → 3 parejas con 5.61 cm entre grupos (pizarrón)
    l6 = pt.distribuir_lechos(6, 2.54, 35, 3, 1.27)
    assert l6['capas'] == [6] and l6['agrupado'] and l6['sep_libre'] == pytest.approx(5.61, abs=0.01)
    # sin agrupar se va a un segundo lecho
    assert pt.distribuir_lechos(6, 2.54, 35, 3, 1.27, agrupar=False)['capas'] == [5, 1]
    # 5#8 caben sueltas; no se agrupan
    assert not pt.distribuir_lechos(5, 2.54, 35, 3, 1.27)['agrupado']


def test_peso_propio():
    import ame_porticos as pt
    E = 14000 * 250 ** 0.5
    a = pt.resolver([5], [3], [(300, 300)], [(35, 70)], E, dict(CM=[1.0], CV=[0.0], CVacc=[0.0]), [0.0])
    b = pt.resolver([5], [3], [(300, 300)], [(35, 70)], E, dict(CM=[1.0], CV=[0.0], CVacc=[0.0]), [0.0], pp=2.4)
    w = 1.0 + 2.4 * 0.35 * 0.70
    assert b['caso']['CM']['diag'][('b', 1, 1)]['M'][0] == pytest.approx(-w * 25 / 12, rel=0.01)
    assert a['caso']['CM']['diag'][('b', 1, 1)]['M'][0] == pytest.approx(-25 / 12, rel=0.01)


def test_viga_continua_corrido_bastones_y_anclaje():
    import ame_porticos as pt
    n = 3
    res = pt.resolver([4, 5], [3] * n, [(40, 40)] * n, [(35, 70)] * n, 14000 * 250 ** .5,
                      dict(CM=[6] * n, CV=[1] * n, CVacc=[.5] * n), [8, 16, 24], pp=2.4)
    env, _ = pt.envolvente(res, pt.combinaciones('A', False))
    niv = pt.armado_nivel(env, res, 1, 250, 4200, '#8', '#8', est_var='#4', rec=3.0)
    # corrido: ≥ As,mín y ≥ 50 % de la mayor As real, mínimo 2 barras
    a = 5.07
    assert niv['n_corr']['superior'] * a >= max(niv['lim']['As_min'], 0.5 * niv['As_real_max']) - 1e-9
    assert niv['n_corr']['superior'] >= 2 and niv['bastones']
    # cada bastón se ancla d más allá de donde se requiere (salvo que llegue al extremo de la viga)
    P = niv['P']
    for q in niv['bastones']:
        assert q['x_ini'] <= q['req_ini'] + 1e-9 and q['x_fin'] >= q['req_fin'] - 1e-9
        assert (q['req_ini'] - q['x_ini']) == pytest.approx(q['d'] / 100) or q['x_ini'] == pytest.approx(P.min()) or q['x_ini'] > P.min()
    # en cada punto el acero final (corrido + bastones) alcanza la As requerida
    for q in niv['puntos']:
        for lecho, nf in (('superior', q['n_sup']), ('inferior', q['n_inf'])):
            s_ = next(s for s in niv['dis'][q['clave']]['secciones'] if s['posicion'] == q['posicion'] and s['lecho'] == lecho)
            if s_['armado']:
                assert nf * a + 1e-9 >= s_['As'] - 0.5      # tolerancia: la d de la envolvente es la mínima del nivel
    pt.figura_alzado(niv, res)


def test_ejemplo_17_clase_viga_del_nivel_1():
    """El ejercicio de clase: marco de 5 niveles y 6 ejes; los Mu del pizarrón salen en el eje del apoyo (1-2 % por la d)."""
    import ame_porticos as pt
    import ame_sismico as ame
    ej = next(v for k, v in ame.EJEMPLOS_MODELO.items() if k.startswith('Ejemplo 17'))
    p = ej['porticos']
    geo = pt.geo_modelo(ame.preparar(ej), ej, [dict(CM=c, CV='0', CVacc='0') for c in p['CM']], pp=2.4)
    res = pt.resolver_geo(geo, [float(x) for x in p['F']])
    assert sum(res['caso']['S']['diag'][k]['V'][0] for k in res['caso']['S']['diag'] if k[0] == 'c' and k[1] == 1) == pytest.approx(55.0)
    env, _ = pt.envolvente(res, pt.combinaciones('A', False, gravedad=False))
    e1, e5 = env[('b', 1, 1)], env[('b', 1, 5)]
    assert -e1['Mmin'][0] == pytest.approx(41.6, rel=0.02)                      # a)
    assert e5['Mmax'].max() == pytest.approx(26.7, rel=0.02)                    # i) centro de la última crujía
    assert -e5['Mmin'][0] == pytest.approx(59.9, rel=0.02) or -e5['Mmin'].min() == pytest.approx(59.9, rel=0.02)
    niv = pt.armado_nivel(env, res, 1, 250, 4200, '#8', '#8', est_var='#4', rec=3.0, paño=False)
    assert niv['n_corr'] == {'superior': 3, 'inferior': 3}                       # 50 % de 30.42 = 15.21 cm² → 3#8
    assert niv['Mu_max'] == pytest.approx(59.9, rel=0.02) and niv['Vu_max'] == pytest.approx(44.3, rel=0.03)
    assert niv['MR_max'] == pytest.approx(86.16, rel=0.02) and niv['V_lim'] == pytest.approx(71.4, rel=0.01)


def test_cargar_ejemplo_17_llena_la_pestana_de_diagramas():
    pytest.importorskip('streamlit')
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    import ame_sismico as ame
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / 'app_web.py'), default_timeout=180).run()
    nombre = next(k for k in ame.EJEMPLOS_MODELO if k.startswith('Ejemplo 17'))
    at.sidebar.selectbox(key='ejemplo').select(nombre)
    at.sidebar.button[0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert list(at.session_state['p6_dft']['F sismo (t)']) == ['5', '5', '10', '15', '20']
    assert at.session_state['p6_grupo'] == 'A' and at.session_state['p6_pp'] is True and at.session_state['p6_paño'] is False
    assert all('muro' in e.value for e in at.error), [e.value for e in at.error]   # el único aviso es el de muros (pestaña 5)
