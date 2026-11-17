"""Examen 3 · viga por secciones y armado de toda la viga (ame_seccion)."""
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ame_seccion as sx  # noqa: E402

# Viga de la hoja de clase (b = 40, h = 80, r = 5, Mu = 35.03): ver test_parcial3 / test_verificacion
D = dict(b=40, h=80, r=5, fc=250, fy=4200, var='#8', est='#4', ramas=4, fy_est=4200, s_tope=25)
P = [dict(punto='a', Mu=35.03, lecho='superior'), dict(punto='b', Mu=11.7, lecho='inferior'),
     dict(punto='c', Mu=20.84, lecho='superior')]
T = [dict(tramo='1', Vu=20.07), dict(tramo='2', Vu=14.23, s_prop=25)]

# El pizarrón de la viga de 5 claros (35 × 70, E#4 de 2 ramas, d = 65); las longitudes de los claros son supuestas
FOTO = [dict(tipo='Apoyo', Mu=41.6), dict(tipo='Centro de claro', Mu=15.26),
        dict(tipo='Apoyo', Mu=49.4, L=6.7, Vu=36.2), dict(tipo='Centro de claro', Mu=18.8),
        dict(tipo='Apoyo', Mu=48.22, L=6.8, Vu=39.6), dict(tipo='Centro de claro', Mu=13.55),
        dict(tipo='Apoyo', Mu=40.1, L=6.5, Vu=35.36), dict(tipo='Centro de claro', Mu=15.4),
        dict(tipo='Apoyo', Mu=59.9, L=7.2, Vu=36.5), dict(tipo='Centro de claro', Mu=26.7),
        dict(tipo='Apoyo', Mu=58.6, L=7.8, Vu=44.3)]
DF = dict(b=35, h=70, r=5, fc=250, fy=4200, var='#8', est='#4', ramas=2, fy_est=4200, s_tope=25)


def test_viga_de_la_hoja_de_clase_por_secciones():
    r = sx.calcular(D, P, T)
    assert r['ok'] and r['error'] is None
    assert r['lim']['Asb'] == pytest.approx(75.89, abs=0.01)
    assert r['lim']['As_max'] == pytest.approx(56.92, abs=0.01)
    assert r['lim']['As_min'] == pytest.approx(10.0, abs=0.01)
    assert r['viga']['MR_max'] == pytest.approx(131.11, abs=0.01)
    assert r['puntos'][0]['ficha']['armado'] == '3#8'
    assert all(e['ficha']['MR'] + 1e-9 >= e['Mu'] for e in r['puntos'])


def test_corrido_bastones_y_estribos_de_toda_la_viga():
    r = sx.calcular(D, P, T)
    cor = r['corrido']
    mayor = max(e['ficha']['As_real'] for e in r['puntos'])
    assert cor['As_max_real'] == pytest.approx(mayor)
    assert cor['As_corr'] == pytest.approx(max(r['lim']['As_min'], 0.5 * mayor))
    assert [p for p, _ in cor['bastones']] == [e['punto'] for e in r['puntos'] if e['ficha']['n'] > cor['n']]
    sup = max(e['ficha']['As_real'] for e in r['puntos'] if e['lecho'] == 'superior')
    assert r['As_auto'] == pytest.approx(sup)
    assert r['tramos'][0]['cort']['rho'] == pytest.approx(sup / (40 * 75), rel=1e-9)
    assert r['tramos'][1]['rev'] is not None
    assert [x[0] for x in sx.resumen(r)][:1] == ['Varillas corridas']


def test_as_propio_de_un_tramo_y_armado_propuesto_con_tanteo():
    r = sx.calcular(D, [dict(P[0], prop='4#8', c_prop=10)] + P[1:], [dict(T[0], As=30.0)] + T[1:])
    assert r['puntos'][0]['ficha']['armado'] == '4#8' and r['puntos'][0]['ficha']['propuesto']
    assert r['puntos'][0]['tanteo_manual']['c'] == 10
    assert r['tramos'][0]['As'] == 30.0 and r['tramos'][0]['cort']['rho'] == pytest.approx(30 / (40 * 75))


def test_punto_que_no_alcanza_no_tumba_a_los_demas_y_se_exporta():
    r = sx.calcular({**D, 'b': 20, 'h': 30}, [dict(punto='a', Mu=60, lecho='superior'), dict(punto='b', Mu=2, lecho='inferior')], T[:1])
    assert not r['ok'] and r['puntos'][0]['error'] and not r['puntos'][1]['error']
    assert sx.pdf(r)[:4] == b'%PDF' and 'Viga de concreto reforzado' in sx.texto(r)
    todos = sx.calcular({**D, 'b': 20, 'h': 30}, [dict(punto='a', Mu=60, lecho='superior')], T[:1])
    assert todos['error'] and sx.pdf(todos)[:4] == b'%PDF'


def test_datos_absurdos_se_rechazan():
    with pytest.raises(ValueError):
        sx.calcular({**D, 'r': 60}, P, T)
    with pytest.raises(ValueError):
        sx.calcular({**D, 'b': 0}, P, T)
    with pytest.raises(ValueError):
        sx.calcular(D, [], T)
    with pytest.raises(ValueError):
        sx.calcular(D, P, [])


# ------------------------------------------------------------------ la viga completa del pizarrón
def test_el_pizarron_da_las_varillas_y_los_bastones_de_la_foto():
    r = sx.calcular_viga(DF, FOTO)
    assert r['ok'] and r['corrido']['n'] == 3                                     # corrido 3#8 arriba y abajo
    assert [b['n'] for b in r['bastones']] == [1, 2, 2, 1, 3, 3]                  # 1#8, 2#8, 2#8, 1#8, 3#8, 3#8
    assert [b['punto'] for b in r['bastones']] == ['a', 'c', 'e', 'g', 'i', 'k']
    for b in r['bastones']:                                                        # ≈ 2.15 – 2.45 m a cada lado, como en la foto
        for lado in (b['izq'], b['der']):
            assert lado == 0.0 or 2.0 < lado < 2.8
    assert r['bastones'][0]['gancho_izq'] and r['bastones'][-1]['gancho_der']     # ganchos en los extremos de la viga
    assert r['d_ef'] == 65
    assert [e['s'] for e in r['estribos'] if e['confinado']] and all(e['s'] <= 25 for e in r['estribos'] if e['s'])


def test_el_momento_del_claro_pasa_por_los_tres_valores_capturados():
    r = sx.calcular_viga(DF, FOTO)
    env = r['env']
    for i, s in enumerate(r['geom']['spans']):
        assert env['Mmax'][i][0] == pytest.approx(-s['Ml']) and env['Mmax'][i][-1] == pytest.approx(-s['Mr'])
        assert env['Mmax'][i][len(env['Mmax'][i]) // 2] == pytest.approx(s['Mc'])
        assert env['Vmax'][i][0] == pytest.approx(s['Vu']) and env['Vmax'][i][-1] == pytest.approx(-s['Vu'])


@pytest.mark.parametrize('lista, texto', [
    ([], 'al menos dos apoyos'),
    ([dict(tipo='Centro de claro', Mu=10)], 'primera sección'),
    ([dict(tipo='Apoyo', Mu=10), dict(tipo='Apoyo', Mu=10, L=5, Vu=5)], 'Falta el momento positivo'),
    ([dict(tipo='Apoyo', Mu=10), dict(tipo='Centro de claro', Mu=5), dict(tipo='Apoyo', Mu=10)], 'L y su Vu'),
    ([dict(tipo='Apoyo', Mu=10), dict(tipo='Centro de claro', Mu=5)], 'no está cerrado'),
    ([dict(tipo='Apoyo', Mu=10), dict(tipo='Centro de claro', Mu=5), dict(tipo='Centro de claro', Mu=5)], 'sobra'),
    ([dict(tipo='Apoyo', Mu=-10)], 'positivos'),
])
def test_secciones_incompletas_dan_un_mensaje_claro(lista, texto):
    with pytest.raises(ValueError, match=texto):
        sx.calcular_viga(DF, lista)


def test_una_viga_de_un_solo_claro_y_la_figura_se_dibuja():
    import matplotlib.pyplot as plt
    r = sx.calcular_viga(DF, [dict(tipo='Apoyo', Mu=20), dict(tipo='Centro de claro', Mu=25), dict(tipo='Apoyo', Mu=20, L=6, Vu=22)])
    assert len(r['geom']['spans']) == 1 and r['bastones'] == [] or all(b['n'] >= 1 for b in r['bastones'])
    fig = sx.figura_hoja(r)
    assert len(fig.axes) >= 4
    plt.close(fig)


def test_exportaciones_pdf_excel_y_texto():
    from openpyxl import load_workbook
    r = sx.calcular_viga(DF, FOTO)
    assert sx.pdf(r, 'Prueba', 'Autora')[:4] == b'%PDF'
    ws = load_workbook(io.BytesIO(sx.excel(r))).active
    valores = [str(c.value) for fila in ws.iter_rows() for c in fila if c.value is not None]
    assert any('PASO 5' in v for v in valores) and any('Bastones' in v for v in valores)
    txt = sx.texto(r)
    for paso in ('PASO 1', 'PASO 2', 'PASO 3', 'PASO 4', 'PASO 5', 'PASO 6'):
        assert paso in txt
    assert 'ARMADO DE TODA LA VIGA' in txt and 'Estribos' in txt and 'Varillas corridas' in txt and 'Bastones' in txt


def test_columnas_de_captura_arman_la_lista_de_secciones():
    assert sx.parse_columna('41.6' + chr(10) + '15,26' + chr(10) + ' 49.4', 'M') == [41.6, 15.26, 49.4]
    assert sx.parse_columna('6.7 6.8, 6.5', 'L') == [6.7, 6.8, 6.5]
    assert sx.parse_columna('', 'L') == []
    with pytest.raises(ValueError, match='Momentos'):
        sx.parse_columna('4x', 'Momentos')
    lista = sx.lista_desde_columnas([41.6, 15.26, 49.4, 18.8, 48.22], [6.7, 6.8], [36.2, 39.6])
    assert [x['tipo'] for x in lista] == ['Apoyo', 'Centro de claro', 'Apoyo', 'Centro de claro', 'Apoyo']
    assert lista[2]['L'] == 6.7 and lista[2]['Vu'] == 36.2 and lista[4]['L'] == 6.8 and lista[4]['Vu'] == 39.6
    assert [x['Vu'] for x in sx.lista_desde_columnas([1, 2, 3, 4, 5], [6, 6], [30])[2::2]] == [30, 30]      # un Vu para todas
    with pytest.raises(ValueError, match='5 momentos'):
        sx.lista_desde_columnas([1, 2, 3], [6, 6], [30])
    with pytest.raises(ValueError, match='Vu por crujía'):
        sx.lista_desde_columnas([1, 2, 3, 4, 5], [6, 6], [30, 30, 30])


def test_la_pagina_arma_el_render_desde_tres_columnas():
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app_web.py'), default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any('Armado de toda la viga' in m.value for m in at.markdown)           # abre con el ejemplo del pizarrón
    area = {a.label: a for a in at.text_area}
    area['Momentos Mu (t·m), en positivo'].set_value(chr(10).join(['20', '25', '20']))
    area['Largo de cada crujía (m)'].set_value('6')
    area['Vu de cada crujía (t)'].set_value('22')
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any('Armado de toda la viga' in m.value for m in at.markdown)
    area = {a.label: a for a in at.text_area}
    area['Momentos Mu (t·m), en positivo'].set_value(chr(10).join(['20', '25']))   # faltan datos: mensaje claro, sin romper
    at.run()
    assert not at.exception and any('necesitas 3 momentos' in w.value for w in at.warning)
