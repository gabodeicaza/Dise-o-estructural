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
    assert cor['n'] * 5.07 >= cor['As_corr'] and (cor['n'] - 1) * 5.07 < cor['As_corr'] or cor['n'] == 2
    assert [p for p, _ in cor['bastones']] == [e['punto'] for e in r['puntos'] if e['ficha']['n'] > cor['n']]
    # el As en tensión por omisión es el mayor As real superior y alimenta el cortante
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


def test_una_seccion_sola_es_un_punto_y_un_tramo():
    r = sx.calcular(D, P[:1], T[:1])
    assert r['ok'] and len(r['puntos']) == 1 and len(r['tramos']) == 1


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


def test_exportaciones_pdf_excel_y_texto():
    from openpyxl import load_workbook
    r = sx.calcular(D, P, T)
    assert sx.pdf(r, 'Prueba', 'Autora')[:4] == b'%PDF'
    ws = load_workbook(io.BytesIO(sx.excel(r))).active
    valores = [str(c.value) for fila in ws.iter_rows() for c in fila if c.value is not None]
    assert any('3#8' in v for v in valores) and any('PASO 5' in v for v in valores)
    txt = sx.texto(r)
    for paso in ('PASO 1', 'PASO 2', 'PASO 3', 'PASO 4', 'PASO 5', 'PASO 6'):
        assert paso in txt
    assert 'ARMADO DE TODA LA VIGA' in txt and 'Estribos' in txt and 'Varillas corridas' in txt
