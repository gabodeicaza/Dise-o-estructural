"""Pruebas automáticas del Análisis Modal Espectral.

Ejecutar desde la carpeta del proyecto:   python -m pytest -q

Los casos de las hojas (14, 18 y 19) son los resultados hechos a mano que sirven de referencia: si una
modificación futura los cambia, estas pruebas fallan.
"""
import copy
import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ame_pdf  # noqa: E402
import ame_sismico as ame  # noqa: E402
from ame_marco import rigidez_marco  # noqa: E402
from ame_revision import combinar, irregularidades, rho_cqc, theta_pdelta  # noqa: E402
from ame_secciones import parse_secciones  # noqa: E402

EJ = ame.EJEMPLOS_MODELO
EJ1, EJ2, EJ3, EJ4, EJ5 = list(EJ.values())


def correr(d):
    return ame.calcular(ame.preparar_modelo(d))


# ---------------------------------------------------------------- casos de las hojas
def test_hoja14_marco_de_un_crujia():
    r = correr(EJ1)
    p = r['p']
    assert p['K'][0, 0] == pytest.approx(20985.1162, rel=1e-5)
    assert p['K'][1, 1] == pytest.approx(17088.7895, rel=1e-5)
    assert p['K'][2, 2] == pytest.approx(2876.65, rel=1e-5)
    assert np.diag(p['M']) == pytest.approx([12.2324, 12.2324, 9.1743], rel=1e-4)
    assert r['T'] == pytest.approx([0.5434, 0.2679, 0.1200], abs=5e-5)
    assert r['gam'] == pytest.approx([0.5851, 0.3022, 0.1127], abs=1e-4)
    assert r['Sa_modal'][:, 0] == pytest.approx([60.475, 59.00, 59.452], abs=1e-2)
    assert r['F_final'] == pytest.approx([491.54, 632.73, 800.56], abs=0.02)
    t = r['tabla_din']
    assert t['Vu'][-1] == pytest.approx(2.117, abs=1e-3)
    assert t['Mvu'].sum() == pytest.approx(19.941, abs=2e-3)


def test_hoja18_dos_crujias():
    r = correr(EJ2)
    assert np.diag(r['p']['M']) == pytest.approx([43.0431, 42.8128, 31.8807], abs=0.011)
    assert r['p']['K'][1, 1] == pytest.approx(69096.21, rel=1e-6)
    assert r['T'] == pytest.approx([0.6800, 0.1912, 0.1230], abs=5e-5)
    assert r['gam'] == pytest.approx([0.83623, 0.14614, 0.01763], abs=1e-4)
    assert r['Sa_modal'][:, 0] == pytest.approx([87.696, 86.218, 87.11], abs=2e-3)
    assert r['F_final'] == pytest.approx([3203.4, 3884.5, 3164.8], abs=0.2)
    t = r['tabla_din']
    assert t['Fu'] == pytest.approx([3.48, 4.27, 3.52], abs=0.01)
    assert t['Vu'][-1] == pytest.approx(11.28, abs=0.005)
    assert t['Mvu'].sum() == pytest.approx(95.72, abs=0.01)


def datos_hoja19(Q='1', sismo='B'):
    K = np.array([[300, -150, 0, 0], [-150, 225, -75, 0], [0, -75, 150, -75], [0, 0, -75, 75]]) * 1000
    M = np.diag([3, 3, 1.5, 1.5]) * 1000
    fmt = lambda A: '\n'.join(' '.join(map(str, f)) for f in A)
    return dict(K=fmt(K), M=fmt(M), niveles='5\n5\n5\n5', grupo='A', sismo=sismo, k1='0.8', Q=Q, a0='80', c='500',
                Ta='0.4', Tb='1.6', k='0.605', Ts='0.53', factor_Fu='1.1', mult_V='1')


def test_hoja19_ocupacion_inmediata_con_K_y_M_manuales():
    r = ame.calcular(ame.preparar_datos(datos_hoja19()))
    assert r['T'] == pytest.approx([2.1790, 0.9939, 0.5735, 0.5020], abs=5e-5)
    assert r['Sa_modal'][:, 0] == pytest.approx([210.02, 476.19, 476.19, 476.19], abs=0.01)
    assert r['F_final'] / 1000 == pytest.approx([608.39, 763.3, 426.32, 601.01], abs=0.02)
    t = r['tabla_din']
    assert t['Vu'] == pytest.approx([661.11, 1130.1, 1969.7, 2638.9], abs=0.05)
    assert t['Mvu'].sum() == pytest.approx(31999, abs=1)


def test_estado_ocupacion_inmediata_equivale_a_Q_uno():
    base = ame.calcular(ame.preparar_datos(datos_hoja19(Q='1')))
    d = datos_hoja19(Q='2')
    d['estado'] = 'Ocupación inmediata'
    oi = ame.calcular(ame.preparar_datos(d))
    assert oi['F_final'] == pytest.approx(base['F_final'])
    assert oi['Qs'] == pytest.approx([1.0])


# ---------------------------------------------------------------- modelo por ejes
def test_ejemplo3_columna_faltante_circulares_y_cargas_por_crujia():
    p = ame.preparar_modelo(EJ3)
    E, I, Ic = 158000, 25 * 50 ** 3 / 12, np.pi * 50 ** 4 / 64
    esperado = [3 * 12 * E * I / 500 ** 3, 2 * 12 * E * I / 350 ** 3, 3 * 12 * E * Ic / 350 ** 3]
    assert [r['k'] for r in p['info']] == pytest.approx(esperado)
    assert [r['ncol'] for r in p['info']] == [3, 2, 3]
    assert [r['W'] for r in p['info']] == pytest.approx([4.7 * 9, 4.0 * 4 + 5.5 * 5, 3.0 * 4 + 3.8 * 5])


@pytest.mark.parametrize('cambio, texto', [
    (lambda d: d['niveles'][1].update(ejes='1 5'), 'el eje 5 no existe'),
    (lambda d: d['niveles'][1].update(seccion='25x50 30x30 40x40'), 'escribiste 3 secciones'),
    (lambda d: d['niveles'][1].update(artic='2'), 'no tiene columna en ese nivel'),
    (lambda d: d['niveles'][0].update(cargas='1 2 3'), 'escribiste 3 cargas'),
    (lambda d: d['niveles'][2].update(seccion='redonda'), 'no es válida'),
    (lambda d: d['niveles'][2].update(seccion=''), 'falta la sección'),
    (lambda d: d['niveles'][2].update(ejes='abc'), 'no es un eje válido'),
    (lambda d: d['niveles'][0].update(h='0'), 'mayor que cero'),
    (lambda d: d.update(crujias='4 -5'), 'mayores que cero'),
    (lambda d: d.update(E='', fc=''), "Indica E o f'c"),
    (lambda d: d.update(estado='inventado'), 'Estado límite no válido'),
    (lambda d: d.update(combinacion='XYZ'), 'Combinación modal no válida'),
    (lambda d: d.update(vigas='flexibles'), 'falta la sección'),  # sin viga en los niveles
])
def test_errores_de_captura_con_mensaje_claro(cambio, texto):
    d = copy.deepcopy(EJ3)
    cambio(d)
    with pytest.raises(ValueError, match=texto):
        ame.preparar_modelo(d)


def test_variantes_de_escritura_aceptadas():
    d = copy.deepcopy(EJ3)
    d['niveles'][2]['seccion'] = 'ø 50, d50 Ø50'
    d['niveles'][1]['seccion'] = '25 X 50'
    d['niveles'][1]['ejes'] = '1,3'
    p = ame.preparar_modelo(d)
    assert [r['seccion'] for r in p['info']] == ['25x50', '25x50', 'Ø50']


def test_proyecto_en_formato_anterior_se_convierte():
    viejo = dict(niveles=[dict(h='4.5', ncol='2', b='25', hs='50', nart='1', w='2.0', L='6', W=''),
                          dict(h='3.7', ncol='2', b='25', hs='45', nart='0', w='2.0', L='6', W=''),
                          dict(h='5.2', ncol='2', b='20', hs='40', nart='0', w='1.5', L='6', W='')],
                 E='158000', fc='', tipo='Grupo B', Q='4', k1='1.0', a0='153', c='636', Ta='0.6', Tb='1.8',
                 k='0.505', Ts='0.9')
    assert correr(ame.convertir_legacy(viejo))['T'] == pytest.approx(correr(EJ1)['T'])


# ---------------------------------------------------------------- secciones
def test_propiedades_de_secciones():
    rect = parse_secciones('25x50', 's')[0]
    assert rect['Ix'] == pytest.approx(25 * 50 ** 3 / 12) and rect['Iy'] == pytest.approx(50 * 25 ** 3 / 12)
    circ = parse_secciones('Ø60', 's')[0]
    assert circ['Ix'] == pytest.approx(np.pi * 60 ** 4 / 64) and circ['A'] == pytest.approx(np.pi * 900)
    tubo = parse_secciones('Ø60e5', 's')[0]
    assert tubo['Ix'] == pytest.approx(np.pi * (60 ** 4 - 50 ** 4) / 64)
    hueca = parse_secciones('30x40e4', 's')[0]
    assert hueca['Ix'] == pytest.approx((30 * 40 ** 3 - 22 * 32 ** 3) / 12)
    # perfil I armado: fórmula cerrada de la inercia fuerte y débil
    bf, tf, d, tw = 30, 1.6, 40, 1.0
    H = parse_secciones('H:30x1.6x40x1', 's')[0]
    fuerte = bf * d ** 3 / 12 - (bf - tw) * (d - 2 * tf) ** 3 / 12
    assert H['Ix'] == pytest.approx(fuerte)
    assert H['Iy'] < H['Ix'] / 5
    assert H['A'] == pytest.approx(2 * bf * tf + (d - 2 * tf) * tw)
    # sección T: el eje neutro está en el centroide
    T = parse_secciones('T:50x10x60x15', 's')[0]
    A = 50 * 10 + 50 * 15
    yc = (50 * 10 * 55 + 50 * 15 * 25) / A
    esperado = 50 * 10 ** 3 / 12 + 500 * (55 - yc) ** 2 + 15 * 50 ** 3 / 12 + 750 * (25 - yc) ** 2
    assert T['Ix'] == pytest.approx(esperado)
    gen = parse_secciones('I:260000/A:1250/I2:45000', 's')[0]
    assert (gen['Ix'], gen['Iy'], gen['A']) == (260000, 45000, 1250)


def test_direccion_del_sismo_gira_las_secciones():
    d = copy.deepcopy(EJ2)
    kx = ame.preparar_modelo(d)['info'][0]['k']
    d['direccion'] = 'Y'
    ky = ame.preparar_modelo(d)['info'][0]['k']
    assert ky / kx == pytest.approx((50 * 25 ** 3) / (25 * 50 ** 3))  # (h·b³)/(b·h³) = (b/h)²


def test_materiales():
    d = copy.deepcopy(EJ2)
    d.update(E='', fc='250')
    assert ame.preparar_modelo(d)['E'] == pytest.approx(14000 * np.sqrt(250))
    d['clase'] = 'Clase 2 (8000·√f\'c)'
    assert ame.preparar_modelo(d)['E'] == pytest.approx(8000 * np.sqrt(250))
    d.update(material='Acero', E='')
    assert ame.preparar_modelo(d)['E'] == 2_040_000
    d.update(material='Mampostería', E='')
    with pytest.raises(ValueError, match='mampostería'):
        ame.preparar_modelo(d)


# ---------------------------------------------------------------- vigas flexibles
def test_marco_con_vigas_rigidas_coincide_con_modelo_de_cortante():
    E, I = 158000, 25 * 50 ** 3 / 12
    h = [500, 350, 350]
    cols = [[dict(eje=1, A=1e9, I=I, artic=False), dict(eje=2, A=1e9, I=I, artic=False)]] * 3
    K = rigidez_marco(E, h, [0, 600], cols, [dict(I=1e13, A=1e9)] * 3)
    ks = [2 * 12 * E * I / x ** 3 for x in h]
    Kc = np.array([[ks[0] + ks[1], -ks[1], 0], [-ks[1], ks[1] + ks[2], -ks[2]], [0, -ks[2], ks[2]]])
    assert K == pytest.approx(Kc, rel=1e-5, abs=0.05)


@pytest.mark.parametrize('nivel_art', [0, 1])
def test_articulacion_en_cualquier_nivel_con_vigas_rigidas(nivel_art):
    E, I = 158000, 25 * 50 ** 3 / 12
    h = [500, 350, 350]
    cols = [[dict(eje=1, A=1e9, I=I, artic=False), dict(eje=2, A=1e9, I=I, artic=(i == nivel_art))] for i in range(3)]
    K = rigidez_marco(E, h, [0, 600], cols, [dict(I=1e13, A=1e9)] * 3)
    k = [(12 + (3 if i == nivel_art else 12)) * E * I / h[i] ** 3 for i in range(3)]
    assert K[0, 0] == pytest.approx(k[0] + k[1], rel=1e-5)


def test_vigas_flexibles_dan_marco_mas_flexible_y_matriz_simetrica():
    rig = correr(EJ4 | dict(vigas='rigidas'))
    fle = correr(EJ4)
    assert fle['T'][0] > rig['T'][0]
    K = fle['p']['K']
    assert K == pytest.approx(K.T)
    assert np.all(np.linalg.eigvalsh(K) > 0)  # estable
    assert fle['p']['vigas'] == 'flexibles'


def test_marco_de_acero_corre():
    r = correr(EJ5)
    assert r['p']['E'] == 2_040_000
    assert np.all(np.diff(r['T']) < 0) and r['T'][0] > 0.1


# ---------------------------------------------------------------- combinación modal
def test_combinaciones_modales():
    w = np.array([10.0, 30.0, 60.0])
    v = np.array([[3.0, 2.0, 1.0], [-1.0, 4.0, 2.0]])
    srss = combinar(v, w, 0.05, 'SRSS')
    assert srss == pytest.approx(np.sqrt((v ** 2).sum(axis=1)))
    assert combinar(v, w, 0.05, 'Suma absoluta') == pytest.approx(np.abs(v).sum(axis=1))
    # modos bien separados: CQC ≈ SRSS; con amortiguamiento ~0 son idénticos
    assert combinar(v, w, 0.05, 'CQC') == pytest.approx(srss, rel=0.05)
    assert combinar(v, w, 1e-6, 'CQC') == pytest.approx(srss, rel=1e-6)
    # modos casi iguales: CQC suma casi como valores absolutos
    cercanos = np.array([[1.0, 1.0]])
    assert combinar(cercanos, np.array([10.0, 10.2]), 0.05, 'CQC')[0] > 1.9
    rho = rho_cqc(w, 0.05)
    assert rho == pytest.approx(rho.T) and np.diag(rho) == pytest.approx(1.0)
    assert np.all(rho <= 1.0 + 1e-12)


def test_srss_cqc_y_suma_absoluta_ordenados_en_un_caso_real():
    d = copy.deepcopy(EJ2)
    resultados = {}
    for comb in ('SRSS', 'CQC', 'Suma absoluta'):
        d['combinacion'] = comb
        resultados[comb] = correr(d)['V_din']
    assert resultados['Suma absoluta'] >= resultados['SRSS'] - 1e-9
    assert resultados['Suma absoluta'] >= resultados['CQC'] - 1e-9


# ---------------------------------------------------------------- estados límite y revisiones
def test_limitacion_de_danos_usa_Sa_por_Ks():
    d = copy.deepcopy(EJ2)
    d.update(estado='Limitación de daños', Ks='0.25')
    r = correr(d)
    sa_el, _ = ame.espectro(r['p'], 4.0, 1.0, r['serie'], dl=True)
    assert r['Sa_modal'][:, 0] == pytest.approx(np.interp(r['T'], r['serie'], sa_el))
    assert r['a_min'] == 0.0


def test_distorsion_dl_es_Ks_entre_FC_de_la_de_vida():
    r = correr(EJ2)
    est = r['revision']['estados']
    cociente = est['DL']['dist'] / est['SV']['dist']
    assert cociente == pytest.approx(0.25 / 1.1, rel=1e-3)  # en las hojas: Ks = 0.22727
    # desplazamientos iguales: vida y ocupación inmediata coinciden
    assert est['OI']['dist'] == pytest.approx(est['SV']['dist'], rel=1e-6)


def test_theta_pdelta_a_mano():
    th, P = theta_pdelta([10.0, 8.0, 6.0], np.array([0.5, 0.4, 0.3]), np.array([20.0, 14.0, 7.0]),
                         np.array([400.0, 350.0, 350.0]))
    assert P == pytest.approx([24, 14, 6])
    assert th == pytest.approx([24 * 0.5 / (20 * 400), 14 * 0.4 / (14 * 350), 6 * 0.3 / (7 * 350)])


def test_irregularidades():
    masa, rigidez, k = irregularidades([10.0, 20.0, 9.0, 5.0], np.array([30.0, 20.0, 12.0, 5.0]),
                                       np.array([1.0, 1.0, 1.0, 1.0]))
    assert masa == [2]  # 20 > 1.5 * 10
    assert rigidez == []
    _, rigidez, k = irregularidades([10.0] * 4, np.array([30.0, 20.0, 12.0, 5.0]), np.array([3.0, 1.0, 1.0, 1.0]))
    assert k[:2] == pytest.approx([10.0, 20.0])
    assert rigidez == [1]  # k1 = 10 < 0.7 * k2 = 14
    _, rigidez, _ = irregularidades([10.0] * 4, np.array([30.0, 20.0, 12.0, 5.0]), np.array([2.0, 1.0, 1.0, 1.0]))
    assert rigidez == []  # k1 = 15 > 14


def test_los_ejemplos_marcan_los_avisos_esperados():
    textos = ' '.join(t for _, t in correr(EJ4)['avisos'])
    assert 'Seguridad de vida: la distorsión supera' in textos  # nivel 1 con 0.0306 > 0.03
    assert 'piso blando' in textos.lower()


def test_avisos_de_parametros_incoherentes():
    d = copy.deepcopy(EJ2)
    d.update(c='0.99', Ta='2.0', Tb='1.0', k='1.5', Q='7')
    p = ame.preparar_modelo(d)
    sev = {t[:30]: s for s, t in ame.calcular(p)['avisos']}
    textos = ' '.join(sev)
    assert 'parece estar en g' in ' '.join(t for _, t in ame.calcular(p)['avisos'])
    assert 'Ta (' in textos
    assert any('k = 1.5' in t for _, t in ame.calcular(p)['avisos'])
    assert any('Q = 7' in t for _, t in ame.calcular(p)['avisos'])


# ---------------------------------------------------------------- consistencia interna
@pytest.mark.parametrize('ej', [EJ1, EJ2, EJ3, EJ4, EJ5])
def test_propiedades_modales(ej):
    r = correr(ej)
    M, K, fi = r['p']['M'], r['p']['K'], r['fi']
    assert r['gam'].sum() > 0 and np.all(r['T'][:-1] > r['T'][1:])  # periodos decrecientes
    # ortogonalidad de los modos respecto a M y K
    Mm, Km = fi.T @ M @ fi, fi.T @ K @ fi
    fuera = lambda A: np.abs(A - np.diag(np.diag(A))).max() / np.abs(np.diag(A)).max()
    assert fuera(Mm) < 1e-8 and fuera(Km) < 1e-8
    # la masa efectiva modal acumulada suma la masa total
    m_ef = [(fi[:, j] @ M @ np.ones(r['p']['n'])) ** 2 / (fi[:, j] @ M @ fi[:, j]) for j in range(r['p']['n'])]
    assert sum(m_ef) == pytest.approx(np.trace(M))
    # el desplazamiento y las fuerzas combinados son coherentes con el cortante basal
    assert r['tabla_din']['Vu'][-1] == pytest.approx(r['p']['factor_Fu'] * r['F_final'].sum() / 1000)


def test_modo_principal_de_estado_coincide_con_la_revision():
    r = correr(EJ2)  # estado SV: d del análisis principal = d del estado SV sin amplificar
    assert r['d'] == pytest.approx(r['revision']['estados']['SV']['d'])


# ---------------------------------------------------------------- exportación
def test_excel_y_pdf_se_generan():
    from openpyxl import load_workbook
    for ej in (EJ1, EJ4):
        r = correr(ej)
        buf = io.BytesIO()
        ame.exportar_excel(r, buf)
        hojas = load_workbook(io.BytesIO(buf.getvalue())).sheetnames
        assert {'Datos', 'Modal', 'Espectro', 'Dinámico', 'Estático', 'Revisiones', 'Gráficas'} <= set(hojas)
        pdf = io.BytesIO()
        ame_pdf.memoria_pdf(r, pdf, proyecto='Prueba', autor='Pruebas')
        assert pdf.getvalue().startswith(b'%PDF') and len(pdf.getvalue()) > 30_000


def test_pdf_con_formulas_y_letras_griegas():
    pymupdf = pytest.importorskip('pymupdf')
    buf = io.BytesIO()
    ame_pdf.memoria_pdf(correr(EJ2), buf, proyecto='Torre Ø', autor='Á')
    texto = ''.join(pg.get_text() for pg in pymupdf.open(stream=buf.getvalue(), filetype='pdf'))
    for esperado in ('Memoria de cálculo', 'λ', 'γ', 'Σ', 'T < Tₐ', 'Torre Ø'):
        assert esperado in texto


def test_figuras_se_dibujan():
    r = correr(EJ3)
    for f in (ame.fig_estructura(r['p']), ame.fig_modal(r), ame.fig_espectro(r), ame.fig_fuerzas(r),
              ame.fig_distorsiones(r)):
        buf = io.BytesIO()
        f.savefig(buf, format='png')
        assert len(buf.getvalue()) > 5_000


# ---------------------------------------------------------------- lectura flexible de matrices (modo manual)
def test_lectura_flexible_de_matrices():
    for texto in ('1 2\n3 4', '1,2;3,4', '[1 2; 3 4]', 'K = 1\t2\n3\t4', '1.0, 2.0\n3.0, 4.0'):
        assert ame.parse_matriz(texto, 'K') == pytest.approx(np.array([[1, 2], [3, 4]]))
    assert ame.parse_matriz('20,985.1162 -14,212.1395', 'K')[0] == pytest.approx([20985.1162, -14212.1395])
    with pytest.raises(ValueError, match='rectangular'):
        ame.parse_matriz('1 2\n3', 'K')


# ---------------------------------------------------------------- página web
def test_pagina_web_corre_con_todos_los_ejemplos():
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    raiz = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(str(raiz / 'app_web.py'), default_timeout=120).run()
    assert not at.exception
    for nombre in EJ:
        at.sidebar.selectbox(key='ejemplo').select(nombre)
        at.sidebar.button[0].click()
        at.run()
        assert not at.exception, nombre
    # controles nuevos
    at.sidebar.selectbox(key='combinacion').select('CQC').run()
    at.sidebar.selectbox(key='estado').select('Limitación de daños').run()
    at.sidebar.radio(key='direccion').set_value('Y').run()
    at.sidebar.selectbox(key='material').select('Acero').run()
    assert not at.exception
    at.text_input(key='crujias').set_value('').run()
    assert at.error and not at.exception  # mensaje claro, sin caerse
