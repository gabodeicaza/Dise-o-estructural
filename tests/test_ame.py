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

import ame_ntc as ntc  # noqa: E402
import ame_pdf  # noqa: E402
import ame_sismico as ame  # noqa: E402
from ame_marco import rigidez_marco  # noqa: E402
from ame_revision import combinar, irregularidades, rho_cqc, theta_pdelta  # noqa: E402
from ame_secciones import parse_secciones  # noqa: E402

EJ = ame.EJEMPLOS_MODELO
EJ1, EJ2, EJ3, EJ4, EJ5, EJ6 = list(EJ.values())


def correr(d):
    return ame.calcular(ame.preparar(d))


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


def un_nivel(**extra):
    d = copy.deepcopy(EJ2)
    d['niveles'] = [dict(h='3.5', ejes='todos', seccion='40x40', artic='', cargas='3', W='')]
    d.update(extra)
    return d


def test_distorsion_sv_amplifica_con_FC_Q_Rprima_del_periodo_fundamental():
    """Apuntes p. 13 y 15: desplazamientos × FC·Q·R′, con R′ evaluado en el periodo fundamental."""
    r = correr(un_nivel())
    p, T1 = r['p'], r['T'][0]
    R1 = float(ame._R(p, 4.0, T1))
    est = r['revision']['estados']
    assert est['SV']['amp'] == pytest.approx(1.1 * 4.0 * R1)
    # con un solo modo: γ_DL / γ_SV = Ks·Q′(T) / (FC·Q)   (DL: Sa·Ks sin FC; SV: Sa/(Q′R′) · FC·Q·R′)
    q = float(ame._q_r(p, 4.0, T1)[0][0])
    assert est['DL']['dist'][0] / est['SV']['dist'][0] == pytest.approx(0.25 * q / (1.1 * 4.0), rel=1e-6)


def test_ocupacion_inmediata_usa_Q_uno_y_R_prima_075R():
    r = correr(un_nivel(subgrupo='A1'))
    p = r['p']
    assert p['estado'] == 'Ocupación inmediata' and r['Qs'] == pytest.approx([1.0])
    R = float(ame._R(dict(p, Q=np.array([1.0])), 1.0, r['T'][0]))
    assert r['revision']['estados']['OI']['amp'][0] == pytest.approx(1.1 * 1.0 * 0.75 * R)
    fila = {f[0]: f[1] for f in r['normativa']['factores']}
    assert float(fila["R′ = 0.75·R"]) == pytest.approx(0.75 * R, abs=1e-4)


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
    r = correr(EJ4)
    textos = ' '.join(t for _, t in r['avisos'])
    assert 'piso blando' in textos.lower()
    # el estado de diseño (seguridad de vida) cumple su límite 0.03; la limitación de daños solo es referencia
    assert r['revision']['estados']['SV']['dist'].max() < 0.03
    assert 'Referencia (no es tu estado de diseño) · Limitación de daños' in textos
    assert not [t for s_, t in r['avisos'] if s_ == 'alerta']


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


# ---------------------------------------------------------------- marco normativo NTC-S 2023 (apuntes)
@pytest.mark.parametrize('grupo, intensidad, clave', [
    ('B', 'Frecuente', 'DL'), ('A', 'Frecuente', 'DL'), ('B', 'Base de diseño', 'SV'), ('A', 'Base de diseño', 'OI'),
    ('A', 'Infrecuente', 'SV'), ('B', 'Infrecuente', 'SV')])
def test_objetivo_de_diseno_segun_la_matriz_de_los_apuntes(grupo, intensidad, clave):
    assert ntc.objetivo_diseno(grupo, intensidad)['clave'] == clave


def test_prevencion_de_colapso_del_grupo_B_avisa_que_requiere_acelerogramas():
    assert 'acelerogramas' in ntc.objetivo_diseno('B', 'Infrecuente')['nota']
    assert ntc.objetivo_diseno('B', 'Infrecuente')['desempeno'] == 'Prevención de colapso'


def test_tipo_del_formato_anterior_se_traduce_a_grupo_e_intensidad():
    d = copy.deepcopy(EJ2)
    d['tipo'] = 'Grupo A - sismo infrecuente'
    p = ame.preparar_modelo(d)
    assert (p['grupo'], p['intensidad'], p['estado']) == ('A', 'Infrecuente', 'Seguridad de vida')
    d['tipo'] = 'Grupo A - sismo base'
    assert ame.preparar_modelo(d)['estado'] == 'Ocupación inmediata'


def test_subgrupos_y_categoria_de_riesgo_asce():
    assert [ntc.SUBGRUPOS[s] for s in ('B2', 'B1', 'A2', 'A1')] == ['B', 'B', 'A', 'A']
    assert [ntc.RIESGO_ASCE[s] for s in ('B2', 'B1', 'A2', 'A1')] == ['I', 'II', 'III', 'IV']
    d = copy.deepcopy(EJ2)
    d['subgrupo'] = 'A2'
    p = ame.preparar_modelo(d)
    assert p['grupo'] == 'A' and p['subgrupo'] == 'A2'
    d['subgrupo'] = 'Z9'
    with pytest.raises(ValueError, match='Subgrupo no válido'):
        ame.preparar_modelo(d)


def test_espectro_por_intensidad():
    d = copy.deepcopy(EJ2)
    bd = dict(a0='224', c='975', Ta='0.8', Tb='1.7', k='0.445', Ts='1.0')
    inf = dict(a0='300', c='1300', Ta='0.8', Tb='1.7', k='0.445', Ts='1.0')
    d['espectros'] = {'Base de diseño': bd, 'Infrecuente': inf}
    v_bd = correr(dict(d, intensidad='Base de diseño'))['V_din']
    v_inf = correr(dict(d, intensidad='Infrecuente'))['V_din']
    assert v_inf / v_bd == pytest.approx(1300 / 975, rel=0.05)  # sube el plato del espectro
    with pytest.raises(ValueError, match='espectro SASID de la intensidad "Frecuente"'):
        ame.preparar_modelo(dict(d, intensidad='Frecuente'))


def test_limitacion_de_danos_sin_factor_de_carga():
    r = correr(dict(copy.deepcopy(EJ2), intensidad='Frecuente'))
    t = r['tabla_din']
    assert r['p']['estado'] == 'Limitación de daños'
    assert t['Fu'] == pytest.approx(t['F'])  # "sin FC" (apuntes p. 12)
    assert r['revision']['estados']['DL']['gamma_c'] == 1.0


@pytest.mark.parametrize('irreg, fuerte_t, fuerte_e, esperado', [
    ([], False, False, 1.0), (['5.3.2'], False, False, 0.8), (['5.2.1', '5.2.3'], False, False, 0.7),
    (['5.2.1', '5.2.3', '5.3.1'], False, False, 0.6), ([], True, False, 0.6), (['5.3.1'], True, False, 0.5),
    (['5.2.1'], True, False, 0.6),  # 5.2.1 no cuenta como condición adicional de la fuerte irregularidad por torsión
    ([], False, True, 0.33)])
def test_gamma_c_segun_las_tablas_c542_c553_c563(irreg, fuerte_t, fuerte_e, esperado):
    assert ntc.gamma_c(irreg, fuerte_t, fuerte_e)[0] == pytest.approx(esperado)


def test_irregularidad_reduce_el_limite_de_ocupacion_inmediata_y_vida_pero_no_el_de_danos():
    r = correr(dict(copy.deepcopy(EJ2), irreg=['5.3.2', '5.3.1']))
    est = r['revision']['estados']
    assert est['SV']['limite'] == pytest.approx(0.03 * 0.7)
    assert est['OI']['limite'] == pytest.approx(0.005 * 0.7)
    assert est['DL']['limite'] == pytest.approx(0.004)
    with pytest.raises(ValueError, match='no reconocida'):
        ntc.gamma_c(['9.9.9'])


def test_excentricidad_accidental_de_los_apuntes():
    assert ntc.excentricidad_accidental(3, 12) == pytest.approx([0.6, 0.9, 1.2])   # 0.05b, 0.075b, 0.10b
    assert ntc.excentricidad_accidental(1, 10) == pytest.approx([0.5])
    r = correr(dict(copy.deepcopy(EJ2), b_planta='12'))
    tor = r['normativa']['torsion']
    assert tor['Mt'] == pytest.approx(tor['e_a'] * tor['Fu'])
    assert 'torsion' not in correr(EJ2)['normativa']


def test_aplicabilidad_del_metodo_estatico():
    assert ntc.altura_maxima_estatico('I', True) == 40 and ntc.altura_maxima_estatico('I', False) == 30
    assert ntc.altura_maxima_estatico('II', True) == 30 and ntc.altura_maxima_estatico('III', False) == 20
    assert ntc.altura_maxima_estatico('', True) is None
    assert correr(dict(copy.deepcopy(EJ2), zona='III'))['normativa']['estatico']['aplica']       # H = 12 m ≤ 30 m
    assert not correr(dict(copy.deepcopy(EJ2), zona='III', subgrupo='A1'))['normativa']['estatico']['aplica']
    alto = copy.deepcopy(EJ2)
    alto['niveles'] = [dict(alto['niveles'][1]) for _ in range(9)]          # 31.5 m en zona III regular
    assert not correr(dict(alto, zona='III'))['normativa']['estatico']['aplica']
    with pytest.raises(ValueError, match='zona geotécnica'):
        ame.preparar_modelo(dict(copy.deepcopy(EJ2), zona='IV'))


def test_cimentacion_con_065_R_prima():
    r = correr(EJ2)
    c = r['normativa']['cimentacion']
    R1 = float(ame._R(r['p'], 4.0, r['T'][0]))
    assert c['factor'] == pytest.approx(0.65 * R1)
    assert c['V'] == pytest.approx(c['factor'] * r['V_din'])
    assert 'cimentacion' not in correr(dict(copy.deepcopy(EJ2), intensidad='Frecuente'))['normativa']


def test_cortante_basal_minimo():
    r = correr(EJ2)
    v = r['normativa']['vmin']
    assert v['V_min'] == pytest.approx(1.1 * r['a_min_g'] * sum(i['W'] for i in r['p']['info']))
    assert v['cumple'] == (r['V_din'] >= v['V_min'])
    assert v['factor'] >= 1.0
    # un edificio muy rígido y pesado con espectro bajo debe poder incumplirlo y avisarlo
    d = copy.deepcopy(EJ2)
    d.update(a0='1', c='5.1', Ts='2')
    r2 = correr(d)
    assert (not r2['normativa']['vmin']['cumple']) == any('Cortante basal mínimo' in t for _, t in r2['avisos'])


def test_R_igual_a_uno_en_materiales_distintos_del_concreto():
    r = correr(EJ5)
    assert r['p']['R_unitaria'] and all(f[4] == pytest.approx(1.0 * 1.0) for f in r['QR'])
    d = copy.deepcopy(EJ5)
    d['R1_otros'] = False
    r2 = correr(d)
    assert not r2['p']['R_unitaria'] and r2['QR'][0][4] > 1.7
    assert r2['V_din'] < r['V_din']                  # con R = 1 las fuerzas de diseño son mayores
    assert not correr(EJ2)['p']['R_unitaria']        # concreto: R = k1·R0 + k2


def test_amortiguamiento_por_material():
    assert correr(EJ2)['p']['zeta'] == 0.05
    d = copy.deepcopy(EJ5)
    d.pop('zeta')
    assert correr(d)['p']['zeta'] == 0.03
    d['zeta'] = '0.02'
    assert correr(d)['p']['zeta'] == 0.02


def test_participacion_modal_acumulada():
    r = correr(EJ2)
    m = r['normativa']['modos']
    assert m['acum'][-1] == pytest.approx(1.0) and m['n95'] <= 3 and m['n_T04'] == 1


def test_avisos_normativos():
    textos = lambda r: ' | '.join(t for _, t in r['avisos'])
    # k1 distinto del sugerido para el número de crujías (hoja 14: 1 crujía con k1 = 1.0)
    assert 'k1 = 0.8 con menos de 3 crujías' in textos(correr(EJ1))
    d3 = copy.deepcopy(EJ2)
    d3['crujias'] = '4 5 4'
    d3['niveles'] = [dict(n, cargas='4') for n in d3['niveles']]
    assert 'k1 = 0.8' not in textos(correr(d3))
    # grupo B, infrecuente: prevención de colapso
    assert 'acelerogramas' in textos(correr(dict(copy.deepcopy(EJ2), intensidad='Infrecuente')))
    # grupo A: el estático no aplica
    assert 'método estático no aplica' in textos(correr(dict(copy.deepcopy(EJ2), subgrupo='A1')))
    # R = 1 y ζ del acero
    assert 'R = 1' in textos(correr(EJ5))
    # intensidad distinta de la base: recordar el espectro
    assert 'espectro SASID de esa intensidad' in textos(correr(dict(copy.deepcopy(EJ2), intensidad='Infrecuente')))
    # solo el estado de diseño marca alerta; los demás son referencia
    r = correr(EJ4)
    alertas = [t for s_, t in r['avisos'] if s_ == 'alerta']
    assert all('Seguridad de vida' in t for t in alertas)
    assert any('Referencia' in t for s_, t in r['avisos'] if s_ == 'info')


def test_deteccion_de_fuerte_irregularidad_en_elevacion():
    d = copy.deepcopy(EJ2)
    d['niveles'][1] = dict(d['niveles'][1], seccion='15x15')     # el nivel 2 queda muchísimo más flexible
    r = correr(d)
    textos = ' | '.join(t for _, t in r['avisos'])
    assert r['revision']['irreg_rigidez'] == [2]
    assert 'piso blando' in textos.lower() and '(5.3.3)' in textos
    # al marcarla, el aviso desaparece y se aplica γc = 0.33 a los límites de OI y SV
    r2 = correr(dict(d, fuerte_elev=True))
    assert 'posible irregularidad fuerte' in textos
    assert 'posible irregularidad fuerte' not in ' | '.join(t for _, t in r2['avisos'])
    assert r2['p']['gamma_c'] == pytest.approx(0.33)
    assert r2['revision']['estados']['SV']['limite'] == pytest.approx(0.03 * 0.33)


# ---------------------------------------------------------------- modo "K y M dadas" (hoja 19)
def test_modo_matrices_reproduce_la_hoja_19():
    r = correr(EJ6)
    p = r['p']
    assert p['manual'] and p['estado'] == 'Ocupación inmediata' and r['Qs'] == pytest.approx([1.0])
    assert r['T'] == pytest.approx([2.1790, 0.9939, 0.5735, 0.5020], abs=5e-5)
    assert r['Sa_modal'][:, 0] == pytest.approx([210.02, 476.19, 476.19, 476.19], abs=0.01)
    assert r['F_final'] / 1000 == pytest.approx([608.39, 763.3, 426.32, 601.01], abs=0.02)
    assert r['tabla_din']['Vu'] == pytest.approx([661.11, 1130.1, 1969.7, 2638.9], abs=0.05)
    assert r['tabla_din']['Mvu'].sum() == pytest.approx(31999, abs=1)


def test_modo_matrices_pesos_y_unidades():
    r = correr(EJ6)
    assert r['p']['cargas'] == pytest.approx(np.array([3, 3, 1.5, 1.5]) * 981)     # W = m·g (t)
    kg = dict(EJ6, unidades='kg, cm',
              K='\n'.join(' '.join(str(x * 1000) for x in f) for f in
                          [[300, -150, 0, 0], [-150, 225, -75, 0], [0, -75, 150, -75], [0, 0, -75, 75]]),
              M='3000 3000 1500 1500')  # M como vector de masas
    assert correr(kg)['F_final'] == pytest.approx(r['F_final'])
    r2 = correr(dict(EJ6, pesos='100 100 100 100'))
    assert r2['p']['cargas'] == pytest.approx([100] * 4) and r2['V_est'] != pytest.approx(r['V_est'])


@pytest.mark.parametrize('cambio, texto', [
    (dict(alturas='5 5 5'), 'Hay 3 alturas'), (dict(alturas='5 5 5 -1'), 'mayores que cero'),
    (dict(pesos='1 2'), 'Hay 2 pesos'), (dict(K='1 2\n3 4'), 'M debe ser'), (dict(K='1 2 3\n4 5 6'), 'cuadrada')])
def test_modo_matrices_errores_de_captura(cambio, texto):
    with pytest.raises(ValueError, match=texto):
        ame.preparar(dict(EJ6, **cambio))


def test_modo_matrices_avisa_si_K_es_incoherente():
    sim = correr(dict(EJ6, K='300 -150 0 0\n-100 225 -75 0\n0 -75 150 -75\n0 0 -75 75'))
    assert any('K no es simétrica' in t for s_, t in sim['avisos'] if s_ == 'alerta')
    neg = correr(dict(EJ6, K='-300 150 0 0\n150 -225 75 0\n0 75 -150 75\n0 0 75 -75'))
    assert any('definida positiva' in t for s_, t in neg['avisos'])


def test_modo_matrices_con_formato_anterior_sigue_funcionando():
    r = ame.calcular(ame.preparar_datos(datos_hoja19()))
    assert r['T'][0] == pytest.approx(2.1790, abs=5e-5)


def test_modo_matrices_exporta_excel_y_pdf():
    r = correr(EJ6)
    buf = io.BytesIO()
    ame.exportar_excel(r, buf)
    pdf = io.BytesIO()
    ame_pdf.memoria_pdf(r, pdf, proyecto='K y M', autor='Pruebas')
    assert buf.getvalue()[:2] == b'PK' and pdf.getvalue().startswith(b'%PDF')


# ---------------------------------------------------------------- exportación
def test_excel_y_pdf_se_generan():
    from openpyxl import load_workbook
    for ej in (EJ1, EJ4):
        r = correr(ej)
        buf = io.BytesIO()
        ame.exportar_excel(r, buf)
        hojas = load_workbook(io.BytesIO(buf.getvalue())).sheetnames
        assert {'Datos', 'Modal', 'Espectro', 'Dinámico', 'Estático', 'Normativa', 'Revisiones', 'Gráficas'} <= set(hojas)
        pdf = io.BytesIO()
        ame_pdf.memoria_pdf(r, pdf, proyecto='Prueba', autor='Pruebas')
        assert pdf.getvalue().startswith(b'%PDF') and len(pdf.getvalue()) > 30_000


def test_pdf_con_formulas_y_letras_griegas():
    pymupdf = pytest.importorskip('pymupdf')
    buf = io.BytesIO()
    ame_pdf.memoria_pdf(correr(EJ2), buf, proyecto='Torre Ø', autor='Á')
    texto = ''.join(pg.get_text() for pg in pymupdf.open(stream=buf.getvalue(), filetype='pdf'))
    for esperado in ('Memoria de cálculo', 'λ', 'γ', 'Σ', 'T < Tₐ', 'Torre Ø', 'Marco normativo (NTC-S 2023)',
                     'Cortante basal mínimo', 'Cimentación'):
        assert esperado in texto


def test_figuras_se_dibujan():
    r = correr(EJ3)
    for f in (ame.fig_estructura(r['p']), ame.fig_modal(r), ame.fig_espectro(r), ame.fig_fuerzas(r), ame.fig_diagramas(r),
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
    # controles nuevos (sobre un ejemplo por secciones: con matrices algunos controles se deshabilitan)
    at.sidebar.selectbox(key='ejemplo').select(list(EJ)[1])
    at.sidebar.button[0].click()
    at.run()
    at.sidebar.selectbox(key='combinacion').select('CQC').run()
    at.sidebar.selectbox(key='estado').select('Limitación de daños').run()
    at.sidebar.radio(key='direccion').set_value('Y').run()
    at.sidebar.selectbox(key='material').select('Acero').run()
    assert not at.exception
    # objetivo de diseño: grupo A / base -> ocupación inmediata; infrecuente exige su propio espectro
    at.sidebar.selectbox(key='subgrupo').select('A1').run()
    assert not at.exception and not any('Revisa los datos' in e.value for e in at.error)
    # modo matrices: se puede alternar sin perder los datos de secciones
    at.sidebar.selectbox(key='ejemplo').select(list(EJ)[5])
    at.sidebar.button[0].click()
    at.run()
    assert not at.exception and at.radio(key='modo').value.startswith('Matrices')
    at.radio(key='modo').set_value('Por secciones y cargas').run()
    assert not at.exception and at.text_input(key='crujias').value
    at.sidebar.selectbox(key='intensidad').select('Infrecuente').run()
    assert at.error and 'espectro SASID' in at.error[0].value and not at.exception
    for k, v in dict(a0='300', c='1300', Ta='0.8', Tb='1.7', k='0.445', Ts='1.0').items():
        at.sidebar.text_input(key=k).set_value(v)
    at.run()
    assert not at.exception and not any('Revisa los datos' in e.value for e in at.error)
    at.sidebar.selectbox(key='intensidad').select('Base de diseño').run()
    assert at.sidebar.text_input(key='a0').value == '80'         # el espectro base (el del ejemplo 6 cargado antes) se conservó
    at.sidebar.multiselect(key='irreg').set_value(['5.3.2']).run()
    at.sidebar.text_input(key='b_planta').set_value('12').run()
    at.sidebar.selectbox(key='zona').select('III').run()
    assert not at.exception
    at.text_input(key='crujias').set_value('').run()
    assert at.error and not at.exception  # mensaje claro, sin caerse
