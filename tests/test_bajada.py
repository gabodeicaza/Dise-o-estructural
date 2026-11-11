import pytest

import ame_bajada as bj


def _ej1():
    tipos = {'Azotea': dict(CG=(530, 100), CA=(530, 70), CS=(530, 15)),
             'Intermedio': dict(CG=(610, 350), CA=(610, 250), CS=(610, 40))}
    niv = [dict(nombre='N3 (azotea)', At=11.25, tipo='Azotea', Lv=7, hc=0),
           dict(nombre='N2', At=11.25, tipo='Intermedio', Lv=7, hc=3),
           dict(nombre='N1', At=11.25, tipo='Intermedio', Lv=7, hc=3),
           dict(nombre='PB', At=0, tipo='Intermedio', Lv=0, hc=4.5)]
    return bj.bajada(niv, tipos, bj.pp_lineal(25, 50), bj.pp_lineal(40, 45))


def test_ejemplo_columna_B2_grupo_A():
    f = _ej1()
    assert [round(x['CG']['cm'], 2) for x in f] == [8062.5, 18321.0, 28579.5, 30523.5]
    assert [round(x['CG']['cv'], 2) for x in f] == [1125.0, 5062.5, 9000.0, 9000.0]
    b = bj.cargas_base(f, 'A', 250)
    assert b['casos']['CG']['Pn'] == pytest.approx(39523.5)
    assert b['casos']['CG']['Pu'] == pytest.approx(61085.25)


def test_ejemplo_columna_A1_grupo_B():
    tipos = {'Azotea': dict(CG=(530, 100), CA=(530, 70), CS=(530, 15)),
             'Oficinas': dict(CG=(670, 250), CA=(670, 180), CS=(670, 100))}
    niv = [dict(nombre='Azotea', At=4, tipo='Azotea', Lv=4, hc=0), dict(nombre='N2', At=4, tipo='Oficinas', Lv=4, hc=3),
           dict(nombre='N1', At=4, tipo='Oficinas', Lv=4, hc=3), dict(nombre='PB', At=0, tipo='Oficinas', Lv=0, hc=4)]
    f = bj.bajada(niv, tipos, bj.pp_lineal(20, 50), bj.pp_lineal(50, 50))
    assert [round(x['CG']['cm']) for x in f] == [3080, 8520, 13960, 16360]
    b = bj.cargas_base(f, 'B', 250)
    assert b['casos']['CG']['CV'] == pytest.approx(2400)
    assert b['casos']['CG']['Pu'] == pytest.approx(24868)


def test_tipo_inexistente():
    with pytest.raises(ValueError):
        bj.bajada([dict(nombre='x', At=1, tipo='no', Lv=0, hc=0)], {}, 0, 0)


def test_planta_areas_columnas_de_los_apuntes():
    import numpy as np

    import ame_planta as pl
    m = np.ones((2, 2), bool)
    assert pl.columna(m, [5, 4], [3, 2], 1, 1) == (11.25, 7.0)          # B-2 (ejemplo 1)
    m2 = np.ones((2, 3), bool)
    assert pl.columna(m2, [4, 2.5, 5.4], [4, 2], 0, 0) == (4.0, 4.0)    # A-1 (ejemplo 2)
    m3 = np.ones((2, 2), bool); m3[1, 1] = False                        # planta en L: la columna B-2 es de esquina entrante
    assert pl.columna(m3, [5, 4], [3, 2], 1, 1)[0] == pytest.approx(5 / 2 * 3 / 2 + 4 / 2 * 3 / 2 + 5 / 2 * 2 / 2)
    assert not pl.nodos_existentes(m3)[2, 2]


def test_planta_viga_area_45():
    import numpy as np

    import ame_planta as pl
    t = pl.tramos_viga(np.ones((1, 1), bool), [5], [4])
    largo = next(x for x in t if x['tipo'] == 'x')
    corto = next(x for x in t if x['tipo'] == 'y')
    assert largo['A'] == pytest.approx((2 * 5 - 4) * 4 / 4) and corto['A'] == pytest.approx(4)


def test_regiones_45_coinciden_con_areas_de_viga():
    import ame_planta as pl
    for w, h in [(5, 4), (4, 5), (4, 4), (6, 2), (2, 6)]:
        reg = pl.regiones_tablero(w, h)
        assert sum(a for _, _, a in reg) == pytest.approx(w * h)
        lado_largo = {'arriba': 'x', 'abajo': 'x'} if w >= h else {'izq': 'y', 'der': 'y'}
        suma_largos = sum(a for _, v, a in reg if v in lado_largo)
        assert suma_largos == pytest.approx(2 * pl.area_a_viga(min(w, h), max(w, h), True))


def test_voladizo_y_edificio_del_pizarron():
    import numpy as np

    import ame_edificio as ed
    import ame_planta as pl
    dx, dy = [8, 6, 8, 8, 8, 6], [4, 6, 6]
    m = np.full((3, 6), pl.LOSA)
    m[0, :] = pl.VACIO
    m[0, :3] = pl.VOL
    # el voladizo no genera columnas en el eje A; su área (4×22) baja a la viga B 1–4
    ex = pl.nodos_existentes(m)
    assert not ex[0].any() and ex[1:].all()
    area_total = sum(dx) * 12 + 22 * 4
    ats = [pl.columna(m, dx, dy, i, j)[0] for j in range(4) for i in range(7) if ex[j, i]]
    assert sum(ats) == pytest.approx(area_total)
    tr = pl.tramos_viga(m, dx, dy)
    vB = next(t for t in tr if t['tipo'] == 'x' and t['j'] == 1 and t['i'] == 0)
    assert vB['A'] == pytest.approx(8 * 4 + (2 * 8 - 6) * 6 / 4)       # voladizo completo + trapecio de la losa de abajo
    tipos = {'O': dict(CG=(300, 250), CA=(300, 180), CS=(300, 100))}
    niv = [dict(nombre=f'N{k}', tipo='O', h=3.8) for k in range(8, 0, -1)]
    todos = ed.columnas_edificio(niv, [m] * 8, dx, dy, tipos, 768, 1000)
    f = todos[(2, 3)]                                   # columna interior C-4
    assert len(f) == 9 and f[-1]['nombre'] == 'Base'
    assert f[0]['hc'] == 0 and f[1]['hc'] == 3.8 and f[-1]['hc'] == 3.8
    # un nivel sin esa columna no suma su peso ni deja huérfanos de columna
    m2 = m.copy(); m2[1:, 3:] = pl.VACIO
    t2 = ed.columnas_edificio([niv[0], niv[1]], [m2, m], dx, dy, tipos, 768, 1000)
    assert t2[(2, 5)][0]['At'] == 0 and t2[(2, 5)][0]['hc'] == 0


def test_vista_columna_y_rapido():
    f = _ej1()
    rap = bj.bajada_rapida([dict(nombre=x['nombre'], At=x['At'], Lv=x['Lv'], hc=x['hc']) for x in f], 1.3, True)
    assert rap[-1]['P'] == pytest.approx(3 * 11.25 * 1300)
    assert bj.figura_columna([('P3', 'P3 = a', 1.0), ('P2', 'P2 = b', 2.0), ('P0', 'P0 = c', 3.0)], 't', 'Pu = 1') is not None


def test_eje_C_del_pizarron_equilibrio_y_cargas():
    import numpy as np

    import ame_edificio as ed
    import ame_planta as pl
    import ame_porticos as pt
    dx, dy = [8, 6, 8, 8, 8, 6], [4, 6, 6]
    m = np.full((3, 6), pl.LOSA); m[0, :] = pl.VACIO; m[0, :3] = pl.VOL
    tipos = {'O': dict(CG=(300, 250), CA=(300, 180), CS=(300, 100))}
    niv = [dict(nombre=f'N{k}', tipo='O', h=3.8) for k in range(8, 0, -1)]
    d = ed.cargas_eje(niv, [m] * 8, dx, dy, tipos, bj.pp_lineal(40, 80), 2)
    assert d['spans'] == dx and len(d['alturas']) == 8
    # tramo de 8 m entre dos tableros de 6 m: trapecios (2·8−6)·6/4 = 15 m² por lado → 30 m²
    assert d['cargas']['CM'][0][0] == pytest.approx((300 * 30 / 8 + 768) / 1000)
    F = ed.fuerzas_triangulares([1000.0] * 8, [3.8 * k for k in range(1, 9)], 0.1)
    assert sum(F) == pytest.approx(0.1 * 8)
    res = pt.resolver(d['spans'], d['alturas'], [(70, 70)] * 8, [(40, 80)] * 8, 14000 * 250 ** .5, d['cargas'], F)
    env, _ = pt.envolvente(res, {'g': (1.3, 1.5, 0, 0), '+': (1.1, 0, 1.1, 1.1), '-': (1.1, 0, 1.1, -1.1)})
    assert len([k for k in env if k[0] == 'b' and k[1] == 2]) == 6
    with pytest.raises(ValueError):
        ed.cargas_eje(niv, [m] * 8, dx, dy, tipos, 0, 0)         # el eje A solo tiene voladizo: sin vigas


def test_elevacion_se_dibuja_por_eje_y_respeta_columnas_ausentes():
    import numpy as np

    import ame_edificio as ed
    import ame_planta as pl
    dx, dy = [5, 4], [3, 2]
    m = np.full((2, 2), pl.LOSA)
    m_arriba = m.copy(); m_arriba[:, 1] = pl.VACIO
    niv = [dict(nombre='N2', tipo='O', h=3), dict(nombre='N1', tipo='O', h=3)]
    for eje in (('X', 1), ('Y', 0), ('Y', 2)):
        assert ed.figura_elevacion(niv, [m_arriba, m], dx, dy, eje, (1, 1), 't') is not None


def test_formato_de_apuntes_totales_y_columnas_sin_losa():
    import numpy as np

    import ame_edificio as ed
    import ame_planta as pl
    import ui_edificio as ue
    # los totales del formato de los apuntes (p. 5 y 7): Azotea 530/630, Intermedio 610/960, Oficinas 670/920, Terraza 700/950
    esperado = {'Azotea': (530, 100), 'Intermedio': (610, 350), 'Oficinas': (670, 250), 'Terraza': (700, 250)}
    vistos = {}
    for e in ue.EJEMPLOS.values():
        for nom, comp, cv in e['tipos']:
            if nom in esperado:
                t = bj.tipo_desde_componentes([(c, v, '', '') for c, v in comp], cv)
                vistos[nom] = (t['CG'][0], t['CG'][1])
                assert t['CA'][0] == t['CG'][0] and t['CA'][1] == cv[1] and t['CS'][1] == cv[2]
    assert vistos == esperado
    # un piso con solo parte de la losa y columnas que siguen hasta arriba (croquis de los apuntes)
    dx, dy = [4, 4], [4]
    arriba = np.full((1, 2), pl.LOSA)
    abajo = np.array([[pl.LOSA, pl.VACIO]])
    extra_abajo = np.zeros((2, 3), int); extra_abajo[:, 2] = 1          # columna forzada en el eje 3 sin losa alrededor
    tipos = {'O': dict(CG=(600, 250), CA=(600, 180), CS=(600, 100))}
    niv = [dict(nombre='N2', tipo='O', h=3), dict(nombre='N1', tipo='O', h=3)]
    sin = ed.columnas_edificio(niv, [arriba, abajo], dx, dy, tipos, 0, 1000)
    con = ed.columnas_edificio(niv, [arriba, abajo], dx, dy, tipos, 0, 1000, [None, extra_abajo])
    assert (0, 2) in sin and sin[(0, 2)][1]['hc'] == 0                  # sin forzar, la columna del eje 3 no baja del nivel 2
    assert con[(0, 2)][1]['hc'] == 3 and con[(0, 2)][1]['At'] == 0 and con[(0, 2)][-1]['nombre'] == 'Base'
    assert ed.figura_3d(niv, [arriba, abajo], dx, dy, [None, extra_abajo], (0, 2), 24, -62, 't', {'N1'}) is not None
