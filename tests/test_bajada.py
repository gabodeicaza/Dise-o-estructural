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
