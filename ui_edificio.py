"""Bajada de cargas de un edificio de varios niveles: una planta (cuadrícula) editable por nivel, con copia entre niveles."""
import numpy as np
import pandas as pd
import streamlit as st

import ame_bajada as bj
import ame_edificio as ed
import ame_planta as pl
import ame_porticos as pt

COL_T = ['Tipo de piso', 'CM CG', 'CV máx CG', 'CM CA', 'CV acc CA', 'CM CS', 'CV med CS']
COL_N = ['Nivel', 'Tipo de piso', 'h entrepiso debajo (m)']
OPC = ['Losa', 'Voladizo', 'Vacío']
MODOS = ['Área de la columna elegida', 'Áreas de todas las columnas', 'Áreas de las vigas (reparto a 45°)']


def _num(x):
    return float(str(x).strip().replace(',', '.'))


def _kg(x):
    return f'{x:,.0f}' if abs(x - round(x)) < 1e-9 else f'{x:,.2f}'


def _df_mask(codigos, dx, dy):
    ny = len(dy)
    return pd.DataFrame([[pl.NOMBRE[int(c)] for c in fila] for fila in codigos],
                        index=[f'{pl.letras(ny + 1)[k]}–{pl.letras(ny + 1)[k + 1]}' for k in range(ny)],
                        columns=[f'{k + 1}–{k + 2}' for k in range(len(dx))])


EJEMPLOS = {
    'Ejercicio del pizarrón (8 niveles, oficinas, voladizo en A–B)': dict(
        cx='8 6 8 8 8 6', cy='4 6 6', grupo='B', vig=(40, 80), col=(70, 70), sel=('N2', 'C-4'),
        tipos=[('Oficinas', 300, 250, 300, 180, 300, 100), ('Azotea', 300, 100, 300, 70, 300, 15)],
        niveles=[(f'N{k}', 'Azotea' if k == 8 else 'Oficinas', 3.8) for k in range(8, 0, -1)],
        mask=lambda: _mask_pizarron()),
    'Apuntes · Ejemplo 1: columna B-2, biblioteca, Grupo A (p. 5)': dict(
        cx='5 4', cy='3 2', grupo='A', vig=(25, 50), col=(40, 45), sel=('N1', 'B-2'),
        tipos=[('Azotea', 530, 100, 530, 70, 530, 15), ('Intermedio', 610, 350, 610, 250, 610, 40)],
        niveles=[('N3 (azotea)', 'Azotea', 3), ('N2', 'Intermedio', 3), ('N1', 'Intermedio', 4.5)],
        mask=lambda: np.full((2, 2), pl.LOSA)),
    'Apuntes · Ejemplo 2: columna A-1, oficinas, Grupo B (p. 8)': dict(
        cx='4 2.5 5.4', cy='4 2', grupo='B', vig=(20, 50), col=(50, 50), sel=('N1', 'A-1'),
        tipos=[('Azotea', 530, 100, 530, 70, 530, 15), ('Oficinas', 670, 250, 670, 180, 670, 100)],
        niveles=[('Azotea', 'Azotea', 3), ('N2', 'Oficinas', 3), ('N1', 'Oficinas', 4)],
        mask=lambda: np.full((2, 3), pl.LOSA)),
}


def _mask_pizarron():
    m = np.full((3, 6), pl.LOSA)
    m[0, :] = pl.VACIO
    m[0, :3] = pl.VOL
    return m


def _cargar_ejemplo(clave=None):
    """Llena claros, niveles, cargas, secciones y la cuadrícula de cada nivel con uno de los ejemplos."""
    ss = st.session_state
    e = EJEMPLOS[clave or ss.get('ed_ej') or list(EJEMPLOS)[0]]
    ss['ed_ver'] = ss.get('ed_ver', 0) + 1
    ss['ed_cx'], ss['ed_cy'] = e['cx'], e['cy']
    ss['ed_tipos'] = pd.DataFrame([[t[0]] + [str(v) for v in t[1:]] for t in e['tipos']], columns=COL_T)
    ss['ed_niv'] = pd.DataFrame([[n[0], n[1], str(n[2])] for n in e['niveles']], columns=COL_N)
    dx, dy = pl.parse_claros(e['cx']), pl.parse_claros(e['cy'])
    ss['ed_masks'] = {n[0]: _df_mask(e['mask'](), dx, dy) for n in e['niveles']}
    ss['ed_key_dims'] = (len(dx), len(dy))
    ss.update({'ed_grupo': e['grupo'], 'ed_vb': float(e['vig'][0]), 'ed_vh': float(e['vig'][1]), 'ed_cb': float(e['col'][0]),
               'ed_ch': float(e['col'][1]), 'ed_sel_niv': e['sel'][0], 'ed_col': e['sel'][1], 'ed_hlosa': 0.0})


def _copiar(origen, destinos):
    ss = st.session_state
    base = ss['ed_masks'][origen].copy()
    for d in destinos:
        if d != origen:
            ss['ed_masks'][d] = base.copy()
    ss['ed_ver'] = ss.get('ed_ver', 0) + 1


def vista_columna(filas, tipos, caso, pp_v, pp_c, grupo, fc, clave):
    """Vista de la columna como en los apuntes (p. 4): P_k = P_{k+1} + A_T·w + p.p. vigas + p.p. columna, y Pu = FC·P0, A = Pu/(0.25 f′c).
    Incluye el predimensionado rápido con w = 1.3 t/m² (que ya incluye el peso propio)."""
    metodo = st.radio('Método de la vista', ['Detallado (CM y CV de cada piso)', 'Rápido: w = 1.3 t/m² incluyendo peso propio'],
                      horizontal=True, key=f'{clave}_met')
    n = len(filas)
    nombres = [f"P{n - 1 - k}" for k in range(n)]
    if metodo.startswith('Detallado'):
        items = []
        for k, f in enumerate(filas):
            d = f[caso]
            cm_l, cv_l = tipos[f['tipo']][caso]
            w = (cm_l + cv_l) / 1000
            ant = f"P{n - k} + " if k > 0 else ''
            partes = ([f"{f['At']:g}({w:.3f})"] if f['At'] else []) + ([f"{f['pp_vig'] / 1000:.2f}"] if f['pp_vig'] else []) + \
                     ([f"{f['pp_col'] / 1000:.2f}"] if f['pp_col'] else [])
            items.append((nombres[k], f"{nombres[k]} = {ant}{' + '.join(partes) if partes else '0'}", (d['cm'] + d['cv']) / 1000))
        b = bj.cargas_base(filas, grupo, fc)
        g = b['casos'][caso]
        cierre = (f"Pu = {g['FC_CM']:g}({g['CM'] / 1000:.2f}) + {g['FC_CV']:g}({g['CV'] / 1000:.2f}) = {g['Pu'] / 1000:.2f} t"
                  f"     A = Pu / 0.25f′c = {g['Pu'] / (0.25 * fc):,.0f} cm²")
    else:
        c1, c2, c3 = st.columns(3)
        w = c1.number_input('w (t/m²)', 0.3, 5.0, 1.3, 0.1, key=f'{clave}_w')
        inc = c2.checkbox('w ya incluye el peso propio de vigas y columnas', True, key=f'{clave}_inc')
        fcm, fcv = bj.FC_GRUPO[grupo]
        fcu = c3.number_input('FC para Pu', 1.0, 2.0, float(fcv), 0.1, key=f'{clave}_fc',
                              help='Los apuntes piden cuidar el FC según el tipo de carga; con una sola w se toma por omisión el de la CV (el mayor).')
        rap = bj.bajada_rapida(filas, w, inc, pp_v, pp_c)
        items = []
        for k, r in enumerate(rap):
            ant = f"P{n - k} + " if k > 0 else ''
            partes = ([f"{r['At']:g}({w:g})"] if r['At'] else []) + ([f"{r['pv'] / 1000:.2f}"] if r['pv'] else []) + \
                     ([f"{r['pc'] / 1000:.2f}"] if r['pc'] else [])
            items.append((nombres[k], f"{nombres[k]} = {ant}{' + '.join(partes) if partes else '0'}", r['P'] / 1000))
        Pu = fcu * rap[-1]['P']
        cierre = f"Pu = {fcu:g}({rap[-1]['P'] / 1000:.2f}) = {Pu / 1000:.2f} t     A = Pu / 0.25f′c = {Pu / (0.25 * fc):,.0f} cm²"
    import matplotlib.pyplot as plt
    st.pyplot(bj.figura_columna(items, 'Bajada de cargas de la columna (nominal P_k acumulada)', cierre), width='stretch')
    plt.close('all')


def mostrar(res_sismo=None):
    ss = st.session_state
    st.caption('Edificio de varios niveles: cada nivel tiene su propia cuadrícula (puedes editarla o copiarla de otro nivel). La app baja '
               'las cargas de todas las columnas, da los pesos por nivel (para el análisis sísmico) y la carga repartida de cada viga.')
    if 'ed_niv' not in ss:
        _cargar_ejemplo(list(EJEMPLOS)[0])
    c1, c2 = st.columns([3, 1])
    c1.selectbox('Ejemplo', list(EJEMPLOS), key='ed_ej')
    c2.write('')
    c2.button('Cargar ejemplo', key='ed_cargar', on_click=_cargar_ejemplo)

    c1, c2, c3, c4, c5 = st.columns(5)
    grupo = c1.radio('Grupo', ['A', 'B'], horizontal=True, key='ed_grupo')
    fc = c2.number_input("f′c (kg/cm²)", 100.0, 1000.0, 250.0, key='ed_fc')
    gam = c3.number_input('γ concreto (t/m³)', 2.0, 2.6, 2.4, 0.05, key='ed_gam')
    hl = c4.number_input('Espesor de losa a sumar a CM (cm)', 0.0, 40.0, 0.0, 1.0, key='ed_hlosa',
                         help='0 si el CM que capturaste ya incluye el peso de la losa; si no, se suma h·γ a todos los CM.')
    c5.write('')
    c1, c2, c3, c4 = st.columns(4)
    vb = c1.number_input('Viga b (cm)', 10.0, 200.0, 40.0, key='ed_vb')
    vh = c2.number_input('Viga h (cm)', 10.0, 300.0, 80.0, key='ed_vh')
    cb = c3.number_input('Columna b (cm)', 10.0, 300.0, 70.0, key='ed_cb')
    ch = c4.number_input('Columna h (cm)', 10.0, 300.0, 70.0, key='ed_ch')
    pp_v, pp_c = bj.pp_lineal(vb, vh, gam), bj.pp_lineal(cb, ch, gam)
    st.caption(f'Peso propio: viga {pp_v:g} kg/m · columna {pp_c:g} kg/m. Predimensionado de los apuntes: h = L/10, b = h/2.')

    v = ss.get('ed_ver', 0)
    st.markdown('##### Cargas por m² de cada tipo de piso (kg/m²)')
    conf = {c: st.column_config.TextColumn(c) for c in COL_T}
    tdf = st.data_editor(ss['ed_tipos'], num_rows='dynamic', hide_index=True, width='stretch', key=f'ed_t_{v}', column_config=conf).dropna()
    st.markdown('##### Niveles, de arriba hacia abajo')
    st.caption('«h entrepiso debajo» es la altura entre ese nivel y el de abajo (para el primer nivel, hasta el desplante). '
               'Agrega o quita filas para cambiar el número de niveles.')
    conf = {c: st.column_config.TextColumn(c) for c in COL_N}
    ndf = st.data_editor(ss['ed_niv'], num_rows='dynamic', hide_index=True, width='stretch', key=f'ed_n_{v}', column_config=conf).dropna()

    c1, c2 = st.columns(2)
    cx = c1.text_input('Claros en X (m), de izquierda a derecha — ejes 1, 2, 3…', key='ed_cx')
    cy = c2.text_input('Claros en Y (m), de arriba hacia abajo — ejes A, B, C…', key='ed_cy')
    try:
        dx, dy = pl.parse_claros(cx), pl.parse_claros(cy)
        extra = hl / 100 * gam * 1000
        tipos = {}
        for _, r in tdf.iterrows():
            tipos[str(r['Tipo de piso']).strip()] = dict(CG=(_num(r['CM CG']) + extra, _num(r['CV máx CG'])),
                                                         CA=(_num(r['CM CA']) + extra, _num(r['CV acc CA'])),
                                                         CS=(_num(r['CM CS']) + extra, _num(r['CV med CS'])))
        niveles = [dict(nombre=str(r['Nivel']).strip(), tipo=str(r['Tipo de piso']).strip(), h=_num(r['h entrepiso debajo (m)']))
                   for _, r in ndf.iterrows()]
        if not niveles:
            st.info('Captura al menos un nivel.')
            return
        for n in niveles:
            if n['tipo'] not in tipos:
                raise ValueError(f"El tipo de piso «{n['tipo']}» del nivel {n['nombre']} no está en la tabla de cargas.")
        if len({n['nombre'] for n in niveles}) != len(niveles):
            raise ValueError('Los nombres de los niveles deben ser distintos.')
    except (ValueError, KeyError) as e:
        st.error(str(e) if 'convert' not in str(e) else 'Escribe números en las tablas (decimales con punto o coma).')
        return

    nx, ny = len(dx), len(dy)
    if ss.get('ed_key_dims') != (nx, ny):                      # cambiaron los ejes: cuadrículas nuevas, todo con losa
        ss['ed_masks'] = {}
        ss['ed_key_dims'] = (nx, ny)
        ss['ed_ver'] = v = ss.get('ed_ver', 0) + 1
    vacia = _df_mask(np.full((ny, nx), pl.LOSA), dx, dy)
    nombres = [n['nombre'] for n in niveles]
    for n in nombres:
        ss['ed_masks'].setdefault(n, vacia.copy())

    st.markdown('##### Planta de cada nivel')
    c1, c2 = st.columns([1, 2])
    sel_niv = c1.selectbox('Nivel a editar y ver', nombres, key='ed_sel_niv')
    ss['ed_masks'][sel_niv] = ed_df = st.data_editor(
        ss['ed_masks'][sel_niv], width='stretch', key=f'ed_m_{sel_niv}_{v}',
        column_config={c: st.column_config.SelectboxColumn(c, options=OPC, required=True) for c in vacia.columns})
    st.caption('Cada casilla es un tablero entre ejes: «Losa» (apoyada en vigas), «Voladizo» (su carga baja a la viga del lado donde '
               'se apoya; no genera columnas) o «Vacío».')
    with c2:
        k1, k2 = st.columns(2)
        origen = k1.selectbox('Copiar la cuadrícula de', nombres, index=nombres.index(sel_niv), key='ed_cp_o')
        dest = k2.multiselect('a los niveles', [n for n in nombres if n != origen], key='ed_cp_d')
        b1, b2 = st.columns(2)
        b1.button('Copiar a los niveles elegidos', key='ed_cp_b', disabled=not dest, on_click=lambda: _copiar(origen, dest))
        b2.button('Copiar a todos los niveles', key='ed_cp_t', on_click=lambda: _copiar(origen, nombres))
    masks = [pl.a_mask(ss['ed_masks'][n].to_numpy()) for n in nombres]
    mk = masks[nombres.index(sel_niv)]
    if not (mk == pl.LOSA).any():
        st.info(f'El nivel {sel_niv} no tiene losa apoyada: marca al menos un tablero como «Losa».')
        return
    if any(((m == pl.VOL).any() and not (m == pl.LOSA).any()) for m in masks):
        st.warning('Algún nivel solo tiene voladizos; un voladizo necesita una losa apoyada a su lado.')

    todos = ed.columnas_edificio(niveles, masks, dx, dy, tipos, pp_v, pp_c)
    L = pl.letras(ny + 1)
    ex = pl.nodos_existentes(mk)
    cols_nivel = [f'{L[j]}-{i + 1}' for j in range(ny + 1) for i in range(nx + 1) if ex[j, i]]
    c1, c2, c3 = st.columns([1, 1, 2])
    col_sel = c1.selectbox('Columna a bajar', cols_nivel, key='ed_col')
    cj, ci = L.index(col_sel.split('-')[0]), int(col_sel.split('-')[1]) - 1
    tr = pl.tramos_viga(mk, dx, dy)
    nom_tr = [pl.nombre_tramo(t, L) for t in tr]
    ver_tr = c2.selectbox('Tramo de viga a resaltar', ['—'] + nom_tr, key='ed_tramo')
    modo_txt = c3.radio('Qué marcar en la planta', MODOS, horizontal=True, key='ed_modo')
    modo = 'todas' if 'todas' in modo_txt else 'vigas' if 'vigas' in modo_txt else 'sel'
    At, Lv = pl.columna(mk, dx, dy, ci, cj)
    ca, cb_ = st.columns([3, 2])
    with ca:
        import matplotlib.pyplot as plt
        st.pyplot(pl.figura_planta(mk, dx, dy, (ci, cj), tr[nom_tr.index(ver_tr)] if ver_tr != '—' else None, modo, f'Planta {sel_niv}'),
                  width='stretch')
        plt.close('all')
    with cb_:
        st.metric(f'Área tributaria de {col_sel} en {sel_niv}', f'{At:g} m²')
        st.metric('Metros de viga que bajan a la columna', f'{Lv:g} m')
        res = pl.resumen_nivel(mk, dx, dy)
        st.metric('Losa del nivel', f"{res['area']:g} m²")
        st.caption(f"{res['columnas']} columnas y {res['vigas']:g} m de viga en {sel_niv}.")

    st.markdown('##### Vista en elevación (corte por un eje)')
    cortes = [f'Eje {x}' for x in L] + [f'Eje {k + 1}' for k in range(nx + 1)]
    corte = st.selectbox('Corte por el', cortes, index=min(2, len(L) - 1), key='ed_corte')
    if corte.split()[1].isalpha():
        eje_c = ('X', L.index(corte.split()[1]))
    else:
        eje_c = ('Y', int(corte.split()[1]) - 1)
    import matplotlib.pyplot as plt
    st.pyplot(ed.figura_elevacion(niveles, masks, dx, dy, eje_c, (cj, ci), f'{corte} · {len(niveles)} niveles'), width='stretch')
    plt.close('all')
    st.caption('Niveles a la izquierda con la altura de cada entrepiso, E1, E2… a la derecha (E1 = el de abajo), columnas y vigas del eje, y '
               'apoyos empotrados. En naranja, la columna elegida para la bajada cuando está en este eje. Si una columna o viga no '
               'aparece en un nivel es porque esa casilla de la planta está «Vacío».')

    tab_a, tab_b, tab_c, tab_d, tab_e = st.tabs(['Bajada de la columna', 'Todas las columnas', 'Pesos por nivel',
                                                 'Cargas de las vigas', 'Análisis de un eje (vigas)'])
    with tab_a:
        filas = todos.get((cj, ci))
        if not filas:
            st.info('Esa columna no existe en ningún nivel.')
        else:
            caso = st.radio('Combinación a mostrar', bj.CASOS, format_func=lambda c: bj.NOMBRE_CASO[c], horizontal=True, key='ed_caso')
            for f in filas:
                d = f[caso]
                cm_l, cv_l = tipos[f['tipo']][caso]
                partes = [f"{_kg(f['At'])}({_kg(cm_l)})"] if f['At'] else []
                if f['Lv']:
                    partes.append(f"{_kg(f['Lv'])}({_kg(pp_v)})")
                if f['hc']:
                    partes.append(f"{_kg(f['hc'])}({_kg(pp_c)})")
                if not partes:
                    continue
                st.markdown(f"**{f['nombre']}**")
                ant = f" {_kg(d['cm_ant'])} +" if d['cm_ant'] else ''
                st.latex(rf"CM = {ant}{' + '.join(partes)} = {_kg(d['cm'])}\ \text{{kg}}")
                if f['At']:
                    antv = f" {_kg(d['cv_ant'])} +" if d['cv_ant'] else ''
                    st.latex(rf"CV = {antv}{_kg(f['At'])}({_kg(cv_l)}) = {_kg(d['cv'])}\ \text{{kg}}")
            vista_columna(filas, tipos, caso, pp_v, pp_c, grupo, fc, 'ed_vc')
            b = bj.cargas_base(filas, grupo, fc)
            g = b['casos']['CG']
            st.latex(rf"P_0 = {_kg(g['CM'])} + {_kg(g['CV'])} = {_kg(g['Pn'])}\ \text{{kg (nominal)}}")
            st.latex(rf"P_{{0u}} = {_kg(g['CM'])}({g['FC_CM']:g}) + {_kg(g['CV'])}({g['FC_CV']:g}) = {_kg(g['Pu'])}\ \text{{kg}}")
            st.latex(rf"A = \dfrac{{P_u}}{{0.25 f'_c}} = {_kg(b['A_col'])}\ \text{{cm}}^2 \Rightarrow {b['lado']:.1f}\times{b['lado']:.1f}\ \text{{cm}}")
            if cb * ch < b['A_col']:
                st.warning(f'Tu columna de {cb:g}×{ch:g} cm ({cb * ch:,.0f} cm²) es menor que el área requerida ({b["A_col"]:,.0f} cm²).')
    with tab_b:
        rows = []
        for (j, i), fl in todos.items():
            b = bj.cargas_base(fl, grupo, fc)
            rows.append([f'{L[j]}-{i + 1}', sum(1 for f in fl if f['At'] > 0), b['casos']['CG']['Pn'] / 1000, b['casos']['CG']['Pu'] / 1000,
                         b['casos']['CA']['Pu'] / 1000, b['rige'], b['A_col'], b['lado']])
        df = pd.DataFrame(rows, columns=['Columna', 'Niveles', 'Pn CG (t)', 'Pu CG (t)', 'Pu CA (t)', 'Rige', 'A req (cm²)', 'Lado (cm)'])
        st.dataframe(df.style.format({'Pn CG (t)': '{:,.1f}', 'Pu CG (t)': '{:,.1f}', 'Pu CA (t)': '{:,.1f}', 'A req (cm²)': '{:,.0f}', 'Lado (cm)': '{:.1f}'}),
                     hide_index=True, width='stretch')
        st.caption('Carga axial en la base de cada columna (sin sismo), con el factor de tu grupo. «A req» = Pu / (0.25 f′c): predimensionado de los apuntes.')
    with tab_c:
        pesos = ed.pesos_por_nivel(niveles, masks, dx, dy, tipos, pp_v, pp_c)
        dfp = pd.DataFrame([[p['nombre'], p['area'], p['vigas_m'], p['columnas'], p['losa'] / 1000, p['vigas'] / 1000, p['cols'] / 1000, p['W'] / 1000]
                            for p in pesos], columns=['Nivel', 'Losa (m²)', 'Viga (m)', 'Columnas', 'Losa (t)', 'Vigas (t)', 'Columnas (t)', 'W = CM+CVacc (t)'])
        st.dataframe(dfp.style.format({c: '{:,.1f}' for c in dfp.columns[4:]}), hide_index=True, width='stretch')
        st.metric('Peso total de la estructura', f"{sum(p['W'] for p in pesos) / 1000:,.1f} t")
        st.caption('Peso sísmico por nivel con CM + CVacc (losa, vigas y la mitad de las columnas de arriba y de abajo): son las masas '
                   'para el método modal espectral del examen 2 (m = W/g).')
    with tab_d:
        nombre_lvl = sel_niv
        f = ed.cargas_tramos(mk, dx, dy, tipos[niveles[nombres.index(nombre_lvl)]['tipo']], pp_v, grupo)
        eje = st.selectbox('Mostrar vigas del eje', ['Todos'] + [x for x in L if any(r['tipo'] == 'x' and L[r['j']] == x for r in f)] +
                           [str(i + 1) for i in range(nx + 1) if any(r['tipo'] == 'y' and r['i'] == i for r in f)], key='ed_eje')
        def en_eje(r):
            return eje == 'Todos' or (r['tipo'] == 'x' and L[r['j']] == eje) or (r['tipo'] == 'y' and str(r['i'] + 1) == eje)
        df = pd.DataFrame([[r['tramo'], r['L'], r['A'], r['CM'], r['CV'], r['wu'], r['wacc']] for r in f if en_eje(r)],
                          columns=['Tramo', 'L (m)', 'A losa (m²)', 'CM (t/m)', 'CV máx (t/m)', f'wu gravedad (t/m) [{bj.FC_GRUPO[grupo][0]:g}/{bj.FC_GRUPO[grupo][1]:g}]',
                                   'w accidental 1.1(CM+CVacc) (t/m)'])
        st.dataframe(df.style.format({c: '{:,.3f}' for c in df.columns[2:]}), hide_index=True, width='stretch')
        st.caption(f'Nivel {nombre_lvl}. CM incluye el peso propio de la viga. Con estas cargas y los claros captura la viga en «Examen 3 → Viga continua» '
                   '(la acción del sismo se suma con el pórtico).')


    with tab_e:
        analisis_eje(niveles, masks, dx, dy, tipos, pp_v, pp_c, vb, vh, cb, ch, fc, grupo, res_sismo, L)


def analisis_eje(niveles, masks, dx, dy, tipos, pp_v, pp_c, vb, vh, cb, ch, fc, grupo, res_sismo, L):
    """Pórtico plano de un eje del edificio (p. ej. el eje C): análisis con las cargas de la bajada y el sismo, y diseño de sus vigas."""
    import ui_porticos as up
    st.caption('Arma el pórtico plano del eje con las cargas por tramo que salen de la planta, le aplica el sismo y diseña las vigas '
               'como en la pestaña «Examen 3 → Pórtico»: envolvente, diagramas de M y V, armado y alzado de la viga continua del nivel.')
    c1, c2, c3 = st.columns(3)
    eje = c1.selectbox('Eje a analizar (letra)', L, index=min(2, len(L) - 1), key='ed_eje_an')
    fila = L.index(eje)
    n = len(niveles)
    base = c2.radio('Base de las columnas', ['Empotrada', 'Articulada'], horizontal=True, key='ed_eje_base')
    fi = c3.number_input('Factor de I en vigas y columnas (agrietamiento)', 0.1, 1.0, 1.0, 0.05, key='ed_eje_fi')
    try:
        d = ed.cargas_eje(niveles, masks, dx, dy, tipos, pp_v, fila)
    except ValueError as e:
        st.error(str(e))
        return
    st.markdown('**Fuerzas sísmicas del eje**')
    c1, c2 = st.columns(2)
    origen = c1.radio('Origen', ['Capturar (t)', 'Traer del análisis del Examen 2'], key='ed_eje_orig',
                      help='Las fuerzas del examen salen del modal espectral. Si modelaste todo el edificio en el Examen 2, multiplica por la '
                           'fracción que toma este eje.')
    pesos = ed.pesos_por_nivel(niveles, masks, dx, dy, tipos, pp_v, pp_c)[::-1]             # de abajo hacia arriba
    zs = np.cumsum(d['alturas'])
    if origen.startswith('Traer'):
        fuente = c2.selectbox('Fuerzas tomadas de', list(up.FUENTES_F), key='ed_eje_fuente')
        frac = c2.number_input('Fracción que toma este eje', 0.01, 1.0, 1.0, 0.01, format='%.3f', key='ed_eje_frac')
        if res_sismo is None or up.FUENTES_F[fuente] not in res_sismo:
            st.warning('Todavía no hay análisis sísmico en el Examen 2 (completa «1 · Estructura» y «2 · Sismo y norma»).')
            return
        Fv = np.asarray(res_sismo[up.FUENTES_F[fuente]]['F'], float)[::-1] * frac
        if len(Fv) != n:
            st.warning(f'El análisis del Examen 2 tiene {len(Fv)} niveles y este edificio {n}.')
            return
        F = [float(x) for x in Fv]
    else:
        coef = c2.number_input('Cortante basal de partida V/W (solo reparte la tabla; cámbiala por tus fuerzas)', 0.0, 1.0, 0.10, 0.01,
                               key='ed_eje_coef')
        frac = c1.number_input('Fracción que toma este eje del sismo en X', 0.01, 1.0, 0.333, 0.001, format='%.3f', key='ed_eje_frac2',
                               help='Sin excentricidades y con ejes iguales, cada eje toma 1/(número de ejes en la dirección): 1/3 para B, C y D.')
        k_ver = f"{n}_{coef}_{frac}_{sum(p['W'] for p in pesos):.0f}"
        F0 = [x * frac for x in ed.fuerzas_triangulares([p['W'] for p in pesos], list(zs), coef)]
        df0 = pd.DataFrame({'Nivel': d['nombres'], 'F sismo del eje (t)': [f'{x:.2f}' for x in F0]})
        st.caption('De abajo hacia arriba. Valores de partida: reparto triangular de V = c·W. Escribe aquí las fuerzas de tu modal espectral '
                   '(por eje: si el edificio se reparte entre 3 ejes iguales, cada uno toma 1/3 del total).')
        dfe = st.data_editor(df0, hide_index=True, width='stretch', disabled=['Nivel'], key=f'ed_eje_F_{k_ver}',
                             column_config={'F sismo del eje (t)': st.column_config.TextColumn('F sismo del eje (t)')})
        try:
            F = [float(str(x).replace(',', '.')) for x in dfe['F sismo del eje (t)']]
        except ValueError:
            st.error('Escribe las fuerzas como números.')
            return
    st.markdown('**Combinaciones** (apuntes p. 2): gravitacional, y CM + CVacc ± sismo con FC = 1.1')
    fcm, fcv = bj.FC_GRUPO[grupo]
    combos = {'Gravedad': (fcm, fcv, 0.0, 0.0),
              'CM+CVacc + 100%SX (+30%SY)': (1.1, 0.0, 1.1, 1.1), 'CM+CVacc − 100%SX': (1.1, 0.0, 1.1, -1.1),
              'CM+CVacc − 30%SX (+100%SY)': (1.1, 0.0, 1.1, -0.33), 'CM+CVacc + 30%SX': (1.1, 0.0, 1.1, 0.33)}
    st.caption('Las vigas de este eje trabajan en el plano del pórtico: el sismo en la otra dirección (SY) casi no les produce flexión, así que se '
               'toma solo la componente SX de cada combinación (100 % o 30 %). Con ± 100 % SX la envolvente ya cubre las demás.')
    try:
        res = pt.resolver(d['spans'], d['alturas'], [(cb, ch)] * n, [(vb, vh)] * n, 14000 * np.sqrt(fc), d['cargas'], F, base, fi, fi, 0.0)
    except (np.linalg.LinAlgError, ValueError) as e:
        st.error(f'No se pudo resolver el pórtico: {e}')
        return
    st.caption('Numeración de pisos en las pestañas de abajo: ' + ', '.join(f'piso {k + 1} = {nm}' for k, nm in enumerate(d['nombres'])) + '.')
    c1, c2, c3 = st.columns(3)
    est = c1.selectbox('Estribo', ['#3', '#4'], index=1, key='ed_eje_est')
    rec = c2.number_input('Recubrimiento libre (cm)', 1.5, 8.0, 4.0, 0.5, key='ed_eje_rec')
    agr = c3.checkbox('Permitir barras de 2 en 2 en un lecho', True, key='ed_eje_agr')
    up._resultados(res, res['geo'], F, combos, {'Gravedad': (fcm, fcv, 0.0, 0.0)}, 'CM+CVacc + 100%SX (+30%SY)', 'CM+CVacc − 100%SX', fc,
                   4200.0, est, rec, agr, True, pref='ed_eje')
