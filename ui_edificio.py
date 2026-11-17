"""Bajada de cargas de un edificio de varios niveles: una planta (cuadrícula) editable por nivel, con copia entre niveles."""
import numpy as np
import pandas as pd
import streamlit as st

import ame_bajada as bj
import ame_edificio as ed
import ame_planta as pl
import ame_porticos as pt

COL_C = ['Concepto', 'CG', 'CA', 'CS']
COL_N = ['Incluir', 'Nivel', 'Tipo de piso', 'h entrepiso debajo (m)']
OPC = ['Losa', 'Voladizo', 'Vacío']
OPC_COL = ['Auto', 'Sí', 'No']
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


_AZOTEA = ('Azotea', 10, [('Rellenos', 150), ('Instalaciones', 100), ('RCDF', 40)], (100, 70, 15), 'Azotea')
EJEMPLOS = {
    'Ejercicio del pizarrón (8 niveles, oficinas, voladizo en A–B)': dict(
        cx='8 6 8 8 8 6', cy='4 6 6', grupo='B', vig=(40, 80), col=(70, 70), sel=('N2', 'C-4'),
        tipos=[('Oficinas', 0, [('CM dada', 300)], (250, 180, 100), 'Oficinas'), ('Azotea', 0, [('CM dada', 300)], (100, 70, 15), 'Azotea')],
        niveles=[(f'N{k}', 'Azotea' if k == 8 else 'Oficinas', 3.8) for k in range(8, 0, -1)],
        mask=lambda: _mask_pizarron()),
    'Apuntes · Ejemplo 1: columna B-2, biblioteca, Grupo A (p. 5)': dict(
        cx='5 4', cy='3 2', grupo='A', vig=(25, 50), col=(40, 45), sel=('N1', 'B-2'),
        tipos=[_AZOTEA, ('Intermedio', 10, [('Recubrimientos', 170), ('Instalaciones', 60), ('M. divisorios', 100), ('RCDF', 40)],
                         (350, 250, 40), 'Biblioteca')],
        niveles=[('N3 (azotea)', 'Azotea', 3), ('N2', 'Intermedio', 3), ('N1', 'Intermedio', 4.5)],
        mask=lambda: np.full((2, 2), pl.LOSA)),
    'Apuntes · Ejemplo 2: columna A-1, oficinas, Grupo B (p. 8)': dict(
        cx='4 2.5 5.4', cy='4 2', grupo='B', vig=(20, 50), col=(50, 50), sel=('N1', 'A-1'),
        tipos=[_AZOTEA, ('Oficinas', 10, [('Recubrimientos', 170), ('Instalaciones', 120), ('M. divisorios', 100), ('RCDF', 40)],
                         (250, 180, 100), 'Oficinas'),
               ('Terraza', 10, [('Recubrimientos', 170), ('Rellenos', 150), ('Instalaciones', 100), ('RCDF', 40)], (250, 180, 100), 'Oficinas')],
        niveles=[('Azotea', 'Azotea', 3), ('N2', 'Oficinas', 3), ('N1', 'Oficinas', 4)],
        mask=lambda: np.full((2, 3), pl.LOSA)),
}
EJEMPLOS['Tutorial ETABS · edificio 20×10 m, 3 niveles, oficinas (guía con comprobación)'] = dict(
    cx='5 5 5 5', cy='5 5', grupo='B', vig=(25, 50), col=(40, 40), sel=('N1', 'B-3'),
    tipos=[('Azotea', 15, [('CM adicional', 140)], (100, 70, 15), 'Azotea'), ('Oficinas', 15, [('CM adicional', 140)], (250, 180, 100), 'Oficinas')],
    niveles=[('N3 (azotea)', 'Azotea', 3), ('N2', 'Oficinas', 3), ('N1', 'Oficinas', 3)],
    mask=lambda: np.full((2, 4), pl.LOSA))
EJEMPLOS['Guía ETABS · Ejercicio 1: casa de 2 niveles, vivienda, tableros 5×5'] = dict(
    cx='5 5', cy='5 5', grupo='B', vig=(20, 40), col=(30, 30), sel=('N1', 'B-2'),
    tipos=[('Azotea', 12, [('CM adicional', 150)], (100, 70, 15), 'Azotea'), ('Vivienda', 12, [('CM adicional', 150)], (170, 90, 70), 'Vivienda')],
    niveles=[('N2 (azotea)', 'Azotea', 3), ('N1', 'Vivienda', 3)],
    mask=lambda: np.full((2, 2), pl.LOSA))
EJEMPLOS['Guía ETABS · Ejercicio 2: oficinas de 3 niveles, tableros 6×4 (trapecial)'] = dict(
    cx='6 6', cy='4 4', grupo='B', vig=(25, 50), col=(40, 40), sel=('N1', 'B-2'),
    tipos=[('Azotea', 10, [('CM adicional', 200)], (100, 70, 15), 'Azotea'), ('Oficinas', 10, [('CM adicional', 170)], (250, 180, 100), 'Oficinas')],
    niveles=[('N3 (azotea)', 'Azotea', 3), ('N2', 'Oficinas', 3), ('N1', 'Oficinas', 4)],
    mask=lambda: np.full((2, 2), pl.LOSA))
USOS = {'Oficinas': (250, 180, 100), 'Azotea': (100, 70, 15), 'Biblioteca': (350, 250, 40), 'Vivienda': (170, 90, 70),
        'Otro (capturar CV)': None}


def _mask_pizarron():
    m = np.full((3, 6), pl.LOSA)
    m[0, :] = pl.VACIO
    m[0, :3] = pl.VOL
    return m


def _df_pisos(comp):
    return pd.DataFrame([[c, str(v), '', ''] for c, v in comp], columns=COL_C)


def _cargar_ejemplo(clave=None):
    """Llena claros, niveles, cargas, secciones y la cuadrícula de cada nivel con uno de los ejemplos."""
    st.session_state['ed_firma'] = None
    ss = st.session_state
    e = EJEMPLOS[clave or ss.get('ed_ej') or list(EJEMPLOS)[0]]
    ss['ed_ver'] = ss.get('ed_ver', 0) + 1
    ss['ed_cx'], ss['ed_cy'] = e['cx'], e['cy']
    ss['ed_pisos'] = {t[0]: dict(hl=str(t[1]), comp=_df_pisos(t[2]), cv=[str(x) for x in t[3]], uso=t[4]) for t in e['tipos']}
    ss['ed_piso_sel'] = e['tipos'][0][0]
    ss['ed_niv'] = pd.DataFrame([[True, n[0], n[1], str(n[2])] for n in e['niveles']], columns=COL_N)
    dx, dy = pl.parse_claros(e['cx']), pl.parse_claros(e['cy'])
    ss['ed_masks'] = {n[0]: _df_mask(e['mask'](), dx, dy) for n in e['niveles']}
    ss['ed_cols'] = {}
    ss['ed_key_dims'] = (len(dx), len(dy))
    ss.update({'ed_grupo': e['grupo'], 'ed_vb': float(e['vig'][0]), 'ed_vh': float(e['vig'][1]), 'ed_cb': float(e['col'][0]),
               'ed_ch': float(e['col'][1]), 'ed_sel_niv': e['sel'][0], 'ed_col': e['sel'][1]})


def _copiar(origen, destinos):
    ss = st.session_state
    base = ss['ed_masks'][origen].copy()
    cols = ss['ed_cols'].get(origen)
    for d in destinos:
        if d != origen:
            ss['ed_masks'][d] = base.copy()
            if cols is not None:
                ss['ed_cols'][d] = cols.copy()
    ss['ed_ver'] = ss.get('ed_ver', 0) + 1


def _agregar_tipo():
    ss = st.session_state
    nom = str(ss.get('ed_piso_nuevo', '')).strip()
    if nom and nom not in ss['ed_pisos']:
        ss['ed_pisos'][nom] = dict(hl='10', comp=pd.DataFrame([['Acabados', '0', '', '']], columns=COL_C), cv=['0', '0', '0'], uso='Otro (capturar CV)')
        ss['ed_piso_sel'] = nom
        ss['ed_ver'] = ss.get('ed_ver', 0) + 1
    ss['ed_piso_nuevo'] = ''


def _quitar_tipo():
    ss = st.session_state
    if len(ss['ed_pisos']) > 1:
        ss['ed_pisos'].pop(ss.get('ed_piso_sel'), None)
        ss['ed_piso_sel'] = next(iter(ss['ed_pisos']))
        ss['ed_ver'] = ss.get('ed_ver', 0) + 1


def _tabla_html(encab, filas):
    """Tabla con el aspecto de los apuntes: encabezado azul, renglones alternados y totales en negritas. filas = (c1, c2, c3, c4, negrita)."""
    import html
    estilo_th = 'background:#0b3558;color:#fff;padding:9px 12px;text-align:{a};font-weight:700'
    out = ['<table style="width:100%;border-collapse:collapse;font-size:15px;color:#1b2733">', '<tr>']
    for k, h in enumerate(encab):
        out.append(f'<th style="{estilo_th.format(a="left" if k < 2 else "right")}">{html.escape(h)}</th>')
    out.append('</tr>')
    for n, (a, b, c, d, neg) in enumerate(filas):
        bg = '#dbe7f4' if neg else ('#f1f5fa' if n % 2 == 0 else '#ffffff')
        w = '700' if neg else '400'
        out.append(f'<tr style="background:{bg};font-weight:{w};border-bottom:1px solid #cfd9e6">'
                   f'<td style="padding:8px 12px">{html.escape(a)}</td><td style="padding:8px 12px">{html.escape(b)}</td>'
                   f'<td style="padding:8px 12px;text-align:right">{html.escape(c)}</td><td style="padding:8px 12px;text-align:right">{html.escape(d)}</td></tr>')
    out.append('</table>')
    return ''.join(out)


def _fmt(x, dec=0):
    return f'{x:,.{dec}f}'


def _comp(p):
    comp = [(str(r['Concepto']), _num(r['CG']), r['CA'], r['CS']) for _, r in p['comp'].iterrows() if str(r['CG']).strip() != '']
    return [(c, cg, None if str(ca).strip() == '' else _num(ca), None if str(cs).strip() == '' else _num(cs)) for c, cg, ca, cs in comp]


def _tipo(p, gam=2.4):
    """Tipo de piso: PP de la losa (espesor × γ) + componentes de CM + CV, para los tres casos."""
    hl = _num(p.get('hl', '0') or 0)
    comp = _comp(p)
    if hl > 0:
        comp = [('PP losa', hl / 100 * gam * 1000, None, None)] + comp
    return bj.tipo_desde_componentes(comp, [_num(x) for x in p['cv']])


def _editor_pisos(v, gam):
    """Calculadora de cargas por tipo de piso, con el formato de los apuntes: PP de la losa (espesor × γ), componentes de CM,
    CM total, CV según el uso y total; las tres combinaciones (CG, CA, CS)."""
    ss = st.session_state
    pisos = ss['ed_pisos']
    st.markdown('##### Cargas de cada tipo de piso (calculadora)')
    st.caption('Configura cada tipo de piso: el espesor de la losa da su peso propio solo (espesor × γ), agrega los demás componentes de la carga muerta '
               'y elige el uso para la carga viva. El resultado se ve como en los apuntes y alimenta toda la bajada de cargas.')
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    nom = c1.selectbox('Tipo de piso a editar', list(pisos), key='ed_piso_sel')
    c2.text_input('Nombre de un tipo nuevo', key='ed_piso_nuevo')
    c3.write('')
    c3.button('Agregar', key='ed_piso_add', on_click=_agregar_tipo)
    c4.write('')
    c4.button('Quitar este', key='ed_piso_del', on_click=_quitar_tipo, disabled=len(pisos) < 2)
    p = pisos[nom]
    p.setdefault('hl', '0')
    p.setdefault('uso', 'Otro (capturar CV)')
    c1, c2 = st.columns(2)
    p['hl'] = c1.text_input('Espesor de la losa (cm) — 0 si el piso no lleva losa propia', p['hl'], key=f'ed_hl_{nom}_{v}')
    usos = list(USOS)
    p['uso'] = c2.selectbox('Uso (carga viva)', usos, index=usos.index(p['uso']) if p['uso'] in usos else len(usos) - 1, key=f'ed_uso_{nom}_{v}')
    if USOS[p['uso']] is not None:
        p['cv'] = [str(x) for x in USOS[p['uso']]]
        c2.caption(f"CVmáx {p['cv'][0]} · CVacc {p['cv'][1]} · CVmed {p['cv'][2]} kg/m² (tabla de los apuntes)")
    else:
        d1, d2, d3 = st.columns(3)
        p['cv'] = [d1.text_input('CVmáx (CG)', p['cv'][0], key=f'ed_cv0_{nom}_{v}'), d2.text_input('CVacc (CA)', p['cv'][1], key=f'ed_cv1_{nom}_{v}'),
                   d3.text_input('CVmed (CS)', p['cv'][2], key=f'ed_cv2_{nom}_{v}')]
    st.markdown('**Carga muerta adicional** (acabados, rellenos, instalaciones, plafones, muros divisorios, reglamento…)')
    var = st.checkbox('Algún componente vale distinto en accidental o servicio', value=bool((p['comp'][['CA', 'CS']].astype(str).apply(lambda s: s.str.strip()) != '').any().any()),
                      key=f'ed_var_{nom}_{v}')
    conf = {'Concepto': st.column_config.TextColumn('Concepto'), 'CG': st.column_config.TextColumn('kg/m² (gravedad)'),
            'CA': st.column_config.TextColumn('kg/m² (accidental)', help='Vacío = igual que en gravedad'),
            'CS': st.column_config.TextColumn('kg/m² (servicio)', help='Vacío = igual que en gravedad')}
    ed_comp = st.data_editor(p['comp'], num_rows='dynamic', hide_index=True, width='stretch', key=f'ed_comp_{nom}_{v}', column_config=conf,
                             column_order=COL_C if var else COL_C[:2])
    p['comp'] = ed_comp.fillna('')
    caso = st.radio('Combinación a mostrar', bj.CASOS, format_func=lambda c: bj.NOMBRE_CASO[c], horizontal=True, key='ed_piso_caso')
    try:
        filas, cm, cv = bj.desglose_piso(_num(p['hl'] or 0), gam, _comp(p), [_num(x) for x in p['cv']], caso, p['uso'] if USOS[p['uso']] else '')
    except ValueError:
        st.warning('Escribe números en las cargas de este tipo de piso.')
        return
    st.markdown(_tabla_html(['Carga', 'Cálculo', 'kg/m²', 't/m²'], [(a, b, _fmt(c), f'{c / 1000:.3f}', neg) for a, b, c, neg in filas]),
                unsafe_allow_html=True)
    todos = _tipo(p, gam)
    st.caption('Resumen de este tipo de piso → ' + ' · '.join(f"{c}: CM {todos[c][0]:,.0f} + CV {todos[c][1]:,.0f} = {sum(todos[c]):,.0f} kg/m²"
                                                           for c in bj.CASOS))


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

    c1, c2, c3 = st.columns(3)
    grupo = c1.radio('Grupo', ['A', 'B'], horizontal=True, key='ed_grupo')
    fc = c2.number_input("f′c (kg/cm²)", 100.0, 1000.0, 250.0, key='ed_fc')
    gam = c3.number_input('γ concreto (t/m³)', 2.0, 2.6, 2.4, 0.05, key='ed_gam')
    c1, c2, c3, c4 = st.columns(4)
    vb = c1.number_input('Viga b (cm)', 10.0, 200.0, 40.0, key='ed_vb')
    vh = c2.number_input('Viga h (cm)', 10.0, 300.0, 80.0, key='ed_vh')
    cb = c3.number_input('Columna b (cm)', 10.0, 300.0, 70.0, key='ed_cb')
    ch = c4.number_input('Columna h (cm)', 10.0, 300.0, 70.0, key='ed_ch')
    pp_v, pp_c = bj.pp_lineal(vb, vh, gam), bj.pp_lineal(cb, ch, gam)
    st.caption(f'Peso propio: viga {pp_v:g} kg/m · columna {pp_c:g} kg/m. Predimensionado de los apuntes: h = L/10, b = h/2.')

    v = ss.get('ed_ver', 0)
    _editor_pisos(v, gam)
    st.markdown('##### Niveles, de arriba hacia abajo')
    st.caption('«Incluir» activa o desactiva un nivel sin borrarlo (así puedes analizar un piso sin los demás y volver a encenderlos). '
               '«h entrepiso debajo» es la altura entre ese nivel y el de abajo (para el primer nivel, hasta el desplante). '
               'Agrega o quita filas para cambiar el número de niveles.')
    conf = {c: st.column_config.TextColumn(c) for c in COL_N[1:]}
    conf['Incluir'] = st.column_config.CheckboxColumn('Incluir', default=True)
    conf['Tipo de piso'] = st.column_config.SelectboxColumn('Tipo de piso', options=list(ss['ed_pisos']), required=True)
    ndf = st.data_editor(ss['ed_niv'], num_rows='dynamic', hide_index=True, width='stretch', key=f'ed_n_{v}', column_config=conf)
    ndf = ndf.dropna(subset=COL_N[1:])

    c1, c2 = st.columns(2)
    cx = c1.text_input('Claros en X (m), de izquierda a derecha — ejes 1, 2, 3…', key='ed_cx')
    cy = c2.text_input('Claros en Y (m), de arriba hacia abajo — ejes A, B, C…', key='ed_cy')
    try:
        dx, dy = pl.parse_claros(cx), pl.parse_claros(cy)
        tipos = {nom: _tipo(p, gam) for nom, p in ss['ed_pisos'].items()}
        todos_niv = [dict(nombre=str(r['Nivel']).strip(), tipo=str(r['Tipo de piso']).strip(), h=_num(r['h entrepiso debajo (m)']),
                          incluir=bool(r['Incluir'])) for _, r in ndf.iterrows()]
        if len({n['nombre'] for n in todos_niv}) != len(todos_niv):
            raise ValueError('Los nombres de los niveles deben ser distintos.')
        niveles = [n for n in todos_niv if n['incluir']]
        if not niveles:
            st.info('Activa «Incluir» en al menos un nivel.')
            return
        for n in niveles:
            if n['tipo'] not in tipos:
                raise ValueError(f"El tipo de piso «{n['tipo']}» del nivel {n['nombre']} no está definido arriba.")
    except (ValueError, KeyError) as e:
        st.error(str(e) if 'convert' not in str(e) else 'Escribe números en las tablas (decimales con punto o coma).')
        return

    nx, ny = len(dx), len(dy)
    L = pl.letras(ny + 1)
    if ss.get('ed_key_dims') != (nx, ny):                      # cambiaron los ejes: cuadrículas nuevas, todo con losa
        ss['ed_masks'] = {}
        ss['ed_cols'] = {}
        ss['ed_key_dims'] = (nx, ny)
        ss['ed_ver'] = v = ss.get('ed_ver', 0) + 1
    vacia = _df_mask(np.full((ny, nx), pl.LOSA), dx, dy)
    auto = pd.DataFrame('Auto', index=L, columns=[str(k + 1) for k in range(nx + 1)])
    ss.setdefault('ed_cols', {})
    for n in todos_niv:
        ss['ed_masks'].setdefault(n['nombre'], vacia.copy())
        ss['ed_cols'].setdefault(n['nombre'], auto.copy())
    nombres = [n['nombre'] for n in niveles]

    st.markdown('##### Planta de cada nivel')
    c1, c2 = st.columns([1, 2])
    sel_niv = c1.selectbox('Nivel a editar y ver', nombres, key='ed_sel_niv')
    ss['ed_masks'][sel_niv] = st.data_editor(
        ss['ed_masks'][sel_niv], width='stretch', key=f'ed_m_{sel_niv}_{v}',
        column_config={c: st.column_config.SelectboxColumn(c, options=OPC, required=True) for c in vacia.columns})
    st.caption('Cada casilla es un tablero entre ejes: «Losa» (apoyada en vigas), «Voladizo» (su carga baja a la viga del lado donde '
               'se apoya; no genera columnas) o «Vacío». Un nivel puede quedar sin losa en partes o por completo.')
    with st.expander('Columnas de este nivel (por defecto salen de las losas)'):
        ss['ed_cols'][sel_niv] = st.data_editor(
            ss['ed_cols'][sel_niv], width='stretch', key=f'ed_c_{sel_niv}_{v}',
            column_config={c: st.column_config.SelectboxColumn(c, options=OPC_COL, required=True) for c in auto.columns})
        st.caption('Filas = ejes de letra, columnas = ejes de número. «Auto»: hay columna si el nodo toca una losa. «Sí»: fuerza una columna aunque '
                   'no haya losa (como en un piso que solo tiene una parte de losa). «No»: quita la columna de ese nodo en este nivel.')
    with c2:
        k1, k2 = st.columns(2)
        origen = k1.selectbox('Copiar la cuadrícula de', nombres, index=nombres.index(sel_niv), key='ed_cp_o')
        dest = k2.multiselect('a los niveles', [n for n in nombres if n != origen], key='ed_cp_d')
        b1, b2 = st.columns(2)
        b1.button('Copiar a los niveles elegidos', key='ed_cp_b', disabled=not dest, on_click=lambda: _copiar(origen, dest))
        b2.button('Copiar a todos los niveles', key='ed_cp_t', on_click=lambda: _copiar(origen, nombres))
    masks = [pl.a_mask(ss['ed_masks'][n].to_numpy()) for n in nombres]
    extras = [pl.a_extra(ss['ed_cols'][n].to_numpy()) for n in nombres]
    mk, ek = masks[nombres.index(sel_niv)], extras[nombres.index(sel_niv)]
    if any(((m == pl.VOL).any() and not (m == pl.LOSA).any()) for m in masks):
        st.warning('Algún nivel solo tiene voladizos; un voladizo necesita una losa apoyada a su lado.')
    if not (mk == pl.LOSA).any() and not (ek == 1).any():
        st.info(f'El nivel {sel_niv} no tiene losa apoyada ni columnas forzadas: no hay nada que bajar en este nivel.')

    firma = repr((tipos, niveles, [m.tolist() for m in masks], [x.tolist() for x in extras], dx, dy, pp_v, pp_c, vb, vh, cb, ch, fc, grupo))
    st.divider()
    b1, b2 = st.columns([1, 3])
    if b1.button('Calcular', type='primary', key='ed_calcular', width='stretch'):
        ss['ed_firma'] = firma
    if ss.get('ed_firma') != firma:
        b2.info('Configura los pisos, los niveles y las plantas con calma; cuando termines presiona **Calcular** para ver la bajada, '
                'los pesos, las vistas y el análisis. Si cambias algo después, vuelve a presionarlo.')
        return
    b2.success('Resultados al día con la configuración actual.')

    @st.fragment
    def resultados():
        todos = ed.columnas_edificio(niveles, masks, dx, dy, tipos, pp_v, pp_c, extras)
        ex = np.any([pl.nodos_existentes(m, x) for m, x in zip(masks, extras)], axis=0)
        cols_nivel = [f'{L[j]}-{i + 1}' for j in range(ny + 1) for i in range(nx + 1) if ex[j, i]]
        if not cols_nivel:
            st.info('Ningún nivel tiene columnas todavía.')
            return
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
            st.pyplot(pl.figura_planta(mk, dx, dy, (ci, cj), tr[nom_tr.index(ver_tr)] if ver_tr != '—' else None, modo, f'Planta {sel_niv}', ek),
                      width='stretch')
            plt.close('all')
        with cb_:
            st.metric(f'Área tributaria de {col_sel} en {sel_niv}', f'{At:g} m²')
            st.metric('Metros de viga que bajan a la columna', f'{Lv:g} m')
            res = pl.resumen_nivel(mk, dx, dy, ek)
            st.metric('Losa del nivel', f"{res['area']:g} m²")
            st.caption(f"{res['columnas']} columnas y {res['vigas']:g} m de viga en {sel_niv}.")

        st.markdown('##### Vista del edificio')
        import matplotlib.pyplot as plt
        vista = st.radio('Vista', ['3D', 'Elevación (corte por un eje)'], horizontal=True, key='ed_vista')
        if vista == '3D':
            c1, c2, c3 = st.columns([2, 1, 1])
            mostrar = c1.multiselect('Niveles a mostrar', nombres, default=nombres, key=f'ed_3d_niv_{abs(hash(tuple(nombres)))}',
                                     help='Quita niveles para ver un piso solo o varios sin los demás.')
            elev = c2.slider('Inclinación', 0, 90, 24, key='ed_3d_el')
            azim = c3.slider('Giro', -180, 180, -62, key='ed_3d_az')
            st.pyplot(ed.figura_3d(niveles, masks, dx, dy, extras, (cj, ci), elev, azim, f'{len(mostrar)} de {len(niveles)} niveles', set(mostrar)),
                      width='stretch')
            plt.close('all')
            st.caption('Azul: losa; naranja claro: voladizo. En naranja fuerte, la columna elegida para la bajada con su área tributaria en cada '
                       'nivel. Cada nivel se dibuja con su propia planta y sus columnas. Mueve «Giro» e «Inclinación» para rotar la vista.')
        else:
            cortes = [f'Eje {x}' for x in L] + [f'Eje {k + 1}' for k in range(nx + 1)]
            corte = st.selectbox('Corte por el', cortes, index=min(2, len(L) - 1), key='ed_corte')
            eje_c = ('X', L.index(corte.split()[1])) if corte.split()[1].isalpha() else ('Y', int(corte.split()[1]) - 1)
            st.pyplot(ed.figura_elevacion(niveles, masks, dx, dy, eje_c, (cj, ci), f'{corte} · {len(niveles)} niveles', extras), width='stretch')
            plt.close('all')
            st.caption('Niveles a la izquierda con la altura de cada entrepiso, E1, E2… a la derecha (E1 = el de abajo), columnas y vigas del eje, y '
                       'apoyos empotrados. En naranja, la columna elegida cuando está en este eje.')

        tab_a, tab_b, tab_c, tab_d, tab_e, tab_f = st.tabs(['Bajada de la columna', 'Todas las columnas', 'Pesos por nivel',
                                                            'Cargas de las vigas', 'Análisis de un eje (vigas)', 'Comprobación (peso total)'])
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
            pesos = ed.pesos_por_nivel(niveles, masks, dx, dy, tipos, pp_v, pp_c, extras)
            st.markdown('**Peso y masa de cada nivel** (CM + CVacc: lo que pide el análisis sísmico)')
            niv_p = st.selectbox('Nivel', nombres, index=nombres.index(sel_niv), key='ed_peso_niv')
            p = pesos[nombres.index(niv_p)]
            cm_k, cv_k = p['cm'], p['cv']
            filas = [('Losa', f"{p['area']:,.2f} m² × (CM {cm_k:,.0f} + CVacc {cv_k:,.0f}) kg/m²", _fmt(p['losa']), f"{p['losa'] / 1000:,.2f}", False),
                     ('Vigas', f"{p['vigas_m']:,.2f} m × {pp_v:,.0f} kg/m", _fmt(p['vigas']), f"{p['vigas'] / 1000:,.2f}", False),
                     ('Columnas (mitad de arriba y de abajo)', f"{p['columnas']} col × {p['h_col']:g} m × {pp_c:,.0f} kg/m", _fmt(p['cols']),
                      f"{p['cols'] / 1000:,.2f}", False),
                     ('Peso del nivel W', 'Losa + vigas + columnas', _fmt(p['W']), f"{p['W'] / 1000:,.2f}", True)]
            st.markdown(_tabla_html(['Concepto', 'Cálculo', 'kg', 't'], filas), unsafe_allow_html=True)
            st.latex(rf"m = \dfrac{{W}}{{g}} = \dfrac{{{p['W'] / 1000:,.2f}}}{{9.81}} = {p['W'] / 1000 / 9.81:,.3f}\ \text{{t·s}}^2/\text{{m}}"
                     rf"\qquad = {p['W'] / 981:,.4f}\ \text{{kg·s}}^2/\text{{cm}}")
            st.markdown('**Todos los niveles**')
            dfp = pd.DataFrame([[q['nombre'], q['area'], q['vigas_m'], q['columnas'], q['losa'] / 1000, q['vigas'] / 1000, q['cols'] / 1000, q['W'] / 1000,
                                 q['W'] / 1000 / 9.81, q['W'] / 981] for q in pesos],
                               columns=['Nivel', 'Losa (m²)', 'Viga (m)', 'Columnas', 'Losa (t)', 'Vigas (t)', 'Columnas (t)', 'W (t)', 'm (t·s²/m)', 'm (kg·s²/cm)'])
            st.dataframe(dfp.style.format({c: '{:,.1f}' for c in dfp.columns[4:8]} | {'m (t·s²/m)': '{:,.3f}', 'm (kg·s²/cm)': '{:,.3f}'}),
                         hide_index=True, width='stretch')
            st.metric('Peso total de la estructura', f"{sum(q['W'] for q in pesos) / 1000:,.1f} t")
            st.caption('Peso sísmico por nivel con CM + CVacc (losa, vigas y la mitad de las columnas de arriba y de abajo). La masa m = W/g es la '
                       'que entra a la matriz M del método modal espectral del examen 2.')
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
            st.caption(f'Nivel {nombre_lvl}. CM incluye el peso propio de la viga. Con los Mu y Vu que te dé ETABS captura la sección en «Examen 3».')


        with tab_e:
            analisis_eje(niveles, masks, dx, dy, tipos, pp_v, pp_c, vb, vh, cb, ch, fc, grupo, res_sismo, L, extras)

        with tab_f:
            _comprobacion(niveles, masks, dx, dy, tipos, pp_v, pp_c, extras, grupo, gam, L)

    resultados()


def analisis_eje(niveles, masks, dx, dy, tipos, pp_v, pp_c, vb, vh, cb, ch, fc, grupo, res_sismo, L, extras=None):
    """Pórtico plano de un eje del edificio (p. ej. el eje C): análisis con las cargas de la bajada y el sismo, y diseño de sus vigas."""
    import ui_porticos as up
    st.caption('Arma el pórtico plano del eje con las cargas por tramo que salen de la planta, le aplica el sismo y diseña las vigas '
               'con envolvente, diagramas de M y V, armado y alzado de la viga continua del nivel.')
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
    pesos = ed.pesos_por_nivel(niveles, masks, dx, dy, tipos, pp_v, pp_c, extras)[::-1]             # de abajo hacia arriba
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


def _comprobacion(niveles, masks, dx, dy, tipos, pp_v, pp_c, extras, grupo, gam, L):
    """El «número de control» de la guía: el peso total del edificio por patrón (PP, CM, CV), contra la suma de las columnas y contra
    lo que da ETABS."""
    ss = st.session_state
    hl_tipo = {nom: _num(p.get('hl', '0') or 0) for nom, p in ss['ed_pisos'].items()}
    c1, c2 = st.columns([2, 1])
    caso = c1.radio('Combinación', bj.CASOS, format_func=lambda c: bj.NOMBRE_CASO[c], horizontal=True, key='ed_cmp_caso',
                    help='La guía llama «servicio» a PP + CM + CV con la carga viva de 250 y 100 kg/m² y factor 1.0: eso es la combinación de '
                         'gravedad (CVmáx) sin factores. La combinación de servicio de la norma usa CVmed.')
    fact = c2.number_input('Factor de la combinación «factorada»', 1.0, 2.0, 1.4, 0.05, key='ed_cmp_fact')
    r = ed.comprobacion(niveles, masks, dx, dy, tipos, pp_v, pp_c, hl_tipo, gam, caso, grupo, extras)
    t = r['tot']
    k = lambda x: f'{x / 1000:,.2f}'
    mas = ' + …' if len(r['filas']) > 2 else ''
    st.markdown('**Peso total del edificio** (tu número de control)')
    filas = [('PP losas', ' + '.join(f"{f['area']:g} m² × {f['pp_l'] / 1000:.3f}" for f in r['filas'][:2]) + mas, k(t['pp_losa']), False),
             ('PP vigas', ' + '.join(f"{f['vigas_m']:g} m × {pp_v / 1000:.3f}" for f in r['filas'][:2]) + mas, k(t['pp_vigas']), False),
             ('PP columnas', ' + '.join(f"{f['n_seg']} × {niveles[n]['h']:g} m × {pp_c / 1000:.3f}" for n, f in enumerate(r['filas'][:2])) + mas,
              k(t['pp_cols']), False),
             ('Peso propio total (PP)', 'Losas + vigas + columnas', k(t['pp']), True),
             ('CM adicional', 'Σ área × CM adicional', k(t['cm_ad']), False),
             ('CV', 'Σ área × CV de cada nivel', k(t['cv']), False),
             (f'TOTAL ({caso}, sin factores)', 'PP + CM + CV', k(t['servicio']), True),
             (f'Factorada (× {fact:g})', f"{k(t['servicio'])} × {fact:g}", k(t['servicio'] * fact), False),
             (f'Con factores del grupo {grupo}' if caso == 'CG' else 'Con los factores de la combinación',
              f"CM×{r['fc'][0]:g} + CV×{r['fc'][1]:g}", k(t['factorada_grupo']), False)]
    st.markdown(_tabla_html(['Concepto', 'Cálculo', 't', ''], [(a, b, c, '', neg) for a, b, c, neg in filas]), unsafe_allow_html=True)
    with st.expander('Desglose por nivel'):
        df = pd.DataFrame([[f['nombre'], f['area'], f['pp_losa'] / 1000, f['vigas_m'], f['pp_vigas'] / 1000, f['n_seg'], f['pp_cols'] / 1000,
                            f['cm_ad'] / 1000, f['cv'] / 1000] for f in r['filas']],
                          columns=['Nivel', 'Losa (m²)', 'PP losa (t)', 'Viga (m)', 'PP vigas (t)', 'Tramos de col.', 'PP col. (t)', 'CM adic. (t)', 'CV (t)'])
        st.dataframe(df.style.format({c: '{:,.2f}' for c in df.columns[1:]}), hide_index=True, width='stretch')
        st.caption('El tramo de columna se asigna al nivel que lo carga (el de arriba); el último nivel carga el tramo hasta el desplante.')

    st.markdown('**La gran comprobación: la suma de las columnas tiene que dar el mismo total**')
    grupos = {}
    for j, i, pn, _ in r['cols']:
        grupos.setdefault(round(pn / 1000, 2), []).append(f'{L[j]}-{i + 1}')
    fg = [(f'{len(n)} columnas', ', '.join(n), f'{pn * len(n):,.2f}', f'{pn:,.2f}', False) for pn, n in sorted(grupos.items())]
    fg.append(('Suma de las columnas', f'{len(r["cols"])} columnas', k(r['suma_cols']), '', True))
    st.markdown(_tabla_html(['Columnas con la misma carga', 'Ejes', 't (todas)', 't (cada una)'], fg), unsafe_allow_html=True)
    dif = r['suma_cols'] - t['servicio']
    (st.success if abs(dif) < 1 else st.error)(
        f"Suma de columnas {k(r['suma_cols'])} t  vs  total {k(t['servicio'])} t  →  diferencia {dif / 1000:+,.3f} t. "
        + ('La carga no se perdió en ningún tablero.' if abs(dif) < 1 else
           'No coinciden: revisa que todos los tableros tengan viga en el contorno y columna en las esquinas.'))

    with st.expander('Comparar con lo que te dio ETABS (Base Reactions, FZ)'):
        st.caption('Captura en toneladas lo que te dio la tabla de reacciones. Menos de 1 % es redondeo; si es más, el patrón que falla te dice dónde '
                   'buscar (CM mal → multiplicador o asignación; PP mal → secciones o elementos faltantes; todo ×1000 → unidades).')
        d1, d2, d3, d4 = st.columns(4)
        vals = [('PP', t['pp'], d1.number_input('PP (t)', 0.0, key='ed_et_pp', format='%.2f')),
                ('CM', t['cm_ad'], d2.number_input('CM (t)', 0.0, key='ed_et_cm', format='%.2f')),
                ('CV', t['cv'], d3.number_input('CV (t)', 0.0, key='ed_et_cv', format='%.2f')),
                ('Total', t['servicio'], d4.number_input('Servicio (t)', 0.0, key='ed_et_tot', format='%.2f'))]
        rows = []
        for nom, mio, et in vals:
            if et > 0:
                p = (et - mio / 1000) / (mio / 1000) * 100 if mio else 0.0
                rows.append(f"**{nom}**: tu cuenta {mio / 1000:,.2f} t · ETABS {et:,.2f} t · diferencia {p:+.2f} % " + ('✅' if abs(p) < 1 else '⚠️'))
        st.markdown('\n\n'.join(rows) if rows else 'Aún no capturas valores de ETABS.')
