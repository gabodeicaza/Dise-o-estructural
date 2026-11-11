"""Examen 1 · Bajada de cargas de una columna (apuntes pp. 5 y 8): tablas editables, resultado por nivel en formato de pizarrón."""
import pandas as pd
import streamlit as st

import ame_bajada as bj

COL_T = ['Tipo de piso', 'CM CG', 'CV máx CG', 'CM CA', 'CV acc CA', 'CM CS', 'CV med CS']
COL_N = ['Nivel', 'Tipo de piso', 'At (m²)', 'L viga (m)', 'h col (m)']

EJEMPLOS = {
    'Ejemplo 1 · columna B-2, biblioteca, Grupo A (apuntes p. 5)': dict(
        grupo='A', vig=(25, 50), col=(40, 45),
        tipos=[('Azotea', 530, 100, 530, 70, 530, 15), ('Intermedio', 610, 350, 610, 250, 610, 40)],
        niveles=[('N3 (azotea)', 'Azotea', 11.25, 7, 0), ('N2', 'Intermedio', 11.25, 7, 3), ('N1', 'Intermedio', 11.25, 7, 3),
                 ('PB', 'Intermedio', 0, 0, 4.5)]),
    'Ejemplo 2 · columna A-1, oficinas, Grupo B (apuntes p. 8)': dict(
        grupo='B', vig=(20, 50), col=(50, 50),
        tipos=[('Azotea', 530, 100, 530, 70, 530, 15), ('Oficinas', 670, 250, 670, 180, 670, 100)],
        niveles=[('Azotea', 'Azotea', 4, 4, 0), ('N2', 'Oficinas', 4, 4, 3), ('N1', 'Oficinas', 4, 4, 3), ('PB', 'Oficinas', 0, 0, 4)]),
}


def _num(x):
    return float(str(x).strip().replace(',', '.'))


def _cargar(clave):
    e = EJEMPLOS[clave]
    st.session_state['bj_ver'] = st.session_state.get('bj_ver', 0) + 1
    st.session_state['bj_tipos'] = pd.DataFrame([[t[0]] + [str(v) for v in t[1:]] for t in e['tipos']], columns=COL_T)
    st.session_state['bj_niv'] = pd.DataFrame([[n[0], n[1]] + [str(v) for v in n[2:]] for n in e['niveles']], columns=COL_N)
    st.session_state.update({'bj_grupo': e['grupo'], 'bj_vb': float(e['vig'][0]), 'bj_vh': float(e['vig'][1]),
                             'bj_cb': float(e['col'][0]), 'bj_ch': float(e['col'][1])})


def _kg(x):
    return f'{x:,.2f}'.rstrip('0').rstrip('.') if abs(x - round(x)) > 1e-9 else f'{x:,.0f}'


def mostrar():
    st.caption('Bajada de cargas de una columna, de arriba hacia abajo, como en los apuntes: en cada nivel se suma la losa del '
               'área tributaria, el peso propio de las vigas y el de la columna. Se lleva CM y CV por separado para aplicar '
               'después el factor de carga que corresponde (Grupo A: 1.5 / 1.7, Grupo B: 1.3 / 1.5, accidental 1.1, servicio 1.0).')
    if 'bj_tipos' not in st.session_state:
        _cargar(list(EJEMPLOS)[0])
    c1, c2 = st.columns([3, 1])
    ej = c1.selectbox('Ejemplo de los apuntes', list(EJEMPLOS), key='bj_ej')
    c2.write('')
    c2.button('Cargar ejemplo', key='bj_cargar', on_click=lambda: _cargar(ej))

    c1, c2, c3, c4, c5 = st.columns(5)
    grupo = c1.radio('Grupo', ['A', 'B'], horizontal=True, key='bj_grupo')
    fc = c2.number_input("f′c (kg/cm²)", 100.0, 1000.0, 250.0, key='bj_fc')
    gam = c3.number_input('γ concreto (t/m³)', 2.0, 2.6, 2.4, 0.05, key='bj_gam')
    vb = c4.number_input('Viga b (cm)', 10.0, 200.0, 25.0, key='bj_vb')
    vh = c5.number_input('Viga h (cm)', 10.0, 300.0, 50.0, key='bj_vh')
    c1, c2, c3 = st.columns(3)
    cb = c1.number_input('Columna b (cm)', 10.0, 300.0, 40.0, key='bj_cb')
    ch = c2.number_input('Columna h (cm)', 10.0, 300.0, 45.0, key='bj_ch')
    pp_v, pp_c = bj.pp_lineal(vb, vh, gam), bj.pp_lineal(cb, ch, gam)
    c3.metric('p.p. viga / columna (kg/m)', f'{pp_v:g} / {pp_c:g}')

    st.markdown('##### Cargas por m² de cada tipo de piso (kg/m²)')
    st.caption('CG = gravedad (CM y CVmáx), CA = accidental (CM y CVacc), CS = servicio (CM y CVmed). Cada CM ya incluye losa, '
               'rellenos, recubrimientos, instalaciones, muros divisorios y 40 kg/m² del reglamento.')
    v = st.session_state.get('bj_ver', 0)
    conf = {c: st.column_config.TextColumn(c) for c in COL_T}
    tipos_df = st.data_editor(st.session_state['bj_tipos'], num_rows='dynamic', hide_index=True, width='stretch',
                              key=f'bj_ed_t_{v}', column_config=conf).dropna()
    st.markdown('##### Niveles, de arriba hacia abajo')
    st.caption('At = área tributaria de la columna en ese nivel; L viga = metros de viga que bajan a la columna; h col = altura de la '
               'columna que se suma en ese nivel (0 en la azotea; agrega una fila «PB» con At = 0 para la columna de planta baja).')
    conf = {c: st.column_config.TextColumn(c) for c in COL_N}
    niv_df = st.data_editor(st.session_state['bj_niv'], num_rows='dynamic', hide_index=True, width='stretch',
                            key=f'bj_ed_n_{v}', column_config=conf).dropna()
    try:
        tipos = {str(r['Tipo de piso']).strip(): dict(CG=(_num(r['CM CG']), _num(r['CV máx CG'])),
                                                      CA=(_num(r['CM CA']), _num(r['CV acc CA'])),
                                                      CS=(_num(r['CM CS']), _num(r['CV med CS']))) for _, r in tipos_df.iterrows()}
        niveles = [dict(nombre=str(r['Nivel']), tipo=str(r['Tipo de piso']).strip(), At=_num(r['At (m²)']), Lv=_num(r['L viga (m)']),
                        hc=_num(r['h col (m)'])) for _, r in niv_df.iterrows()]
        if not niveles:
            st.info('Captura al menos un nivel.')
            return
        filas = bj.bajada(niveles, tipos, pp_v, pp_c)
    except (ValueError, KeyError) as e:
        st.error(str(e) if 'convert' not in str(e) else 'Escribe números en las tablas (decimales con punto o coma).')
        return
    base = bj.cargas_base(filas, grupo, fc)

    st.markdown('##### Bajada de cargas (kg)')
    caso = st.radio('Combinación a mostrar', bj.CASOS, format_func=lambda c: bj.NOMBRE_CASO[c], horizontal=True, key='bj_caso')
    for f in filas:
        d = f[caso]
        cm_l, cv_l = tipos[f['tipo']][caso]
        partes = [f"{_kg(f['At'])}({_kg(cm_l)})"] if f['At'] else []
        if f['Lv']:
            partes.append(f"{_kg(f['Lv'])}({_kg(pp_v)})")
        if f['hc']:
            partes.append(f"{_kg(f['hc'])}({_kg(pp_c)})")
        ant = f" {_kg(d['cm_ant'])} +" if d['cm_ant'] else ''
        st.markdown(f"**{f['nombre']}**")
        st.latex(rf"CM = {ant}{' + '.join(partes) if partes else '0'} = {_kg(d['cm'])}\ \text{{kg}}")
        antv = f" {_kg(d['cv_ant'])} +" if d['cv_ant'] else ''
        st.latex(rf"CV = {antv}{_kg(f['At'])}({_kg(cv_l)}) = {_kg(d['cv'])}\ \text{{kg}}")

    tab = pd.DataFrame([[f['nombre'], f['At'], round(f[caso]['d_cm'], 1), round(f[caso]['d_cv'], 1), round(f[caso]['cm'], 1),
                         round(f[caso]['cv'], 1), round(f[caso]['cm'] + f[caso]['cv'], 1)] for f in filas],
                       columns=['Nivel', 'At (m²)', 'ΔCM', 'ΔCV', 'CM acum.', 'CV acum.', 'CM + CV'])
    st.dataframe(tab, hide_index=True, width='stretch')

    st.markdown('##### Carga en la base de la columna')
    r = pd.DataFrame([[bj.NOMBRE_CASO[c], b['CM'] / 1000, b['CV'] / 1000, b['Pn'] / 1000, f"{b['FC_CM']:g} / {b['FC_CV']:g}", b['Pu'] / 1000]
                      for c, b in base['casos'].items()],
                     columns=['Combinación', 'CM (t)', 'CV (t)', 'Pn nominal (t)', 'FC (CM / CV)', 'Pu de diseño (t)'])
    st.dataframe(r.style.format({c: '{:,.3f}' for c in r.columns[1:4]} | {'Pu de diseño (t)': '{:,.3f}'}), hide_index=True, width='stretch')
    g = base['casos']['CG']
    st.latex(rf"P_0 = {_kg(g['CM'])} + {_kg(g['CV'])} = {_kg(g['Pn'])}\ \text{{kg (nominal)}}")
    st.latex(rf"P_{{0u}} = {_kg(g['CM'])}({g['FC_CM']:g}) + {_kg(g['CV'])}({g['FC_CV']:g}) = {_kg(g['Pu'])}\ \text{{kg de diseño}}")
    st.latex(rf"A = \dfrac{{P_u}}{{0.25 f'_c}} = \dfrac{{{_kg(base['Pu'])}}}{{0.25({fc:g})}} = {_kg(base['A_col'])}\ \text{{cm}}^2"
             rf"\ \Rightarrow\ {base['lado']:.1f}\times{base['lado']:.1f}\ \text{{cm (predimensionado, rige {base['rige']})}}")

    with st.expander('Carga repartida sobre una viga desde la losa (para el diagrama de diseño de la viga)'):
        c1, c2, c3, c4 = st.columns(4)
        wcm = c1.number_input('CM de piso (kg/m²)', 0.0, 3000.0, 670.0, key='bj_wcm')
        wcv = c2.number_input('CV máx (kg/m²)', 0.0, 3000.0, 250.0, key='bj_wcv')
        at = c3.number_input('Área tributaria At (m²)', 0.0, 500.0, 7.2, key='bj_at')
        lv = c4.number_input('Claro de la viga L (m)', 0.5, 50.0, 4.0, key='bj_lv')
        fcm, fcv = bj.FC_GRUPO[grupo]
        w = bj.carga_viga_desde_losa(wcm, wcv, at, lv, pp_v, fcm, fcv)
        st.latex(rf"CM = \dfrac{{{wcm:g}({at:g})}}{{{lv:g}}} + {pp_v:g} = {w['cm']:,.1f}\ \text{{kg/m}}\qquad "
                 rf"CV = \dfrac{{{wcv:g}({at:g})}}{{{lv:g}}} = {w['cv']:,.1f}\ \text{{kg/m}}")
        st.latex(rf"w_u = {fcm:g}({w['cm']:,.1f}) + {fcv:g}({w['cv']:,.1f}) = {w['wu']:,.1f}\ \text{{kg/m}} \approx {w['wu'] / 1000:.2f}\ \text{{t/m}}")
        st.caption('Con esa w_u (y las cargas puntuales de las vigas secundarias) captura la viga en «Viga continua»: diagramas, '
                   'flexión, estribos y la revisión de la flecha.')
