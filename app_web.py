"""Versión web (Streamlit) del Análisis Modal Espectral.  Ejecutar:  streamlit run app_web.py"""
import io
import json

import numpy as np
import pandas as pd
import streamlit as st

import ame_sismico as ame

NAVY, AZUL_OSC, AZUL, AZUL_MED, AZUL_CLARO, PALE, BG = (
    '#0B2545', '#13315C', '#1D4E89', '#3E7CB1', '#81A4CD', '#DCE9F7', '#F1F6FC')

COLS = ['h', 'ncol', 'b', 'hs', 'nart', 'w', 'L', 'W']
ETIQ = {'h': 'h entrepiso (m)', 'ncol': 'Nº columnas', 'b': 'b (cm)', 'hs': 'h sección (cm)',
        'nart': 'Nº articuladas en la base', 'w': 'Carga w (t/m)', 'L': 'Longitud L (m)', 'W': 'o Peso W (t)'}
TIPOS = ['Grupo B', 'Grupo A - sismo base', 'Grupo A - sismo infrecuente']
CAMPOS = ['tipo', 'Q', 'k1', 'E', 'fc', 'factor_Fu', 'mult_V', 'a0', 'c', 'Ta', 'Tb', 'k', 'Ts']

EJEMPLO2 = dict(
    niveles=[dict(h='5.0', ncol='3', b='25', hs='50', nart='0', w='4.6917', L='9', W=''),
             dict(h='3.5', ncol='3', b='25', hs='50', nart='0', w='4.6667', L='9', W=''),
             dict(h='3.5', ncol='3', b='25', hs='50', nart='0', w='3.475', L='9', W='')],
    E='158000', fc='', tipo='Grupo B', Q='4', k1='1.0',
    a0='224', c='975', Ta='0.8', Tb='1.7', k='0.445', Ts='1.0', factor_Fu='1.1', mult_V='1')
EJEMPLOS = {'Ejemplo 1 (3 niveles, 1 crujía)': ame.EJEMPLO, 'Ejemplo 2 (3 niveles, 2 crujías)': EJEMPLO2}

st.set_page_config(page_title='Análisis Modal Espectral', page_icon='🏢', layout='wide')

st.markdown(f"""
<style>
.cabecera {{background:{NAVY};padding:20px 28px;border-radius:12px;margin-bottom:18px}}
.cabecera h1 {{color:white;margin:0;font-size:2rem}}
.cabecera p {{color:{AZUL_CLARO};margin:4px 0 0 0}}
.kpi {{background:white;border:1px solid {AZUL_CLARO};border-top:4px solid {AZUL};border-radius:8px;padding:12px 16px}}
.kpi .t {{color:{AZUL_MED};font-size:.75rem;font-weight:700;letter-spacing:.04em}}
.kpi .n {{color:{AZUL};font-size:1.9rem;font-weight:700;line-height:1.2}}
.kpi .s {{color:{AZUL_MED};font-size:.8rem}}
h3 {{color:{AZUL_OSC}}}
</style>
<div class="cabecera"><h1>Análisis Modal Espectral</h1>
<p>Marcos de cortante · fuerzas sísmicas por nivel, cortante basal y factor de escala</p></div>
""", unsafe_allow_html=True)


def df_de(niveles):
    return pd.DataFrame([{k: (float(r[k]) if str(r.get(k, '')).strip() else np.nan) for k in COLS}
                         for r in niveles])


def cargar(d):
    """Pone un proyecto (ejemplo o archivo) en el estado de la página."""
    for k in CAMPOS:
        st.session_state[k] = d.get(k, '')
    st.session_state['df'] = df_de(d['niveles'])
    st.session_state['ver'] = st.session_state.get('ver', 0) + 1


if 'df' not in st.session_state:
    cargar(ame.EJEMPLO)

# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.markdown('### Proyecto')
    st.selectbox('Ejemplo', list(EJEMPLOS), key='ejemplo')
    st.button('Cargar ejemplo', on_click=lambda: cargar(EJEMPLOS[st.session_state['ejemplo']]), width='stretch')
    arch = st.file_uploader('o abrir proyecto (.json)', type='json')
    if arch is not None and st.session_state.get('_arch') != arch.file_id:
        st.session_state['_arch'] = arch.file_id
        try:
            cargar(json.load(arch))
            st.rerun()
        except (ValueError, KeyError):
            st.error('No se pudo leer ese archivo de proyecto.')

    st.markdown('### Edificación y sismo')
    st.selectbox('Tipo', TIPOS, key='tipo')
    st.text_input('Q (varios valores separados por espacio)', key='Q',
                  help='Se usa el primero para las fuerzas. Ocupación inmediata / limitación de daños: Q = 1.')
    st.text_input('k1', key='k1')

    st.markdown('### Concreto y opciones')
    c1, c2 = st.columns(2)
    c1.text_input('E (kg/cm²)', key='E', help="Si está vacío se usa E = 14000·√f'c")
    c2.text_input("f'c (kg/cm²)", key='fc')
    c1.text_input('Factor Fu', key='factor_Fu')
    c2.text_input('V estático ×', key='mult_V')

    st.markdown('### Espectro del sitio (SASID)')
    c1, c2 = st.columns(2)
    for i, (et, k) in enumerate([('a0 (cm/s²)', 'a0'), ('c (cm/s²)', 'c'), ('Ta (s)', 'Ta'),
                                 ('Tb (s)', 'Tb'), ('k', 'k'), ('Ts (s)', 'Ts')]):
        (c1 if i % 2 == 0 else c2).text_input(et, key=k)

# ------------------------------------------------------------------ estructura
st.markdown('### Estructura (niveles de abajo hacia arriba)')
edit = st.data_editor(
    st.session_state['df'], key=f'niv_{st.session_state["ver"]}', num_rows='dynamic', width='stretch',
    hide_index=False,
    column_config={k: st.column_config.NumberColumn(ETIQ[k], format='%g') for k in COLS})
st.caption('Una fila por nivel (la primera es la base; agrega filas con el + de la tabla). '
           'Nº columnas = crujías + 1 y L = suma de las crujías. “h sección” es la dimensión en la dirección del sismo. '
           'Las articuladas en la base usan 3EI/h³ en vez de 12EI/h³. Si escribes W (t) se usa en lugar de w × L.')

datos = {k: st.session_state[k] for k in CAMPOS}
datos['niveles'] = [{k: ('' if pd.isna(v) else str(v)) for k, v in fila.items()}
                    for _, fila in edit.dropna(how='all').iterrows()]

try:
    res = ame.calcular(ame.preparar_estructura(datos))
except ValueError as e:
    st.error(f'Revisa los datos: {e}')
    st.stop()
except np.linalg.LinAlgError:
    st.error('No se pudo calcular: las matrices son singulares. Revisa las secciones y las alturas.')
    st.stop()

# ------------------------------------------------------------------ resultados
p, n = res['p'], res['p']['n']
fe = res['factor_escala']


def kpi(col, titulo, num, sub):
    col.markdown(f'<div class="kpi"><div class="t">{titulo}</div><div class="n">{num}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


st.markdown('### Resultados')
k = st.columns(4)
kpi(k[0], 'PERIODO FUNDAMENTAL', f'{res["T"][0]:.4f} s', f'{n} niveles · γ₁ = {res["gam"][0]:.3f}')
kpi(k[1], 'CORTANTE BASAL DINÁMICO', f'{res["V_din"]:.3f} t', f'Σ Mvu = {res["tabla_din"]["Mvu"].sum():.2f} t·m')
kpi(k[2], 'CORTANTE BASAL ESTÁTICO', f'{res["V_est"]:.3f} t', f'Sa/(Q′R′) máx = {res["Fmax_est"]:.4f} g')
kpi(k[3], 'FACTOR DE ESCALA', f'{fe:.4f}', 'Escalar fuerzas dinámicas' if fe > 1 else 'El dinámico ya cumple (≤ 1)')
st.write('')

buf = io.BytesIO()
ame.exportar_excel(res, buf)
d1, d2, _ = st.columns([1, 1, 3])
d1.download_button('Descargar Excel', buf.getvalue(), 'resultados_AME.xlsx', type='primary',
                   mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                   width='stretch')
d2.download_button('Guardar proyecto', json.dumps(datos, indent=2, ensure_ascii=False), 'proyecto_AME.json',
                   mime='application/json', width='stretch')


def mostrar(df, dec=4):
    st.dataframe(df, hide_index=True, width='stretch',
                 column_config={c: st.column_config.NumberColumn(format=f'%.{dec}f') for c in df.columns[1:]})


tabs = st.tabs(['Fuerzas por nivel', 'Modos', 'Detalle completo', 'Formas modales', 'Espectro'])
with tabs[0]:
    t = res['tabla_din']
    st.markdown('**Análisis dinámico (modal espectral)**')
    mostrar(pd.DataFrame({'Nivel': t['nivel'], 'h (m)': t['h'], 'F (t)': t['F'], 'Fu (t)': t['Fu'],
                          'Vu (t)': t['Vu'], 'Mvu (t·m)': t['Mvu']}))
    e = res['tabla_est']
    st.markdown('**Análisis estático (para el factor de escala)**')
    mostrar(pd.DataFrame({'Nivel': e['nivel'], 'h (m)': e['h'], 'Wi (t)': e['Wi'], 'Wihi (t·m)': e['hWi'],
                          'Fi (t)': e['Fi'], 'Fu (t)': e['Fu'], 'Vu (t)': e['Vu'], 'Mvu (t·m)': e['Mvu']}))
with tabs[1]:
    qr = {f[0]: f[-1] for f in res['QR'][::len(p['Q'])]}
    mostrar(pd.DataFrame({'Modo': np.arange(1, n + 1), 'T (s)': res['T'], 'f (Hz)': res['f'], 'ω (rad/s)': res['w'],
                          'γ': res['gam'], "Sa/(Q'R') (cm/s²)": res['Sa_modal'][:, 0],
                          "Q·R'": [qr[j + 1] for j in range(n)]}))
with tabs[2]:
    st.code(ame.reporte_texto(res), language=None)
with tabs[3]:
    st.pyplot(ame.fig_modal(res))
with tabs[4]:
    st.pyplot(ame.fig_espectro(res))
