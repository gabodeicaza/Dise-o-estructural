"""Examen 3 · mampostería, viento, viga por puntos y herramientas de una sección de concreto reforzado (motor en ame_parcial3.py)."""
import numpy as np
import pandas as pd
import streamlit as st

import ame_ntc as ntc
import ame_parcial3 as p3


def _numeros(df, cols):
    """Las tablas se capturan como texto (así se pueden escribir decimales como 3.5 o 3,5) y aquí se pasan a número.
    Devuelve el DataFrame numérico o None si alguna celda no es un número."""
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c].astype(str).str.strip().str.replace(',', '.', regex=False), errors='coerce')
    return out if not out[cols].isna().any().any() else None


def _pasos(lista):
    for t in lista:
        st.markdown(f'- {t}')


def _veredicto(ok, texto_ok='PASA', texto_no='NO PASA'):
    (st.success if ok else st.error)(texto_ok if ok else texto_no)


def _nums(txt, nombre):
    try:
        v = [float(x) for x in txt.replace(',', ' ').split()]
    except ValueError:
        raise ValueError(f'«{nombre}» debe ser una lista de números separados por espacios.')
    if not v:
        raise ValueError(f'Escribe «{nombre}».')
    return v


def _tabla(df, dec=3):
    st.dataframe(df, hide_index=True, width='stretch',
                 column_config={c: st.column_config.NumberColumn(format=f'%.{dec}f') for c in df.columns
                                if pd.api.types.is_float_dtype(df[c])})


# ------------------------------------------------------------------ mampostería
def _mamposteria():
    s_comp, s_cort, s_est, s_dist = st.tabs(['Carga vertical (PR)', 'Cortante por muro (VmR)', 'Fuerzas sísmicas (estático)',
                                             'Q, γ y otros límites'])
    with s_comp:
        c1, c2, c3 = st.columns(3)
        t = c1.number_input('t espesor (cm)', 1.0, 100.0, 14.0, key='mc_t')
        H = c2.number_input('H altura (cm)', 1.0, 2000.0, 350.0, key='mc_H')
        L = c3.number_input('L longitud (cm)', 1.0, 5000.0, 400.0, key='mc_L')
        c1, c2, c3 = st.columns(3)
        Pu = c1.number_input('Pu (kg) = 1.3 CM + 1.5 CVmáx', 0.0, 1e7, 8163.0, key='mc_Pu')
        fm = c2.number_input("f′m (kg/cm²)", 1.0, 200.0, 15.0, key='mc_fm')
        tipo = c3.selectbox('Tipo de muro', p3.TIPOS_MURO, index=1, key='mc_tipo')
        c1, c2, c3, c4 = st.columns(4)
        L2L1 = c1.number_input('L₂/L₁ de las losas', 0.0, 10.0, 1.0, key='mc_l2')
        cvcm = c2.number_input('CV/CM', 0.0, 10.0, 0.8, key='mc_cv')
        mov = c3.checkbox('Movimiento perpendicular restringido', True, key='mc_mov')
        vol = c3.checkbox('Sin volados ni cargas perpendiculares', True, key='mc_vol')
        b = c4.number_input('b apoyo de la losa (cm, 0 = t)', 0.0, 100.0, 0.0, key='mc_b')
        r = p3.muro_compresion(t, H, L, Pu, fm, tipo, L2L1, cvcm, mov, vol, b or None)
        cols = st.columns(3)
        cols[0].metric('FE', f'{r["FE"]:.4f}')
        cols[1].metric('PR (kg)', f'{r["PR"]:,.0f}')
        cols[2].metric('Pu (kg)', f'{Pu:,.0f}')
        _veredicto(r['pasa'])
        with st.expander('Procedimiento', expanded=True):
            _pasos(r['pasos'])
    with s_cort:
        c1, c2, c3 = st.columns(3)
        t = c1.number_input('t (cm)', 1.0, 100.0, 14.0, key='mv_t')
        L = c2.number_input('L (cm)', 1.0, 5000.0, 400.0, key='mv_L')
        H = c3.number_input('H (cm)', 1.0, 2000.0, 350.0, key='mv_H')
        c1, c2, c3 = st.columns(3)
        Pn = c1.number_input('Pn (kg) = CM + CVacc sin factor', 0.0, 1e7, 7520.0, key='mv_Pn')
        vm = c2.number_input("v′m (kg/cm²)", 0.1, 20.0, 2.0, key='mv_vm')
        Vu = c3.number_input('Vu (kg)', 0.0, 1e7, 5270.0, key='mv_Vu')
        r = p3.muro_cortante(t, L, H, Pn, vm, Vu=Vu)
        c1, c2, c3 = st.columns(3)
        c1.metric('f', f'{r["f"]:.4f}')
        c2.metric('VmR (kg)', f'{r["VmR"]:,.1f}')
        c3.metric('Vu (kg)', f'{Vu:,.1f}')
        _veredicto(r['pasa'])
        with st.expander('Procedimiento', expanded=True):
            _pasos(r['pasos'])
        st.markdown('**Entrepiso completo (varios muros en una dirección)**')
        conf_m = {c: st.column_config.TextColumn(c) for c in ['t', 'L', 'H', 'Pn', 'Vu']}
        df = st.data_editor(pd.DataFrame({'muro': ['A', 'C', 'D', 'E'], 't': ['14'] * 4, 'L': ['150', '500', '400', '200'],
                                          'H': ['250'] * 4, 'Pn': ['12142.9', '20250', '16200', '8100'],
                                          'Vu': ['6020', '9960', '8340', '3250']}), num_rows='dynamic', key='mv_tabla',
                            hide_index=True, width='stretch', column_config=conf_m)
        df = _numeros(df.dropna(), ['t', 'L', 'H', 'Pn', 'Vu'])
        if df is None:
            st.error('Hay celdas vacías o que no son números. Los decimales se escriben con punto o coma (por ejemplo 3.5).')
            df = pd.DataFrame(columns=['muro', 't', 'L', 'H', 'Pn', 'Vu'])
        muros = [dict(nombre=str(f['muro']), t=f['t'], L=f['L'], H=f['H'], Pn=f['Pn'], Vu=f['Vu'])
                 for _, f in df.dropna().iterrows() if f['L'] > 0 and f['t'] > 0]
        if muros:
            rm = p3.revision_muros(muros)
            filas = pd.DataFrame(rm['filas']).rename(columns={'H_L': 'H/L', 'pasa': 'Pasa'})
            _tabla(filas, 4)
            st.caption(f'ΣVmR = {rm["sum_VmR"]:,.0f} kg  ·  ΣVu = {rm["sum_Vu"]:,.0f} kg')
            _veredicto(rm['entrepiso_pasa'] and rm['todos_pasan'], 'Pasan cada muro y el entrepiso',
                       'Algún muro o el entrepiso no pasa')
            for nombre, frac in rm['concentracion']:
                st.warning(f'Muro {nombre}: aporta {frac:.0%} de la resistencia del entrepiso (> 35 %). La NTC-S §5.7.1 '
                           'pide ×1.25 a sus fuerzas sísmicas; confírmalo con tu profesor.')
    with s_est:
        c1, c2, c3 = st.columns(3)
        Q = c1.number_input('Q', 1.0, 4.0, 2.0, 0.5, key='me_Q')
        estado = c2.selectbox('Estado', ['SV (R′ = R)', 'OI (Q = 1, R′ = 0.75R)'], key='me_est')
        FC = c3.number_input('FC', 0.5, 2.0, 1.1, key='me_fc')
        c1, c2, c3, c4 = st.columns(4)
        c_ = c1.number_input('c (g)', 0.01, 3.0, 0.6483, format='%.4f', key='me_c')
        k_ = c2.number_input('k', 0.05, 1.0, 0.505, format='%.3f', key='me_k')
        R0 = c3.number_input('R0', 1.0, 3.0, 2.0, key='me_R0')
        k1 = c4.number_input('k1', 0.5, 1.5, 1.0, key='me_k1')
        niv = st.data_editor(pd.DataFrame({'Nivel': ['2 (azotea)', '1'], 'W (t)': ['70', '80'], 'h (m)': ['5', '2.5']}),
                             num_rows='dynamic', key='me_tabla', hide_index=True, width='stretch',
                             column_config={c: st.column_config.TextColumn(c) for c in ['W (t)', 'h (m)']})
        niv = _numeros(niv.dropna(), ['W (t)', 'h (m)'])
        if niv is None:
            st.error('Hay celdas vacías o que no son números. Los decimales se escriben con punto o coma (por ejemplo 2.5).')
            niv = pd.DataFrame({'Nivel': [], 'W (t)': [], 'h (m)': []})
        if len(niv) >= 1 and (niv['W (t)'] > 0).all() and (niv['h (m)'] > 0).all():
            e = p3.estatico(niv['W (t)'], niv['h (m)'], c_, k_, Q, R0, k1, 'OI' if estado.startswith('OI') else 'SV', FC)
            tabla = pd.DataFrame({'Nivel': niv['Nivel'].values, 'W': niv['W (t)'].values, 'h': niv['h (m)'].values,
                                  'W·h': e['Wh'], 'F': e['F'], 'Fu': e['Fu'], 'Vu': e['Vu'], 'Fu·h': e['Mvu']})
            _tabla(tabla, 3)
            c1, c2, c3 = st.columns(3)
            c1.metric('c/(Q′R′)', f'{e["C"]:.5f}')
            c2.metric('VBu (t)', f'{e["VBu"]:.3f}')
            c3.metric('Mvu (t·m)', f'{e["MVU"]:.2f}')
            st.caption('Escribe los niveles de arriba hacia abajo. No aplica a Grupo A ni a estructuras fuertemente irregulares '
                       '(altura máxima 30 m regular / 20 m irregular; zona A 40 / 30 m).')
            with st.expander('Procedimiento'):
                _pasos(e['pasos'])
    with s_dist:
        est = st.selectbox('Estructuración (tabla 4.3.3)', list(p3.TABLA_433), index=1, key='md_est')
        c1, c2, c3, c4 = st.columns(4)
        n = c1.number_input('Niveles sobre banqueta', 1, 30, 3, key='md_n')
        irr = c2.selectbox('Condiciones de irregularidad', [0, 1, 2, 3], key='md_irr')
        ft = c3.checkbox('Fuertemente irregular por torsión', key='md_ft')
        fe = c4.checkbox('Fuertemente irregular en elevación', key='md_fe')
        r = p3.distorsion_limite(est, int(n), int(irr), ft, fe)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Q', f'{r["Q"]:g}')
        c2.metric('γSV tabla', f'{r["gamma_SV"]:.4f}')
        c3.metric('γc', f'{r["gamma_c"]:g}')
        c4.metric('γSV corregida', f'{r["gamma_SV_corregida"]:.5f}')
        if r['gamma_OI'] is not None:
            st.caption(f'Ocupación inmediata: γOI = {r["gamma_OI"]:g} → corregida {r["gamma_OI_corregida"]:.5f}.')
        st.caption('Revisión SV: γ(del análisis sin FC) · 1.1 · Q · R′ ≤ γSV corregida. Limitación de daño: elástico × Ks, '
                   'sin FC, γ ≤ 0.002 (0.004 si los no estructurales están desligados).')
        st.markdown('**Ks y a_min**')
        c1, c2 = st.columns(2)
        Ts = c1.number_input('Ts (s)', 0.0, 5.0, 0.8, key='md_ts')
        Rp = c2.number_input("R′", 0.5, 5.0, 2.0, key='md_rp')
        c1, c2, c3 = st.columns(3)
        c1.metric('Ks', f'{ntc.ks_norma(Ts):.4f}')
        c2.metric('Zona', ntc.zona_ks(Ts))
        c3.metric('a_min', f'{p3.a_min(Ts, Rp):.4f}')
        st.markdown('**Excentricidad accidental** (ec. 2.3.2)')
        c1, c2 = st.columns(2)
        nn = c1.number_input('Número de niveles', 1, 40, 4, key='md_nn')
        bb = c2.number_input('b perpendicular al análisis (m)', 0.1, 500.0, 12.0, key='md_bb')
        ea = p3.excentricidades_accidentales(int(nn), bb)
        _tabla(pd.DataFrame({'Entrepiso': range(1, int(nn) + 1), 'ea (m)': ea}), 3)


# ------------------------------------------------------------------ viento
def _viento():
    c1, c2, c3 = st.columns(3)
    zona = c1.selectbox('Zona y velocidad regional VR (m/s)', list(p3.VR_ZONAS) + ['Otra (la capturo)'], key='vi_zona')
    if zona.startswith('Otra'):
        VR = c2.number_input('VR (m/s)', 1.0, 100.0, 30.08, key='vi_vr')
    else:
        grupo = c2.radio('Grupo', ['A (200 años)', 'B (50 años)', 'Temporal (10 años)'], index=1, key='vi_grupo')
        VR = float(p3.VR_ZONAS[zona][['A', 'B', 'T'].index(grupo[0])])
        c3.metric('VR aproximada', f'{VR:g} m/s')
        st.caption('Valores aproximados del apunte p. 27; el reglamento da VR exacta por colonia (Apéndice A). '
                   'Si el problema da VR, usa «Otra».')
    c1, c2, c3 = st.columns(3)
    terreno = c1.selectbox('Rugosidad del terreno', list(p3.TERRENOS), index=2, key='vi_ter')
    a0, d0, c0 = p3.TERRENOS[terreno]
    topo = c2.selectbox('Topografía', list(p3.TOPOGRAFIA), index=2, key='vi_topo')
    FT = p3.TOPOGRAFIA[topo]
    c3.metric('FT', f'{FT:g}')
    c1, c2, c3 = st.columns(3)
    alfa = c1.number_input('α', 0.01, 0.5, a0, format='%.3f', key=f'vi_a_{terreno}')
    delta = c2.number_input('δ (m)', 100.0, 600.0, d0, key=f'vi_d_{terreno}')
    cc = c3.number_input('c', 0.1, 2.0, c0, format='%.3f', key=f'vi_c_{terreno}')
    if terreno.startswith('R1'):
        st.caption('El α de R1 está ilegible en los apuntes (0.06?). Verifícalo en la NTC-Viento.')
    c1, c2, c3, c4 = st.columns(4)
    zt = c1.text_input('Alturas z de los niveles (m)', '3 6 9 12 15', key='vi_z')
    ancho = c2.number_input('Ancho de la cara expuesta (m)', 0.1, 500.0, 25.0, key='vi_ancho')
    largo = c3.number_input('Largo en el sentido del viento (m)', 0.1, 500.0, 35.0, key='vi_largo')
    FC = c4.number_input('FC', 0.5, 2.0, 1.1, key='vi_fc')
    try:
        z = _nums(zt, 'alturas z')
    except ValueError as e:
        st.error(str(e))
        return
    H = max(z)
    c1, c2 = st.columns(2)
    c1.metric('Tipo de estructura', p3.tipo_estructura(H, min(ancho, largo)).split(' · ')[0])
    c2.caption(f'H/A = {H / min(ancho, largo):.2f}: ' + p3.tipo_estructura(H, min(ancho, largo)).split(': ')[-1])
    caras = {'Barlovento': (0.8, False), 'Sotavento': (-0.4, True), 'Paredes laterales': (-0.8, True)}
    tabs = st.tabs(list(caras) + ['Techo plano', 'Servicio'])
    for tab, (nombre, (cp, cte)) in zip(tabs, caras.items()):
        with tab:
            cp = st.number_input(f'Cpe ({nombre})', -3.0, 3.0, cp, key=f'vi_cp_{nombre}')
            ancho_cara = largo if nombre == 'Paredes laterales' else ancho
            v = p3.viento_cara(z, ancho_cara, VR, alfa, delta, cc, FT, cp, cte, FC)
            _tabla(pd.DataFrame({'Nivel': range(1, len(z) + 1), 'z (m)': v['z'], 'Frz': v['Frz'], 'VD (m/s)': v['VD'],
                                 'pz (kg/m²)': v['pz'], 'AT (m²)': v['AT'], 'F (t)': v['F'], 'Fu (t)': v['Fu'],
                                 'Vu (t)': v['Vu'], 'Fu·z (t·m)': v['Mvu']}), 3)
            c1, c2 = st.columns(2)
            c1.metric('VBu (t)', f'{v["VBu"]:.2f}')
            c2.metric('Mvu (t·m)', f'{v["MVU"]:.2f}')
            if nombre == 'Sotavento':
                st.caption('La succión de sotavento jala en el mismo sentido del viento: para el cortante total del edificio '
                           'suma barlovento + sotavento (confírmalo con tu profesor).')
    with tabs[3]:
        cp = st.number_input('Cpe del techo', -3.0, 3.0, -0.8, key='vi_cp_techo')
        t = p3.techo_plano(largo, ancho, VR, alfa, delta, cc, H, FT, cp, FC)
        c1, c2, c3 = st.columns(3)
        c1.metric('pz (kg/m²)', f'{t["pz"]:.2f}')
        c2.metric('F (t)', f'{t["F"] / 1000:.2f}')
        c3.metric('Fu (t)', f'{t["Fu"] / 1000:.2f}')
        st.caption('Es succión (hacia arriba): se revisa contra CM + CVmín con FC = 0.9 (estabilidad).')
    with tabs[4]:
        st.markdown(f'- Desplazamiento máximo: Δmáx ≤ H/500 = {H * 100 / 500:.2f} cm\n'
                    '- Distorsión: γ ≤ 0.005 (sin elementos no estructurales) o γ ≤ 0.002 (hay elementos susceptibles a daño)')


# ------------------------------------------------------------------ concreto
def _desarrollo(f, titulo=None):
    """Muestra el cálculo de una ficha con las fórmulas y los números (LaTeX), paso a paso."""
    if titulo:
        st.markdown(f'##### {titulo}')
    for tit, tx in f['tex']:
        st.markdown(f'**{tit}**')
        st.latex(tx)


def _viga_completa():
    """Procedimiento de clase completo: datos y revisiones, As de todos los puntos del diagrama, armado con la varilla elegida,
    corrido (50 % de la mayor As), render de las secciones y estribos."""
    import ame_porticos as pt
    st.caption('Captura la sección y los Mu del diagrama de momentos (uno por punto: caras de apoyo y centros). La app hace todo el '
               'procedimiento del pizarrón: revisiones, As con la cuadrática, barras, MR, separación, corrido y estribos.')
    c1, c2, c3, c4 = st.columns(4)
    b = c1.number_input('b (cm)', 10.0, 300.0, 35.0, key='vc_b')
    h = c2.number_input('h (cm)', 20.0, 300.0, 70.0, key='vc_h')
    fc = c3.number_input("f′c (kg/cm²)", 100.0, 1000.0, 250.0, key='vc_fc')
    fy = c4.number_input('fy (kg/cm²)', 2000.0, 6000.0, 4200.0, key='vc_fy')
    sug, txt_sug = pt.varilla_sugerida(h)
    c1, c2, c3, c4 = st.columns(4)
    var = c1.selectbox('Varilla longitudinal', list(pt.DIAMETROS), index=list(pt.DIAMETROS).index(sug), key=f'vc_var_{sug}',
                       help=txt_sug)
    est = c2.selectbox('Estribo', ['#3', '#4'], index=1, key='vc_est')
    ramas = c3.selectbox('Ramas del estribo', [2, 4], index=1 if b >= 40 else 0, key=f'vc_ramas_{b >= 40}')
    rec = c4.number_input('Recubrimiento libre (cm)', 1.5, 8.0, 3.0, 0.5, key='vc_rec')
    c1, c2, c3 = st.columns(3)
    r_fija = c1.checkbox('Usar r fija al centroide (como en clase)', True, key='vc_rfija',
                         help='Activado: d = h − r con la r que escribas (clase: r = 5 cm, d = 65). Desactivado: la d se calcula con la '
                              'posición real de las barras.')
    r_val = c2.number_input('r (cm)', 2.0, 20.0, 5.0, 0.5, key='vc_r', disabled=not r_fija)
    agrupar = c3.checkbox('Permitir barras agrupadas de 2 en 2', True, key='vc_agr')
    st.caption(txt_sug)
    modo = st.radio('Armado', ['Automático: la app propone el número de barras', 'Propongo yo el armado (número y varilla de cada punto)'],
                    key='vc_modo', horizontal=True,
                    help='En el modo manual escribe en la columna «Armado propuesto» algo como 4#8; la app calcula As real = n·Ab, el MR '
                         'y la separación, y revisa contra el As requerido. Si dejas una celda vacía, ese punto se propone solo.')
    manual = modo.startswith('Propongo')
    base = pd.DataFrame({'Punto': list('abcdefghijk'),
                         'Mu (t·m)': ['41.6', '15.26', '49.91', '18.8', '48.22', '13.55', '40.1', '13.4', '59.9', '26.7', '58.6'],
                         'Tensión': ['superior', 'inferior', 'superior', 'inferior', 'superior', 'inferior', 'superior', 'inferior',
                                     'superior', 'inferior', 'superior'],
                         'Armado propuesto': [''] * 11})
    ed = st.data_editor(base, num_rows='dynamic', hide_index=True, width='stretch', key='vc_tabla',
                        column_config={'Mu (t·m)': st.column_config.TextColumn('Mu (t·m)'),
                                       'Armado propuesto': st.column_config.TextColumn('Armado propuesto', help='Ejemplo: 4#8. Solo se usa en el modo manual.'),
                                       'Punto': st.column_config.TextColumn('Punto'),
                                       'Tensión': st.column_config.SelectboxColumn('Lecho en tensión', options=['superior', 'inferior'],
                                                                                  required=True)})
    ed = ed.dropna(subset=['Punto', 'Mu (t·m)', 'Tensión'])
    ed['Armado propuesto'] = ed['Armado propuesto'].fillna('')
    ed = _numeros(ed, ['Mu (t·m)']) if len(ed) else None
    if ed is None or not len(ed):
        st.info('Captura al menos un punto con Mu numérico (decimales con punto o coma).')
        return
    c1, c2, c3 = st.columns(3)
    Vu = c1.number_input('Vu máximo (t)', 0.0, 1000.0, 44.3, key='vc_Vu')
    s_tope = c2.number_input('Tope de la separación (cm)', 10.0, 40.0, 25.0, 1.0, key='vc_stope',
                             help='El profesor prefiere 25 cm cuando rige el tope (mayor seguridad en obra). Los apuntes marcan 30 cm: cámbialo si lo necesitas.')
    fyest = c3.selectbox('fy del estribo', [4200.0, 2530.0], key='vc_fyest')
    est_db, a_est = pt.DIAMETROS[est], p3.VARILLAS[est]
    rf = r_val if r_fija else None
    try:
        props = [pt.parse_propuesta(t) if manual else None for t in ed['Armado propuesto']]
    except ValueError as e:
        st.error(str(e))
        return
    fichas = [pt.ficha_punto(float(m), b, h, fc, fy, var, rec, est_db, rf, agrupar, lecho=lc, propuesta=pr)
              for m, lc, pr in zip(ed['Mu (t·m)'], ed['Tensión'], props)]
    viga = pt.ficha_viga(b, h, fc, fy, var, est, ramas, rec, rf, Mu_max=float(ed['Mu (t·m)'].max()), Vu_max=Vu)
    (st.success if viga['ok_flexion'] and viga['ok_cortante'] else st.error)(
        'La sección cumple: puede ser subreforzada en todos los puntos y el cortante no excede el límite.'
        if viga['ok_flexion'] and viga['ok_cortante'] else 'La sección NO cumple una revisión global: cámbiala antes de armar.')
    st.markdown('#### 0 · Datos y revisiones de la viga')
    _desarrollo(viga)
    st.markdown('#### 1 · Flexión en cada punto')
    filas = [[p_, lc, f['Mu'], f['As_req'] if f['As_req'] is not None else np.nan, f['As_usar'] if f['As_usar'] is not None else np.nan,
              f['armado'], f['As_real'] if f['As_real'] is not None else np.nan, f['MR'] if f['MR'] is not None else np.nan,
              f['sep_libre'] if f['sep_libre'] is not None else np.nan, '✔' if f['ok'] else '✘']
             for p_, lc, f in zip(ed['Punto'], ed['Tensión'], fichas)]
    _tabla(pd.DataFrame(filas, columns=['Punto', 'Lecho', 'Mu (t·m)', 'As req (cm²)', 'As usar (cm²)', 'Armado', 'As real (cm²)',
                                        'MR (t·m)', 'Sep. libre (cm)', 'OK']), 2)
    st.markdown('##### Cálculo desarrollado de cada punto')
    for p_, f in zip(ed['Punto'], fichas):
        with st.container(border=True):
            _desarrollo(f, f'{p_})  ' + ('✔' if f['ok'] else '✘ revisa'))
            for t in f['lineas'][-3:]:
                st.caption(t)
    st.markdown('#### 2 · Varillas corridas (a todo lo largo, arriba y abajo, con ganchos al inicio y al final)')
    reales = [f['As_real'] for f in fichas if f['As_real']]
    if reales:
        As_max_real = max(reales)
        ab = p3.VARILLAS[var]
        lim = viga['lim']
        As_corr = max(lim['As_min'], 0.5 * As_max_real)
        n_corr = max(2, int(np.ceil(As_corr / ab - 1e-9)))
        st.markdown(f'- Mayor As real = {As_max_real:.2f} cm²; 50 % = {0.5 * As_max_real:.2f} cm²; As,mín = {lim["As_min"]:.2f} cm²')
        st.markdown(f'- Corrido: máx(As,mín; 50 % As mayor) = {As_corr:.2f} cm² → n = {As_corr:.2f}/{ab} = {As_corr / ab:.2f} → '
                    f'**{n_corr}{var}** ({n_corr * ab:.2f} cm²) arriba y abajo')
        st.markdown('- Bastones: en cada punto se agregan las barras que falten hasta el As real de la tabla; cada bastón se ancla una '
                    'distancia d más allá de donde ya no se requiere en el diagrama de Mu. El alzado con las longitudes sale en «Examen 3» → '
                    '«Viga continua» (o en «Pórtico»).')
        faltan = [f'{p_}) +{f["n"] - n_corr}{var}' for p_, f in zip(ed['Punto'], fichas) if f['n'] and f['n'] > n_corr]
        if faltan:
            st.markdown('- Bastones por punto: ' + ' · '.join(faltan))
    st.markdown('#### 3 · Secciones')
    import matplotlib.pyplot as plt
    fig = pt.figura_puntos(fichas, b, h, rec, est_db, [f'{p_})' for p_ in ed['Punto']])
    st.pyplot(fig, width='stretch')
    plt.close(fig)
    st.caption('Rojo = no cumple la separación libre o no caben las barras. La numeración de pares (agrupadas) sigue la práctica de clase.')
    st.markdown('#### 4 · Estribos (cortante)')
    As_auto = max([f['As_real'] for f in fichas if f['As_real'] and f.get('lecho') == 'superior'] or [max(reales or [1.0])])
    Av = a_est * ramas
    d_c = viga['d']
    st.markdown('**Límites**')
    for tit, tx in p3.limites_cortante_tex(b, d_c, fc, Av, ramas, a_est):
        st.markdown(f'*{tit}*')
        st.latex(tx)
    st.caption('Captura el Vu de cada tramo y el acero en tensión (As) de la sección donde actúa; si dejas el As vacío se usa el mayor '
               f'As real superior calculado arriba ({As_auto:.2f} cm²).')
    vt = pd.DataFrame({'Tramo': ['1', '2', '3', '4', '5'], 'Vu (t)': ['36.2', '39.6', '35.36', '36.5', '44.3'],
                       'As en tensión (cm²)': ['20.28', '25.35', '25.35', '20.28', '30.42']})
    ev = st.data_editor(vt, num_rows='dynamic', hide_index=True, width='stretch', key='vc_tabla_v',
                        column_config={c: st.column_config.TextColumn(c) for c in vt.columns})
    ev = ev.dropna(subset=['Tramo', 'Vu (t)']).copy()
    ev['As en tensión (cm²)'] = ev['As en tensión (cm²)'].fillna('').astype(str).replace('', str(As_auto))
    ev = _numeros(ev, ['Vu (t)', 'As en tensión (cm²)']) if len(ev) else None
    if ev is None or not len(ev):
        st.info('Captura al menos un tramo con Vu numérico.')
        return
    resumen = []
    for i, (tr, vu, as_t) in enumerate(zip(ev['Tramo'], ev['Vu (t)'], ev['As en tensión (cm²)']), 1):
        c_ = p3.cortante_ficha(float(vu) * 1000, b, d_c, fc, fyest, Av, float(as_t), s_tope)
        with st.container(border=True):
            st.markdown(f'##### {i}. Tramo {tr}:  Vu = {vu:g} t')
            for tit, tx in c_['tex']:
                st.markdown(f'*{tit}*')
                st.latex(tx)
            if c_['ok']:
                st.success(f'E {est}, {ramas} ramas @ {c_["s_usar"]} cm  (rige {c_["rige"]} en smáx)')
                if c_['s_usar'] < 10:
                    alt = p3.alternativas_estribo(float(vu) * 1000, b, d_c, fc, fyest, float(as_t), s_tope)
                    st.warning(f'@ {c_["s_usar"]} cm es muy cerrado para colar. '
                               + ('Alternativas con s ≥ 10 cm: ' + ' · '.join(o['texto'] for o in alt[:4]) if alt else
                                  'Ningún estribo #3/#4 de 2–4 ramas da s ≥ 10 cm: agranda la sección.'))
            else:
                st.error('Vu excede el límite de la sección: agranda b o h.')
        resumen.append([tr, vu, as_t, c_['rho'], c_['VcR'], c_['Vs'], c_['s_calc'] if c_['s_calc'] is not None else np.nan, c_['s_max'],
                        f'E {est}, {ramas}r @ {c_["s_usar"]} cm' if c_['ok'] else 'NO'])
    st.markdown('**Resumen**')
    _tabla(pd.DataFrame(resumen, columns=['Tramo', 'Vu (t)', 'As tens. (cm²)', 'ρ', 'VcR (kg)', 'Vu−VcR (kg)', 's calc (cm)', 'smáx (cm)',
                                          'Estribos']), 4)
    st.markdown('#### 5 · Zona de confinamiento (integridad estructural)')
    integ = p3.integridad_estructural(h, d_c, pt.DIAMETROS[var], est_db, s_tope)
    for tit, tx in integ['tex']:
        st.latex(tx)
    st.success(f'En 2h = {integ["Lconf"]:g} cm desde la cara de cada apoyo: E {est}, {ramas} ramas @ mín(s por cortante; {integ["s_usar"]}) cm')
    st.markdown('#### 6 · Acero de piel (temperatura)')
    piel = p3.acero_piel(b, h, r_val if r_fija else rec + est_db + pt.DIAMETROS[var] / 2, est if est in ('#3', '#4') else '#3')
    for tit, tx in piel['tex']:
        st.latex(tx)
    for t in piel['lineas']:
        st.markdown(f'- {t}')


EJEMPLO_MOMENTOS = '41.6\n15.26\n49.4\n18.8\n48.22\n13.55\n40.1\n15.4\n59.9\n26.7\n58.6'
EJEMPLO_LARGOS = '6.7\n6.8\n6.5\n7.2\n7.8'
EJEMPLO_VUS = '36.2\n39.6\n35.36\n36.5\n44.3'


@st.cache_data(show_spinner=False, max_entries=8)
def _exportar_viga(datos, momentos, largos, vus, proyecto, autor):
    """PDF, Excel y texto de la viga (se calculan una vez por cada combinación de datos)."""
    import ame_seccion as sx
    res = sx.calcular_viga(dict(datos), sx.lista_desde_columnas(list(momentos), list(largos), list(vus)))
    return sx.pdf(res, proyecto, autor), sx.excel(res), sx.texto(res).encode('utf-8')


_NL = chr(10)
EJEMPLO_ETABS = dict(
    sx_mom=_NL.join('153.70 10 154.38 10 154.38 10 144.35 10 144.78 10 164.33 10 164.33'.split()),
    sx_largos=_NL.join('7.2 5.2 7.2 7.2 7.2 5.2'.split()),
    sx_vus=_NL.join('52.84 66.05 51.01 51.07 51.14 68.55'.split()),
    sx_b=50.0, sx_h=90.0, sx_r=5.0)


def _cargar_etabs():
    st.session_state.update(EJEMPLO_ETABS)


def mostrar_seccion():
    """Examen 3 · viga completa: capturas la columna de momentos, el largo de cada crujía y el Vu de cada crujía, y la página arma
    el render de la viga (diagramas, corridas, bastones, estribos y secciones) con exportación a PDF, Excel y texto."""
    import matplotlib.pyplot as plt

    import ame_porticos as pt
    import ame_seccion as sx
    st.caption('Escribe los datos de tus diagramas y la página arma el render: los momentos en una columna (de izquierda a derecha), '
               'el largo de cada crujía y el Vu de cada crujía. Al abrir trae el ejemplo del pizarrón (con largos supuestos): '
               'cámbialo por los tuyos.')
    with st.expander('Datos de la viga', expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        b = c1.number_input('b (cm)', 5.0, 300.0, 35.0, key='sx_b')
        h = c2.number_input('h (cm)', 10.0, 300.0, 70.0, key='sx_h')
        r = c3.number_input('r (cm)', 1.0, 20.0, 5.0, key='sx_r')
        fc = c4.number_input("f′c (kg/cm²)", 100.0, 1000.0, 250.0, key='sx_fc')
        fy = c5.number_input('fy (kg/cm²)', 2000.0, 6000.0, 4200.0, key='sx_fy')
        sug, txt_sug = pt.varilla_sugerida(h)
        c1, c2, c3, c4, c5 = st.columns(5)
        var = c1.selectbox('Varilla longitudinal', list(pt.DIAMETROS), index=list(pt.DIAMETROS).index(sug), key=f'sx_var_{sug}', help=txt_sug)
        est = c2.selectbox('Estribo', ['#3', '#4'], index=1, key='sx_est')
        ramas = c3.selectbox('Ramas del estribo', [2, 3, 4], index=2 if b >= 40 else 0, key=f'sx_ramas_{b >= 40}')
        fyest = c4.selectbox('fy del estribo', [4200.0, 2530.0], key='sx_fyest')
        s_tope = c5.number_input('Tope de la separación (cm)', 10.0, 40.0, 25.0, 1.0, key='sx_stope',
                                 help='El profesor prefiere 25 cm cuando rige el tope; los apuntes marcan 30 cm.')

    st.button('Cargar el ejercicio de ETABS (6 vigas de la envolvente)', on_click=_cargar_etabs, key='sx_cargar_etabs',
              help='Mu = |mínimo| de M3 de cada tramo, Vu = máximo |V2|, claro libre = longitud − 0.8 m; sección 50×90 supuesta.')
    c1, c2, c3 = st.columns([1.2, 1, 1])
    t_mom = c1.text_area('Momentos Mu (t·m), en positivo', EJEMPLO_MOMENTOS, height=290, key='sx_mom',
                         help='Una columna, de izquierda a derecha: apoyo, centro del claro, apoyo, centro, …, apoyo. '
                              'Con n crujías van 2n + 1 valores. Puedes pegar la columna de Excel.')
    t_len = c2.text_area('Largo de cada crujía (m)', EJEMPLO_LARGOS, height=290, key='sx_largos',
                         help='Claro libre, de cara a cara de apoyo, uno por crujía.')
    t_vu = c3.text_area('Vu de cada crujía (t)', EJEMPLO_VUS, height=290, key='sx_vus',
                        help='Un valor por crujía. Con un solo valor se usa en todas.')
    try:
        momentos = sx.parse_columna(t_mom, 'Momentos')
        largos = sx.parse_columna(t_len, 'Largo de cada crujía')
        vus = sx.parse_columna(t_vu, 'Vu de cada crujía')
        n = len(largos)
        c1.caption(f'{len(momentos)} momentos' + (f' · {n} crujías → {2 * n + 1} momentos' if n else ''))
        lista = sx.lista_desde_columnas(momentos, largos, vus)
    except ValueError as e:
        st.warning(str(e))
        return

    datos = dict(b=b, h=h, r=r, fc=fc, fy=fy, var=var, est=est, ramas=int(ramas), fy_est=fyest, s_tope=s_tope)
    try:
        res = sx.calcular_viga(datos, lista)
    except ValueError as e:
        st.warning(f'{e}')
        return
    if res['error']:
        st.error(res['error'])
        return
    malos = [t for t, ok in (('flexión', res['ok_flex']), ('cortante', res['ok_cort'])) if not ok]
    _veredicto(res['ok'], 'La viga cumple: subreforzada en todos los puntos, el cortante no excede el límite y el armado alcanza.',
               'La viga NO cumple: ' + ', '.join(malos) + '. Abre los pasos de abajo para ver dónde.')

    fig = sx.figura_hoja(res)
    st.pyplot(fig)
    plt.close(fig)
    st.caption('Las longitudes de los bastones van del punto donde el momento de ese signo se anula hasta una distancia d más allá, '
               'medidas desde el eje de cada apoyo (cara de apoyo, sin ancho de columna). Los claros usan la forma parabólica de una '
               'carga uniforme que pasa por los tres momentos que diste.')

    pr, xl, tx = _exportar_viga(tuple(sorted(datos.items())), tuple(momentos), tuple(largos), tuple(vus),
                                st.session_state.get('proyecto', ''), st.session_state.get('autor', ''))
    d1, d2, d3 = st.columns(3)
    d1.download_button('Descargar PDF', pr, 'viga_concreto.pdf', 'application/pdf', width='stretch', key='sx_dl_pdf')
    d2.download_button('Descargar Excel', xl, 'viga_concreto.xlsx',
                       'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', width='stretch', key='sx_dl_xlsx')
    d3.download_button('Descargar texto', tx, 'viga_concreto.txt', 'text/plain', width='stretch', key='sx_dl_txt')

    st.markdown('#### Armado de toda la viga')
    cols = st.columns(2)
    for i, (et_, val, det_) in enumerate(sx.resumen(res)):
        with cols[i % 2].container(border=True):
            st.markdown(f'**{et_}**')
            st.markdown(f'### {val}')
            st.caption(det_)

    with st.expander('1 · Datos y revisiones de la viga'):
        _desarrollo(res['viga'])
    with st.expander('2 · Flexión en cada punto'):
        enc, filas = sx.tabla_flexion(res)
        st.dataframe(pd.DataFrame(filas, columns=enc), hide_index=True, width='stretch')
        for e in res['puntos']:
            if e['error']:
                st.error(f'Punto {e["punto"]}): {e["error"]}')
                continue
            with st.container(border=True):
                _desarrollo(e['ficha'], f'Punto {e["punto"]}) · Mu = {e["Mu"]:g} t·m · {e["ficha"]["armado"]} ' + ('✔' if e['ficha']['ok'] else '✘ revisa'))
                if e['alternativas']:
                    _tabla(pd.DataFrame(e['alternativas']).rename(columns={'armado': 'Armado', 'As': 'As (cm²)', 'exceso': 'Exceso', 'MR': 'MR (t·m)'}), 2)
        fig2 = sx.figura(res)
        st.pyplot(fig2)
        plt.close(fig2)
        st.caption('Rojo = no cumple la separación libre o no caben las barras. La numeración de pares (agrupadas) sigue la práctica de clase.')
    with st.expander('3 · Tanteo del eje neutro'):
        filas = [[e['punto'], e['ficha']['As_real'], e['tanteo']['c'], e['tanteo']['a'], e['tanteo']['es'],
                  'sí' if e['tanteo']['fluye'] else 'NO', e['tanteo']['MR'] / 1e5] for e in res['puntos'] if e['tanteo']]
        st.dataframe(pd.DataFrame(filas, columns=['Punto', 'As real (cm²)', 'c (cm)', 'a (cm)', 'εs', 'Fluye', 'MR (t·m)']), hide_index=True,
                     width='stretch')
    with st.expander('4 · Varillas corridas y bastones'):
        cor = res['corrido']
        st.markdown(f'- Mayor As real = {cor["As_max_real"]:.2f} cm²; 50 % = {0.5 * cor["As_max_real"]:.2f} cm²; As,mín = {res["lim"]["As_min"]:.2f} cm²')
        st.markdown(f'- Corrido: máx(As,mín; 50 % As mayor) = {cor["As_corr"]:.2f} cm² → n = {cor["As_corr"]:.2f}/{res["ab"]} = '
                    f'{cor["As_corr"] / res["ab"]:.2f} → **{cor["n"]}{var}** ({cor["As"]:.2f} cm²) arriba y abajo')
        if res['bastones']:
            _tabla(pd.DataFrame([[b_['punto'], b_['lecho'], f'{b_["n"]}{var}', b_['izq'], b_['der'], b_['izq'] + b_['der']] for b_ in res['bastones']],
                                columns=['Punto', 'Lecho', 'Varillas', 'Izquierda (m)', 'Derecha (m)', 'Total (m)']), 2)
        else:
            st.markdown('- Bastones: ninguno, las corridas alcanzan en todos los puntos.')
    with st.expander('5 · Cortante: estribos por tramo'):
        for tit, tx_ in res['lim_cort']:
            st.markdown(f'*{tit}*')
            st.latex(tx_)
        st.caption(f'El As en tensión de cada tramo es el mayor As real superior ({res["As_auto"]:.2f} cm²).')
        for i, t in enumerate(res['tramos'], 1):
            c_ = t['cort']
            with st.container(border=True):
                st.markdown(f'##### Tramo {t["tramo"]} · Vu = {t["Vu"]:g} t · ' + (f'E{est} {ramas}r @ {c_["s_usar"]} cm' if c_['ok'] else 'NO PASA'))
                for tit, tx_ in c_['tex']:
                    st.markdown(f'*{tit}*')
                    st.latex(tx_)
                if c_['ok']:
                    st.success(f'E {est}, {ramas} ramas @ {c_["s_usar"]} cm  (rige {c_["rige"]} en smáx)')
                    if t['alt']:
                        st.warning(f'@ {c_["s_usar"]} cm es muy cerrado para colar. Alternativas con s ≥ 10 cm: '
                                   + ' · '.join(o['texto'] for o in t['alt'][:4]))
                    elif c_['s_usar'] < 10:
                        st.warning(f'@ {c_["s_usar"]} cm es muy cerrado para colar y ningún estribo #3/#4 de 2–4 ramas da s ≥ 10 cm: agranda la sección.')
                else:
                    st.error('Vu excede el límite de la sección: agranda b o h.')
        enc, filas = sx.tabla_cortante(res)
        st.markdown('**Resumen de cortante**')
        _tabla(pd.DataFrame([[(x if not isinstance(x, str) or x != '—' else np.nan) for x in f_] for f_ in filas], columns=enc), 4)
    with st.expander('6 · Zona de confinamiento y acero de piel'):
        for tit, tx_ in res['integ']['tex']:
            st.latex(tx_)
        st.success(f'En 2h = {res["integ"]["Lconf"]:g} cm desde la cara de cada apoyo: E {est}, {ramas} ramas @ '
                   f'mín(s por cortante; {res["integ"]["s_usar"]}) cm')
        for tit, tx_ in res['piel']['tex']:
            st.latex(tx_)
        for t in res['piel']['lineas']:
            st.markdown(f'- {t}')


def mostrar_viga_puntos():
    """Examen 3 · Viga por puntos: capturas los Mu y Vu del diagrama (de ETABS o del pórtico) y la app arma todo."""
    _viga_completa()


def mostrar_mamposteria():
    st.caption('Apuntes pp. 22–26: carga vertical, cortante por muro, fuerzas sísmicas por el método estático y distorsiones.')
    _mamposteria()


def mostrar_viento():
    st.caption('Apuntes pp. 27–30: velocidad de diseño, presiones por cara, fuerzas por nivel y techo.')
    _viento()
