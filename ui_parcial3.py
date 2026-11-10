"""Pestaña «5 · Tercer parcial» de la app: mampostería, viento y concreto reforzado (motor en ame_parcial3.py)."""
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
                    'distancia d más allá de donde ya no se requiere en el diagrama de Mu. El alzado con las longitudes sale en la pestaña '
                    '«6 · Diagramas» → «Viga continua del nivel».')
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
            else:
                st.error('Vu excede el límite de la sección: agranda b o h.')
        resumen.append([tr, vu, as_t, c_['rho'], c_['VcR'], c_['Vs'], c_['s_calc'] if c_['s_calc'] is not None else np.nan, c_['s_max'],
                        f'E {est}, {ramas}r @ {c_["s_usar"]} cm' if c_['ok'] else 'NO'])
    st.markdown('**Resumen**')
    _tabla(pd.DataFrame(resumen, columns=['Tramo', 'Vu (t)', 'As tens. (cm²)', 'ρ', 'VcR (kg)', 'Vu−VcR (kg)', 's calc (cm)', 'smáx (cm)',
                                          'Estribos']), 4)


def _concreto():
    s_viga_c, s_flex, s_tant, s_cort, s_viga = st.tabs(['Viga completa (clase)', 'Flexión: As', 'Tanteos (eje neutro)',
                                                         'Cortante: estribos', 'Cargas y diagramas'])
    with s_viga_c:
        _viga_completa()
    with s_flex:
        c1, c2, c3, c4 = st.columns(4)
        b = c1.number_input('b (cm)', 5.0, 300.0, 40.0, key='cf_b')
        h = c2.number_input('h (cm)', 10.0, 300.0, 80.0, key='cf_h')
        r = c3.number_input('r (cm)', 1.0, 20.0, 5.0, key='cf_r')
        Mu = c4.number_input('Mu (t·m)', 0.1, 5000.0, 35.03, key='cf_Mu')
        c1, c2 = st.columns(2)
        fc = c1.number_input("f′c (kg/cm²)", 100.0, 1000.0, 250.0, key='cf_fc')
        fy = c2.number_input('fy (kg/cm²)', 2000.0, 6000.0, 4200.0, key='cf_fy')
        res = p3.diseno_flexion(Mu, b, h, r, fc, fy)
        if not res['ok']:
            st.error('La sección no alcanza con acero subreforzado: cambia sección o concreto.')
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('As calculada', f'{res["As_calc"]:.2f} cm²')
            c2.metric('As de diseño', f'{res["As_dis"]:.2f} cm²')
            c3.metric('As,mín', f'{res["As_min"]:.2f} cm²')
            c4.metric('As,máx', f'{res["As_max"]:.2f} cm²')
            _tabla(pd.DataFrame(p3.proponer_armados(res['As_dis'])).assign(
                MR=lambda d: [p3.momento_resistente(a, b, res['d'], fc, fy) / 1e5 for a in d['As']]).rename(
                columns={'armado': 'Armado', 'As': 'As (cm²)', 'exceso': 'Exceso', 'MR': 'MR (t·m)'}), 2)
            st.caption('MR se verifica con el área REAL del armado. Si b ≥ 40 cm el estribo lleva 4 ramas.')
        with st.expander('Procedimiento', expanded=True):
            _pasos(res['pasos'])
        import ame_porticos as pt
        sug_f, txt_f = pt.varilla_sugerida(h)
        var_f = st.selectbox('Varilla para el armado detallado', list(pt.DIAMETROS), index=list(pt.DIAMETROS).index(sug_f),
                             key=f'cf_var_{sug_f}', help=txt_f)
        prop = st.checkbox('Yo propongo el número de varillas (la app calcula As real = n·Ab)', False, key='cf_prop')
        n_prop = st.number_input('Número de varillas', 1, 30, 4, key='cf_nprop', disabled=not prop)
        f_ = pt.ficha_punto(Mu, b, h, fc, fy, var_f, rec=max(r - 0.95 - pt.DIAMETROS[var_f] / 2, 1.5), est_db=0.95, r_fija=r,
                            propuesta=(int(n_prop), var_f) if prop else None)
        (st.success if f_['ok'] else st.error)(f'{f_["armado"]} = {f_["As_real"]:.2f} cm²; MR = {f_["MR"]:.2f} t·m '
                                                f'{"≥" if f_["MR"] >= Mu else "<"} Mu = {Mu:.2f} t·m' if f_['As_real'] else 'Sin solución')
        st.markdown('##### Cálculo detallado')
        _desarrollo(f_)
    with s_tant:
        c1, c2, c3, c4 = st.columns(4)
        b = c1.number_input('b (cm)', 5.0, 300.0, 35.0, key='ct_b')
        d = c2.number_input('d (cm)', 5.0, 300.0, 55.0, key='ct_d')
        As = c3.number_input('As (cm²)', 0.1, 500.0, 11.94, key='ct_As')
        cp = c4.number_input('Tanteo de c (cm, 0 = resolver)', 0.0, 300.0, 25.0, key='ct_c')
        fc = c1.number_input("f′c", 100.0, 1000.0, 250.0, key='ct_fc')
        fy = c2.number_input('fy', 2000.0, 6000.0, 4200.0, key='ct_fy')
        lim = p3.limites_flexion(b, d, fc, fy)
        if cp > 0:
            e = p3.tanteo_eje_neutro(As, b, d, fc, fy, c_prop=cp)
            st.markdown(f'**Tanteo c = {cp:g} cm:** a = β1·c = {e["a"]:.2f} cm; C = a·b·f″c = {e["C"]:,.1f} kg; '
                        f'εs = 0.003(d − c)/c = {e["es"]:.5f} → {"fluye" if e["fluye"] else "NO fluye"}; fs = {e["fs"]:,.0f}; '
                        f'T = As·fs = {e["T"]:,.1f} kg → {"C > T: baja c" if e["C"] > e["T"] else "C < T: sube c"}.')
        e = p3.tanteo_eje_neutro(As, b, d, fc, fy)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('c (cm)', f'{e["c"]:.3f}')
        c2.metric('a (cm)', f'{e["a"]:.3f}')
        c3.metric('εs', f'{e["es"]:.5f}')
        c4.metric('MR (t·m)', f'{e["MR"] / 1e5:.2f}')
        st.caption(f'Equilibrio C = T = {e["T"]:,.1f} kg; el acero {"fluye → subreforzada" if e["fluye"] else "NO fluye → sobrerreforzada"}. '
                   f'Asb = {lim["Asb"]:.2f}, As,máx = {lim["As_max"]:.2f}, As,mín = {lim["As_min"]:.2f} cm² '
                   f'(con As = {As:g}: {"dentro de los límites" if lim["As_min"] <= As <= lim["As_max"] else "FUERA de los límites"}).')
    with s_cort:
        c1, c2, c3, c4 = st.columns(4)
        b = c1.number_input('b (cm)', 5.0, 300.0, 25.0, key='cc_b')
        h = c2.number_input('h (cm)', 10.0, 300.0, 50.0, key='cc_h')
        r = c3.number_input('r (cm)', 1.0, 20.0, 5.0, key='cc_r')
        L = c4.number_input('Claro L (cm)', 10.0, 3000.0, 500.0, key='cc_L')
        c1, c2, c3, c4 = st.columns(4)
        Vu = c1.number_input('Vu (t)', 0.1, 500.0, 14.83, key='cc_Vu')
        As = c2.number_input('As longitudinal (cm²)', 0.1, 500.0, 11.4, key='cc_As')
        est = c3.selectbox('Estribo', ['#3 (2 ramas)', '#3 (4 ramas)', '#4 (2 ramas)', '#4 (4 ramas)', '#2 (2 ramas)'], key='cc_est')
        fyest = c4.selectbox('fy del estribo', [4200.0, 2530.0], key='cc_fyest')
        c1, c2 = st.columns(2)
        fc = c1.number_input("f′c", 100.0, 1000.0, 250.0, key='cc_fc')
        av_bar = {'#2': 0.32, '#3': 0.71, '#4': 1.27}[est[:2]]
        Av = av_bar * (4 if '4 ramas' in est else 2)
        c2.metric('Av', f'{Av:.2f} cm²')
        res = p3.cortante_viga(Vu * 1000, b, h - r, h, L, fc, fyest, Av, As)
        if res['ok']:
            c1, c2, c3 = st.columns(3)
            c1.metric('VcR (kg)', f'{res["VcR"]:,.0f}')
            c2.metric('smáx (cm)', f'{res["s_max"]:.1f}')
            c3.metric('Estribos', f'{est[:2]} @ {res["s_usar"]} cm')
        else:
            st.error('Vu excede el límite 2.65·FR·b·d·√f′c: agranda la sección.')
        st.caption('Extremos de viga (L = 2h) en condición sísmica: integridad estructural s ≤ min{d/4; 8 db long.; 24 db est.; 30 cm}.')
        with st.expander('Procedimiento', expanded=True):
            _pasos(res['pasos'])
    with s_viga:
        c1, c2, c3 = st.columns(3)
        w = c1.number_input('w (t/m), ya factorizada o no', 0.01, 500.0, 3.55, key='cv_w')
        L = c2.number_input('L (m)', 0.5, 50.0, 5.0, key='cv_L')
        c3.caption('Carga última: wu = FC_CM·CM + FC_CV·CVmáx. Grupo B: 1.3 / 1.5; Grupo A: 1.5 / 1.7.')
        s = p3.viga_simple(w, L)
        e = p3.viga_doble_empotrada(w, L)
        _tabla(pd.DataFrame({'Apoyos': ['Simplemente apoyada', 'Doblemente empotrada'],
                             'Vu (t)': [s['Vu'], e['Vu']], 'Mu − (t·m)': [0.0, e['Mu_neg']],
                             'Mu + (t·m)': [s['Mu'], e['Mu_pos']]}), 3)
        st.markdown('**Wu de una viga (Grupo B/A)**')
        c1, c2, c3, c4 = st.columns(4)
        cm = c1.number_input('CM (t/m)', 0.0, 500.0, 1.0, key='cv_cm')
        cvm = c2.number_input('CVmáx (t/m)', 0.0, 500.0, 0.5, key='cv_cvm')
        pp = c3.number_input('Peso propio (t/m)', 0.0, 50.0, 0.768, key='cv_pp')
        grupo = c4.radio('Grupo', ['B', 'A'], horizontal=True, key='cv_grupo')
        fcm, fcv = (1.3, 1.5) if grupo == 'B' else (1.5, 1.7)
        st.metric(f'wu = {fcm}(CM + pp) + {fcv}·CVmáx', f'{fcm * (cm + pp) + fcv * cvm:.4f} t/m')


def mostrar():
    st.caption('Tercer parcial del Dr. Rivero (apuntes pp. 22–35): mampostería, viento y concreto reforzado. Cada módulo '
               'viene validado contra los ejemplos de tus apuntes. Las normas de material (Mampostería, Viento, Concreto) '
               'se toman de tus apuntes tal cual.')
    t_m, t_v, t_c = st.tabs(['Mampostería', 'Viento', 'Concreto reforzado'])
    with t_m:
        _mamposteria()
    with t_v:
        _viento()
    with t_c:
        _concreto()
