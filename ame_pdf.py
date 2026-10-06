"""Memoria de cálculo en PDF del Análisis Modal Espectral."""
import datetime
import io
import os
import re

import numpy as np
from matplotlib import get_data_path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

import ame_sismico as ame

NAVY, AZUL_OSC, AZUL, AZUL_MED, AZUL_CLARO, PALE, BG = (colors.HexColor(c) for c in (
    '#0B2545', '#13315C', '#1D4E89', '#3E7CB1', '#81A4CD', '#DCE9F7', '#F1F6FC'))
ROJO = colors.HexColor('#B3261E')

_FUENTES_OK = False


def _registrar_fuentes():
    """DejaVu Sans (incluida con matplotlib): trae letras griegas, Ø, ², Σ, etc."""
    global _FUENTES_OK
    if _FUENTES_OK:
        return
    carpeta = os.path.join(get_data_path(), 'fonts', 'ttf')
    for nombre, archivo in (('DejaVu', 'DejaVuSans.ttf'), ('DejaVu-Bold', 'DejaVuSans-Bold.ttf'),
                            ('DejaVu-Italic', 'DejaVuSans-Oblique.ttf'), ('DejaVu-BoldItalic', 'DejaVuSans-BoldOblique.ttf')):
        pdfmetrics.registerFont(TTFont(nombre, os.path.join(carpeta, archivo)))
    pdfmetrics.registerFontFamily('DejaVu', normal='DejaVu', bold='DejaVu-Bold', italic='DejaVu-Italic',
                                  boldItalic='DejaVu-BoldItalic')
    _FUENTES_OK = True


def _estilos():
    base = ParagraphStyle('base', fontName='DejaVu', fontSize=9, leading=12.5, textColor=NAVY, alignment=TA_LEFT)
    return dict(
        base=base,
        h1=ParagraphStyle('h1', parent=base, fontName='DejaVu-Bold', fontSize=14, leading=18, textColor=AZUL,
                          spaceBefore=14, spaceAfter=6),
        h2=ParagraphStyle('h2', parent=base, fontName='DejaVu-Bold', fontSize=10.5, leading=14, textColor=AZUL_OSC,
                          spaceBefore=8, spaceAfter=3),
        nota=ParagraphStyle('nota', parent=base, fontSize=8, leading=11, textColor=AZUL_MED),
        formula=ParagraphStyle('formula', parent=base, fontName='DejaVu-Italic', leftIndent=14, textColor=AZUL_OSC,
                               spaceAfter=2),
        th=ParagraphStyle('th', parent=base, fontName='DejaVu-Bold', fontSize=8, leading=10, textColor=colors.white,
                          alignment=1),
    )


_ETIQUETA = re.compile(r'<(?!/?(?:b|i|u|font|sub|super|br)[\s>/])')


def _escapar(txt):
    """Escapa el '<' de las fórmulas (T<Ta) sin tocar las etiquetas de formato del Paragraph."""
    return _ETIQUETA.sub('&lt;', txt.replace('&', '&amp;'))


def _f(v, dec=4):
    if not isinstance(v, (int, float, np.number)):
        return v  # texto o Paragraph
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if abs(v) >= 1e5:
        return f'{v:,.1f}'
    return f'{v:.{dec}f}'


def _tabla(st, enc, filas, dec=4, anchos=None, fuente=8, zebra=True):
    datos = [[Paragraph(h, st['th']) for h in enc]] + [[_f(v, dec) for v in fila] for fila in filas]
    t = Table(datos, colWidths=anchos, repeatRows=1, hAlign='LEFT')
    estilo = [('FONTNAME', (0, 0), (-1, -1), 'DejaVu'), ('FONTSIZE', (0, 0), (-1, -1), fuente),
              ('TEXTCOLOR', (0, 1), (-1, -1), NAVY), ('BACKGROUND', (0, 0), (-1, 0), AZUL),
              ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
              ('GRID', (0, 0), (-1, -1), 0.4, AZUL_CLARO), ('TOPPADDING', (0, 0), (-1, -1), 3),
              ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]
    if zebra:
        estilo += [('BACKGROUND', (0, i), (-1, i), BG) for i in range(2, len(datos), 2)]
    t.setStyle(TableStyle(estilo))
    return t


def _imagen(fig, ancho_cm):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=160)
    buf.seek(0)
    w, h = fig.get_size_inches()
    ancho = ancho_cm * cm
    return Image(buf, width=ancho, height=ancho * h / w)


def memoria_pdf(res, destino, proyecto='', autor=''):
    """Genera la memoria de cálculo. `destino` es una ruta o un BytesIO."""
    _registrar_fuentes()
    st = _estilos()
    p, n = res['p'], res['p']['n']
    P = lambda txt, est='base': Paragraph(_escapar(txt), st[est])
    Ancho = letter[0] - 3.6 * cm

    def pie(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(AZUL_CLARO)
        canvas.line(1.8 * cm, 1.5 * cm, letter[0] - 1.8 * cm, 1.5 * cm)
        canvas.setFont('DejaVu', 7.5)
        canvas.setFillColor(AZUL_MED)
        canvas.drawString(1.8 * cm, 1.05 * cm, f'Memoria de cálculo · Análisis Modal Espectral'
                                              + (f' · {proyecto}' if proyecto else ''))
        canvas.drawRightString(letter[0] - 1.8 * cm, 1.05 * cm, f'Página {doc.page}')
        canvas.restoreState()

    doc = BaseDocTemplate(destino, pagesize=letter, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.8 * cm,
                          bottomMargin=2 * cm, title='Memoria de cálculo - Análisis Modal Espectral',
                          author=autor or 'AME')
    doc.addPageTemplates([PageTemplate(id='p', frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                                                             id='f')], onPage=pie)])
    E = []

    # ------------------------------------------------------------------ portada
    banda = Table([[Paragraph('<font color="white" size="20"><b>Memoria de cálculo</b></font><br/>'
                              '<font color="#81A4CD" size="11">Análisis Modal Espectral</font>', st['base'])]],
                  colWidths=[Ancho])
    banda.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), NAVY), ('LEFTPADDING', (0, 0), (-1, -1), 16),
                               ('TOPPADDING', (0, 0), (-1, -1), 18), ('BOTTOMPADDING', (0, 0), (-1, -1), 18)]))
    E += [banda, Spacer(1, 8)]
    meta = [['Proyecto', proyecto or '—'], ['Elaboró', autor or '—'],
            ['Fecha', datetime.date.today().strftime('%d/%m/%Y')]]
    t = Table(meta, colWidths=[3 * cm, Ancho - 3 * cm], hAlign='LEFT')
    t.setStyle(TableStyle([('FONTNAME', (0, 0), (0, -1), 'DejaVu-Bold'), ('FONTNAME', (1, 0), (1, -1), 'DejaVu'),
                           ('FONTSIZE', (0, 0), (-1, -1), 9), ('TEXTCOLOR', (0, 0), (-1, -1), NAVY),
                           ('TEXTCOLOR', (0, 0), (0, -1), AZUL), ('LINEBELOW', (0, 0), (-1, -1), 0.4, AZUL_CLARO)]))
    E.append(t)

    # resumen
    fe = res.get('factor_escala')
    resumen = [['Periodo fundamental T₁', f'{res["T"][0]:.4f} s'],
               ['Cortante basal dinámico Vu', f'{res["V_din"]:.3f} t']]
    if fe is not None:
        resumen += [['Cortante basal estático', f'{res["V_est"]:.3f} t'], ['Factor de escala', f'{fe:.4f}']]
    est = res['revision']['estados']
    resumen += [[f'Distorsión máx. · {e["nombre"]}', f'{e["dist"].max():.5f}'
                 + ('' if e['limite'] is None else f'  (límite {e["limite"]:g}: '
                                                   + ('cumple' if e['dist'].max() <= e['limite'] else 'NO cumple') + ')')]
                for e in est.values()]
    E += [P('Resumen de resultados', 'h1'), _tabla(st, ['Concepto', 'Valor'], resumen, anchos=[8 * cm, Ancho - 8 * cm],
                                                   fuente=9)]
    if res['avisos']:
        E += [P('Avisos principales', 'h2')]
        for sev, txt in [a for a in res['avisos'] if a[0] != 'info'][:6]:
            E.append(P(f'<font color="{"#B3261E" if sev == "alerta" else "#3E7CB1"}"><b>'
                       f'{"ALERTA" if sev == "alerta" else "AVISO"}</b></font> · {txt}', 'nota'))

    # ------------------------------------------------------------------ 1. datos
    E += [P('1. Datos de entrada', 'h1')]
    datos = [['Material', p.get('material', 'Concreto')], ['Módulo de elasticidad E', f'{p.get("E", 0):,.0f} kg/cm²'],
             ['Tipo de edificación', f'Grupo {p["grupo"]}' + (f' · sismo {"base" if p["sismo"] == "B" else "infrecuente"}'
                                                                if p['grupo'] == 'A' else '')],
             ['Estado límite de diseño', p['estado'] + (f' (Ks = {p["Ks"]:g})' if res['clave'] == 'DL' else '')],
             ['Q (valores)', ', '.join(f'{q:g}' for q in p['Q'])], ['k₁', f'{p["k1"]:g}'],
             ['Espectro del sitio', f'a₀ = {p["a0"]:g} cm/s² · c = {p["c"]:g} cm/s² · Tₐ = {p["Ta"]:g} s · '
                                    f'T_b = {p["Tb"]:g} s · k = {p["k"]:g} · T_s = {p["Ts"]:g} s'],
             ['Dirección del sismo', p.get('direccion', 'X')],
             ['Rigidez de vigas', 'Flexibles (marco plano)' if p.get('vigas') == 'flexibles' else 'Rígidas (marco de cortante)'],
             ['Combinación modal', p['combinacion'] + (f' (ζ = {p["zeta"]:g})' if p['combinacion'] == 'CQC' else '')],
             ['Factor Fu (FC)', f'{p["factor_Fu"]:g}']]
    E.append(_tabla(st, ['Parámetro', 'Valor'], [[a, Paragraph(b, st['base']) if len(b) > 40 else b] for a, b in datos],
                    anchos=[5.5 * cm, Ancho - 5.5 * cm], fuente=8.5))

    # ------------------------------------------------------------------ 2. estructura
    E += [P('2. Descripción de la estructura', 'h1')]
    if 'geom' in p:
        E.append(_imagen(ame.fig_estructura(p), 17))
    if 'info' in p:
        enc = ['Nivel', 'h (m)', 'Nº col.', 'Nº art.', 'Sección (cm)', 'ΣI (cm⁴)', 'K nivel (kg/cm)', 'W (t)', 'm (kg·s²/cm)']
        filas = [[r['nivel'], r['h'], r['ncol'], r['nart'], r['seccion'], r['I'], r['k'], r['W'], r['m']] for r in p['info']]
        if p.get('vigas') == 'flexibles':
            enc.append('Viga')
            for f, r in zip(filas, p['info']):
                f.append(r['viga'])
        E += [P('Niveles', 'h2'), _tabla(st, enc, filas, dec=3, fuente=7.5)]
        if p.get('vigas') != 'flexibles':
            E.append(P('Rigidez de entrepiso con vigas rígidas: k = Σ 12·E·I / h³ por columna empotrada y '
                       '3·E·I / h³ por columna articulada en su base.', 'nota'))
        else:
            E.append(P('Rigidez lateral: se ensambla la rigidez de columnas y vigas (flexión y axial) de un marco plano '
                       'con diafragmas rígidos y se condensan los giros y desplazamientos verticales para dejar un '
                       'grado de libertad lateral por nivel.', 'nota'))
        if 'geom' in p:
            filas_c = [[c['nivel'], c['eje'], c['x'], c['texto'], c['I'], 'Sí' if c['artic'] else 'No', c['k']]
                       for l in p['geom']['niveles'] for c in l['cols']]
            E += [P('Columnas', 'h2'), _tabla(st, ['Nivel', 'Eje', 'x (m)', 'Sección (cm)', 'I (cm⁴)', 'Base articulada',
                                                  'k (kg/cm)'], filas_c, dec=2, fuente=7.5)]

    # ------------------------------------------------------------------ 3. matrices
    E += [P('3. Matrices de rigidez y de masa', 'h1'),
          P('Ecuación de movimiento libre: M·ü + K·u = 0 → (M⁻¹K − ω²·I)·Φ = 0', 'formula')]
    an = [None] + [None] * n
    fs = 8 if n <= 6 else 6.5
    E += [P('Matriz K (kg/cm)', 'h2'),
          _tabla(st, [''] + [str(j + 1) for j in range(n)], [[i + 1] + list(p['K'][i]) for i in range(n)], dec=2,
                 fuente=fs, zebra=False),
          P('Matriz M (kg·s²/cm) = W / g', 'h2'),
          _tabla(st, [''] + [str(j + 1) for j in range(n)], [[i + 1] + list(p['M'][i]) for i in range(n)], dec=4,
                 fuente=fs, zebra=False)]

    # ------------------------------------------------------------------ 4. modal
    E += [P('4. Análisis modal', 'h1'),
          P('A = M⁻¹·K · λ = eigenvalores de A · ω = √λ · f = ω / 2π · T = 1 / f · '
            'Φ = eigenvectores de A (normalizados con el primer elemento = 1)', 'formula'),
          P('γᵢ = (Φᵢᵀ·M·{1}) / (Φᵢᵀ·M·Φᵢ)', 'formula')]
    E.append(_tabla(st, ['Modo', 'λ (rad/s)²', 'ω (rad/s)', 'f (Hz)', 'T (s)', 'γ'],
                    [[j + 1, res['lam'][j], res['w'][j], res['f'][j], res['T'][j], res['gam'][j]] for j in range(n)], dec=5))
    E.append(P(f'Σγ = {res["gam"].sum():.5f}', 'nota'))
    E += [P('Eigenvectores normalizados (columnas = modos)', 'h2'),
          _tabla(st, ['Nivel'] + [f'Modo {j + 1}' for j in range(n)],
                 [[i + 1] + list(res['fi'][i]) for i in range(n)], dec=5, fuente=fs), Spacer(1, 6),
          _imagen(ame.fig_modal(res), 17)]

    # ------------------------------------------------------------------ 5. espectro
    E += [P('5. Espectro de diseño', 'h1'),
          P(f'Estado límite: <b>{p["estado"]}</b>. '
            + {'SV': "Se usa la ordenada reducida Sa/(Q′·R′).",
               'OI': "Ocupación inmediata: Q′ = 1, se reduce solo por R′.",
               'DL': f"Limitación de daños: se usa Sa·Ks con Ks = {p['Ks']:g}, sin Q′ ni R′."}[res['clave']])]
    E += [P('T < Tₐ: Sa = a₀ + (c − a₀)·T/Tₐ · Tₐ ≤ T < T_b: Sa = c · T ≥ T_b: Sa = c·p·(T_b/T)², '
            'p = k + (1−k)(T_b/T)²', 'formula'),
          P("Q′ = 1 + (Q−1)·√(1/k)·T/Tₐ  (T<Tₐ) · 1 + (Q−1)·√(1/k)  (Tₐ≤T<T_b) · 1 + (Q−1)·√(p/k)  (T≥T_b)", 'formula'),
          P("R = k₁·R₀ + k₂ (R₀ = 1.75 si Q<3, 2 si Q≥3; k₂ = 0.5·(1−√(T/Tₐ)) ≥ 0) · "
            "R′ = R (grupo B o sismo infrecuente) · 0.75·R (grupo A, sismo base)", 'formula')]
    E.append(_imagen(ame.fig_espectro(res), 15))
    E += [P("Ordenadas espectrales de diseño Sa/(Q′R′) (cm/s²)", 'h2'),
          _tabla(st, ['Modo', 'T (s)'] + [f'Q = {q:g}' for q in res['Qs']],
                 [[j + 1, res['T'][j]] + list(res['Sa_modal'][j]) for j in range(n)], dec=4)]
    if res['a_min']:
        E.append(P(f"Ordenada mínima: a_min = {res['a_min']:.4f} cm/s² ({res['a_min_g']:.5f} g), con Ts = {p['Ts']:g} s.",
                   'nota'))
    E += [P("Factores de reducción por modo", 'h2'),
          _tabla(st, ['Modo', 'T (s)', 'Q', 'k₂', "R′", "Q·R′"], res['QR'], dec=4)]

    # ------------------------------------------------------------------ 6. fuerzas modales
    E += [P('6. Desplazamientos y fuerzas modales', 'h1'),
          P('δᵢ = Φᵢ · (Saᵢ·γᵢ / λᵢ) · Fᵢ = K·δᵢ', 'formula'),
          P('Desplazamientos modales δ (cm), columnas = modos', 'h2'),
          _tabla(st, ['Nivel'] + [f'Modo {j + 1}' for j in range(n)], [[i + 1] + list(res['d'][i]) for i in range(n)],
                 dec=5, fuente=fs),
          P('Fuerzas modales F (kg), columnas = modos', 'h2'),
          _tabla(st, ['Nivel'] + [f'Modo {j + 1}' for j in range(n)], [[i + 1] + list(res['F'][i]) for i in range(n)],
                 dec=2, fuente=fs)]
    comb = {'SRSS': 'F = √(Σ Fᵢ²)  (raíz de la suma de los cuadrados)',
            'CQC': 'F = √(Σᵢ Σⱼ ρᵢⱼ·Fᵢ·Fⱼ)  (combinación cuadrática completa, ρ de Der Kiureghian)',
            'Suma absoluta': 'F = Σ |Fᵢ|  (suma de valores absolutos, cota superior)'}[p['combinacion']]
    E += [P('Combinación modal', 'h2'), P(comb, 'formula'),
          _tabla(st, ['Nivel', 'F combinada (kg)'], [[i + 1, res['F_final'][i]] for i in range(n - 1, -1, -1)], dec=2,
                 anchos=[3 * cm, 5 * cm])]

    # ------------------------------------------------------------------ 7. fuerzas por nivel
    t = res['tabla_din']
    E += [P('7. Fuerzas sísmicas por nivel (análisis dinámico)', 'h1'),
          P(f'Fu = F · {p["factor_Fu"]:g}/1000 (t) · Vu = ΣFu (acumulado desde arriba) · Mvu = h · Fu', 'formula'),
          _tabla(st, ['Nivel', 'h (m)', 'F (t)', 'Fu (t)', 'Vu (t)', 'Mvu (t·m)'],
                 [[t['nivel'][i], t['h'][i], t['F'][i], t['Fu'][i], t['Vu'][i], t['Mvu'][i]] for i in range(n)]),
          P(f'Σ Mvu = {t["Mvu"].sum():.4f} t·m', 'nota'), Spacer(1, 4), _imagen(ame.fig_fuerzas(res), 16)]

    # ------------------------------------------------------------------ 8. estático
    if 'tabla_est' in res:
        e = res['tabla_est']
        E += [P('8. Análisis estático y factor de escala', 'h1'),
              P('Fi = Sa_máx · (hᵢ·Wᵢ / Σ hᵢ·Wᵢ) · ΣWᵢ', 'formula'),
              _tabla(st, ['Nivel', 'h (m)', 'Wi (t)', 'Wi·hi (t·m)', 'Fi (t)', 'Fu (t)', 'Vu (t)', 'Mvu (t·m)'],
                     [[e['nivel'][i], e['h'][i], e['Wi'][i], e['hWi'][i], e['Fi'][i], e['Fu'][i], e['Vu'][i],
                       e['Mvu'][i]] for i in range(n)]),
              P(f'Sa/(Q′R′) máximo del espectro = {res["Fmax_est"]:.6f} g', 'nota')]
        filas = [['V estático', f'{res["V_est"]:.4f} t']]
        if p['mult_V'] != 1:
            filas.append(['Multiplicador', f'{p["mult_V"]:g}'])
        filas += [['V dinámico', f'{res["V_din"]:.4f} t'], ['Factor de escala = V est. / V din.', f'{fe:.4f}']]
        E += [Spacer(1, 4), _tabla(st, ['Concepto', 'Valor'], filas, anchos=[8 * cm, 5 * cm], fuente=9),
              P('Si el factor es mayor que 1, las fuerzas del análisis dinámico deben multiplicarse por él.', 'nota')]

    # ------------------------------------------------------------------ 9. revisiones
    rev = res['revision']
    E += [P('9. Revisiones', 'h1'), P('9.1 Distorsiones de entrepiso', 'h2'),
          P('Desplazamiento de diseño = FC·Q′·R′·δ (seguridad de vida y ocupación inmediata); Ks·δ elástico '
            '(limitación de daños). Se combinan las distorsiones modales con el método elegido y γ = Δ / h. '
            'Los límites son editables: confírmalos con tu reglamento.', 'nota')]
    lims = [[e['nombre'], f'{e["limite"]:g}' if e['limite'] is not None else 'sin definir'] for e in est.values()]
    E.append(_tabla(st, ['Estado límite', 'Límite γ'], lims, anchos=[6 * cm, 4 * cm], fuente=8.5))
    enc, filas = ame.tabla_distorsiones(res)
    E += [Spacer(1, 4), _tabla(st, enc, filas, dec=5, fuente=8), Spacer(1, 4), _imagen(ame.fig_distorsiones(res), 12)]
    pdt = ame.tabla_pdelta(res)
    if pdt:
        E += [P('9.2 Efectos P-Δ e irregularidades', 'h2'),
              P('θ = P·δ / (V·h): P = peso acumulado sobre el entrepiso, δ = deriva elástica con las fuerzas de diseño, '
                'V = cortante de entrepiso, h = altura. θ ≤ 0.10: se pueden ignorar; θ > 0.25: posible inestabilidad '
                '(criterio general ASCE 7).', 'nota'),
              _tabla(st, pdt[0], pdt[1], dec=4, fuente=8),
              P(f'Irregularidad de masa (peso > 1.5 veces el de un piso adyacente): '
                f'{", ".join(map(str, rev["irreg_masa"])) or "ninguna"}. '
                f'Piso blando (rigidez efectiva < 70% del nivel superior o < 80% del promedio de tres): '
                f'{", ".join(map(str, rev["irreg_rigidez"])) or "ninguno"}.', 'nota')]

    # ------------------------------------------------------------------ 10. avisos
    E += [P('10. Avisos de coherencia', 'h1')]
    if res['avisos']:
        rows = [[{'alerta': 'ALERTA', 'aviso': 'AVISO', 'info': 'INFO'}[sev], Paragraph(txt, st['base'])]
                for sev, txt in res['avisos']]
        t = Table(rows, colWidths=[2.2 * cm, Ancho - 2.2 * cm])
        t.setStyle(TableStyle([('FONTNAME', (0, 0), (0, -1), 'DejaVu-Bold'), ('FONTSIZE', (0, 0), (0, -1), 8),
                               ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, -1), 0.3, AZUL_CLARO),
                               ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]
                              + [('TEXTCOLOR', (0, i), (0, i), ROJO if sev == 'alerta' else AZUL_MED if sev == 'aviso' else AZUL_CLARO)
                                 for i, (sev, _) in enumerate(res['avisos'])]))
        E.append(t)
    else:
        E.append(P('Sin avisos.'))

    # ------------------------------------------------------------------ alcances
    E += [P('11. Alcances y limitaciones', 'h1')]
    for txt in ('Marco plano en una dirección: un grado de libertad lateral por nivel, masas concentradas y diafragmas rígidos.',
                'No se considera torsión (accidental o por excentricidad), sismo bidireccional, interacción suelo-estructura '
                'ni zonas rígidas en los nudos.',
                'Los criterios de P-Δ e irregularidades son generales (ASCE 7). Los límites de distorsión, Ks y los factores '
                'del espectro deben corresponder al reglamento aplicable; revísalos antes de usar los resultados.',
                'Este documento no sustituye la revisión de un ingeniero responsable.'):
        E.append(P('• ' + txt))

    doc.build(E)
    return destino
