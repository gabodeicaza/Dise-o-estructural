"""Reglas de NTC-S 2023 (Normas Técnicas Complementarias para Diseño por Sismo, CDMX).

Fuente: apuntes de Diseño Estructural (Dr. F. J. Rivero Ángeles, Universidad Iberoamericana), segundo parcial.
Cada función indica la página de los apuntes. Lo que los apuntes no traen (por ejemplo las tablas 4.3.x con los
límites γ_SV de cada sistema estructural) NO se inventa: se captura como dato editable y el programa lo avisa.
"""
import numpy as np

INTENSIDADES = ('Frecuente', 'Ocasional', 'Base de diseño', 'Infrecuente')
CODIGO_INTENSIDAD = {'Frecuente': 'FREC', 'Ocasional': 'OCAS', 'Base de diseño': 'BD', 'Infrecuente': 'INF'}
# "MC" (máximo considerado) es el nombre que algunos exámenes dan a la intensidad infrecuente
ALIAS_INTENSIDAD = {'mc': 'Infrecuente', 'máximo considerado': 'Infrecuente', 'maximo considerado': 'Infrecuente',
                    'máximo creíble': 'Infrecuente', 'bd': 'Base de diseño', 'frec': 'Frecuente', 'ocas': 'Ocasional'}
ETIQUETA_INTENSIDAD = {'Infrecuente': 'Infrecuente (MC)', 'Base de diseño': 'Base de diseño (BD)'}
FACTOR_KS_OCASIONAL = 2.5        # Sa(BD)·2.5·Ks en operación continua con sismo ocasional (p. 12)


def ks_norma(Ts):
    """Factor Ks de la ecuación 3.1.1 (frecuente = base de diseño × Ks): zona A 1/6, zona B 1/[6 − 4(Ts − 0.5)], zona C 1/4.
    Las tres zonas se distinguen por Ts (0.5 s y 1.0 s) y la fórmula es continua entre ellas."""
    if Ts <= 0.5:
        return 1.0 / 6.0
    if Ts >= 1.0:
        return 0.25
    return 1.0 / (6.0 - 4.0 * (Ts - 0.5))


def zona_ks(Ts):
    return 'A' if Ts <= 0.5 else ('C' if Ts >= 1.0 else 'B')


def normalizar_intensidad(texto):
    """Acepta 'MC', 'Infrecuente (MC)', 'bd'… y devuelve el nombre canónico (o el mismo texto si no lo reconoce)."""
    t = str(texto or '').strip()
    if t in INTENSIDADES:
        return t
    base = t.split('(')[0].strip().lower()
    for nombre in INTENSIDADES:
        if base == nombre.lower():
            return nombre
    return ALIAS_INTENSIDAD.get(base) or ALIAS_INTENSIDAD.get(t.lower(), t)


def espectro_fuente(intensidad):
    """Espectro SASID que alimenta cada intensidad: frecuente y ocasional salen del de base de diseño (× Ks)."""
    return 'Infrecuente' if intensidad == 'Infrecuente' else 'Base de diseño'

# Subgrupos NTC y su categoría de riesgo equivalente en ASCE 7 (apuntes, p. 10)
SUBGRUPOS = {'B': 'B', 'A2': 'A', 'A1': 'A'}
RIESGO_ASCE = {'B': 'I–II', 'A2': 'III', 'A1': 'IV'}
ALIAS_SUBGRUPO = {'B1': 'B', 'B2': 'B'}     # proyectos anteriores: las dos B de la norma se unifican en una sola


def normalizar_subgrupo(texto):
    """'B1' y 'B2' (proyectos anteriores) pasan a 'B'; el resto se deja igual (en mayúsculas)."""
    t = str(texto or '').strip().upper()
    return ALIAS_SUBGRUPO.get(t, t)

# Guía de Q (p. 11)
GUIA_Q = {1.0: 'Ductilidad baja · estructura elástica, sin daño (γmáx = 0.005)',
          1.5: 'Ductilidad baja · poco daño',
          2.0: 'Ductilidad media · daño medio',
          3.0: 'Ductilidad media · daño importante',
          4.0: 'Ductilidad alta · daño severo'}

# Hiperestaticidad k1 (p. 15 y 23)
GUIA_K1 = ((0.8, 'menos de 3 crujías resistentes'),
           (1.0, 'marcos y muros de mampostería con 3 o más crujías'),
           (1.25, 'sistemas duales (muros diafragma, contraventeos)'))

# Irregularidades que reducen γmáx (tablas C5.4.2, C5.5.3 y C5.6.3, p. 12)
IRREGULARIDADES = {
    '5.2.1': 'Irregularidad por torsión',
    '5.2.3': 'Forma geométrica irregular en planta',
    '5.2.4': 'Flexibilidad excesiva de un diafragma',
    '5.2.5': 'Discontinuidad en el diafragma',
    '5.3.1': 'Reducciones geométricas en elevación',
    '5.3.2': 'Reducción brusca de rigidez lateral',
}

FACTOR_CIMENTACION = 0.65        # elementos mecánicos de cimentación = 0.65·R′ (p. 9, inciso 3)
FACTOR_R_OI = 0.75               # R′ = 0.75·R en ocupación inmediata, R′ = R en seguridad de vida (p. 10)
ZETA_MATERIAL = {'Concreto': 0.05, 'Acero': 0.03}   # amortiguamiento crítico (p. 14)
LIMITES_ALTURA_ESTATICO = {'I': (40.0, 30.0), 'II': (30.0, 20.0), 'III': (30.0, 20.0)}  # regular, irregular (p. 20)


def objetivo_diseno(grupo, intensidad, subgrupo=None):
    """Nivel de desempeño y espectro que corresponden al subgrupo y a la intensidad (pizarrón de clase y tablas 1.1a / 3.1.1).

                 Op. continua   Ocup. inmediata   Seg. de vida   Prev. de colapso
    Frecuente        B  (A1, A2 también cumplen)
    Ocasional        A1, A2         B
    Base de diseño   A1, A2 (Tabla 3.1.1: objetivos 4 y 6)    B
    MC (infrecuente)                A1                A2             B

    Un grupo A sin subgrupo ('A') se toma como A2. Devuelve un dict con la clave del estado de cálculo
    ('DL', 'OI' o 'SV'), su nombre, el espectro y una nota.
    """
    sub = str(subgrupo or '').strip().upper()
    a1 = grupo == 'A' and sub == 'A1'
    revisa_dl = 'Solo revisión de distorsiones (γ ≤ 0.002 ó 0.004)'
    revisa_r = 'Distorsiones y diseño por resistencia'

    def dl(espectro, nota=''):
        return dict(clave='DL', desempeno='Operación continua (limitación de daño)', espectro=espectro, revisa=revisa_dl,
                    nota=nota)

    def oi(espectro):
        return dict(clave='OI', desempeno='Ocupación inmediata', espectro=espectro, revisa=revisa_r, nota='')

    if intensidad == 'Frecuente':
        return dl('Sa(BD)·Ks, sin factor de carga')
    if intensidad == 'Ocasional':
        return dl('Sa(BD)·2.5·Ks, sin factor de carga') if grupo == 'A' else oi('Sa(BD)/R′ con Q = 1 y R′ = 0.75·R')
    if intensidad == 'Base de diseño':
        if grupo == 'A':
            return oi('Sa(BD)/R′ con Q = 1 y R′ = 0.75·R')
        return dict(clave='SV', desempeno='Seguridad de vida', espectro="Sa(BD)/(Q′·R′) con R′ = R", revisa=revisa_r, nota='')
    if a1:  # MC, A1
        return oi('Sa(INF)/R′ con Q = 1 y R′ = 0.75·R')
    if grupo == 'A':  # MC, A2
        return dict(clave='SV', desempeno='Seguridad de vida', espectro="Sa(INF)/(Q′·R′) con R′ = R", revisa=revisa_r,
                    nota='Usa el espectro de la intensidad infrecuente (MC).')
    # Grupo B, MC: prevención de colapso (evaluación basada en desempeño)
    return dict(clave='SV', desempeno='Prevención de colapso',
                espectro="Sa(INF)/(Q′·R′)", revisa='Evaluación basada en desempeño (acelerogramas)',
                nota='La prevención de colapso del grupo B es una revisión optativa con acelerogramas; '
                     'el programa solo calcula con espectro (como seguridad de vida), a modo de referencia.')


def k1_sugerido(n_crujias):
    """k1 según los apuntes (marcos): 0.8 con menos de 3 crujías, 1.0 con 3 o más. Los sistemas duales usan 1.25."""
    return 0.8 if n_crujias < 3 else 1.0


def gamma_c(irreg, fuerte_torsion=False, fuerte_elevacion=False):
    """Factor γc que reduce el límite de distorsión γmáx por irregularidad (p. 12). Devuelve (factor, texto)."""
    codigos = sorted(set(irreg))
    for c in codigos:
        if c not in IRREGULARIDADES:
            raise ValueError(f'Irregularidad no reconocida: {c}.')
    if fuerte_elevacion:
        return 0.33, ('Fuertemente irregular en elevación (5.3.3): γc = 0.33·γmáx. Se debe diseñar el entrepiso débil '
                      "(y los de abajo) con Q′ = 1 y revisar con análisis no lineal paso a paso.")
    if fuerte_torsion:
        otras = [c for c in codigos if c != '5.2.1']
        if otras:
            return 0.50, 'Fuertemente irregular por torsión (5.2.2) con otras irregularidades: γc = 0.50·γmáx.'
        return 0.60, 'Fuertemente irregular por torsión (5.2.2): γc = 0.60·γmáx.'
    n = len(codigos)
    if n == 0:
        return 1.0, 'Estructura regular: γc = 1.0.'
    factor = {1: 0.8, 2: 0.7}.get(n, 0.6)
    return factor, f'{n} condición(es) de irregularidad: γc = {factor:g}·γmáx.'


def excentricidad_accidental(n, b):
    """e_a,i = [0.05 + 0.05·(i−1)/(n−1)]·b para los niveles i = 1..n (p. 9). b: dimensión perpendicular al análisis."""
    i = np.arange(1, n + 1)
    return (0.05 + (0.05 * (i - 1) / (n - 1) if n > 1 else 0.0 * i)) * b


def altura_maxima_estatico(zona, regular):
    """Altura máxima (m) para el método estático (p. 20). None si la zona no es válida."""
    lim = LIMITES_ALTURA_ESTATICO.get(str(zona).strip().upper())
    return None if lim is None else lim[0 if regular else 1]
