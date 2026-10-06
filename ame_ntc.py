"""Reglas de NTC-S 2023 (Normas Técnicas Complementarias para Diseño por Sismo, CDMX).

Fuente: apuntes de Diseño Estructural (Dr. F. J. Rivero Ángeles, Universidad Iberoamericana), segundo parcial.
Cada función indica la página de los apuntes. Lo que los apuntes no traen (por ejemplo las tablas 4.3.x con los
límites γ_SV de cada sistema estructural) NO se inventa: se captura como dato editable y el programa lo avisa.
"""
import numpy as np

INTENSIDADES = ('Frecuente', 'Base de diseño', 'Infrecuente')
CODIGO_INTENSIDAD = {'Frecuente': 'FREC', 'Base de diseño': 'BD', 'Infrecuente': 'INF'}

# Subgrupos NTC y su categoría de riesgo equivalente en ASCE 7 (apuntes, p. 10)
SUBGRUPOS = {'B2': 'B', 'B1': 'B', 'A2': 'A', 'A1': 'A'}
RIESGO_ASCE = {'B2': 'I', 'B1': 'II', 'A2': 'III', 'A1': 'IV'}

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


def objetivo_diseno(grupo, intensidad):
    """Nivel de desempeño y espectro que corresponden al grupo y a la intensidad (tablas 1.1a y 3.1.1, p. 12).

    Devuelve un dict con la clave del estado de cálculo ('DL', 'OI' o 'SV'), su nombre, el espectro y una nota.
    """
    if intensidad == 'Frecuente':
        return dict(clave='DL', desempeno='Limitación de daño (no estructural)',
                    espectro='Sa(BD)·Ks, sin factor de carga', revisa='Solo revisión de distorsiones (γ ≤ 0.002 ó 0.004)',
                    nota='')
    if grupo == 'A' and intensidad == 'Base de diseño':
        return dict(clave='OI', desempeno='Ocupación inmediata',
                    espectro='Sa(BD)/R′ con Q = 1 y R′ = 0.75·R', revisa='Distorsiones y diseño por resistencia',
                    nota='')
    if grupo == 'A' and intensidad == 'Infrecuente':
        return dict(clave='SV', desempeno='Seguridad de vida',
                    espectro="Sa(INF)/(Q′·R′) con R′ = R", revisa='Distorsiones y diseño por resistencia',
                    nota='Usa el espectro de la intensidad infrecuente.')
    if grupo == 'B' and intensidad == 'Base de diseño':
        return dict(clave='SV', desempeno='Seguridad de vida',
                    espectro="Sa(BD)/(Q′·R′) con R′ = R", revisa='Distorsiones y diseño por resistencia', nota='')
    # Grupo B, infrecuente: prevención de colapso (evaluación basada en desempeño)
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
