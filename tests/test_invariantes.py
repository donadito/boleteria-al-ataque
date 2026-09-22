"""
Invariantes del proyecto: lo que tiene que seguir siendo cierto.

No son pruebas de cobertura, son candados sobre las propiedades que la
validacion estadistica dio por buenas. Cada una corresponde a algo que puede
romperse en silencio al tocar el codigo: un generador que deja de reproducir su
propia secuencia, un truncamiento que devuelve valores fuera de rango, un
proceso de llegadas que se sale de su ventana, una cola donde la paciencia deja
de morder, o boletos que se pierden en la contabilidad.

Uso:
    pytest tests/ -q
    python tests/test_invariantes.py     # sin pytest instalado
"""
import os
import sys

import numpy as np
from scipy import integrate, stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from boleteria.modelo.calendario import obtener_fases
from boleteria.modelo.entidades import AFORO_TOTAL, crear_secciones
from boleteria.modelo.generacion import crear_flota_bots, generar_llegadas, generar_usuarios
from boleteria.modelo.motor import MotorSimulacion
from boleteria.modelo.politicas import Captcha, CupoEstricto, crear_politica
from boleteria.modelo.precios import ControladorPrecios
from boleteria.rng.lcg import LCG, PRESETS, crear_generador_base
from boleteria.rng.poisson import LlegadasNHPP
from boleteria.rng.polar import PolarGenerator
from boleteria.rng.rechazo import RechazoGenerator
from boleteria.rng.truncada import InversaGenerator

SEMILLA = 12345
GENERADORES_NORMALES = [PolarGenerator, RechazoGenerator, InversaGenerator]


# --------------------------------------------------------------- capa 1: LCG

def test_lcg_salto_de_bloque_reproduce_el_bucle():
    """
    El salto de bloque es una optimizacion, no un generador distinto: tiene que
    dar EXACTAMENTE la misma secuencia que aplicar la recurrencia una por una.
    """
    for preset in PRESETS:
        escalar = LCG(2024, preset=preset, bloque=10 ** 9)   # nunca salta
        vectorizado = LCG(2024, preset=preset, bloque=256)   # salta 20 veces
        assert np.array_equal(escalar.uniform(5000), vectorizado.uniform(5000)), preset


def test_lcg_soporte_semiabierto():
    """Nunca debe devolver 1.0: romperia cualquier -log(1-u) o inversa aguas arriba."""
    for preset in PRESETS:
        u = LCG(7, preset=preset).uniform(200_000)
        assert u.min() >= 0.0
        assert u.max() < 1.0


def test_lcg_semilla_cero_no_se_pega():
    """Con c = 0 el estado 0 es absorbente; la semilla debe corregirse sola."""
    u = LCG(0, preset="minstd").uniform(100)
    assert len(np.unique(u)) > 50


def test_randu_reprueba_la_serial_3d_y_pasa_las_demas():
    """
    El control defectuoso tiene que comportarse como tal, si no la capa 1 no
    esta probando nada. RANDU pasa en una dimension y falla en tres.
    """
    from boleteria.analisis.validacion import (prueba_rachas, prueba_serial_3d,
                                               prueba_uniformidad)
    u = crear_generador_base("randu", SEMILLA).uniform(300_000)
    assert prueba_uniformidad(u)["pasa"]
    assert prueba_rachas(u)["pasa"]
    assert not prueba_serial_3d(u)["pasa"]

    sano = crear_generador_base("minstd", SEMILLA).uniform(300_000)
    assert prueba_serial_3d(sano)["pasa"]


# ------------------------------------------------- capa 2: normales truncadas

def test_normales_estandar_ajustan():
    for Gen in GENERADORES_NORMALES:
        z = Gen(crear_generador_base("mt", SEMILLA)).normal(100_000)
        assert abs(z.mean()) < 0.02, Gen.__name__
        assert abs(z.std(ddof=1) - 1.0) < 0.02, Gen.__name__
        assert stats.kstest(z, stats.norm().cdf).pvalue > 0.001, Gen.__name__


def test_truncamiento_respeta_las_cotas():
    """Ni un solo valor fuera de [a, b]: es la propiedad que define el metodo."""
    casos = [(500.0, 220.0, 50.0, np.inf), (90.0, 30.0, 15.0, 600.0), (0.0, 1.0, 2.5, 4.0)]
    for Gen in GENERADORES_NORMALES:
        gen = Gen(crear_generador_base("mt", SEMILLA))
        for mu, sigma, a, b in casos:
            x = gen.normal_truncada(20_000, mu, sigma, a, b)
            assert x.min() >= a - 1e-9, (Gen.__name__, a)
            assert x.max() <= b + 1e-9, (Gen.__name__, b)


def test_inversa_es_exacta_en_la_cola_extrema():
    """
    A 8 sigma, P(X > a) ~ 6e-16: ningun metodo de rechazo puede llegar ahi, y
    la inversa escrita con Phi en vez de la supervivencia se romperia por
    cancelacion. Esta prueba es el candado de esa decision numerica.
    """
    gen = InversaGenerator(crear_generador_base("mt", SEMILLA))
    for a in (5.0, 8.0):
        x = gen.normal_truncada(20_000, 0.0, 1.0, a, np.inf)
        teorica = stats.truncnorm(a, np.inf)
        assert x.min() >= a
        assert abs(x.mean() - teorica.mean()) < 0.01, a
        assert stats.kstest(x, teorica.cdf).pvalue > 0.001, a


# ------------------------------------------------------- capa 3: las llegadas

def test_nhpp_no_se_sale_de_su_ventana():
    """
    Los cortes internos del adelgazamiento son fijos (uno en 480 min); si no se
    recortan a la duracion real, una fase corta genera llegadas despues del
    cierre. Paso de verdad, y esta prueba es lo que lo detuvo.
    """
    nhpp = LlegadasNHPP(crear_generador_base("mt", SEMILLA))
    for duracion in (30.0, 60.0, 240.0, 600.0, 1440.0):
        tiempos = nhpp.generar_fase(600.0, duracion, 60_000)
        assert tiempos, duracion
        assert min(tiempos) >= 600.0, duracion
        assert max(tiempos) <= 600.0 + duracion, duracion


def test_nhpp_reproduce_la_integral_de_lambda():
    """E[N(a,b)] tiene que ser la integral de la intensidad sobre la ventana."""
    nhpp = LlegadasNHPP(crear_generador_base("mt", SEMILLA))
    apertura, duracion, pool = 600.0, 240.0, 60_000
    esperado, _ = integrate.quad(lambda t: nhpp.intensidad(t, apertura, pool),
                                 apertura, apertura + duracion, limit=400)

    conteos = [len(nhpp.generar_fase(apertura, duracion, pool)) for _ in range(40)]
    media = float(np.mean(conteos))
    error = float(np.std(conteos, ddof=1) / np.sqrt(len(conteos)))
    assert abs(media - esperado) < 4 * error, (media, esperado, error)


# ------------------------------------------------------------------- el motor

def _correr_fase_simple(slots=500, secciones=None, humanos_solo=True, semilla=5):
    gen = PolarGenerator(crear_generador_base("mt", semilla))
    nhpp = LlegadasNHPP(crear_generador_base("mt", semilla + 1))
    uni = crear_generador_base("mt", semilla + 2)
    motor = MotorSimulacion(gen, crear_generador_base("mt", semilla + 3),
                            slots_concurrentes=slots)
    fase = obtener_fases()[0]
    secciones = secciones if secciones is not None else crear_secciones(25_000)
    llegadas = generar_usuarios(fase, nhpp, gen, uni) if humanos_solo else None
    motor.correr_fase(fase, secciones, ControladorPrecios(eta=0.0), llegadas)
    return motor, secciones, llegadas


def test_la_paciencia_muerde_cuando_hay_cola():
    """
    Con cajas de sobra nadie espera y nadie abandona; al estrangular las cajas
    la espera crece y los abandonos tienen que aparecer. Si esta prueba pasa
    con abandonos = 0 en los tres casos, la paciencia volvio a ser decorativa.
    """
    abandonos, esperas = [], []
    for slots in (500, 100, 20):
        motor, _, _ = _correr_fase_simple(slots=slots)
        resumen = motor.resumen()
        abandonos.append(resumen["abandonos_impaciencia"])
        esperas.append(resumen["espera_promedio"])

    assert abandonos[0] < abandonos[1] < abandonos[2], abandonos
    assert esperas[0] < esperas[1] < esperas[2], esperas
    assert abandonos[2] > 0


def test_nadie_espera_mas_que_su_paciencia():
    """Quien fue atendido no puede haber esperado mas de lo que aguantaba."""
    motor, _, llegadas = _correr_fase_simple(slots=100)
    paciencia_maxima = max(e.paciencia for e in llegadas)
    assert motor.estadisticas["espera_maxima"] <= paciencia_maxima + 1e-9


def test_los_boletos_no_se_pierden():
    """
    Conservacion: todo boleto liberado termina vendido o disponible. Los
    retenidos por pago fallido vuelven al inventario en la ventana de
    devoluciones, asi que al final la cuenta tiene que cerrar exacta.
    """
    from boleteria.modelo.motor import liberar_inventario

    gen = PolarGenerator(crear_generador_base("mt", 9))
    nhpp = LlegadasNHPP(crear_generador_base("mt", 10))
    uni = crear_generador_base("mt", 11)
    motor = MotorSimulacion(gen, crear_generador_base("mt", 12))
    secciones = crear_secciones(0)
    fases = obtener_fases()
    flota = crear_flota_bots(max(f.bots_admitidos for f in fases), gen)

    motor.correr_simulacion_completa(
        fases, secciones, ControladorPrecios(eta=0.10),
        lambda fase: generar_llegadas(fase, flota, nhpp, gen, uni))

    liberado = sum(f.liberacion_inventario for f in fases)
    disponible = sum(sec.inventario for sec in secciones)
    retenido = sum(sec.pagos_fallidos_retenidos for sec in secciones)
    assert retenido == 0, "las devoluciones deben haber reinyectado todo"
    assert disponible + motor.estadisticas["boletos_vendidos"] == liberado
    assert liberado == AFORO_TOTAL


def test_las_secciones_nunca_quedan_en_negativo():
    motor, secciones, _ = _correr_fase_simple(slots=500)
    for sec in secciones:
        assert sec.inventario >= 0, sec.nombre


def test_misma_semilla_mismo_resultado():
    """Sin esto, la comparacion con numeros aleatorios comunes no significa nada."""
    from experimentos.exp2_precios import _correr_replica
    a = _correr_replica(777, eta=0.10)
    b = _correr_replica(777, eta=0.10)
    assert a["ingreso"] == b["ingreso"]
    assert a["boletos_vendidos"] == b["boletos_vendidos"]


# ---------------------------------------------------------------- politicas

def test_el_cupo_estricto_limita_la_compra():
    politica = CupoEstricto(tope=2)

    class Falso:
        cantidad = 4

    assert politica.tope_compra(Falso()) == 2


def test_el_captcha_bloquea_en_las_proporciones_pedidas():
    """
    Un captcha que bloquea a los humanos en la proporcion de los bots (o al
    reves) invalidaria todo el experimento 3.
    """
    gen = PolarGenerator(crear_generador_base("mt", 3))
    nhpp = LlegadasNHPP(crear_generador_base("mt", 4))
    uni = crear_generador_base("mt", 5)
    fase = obtener_fases()[2]
    flota = crear_flota_bots(fase.bots_admitidos, gen)
    llegadas = generar_llegadas(fase, flota, nhpp, gen, uni)

    n_bots = sum(1 for e in llegadas if not hasattr(e, "valuacion"))
    n_humanos = len(llegadas) - n_bots

    politica = Captcha(deteccion=0.90, falso_positivo=0.05)
    _, bloqueos = politica.filtrar(llegadas, fase, uni)

    assert abs(bloqueos["bots"] / n_bots - 0.90) < 0.02
    assert abs(bloqueos["humanos"] / n_humanos - 0.05) < 0.02


def test_la_loteria_conserva_a_todos_y_reordena():
    """La loteria no bloquea a nadie: solo reparte turnos dentro de la ventana."""
    gen = PolarGenerator(crear_generador_base("mt", 3))
    nhpp = LlegadasNHPP(crear_generador_base("mt", 4))
    uni = crear_generador_base("mt", 5)
    fase = obtener_fases()[2]
    flota = crear_flota_bots(fase.bots_admitidos, gen)
    llegadas = generar_llegadas(fase, flota, nhpp, gen, uni)

    politica = crear_politica("loteria")
    admitidas, bloqueos = politica.filtrar(llegadas, fase, uni)

    assert len(admitidas) == len(llegadas)
    assert bloqueos == {"bots": 0, "humanos": 0}

    # Los bots entraban todos en los primeros 5 s; tras el sorteo deben estar
    # repartidos por la ventana de 30 min.
    bots = [e for e in admitidas if not hasattr(e, "valuacion")]
    desfases = [e.t_llegada - fase.hora_apertura for e in bots]
    assert max(desfases) > 20.0, "el sorteo no disperso a los bots"


def test_las_defensas_mejoran_el_acceso_humano():
    """La prueba de humo del experimento 3: sin defensa 0%, con defensa algo mas."""
    from experimentos.exp2_precios import _correr_replica

    base = _correr_replica(1000, eta=0.10)
    assert base["boletos_humanos"] == 0, "la linea base ya no es 'los bots se llevan todo'"

    con_captcha = _correr_replica(1000, eta=0.10, politica=crear_politica("captcha"))
    assert con_captcha["boletos_humanos"] > 0.5 * con_captcha["boletos_vendidos"]


if __name__ == "__main__":
    fallos = 0
    for nombre, funcion in sorted(globals().items()):
        if not nombre.startswith("test_") or not callable(funcion):
            continue
        try:
            funcion()
            print(f"  ok    {nombre}")
        except AssertionError as error:
            fallos += 1
            print(f"  FALLA {nombre}: {error}")
    print(f"\n{'Todo en orden.' if not fallos else f'{fallos} prueba(s) fallaron.'}")
    sys.exit(1 if fallos else 0)
