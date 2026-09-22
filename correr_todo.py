"""
Corre el proyecto completo y deja todo en resultados/.

Es el punto de entrada unico: valida los generadores, corre los tres
experimentos y produce todas las figuras y CSV que usa el informe. El orden no
es arbitrario -- la validacion va primero porque si un generador no pasa, los
experimentos que se apoyan en el no valen nada.

Uso:
    python correr_todo.py                 # todo (unos 15 min)
    python correr_todo.py --rapido        # menos replicas, para revisar que corre
    python correr_todo.py --solo exp3     # una sola etapa
"""
import argparse
import io
import os
import sys
import time
from contextlib import redirect_stdout

DIR_SALIDA = "resultados"
BITACORA = os.path.join(DIR_SALIDA, "salida_completa.txt")


def _encabezado(titulo):
    print("\n" + "#" * 78)
    print(f"# {titulo}")
    print("#" * 78)


def etapa_validacion(rapido=False):
    from boleteria.analisis import validacion
    if rapido:
        validacion.N_POOL = 100_000
        validacion.N_UNIFORMES = 100_000
    validacion.main()


def etapa_figuras(rapido=False):
    from boleteria.analisis import graficas
    if rapido:
        graficas.N_PLOT = 20_000
    graficas.generar_figuras()
    graficas.figura_reticulo(n_ternas=8_000 if rapido else 30_000)


def etapa_exp1(rapido=False):
    from experimentos import exp1_metodos
    if rapido:
        exp1_metodos.N_CALIDAD = 50_000
        exp1_metodos.N_BENCHMARK = 200_000
        exp1_metodos.N_POR_CORTE = {k: max(500, v // 10)
                                    for k, v in exp1_metodos.N_POR_CORTE.items()}
    exp1_metodos.main()


def etapa_exp2(rapido=False):
    from experimentos import exp2_precios
    n = 20 if rapido else exp2_precios.N_REPLICAS
    print(f"Corriendo {n} replicas (precio fijo vs dinamico, CRN)...")
    filas = exp2_precios.correr_experimento(n)
    resumen = exp2_precios.analizar(filas)
    exp2_precios.guardar_csv(filas)
    exp2_precios.graficar(filas, resumen)


def etapa_exp3(rapido=False):
    from experimentos import exp3_politicas
    n = 10 if rapido else exp3_politicas.N_REPLICAS
    print(f"Corriendo {n} replicas x 5 politicas (CRN)...")
    filas = exp3_politicas.correr_experimento(n)
    exp3_politicas.analizar(filas)
    malla, barrido = exp3_politicas.barrido_captcha(5 if rapido else
                                                    exp3_politicas.REPLICAS_BARRIDO)
    exp3_politicas.guardar_csv(filas, barrido)
    exp3_politicas.graficar(filas, malla)


def etapa_exp4(rapido=False):
    from experimentos import exp4_motores
    n = 5 if rapido else exp4_motores.N_REPLICAS
    print(f"Corriendo {n} replicas x 5 politicas x 3 motores (CRN, precio fijo)...")
    filas = exp4_motores.correr_experimento(n)
    exp4_motores.analizar(filas)
    exp4_motores.guardar_csv(filas)
    exp4_motores.graficar(filas)


def etapa_tablero(rapido=False):
    from boleteria.tablero import vivo
    datos, resumen = vivo.correr_y_grabar()
    vivo.imprimir_resumen(resumen)
    vivo.figura_estatica(datos, resumen)
    if not rapido:
        vivo.animar(datos, resumen, ruta_gif=os.path.join(DIR_SALIDA, "tablero.gif"))


ETAPAS = {
    "validacion": ("Validacion estadistica (3 capas)", etapa_validacion),
    "figuras": ("Figuras de validacion (Q-Q, histogramas, reticulo)", etapa_figuras),
    "exp1": ("Experimento 1: comparacion de metodos", etapa_exp1),
    "exp2": ("Experimento 2: precio fijo vs dinamico", etapa_exp2),
    "exp3": ("Experimento 3: politicas anti-bot", etapa_exp3),
    "exp4": ("Experimento 4: sensibilidad al motor", etapa_exp4),
    "tablero": ("Tablero de la presentacion", etapa_tablero),
}


def main():
    parser = argparse.ArgumentParser(description="Corre el proyecto completo")
    parser.add_argument("--rapido", action="store_true",
                        help="menos replicas y muestras; sirve para verificar que todo corre")
    parser.add_argument("--solo", choices=list(ETAPAS),
                        help="corre una sola etapa")
    args = parser.parse_args()

    os.makedirs(DIR_SALIDA, exist_ok=True)
    etapas = {args.solo: ETAPAS[args.solo]} if args.solo else ETAPAS
    # Una corrida parcial no debe sobrescribir la bitacora de la corrida
    # completa: se guarda con su propio nombre.
    bitacora = (os.path.join(DIR_SALIDA, f"salida_{args.solo}.txt")
                if args.solo else BITACORA)

    # Todo lo que se imprime se guarda ademas en una bitacora, porque las
    # tablas de los experimentos son parte de los resultados del informe y
    # perderlas en el scroll de la terminal seria tonto.
    buffer = io.StringIO()

    class Doble:
        def write(self, texto):
            sys.__stdout__.write(texto)
            buffer.write(texto)

        def flush(self):
            sys.__stdout__.flush()

    inicio_total = time.time()
    with redirect_stdout(Doble()):
        for clave, (titulo, funcion) in etapas.items():
            _encabezado(titulo)
            inicio = time.time()
            funcion(args.rapido)
            print(f"\n[{clave}] listo en {time.time() - inicio:.1f} s")

        print(f"\n{'=' * 78}")
        print(f"TODO LISTO en {time.time() - inicio_total:.1f} s. "
              f"Resultados en {DIR_SALIDA}/")

    with open(bitacora, "w", encoding="utf-8") as fh:
        fh.write(buffer.getvalue())
    print(f"Bitacora guardada en: {bitacora}")


if __name__ == "__main__":
    main()
