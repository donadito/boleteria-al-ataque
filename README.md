# Boletería bajo ataque

Simulación de la venta en línea de 25,000 localidades de un concierto en el
Estadio Cementos Progreso, con una botnet compitiendo contra compradores
humanos por el inventario.

Proyecto 1 — CC2017 Modelización y Simulación.

## Qué se simula

Tres fases de venta en días distintos (Preventa Fan Verificado, Preventa Banco
Aliado, Venta General), cada una con su población elegible y su liberación de
inventario, repartido en cuatro secciones (General, Preferencial, VIP, Palco)
que se agotan de forma independiente. El cuello de botella son 500 cajas de
checkout concurrentes. Al cerrar la venta general hay una ventana de
devoluciones que reinyecta los boletos retenidos por pagos fallidos.

El resultado central del modelo: **sin defensas, los bots se llevan el 100% del
inventario**. No es un desbalance de parámetros, es mecánico. Un bot completa
el checkout en 3 s y un humano en 90 s, y la compra se concreta al *terminar*
el checkout: cuando el primer humano acaba de teclear su tarjeta ya no queda
nada.

## Instalación y ejecución

```bash
pip install -r requirements.txt
python correr_todo.py
```

`correr_todo.py` ejecuta la validación, los tres experimentos y todas las
figuras, y deja los resultados en `resultados/`. Toma unos 15 minutos. Para
correr las piezas por separado:

```bash
python -m boleteria.analisis.validacion     # las tres capas de validación
python -m boleteria.analisis.graficas       # Q-Q, histogramas, retículo 3D
python -m boleteria.analisis.benchmark      # velocidad escalar vs vectorizado
python -m experimentos.exp1_metodos         # comparación de métodos de generación
python -m experimentos.exp2_precios         # precio fijo vs dinámico
python -m experimentos.exp3_politicas       # defensas anti-bot
python -m experimentos.exp4_motores         # sensibilidad al motor de simulación
python -m boleteria.tablero.vivo            # tablero animado (la demo en vivo)
```

El tablero acepta opciones:

```bash
python -m boleteria.tablero.vivo --gif                     # guarda el GIF
python -m boleteria.tablero.vivo --politica captcha        # con defensa
python -m boleteria.tablero.vivo --eta 0                   # precio fijo
```

## Estructura

```
boleteria/
  rng/            generación de variables aleatorias
    lcg.py          LCG propio (vectorizado por saltos de bloque) + Mersenne Twister
    base.py         interfaz común de los generadores normales
    polar.py        método polar de Marsaglia
    rechazo.py      aceptación-rechazo con envolvente exponencial
    truncada.py     transformada inversa (estable en la cola)
    poisson.py      proceso de Poisson no homogéneo por adelgazamiento
  modelo/         el sistema simulado
    calendario.py   las tres fases de venta
    entidades.py    usuarios, bots, secciones
    generacion.py   construye las llegadas de cada fase
    motor.py        motor secuencial: dos pasadas sobre arreglos de tiempos
    motor_simple.py variantes reducidas del modelo de atención (experimento 4)
    precios.py      controlador proporcional de precio dinámico
    politicas.py    defensas anti-bot
  analisis/       validación y medición
    validacion.py   las tres capas de validación estadística
    graficas.py     Q-Q plots, histogramas, retículo 3D
    benchmark.py    velocidad de los generadores
  tablero/
    vivo.py         tablero animado para la presentación
experimentos/
  exp1_metodos.py    comparación de métodos de generación
  exp2_precios.py    precio fijo vs dinámico
  exp3_politicas.py  defensas anti-bot
  exp4_motores.py    sensibilidad al motor de simulación
tests/
  test_invariantes.py  candados sobre las propiedades que la validación dio por buenas
correr_todo.py       punto de entrada único
```

Las pruebas se corren con `pytest tests/ -q` (o `python tests/test_invariantes.py`
si no hay pytest). No son cobertura: fijan los invariantes que pueden romperse
en silencio — que el salto de bloque del LCG reproduzca su propia secuencia, que
el truncamiento respete las cotas, que el NHPP no se salga de su ventana, que la
paciencia siga mordiendo, que no se pierdan boletos en la contabilidad y que la
misma semilla dé el mismo resultado (sin eso, los números aleatorios comunes no
significan nada).

## Cómo se procesa la simulación

El motor **no** usa un calendario de eventos. Sigue lo que plantea la consigna
del curso: se generan los arreglos de tiempos y se procesan secuencialmente, en
dos pasadas.

1. **Asignación de cajas.** Se ordenan las llegadas por tiempo y se recorren.
   Las 500 cajas de checkout son un arreglo `libre_en[k]` con la hora a la que
   se desocupa cada una; a cada persona se le da la que se desocupa primero, y
   su checkout empieza en `max(su llegada, esa hora)`. La diferencia entre
   ambas es su espera, y es lo que se compara contra su paciencia.

2. **Compra.** La pasada anterior deja un arreglo de tiempos de finalización.
   Se ordena y se recorre: cada quien compra cuando *termina* de pagar, no
   cuando empieza. Esa distinción es la que le da sentido al ataque, porque el
   bot paga en 3 s y el humano en 90 s.

El precio dinámico se recalcula avanzando por ventanas conforme se recorre el
arreglo de finalizaciones, sin necesidad de un reloj global.

Una advertencia sobre la estadística de abandonos: la pasada 1 forma la fila
con todas las llegadas de la fase, porque no puede saber de antemano cuándo se
agota el inventario — eso lo decide la pasada 2. Por eso la mayoría de los
abandonos son de gente que hacía fila cuando ya no quedaba nada. La asignación
de boletos no cambia, pero la cifra sí, así que se reportan por separado:
`abandonos_con_inventario` es el número que importa.

## Métodos de generación implementados

| Método | Uso | Aceptación teórica |
|---|---|---|
| LCG propio (MINSTD) | generador uniforme base | — |
| Polar de Marsaglia | normales | π/4 ≈ 0.785 |
| Aceptación-rechazo | normales | √(π/2e) ≈ 0.760 |
| Robert (1995) | normales truncadas en la cola | depende de la cota, no colapsa |
| Transformada inversa | normales truncadas | 1 (no rechaza) |
| Adelgazamiento | proceso de Poisson no homogéneo | — |

Polar y aceptación-rechazo están implementados en versión escalar y vectorizada
por lotes, para poder comparar los dos regímenes. La transformada inversa no
tiene versión escalar porque no rechaza nada: es vectorizada por construcción.

## Validación

Tres capas, en orden, porque cada una solo tiene sentido si la de abajo pasa:

1. **Uniformes base** — χ² de uniformidad, rachas, Ljung-Box y prueba serial en
   3D. Se incluye RANDU como control defectuoso: pasa las tres primeras y
   revienta en la serial 3D, que es lo que demuestra que el banco de pruebas
   tiene poder real. La figura `reticulo_3d.png` lo muestra visualmente.
2. **Distribuciones generadas** — KS sobre submuestras de 20,000 (para evitar
   el falso rechazo que produce KS con millones de datos) y χ² con celdas
   equiprobables, contra las teóricas de scipy.
3. **Modelo completo contra solución analítica** — el número de llegadas del
   NHPP contra la integral de λ(t), su índice de dispersión contra la Poisson,
   y la forma condicional de los tiempos por KS.

## Qué responde cada experimento

**Experimento 1 — ¿qué método de generación conviene?** La calidad no decide:
los tres pasan KS (D ≈ 0.001–0.002). El costo sin truncar favorece a Polar, que
consume 1.27 uniformes por normal contra 3.95 del rechazo, y las fracciones de
aceptación medidas confirman la derivación teórica con error < 0.2 % (0.7853 vs
π/4, 0.7589 vs √(π/2e)).

Pero las tres variables del modelo son normales *truncadas*, y ahí el ranking
se invierte. En una cota de 4σ el descarte ingenuo de Polar gasta **55,158
uniformes por muestra**, contra 2.1 del rechazo con envolvente de Robert y 1.0
de la transformada inversa; en 5σ Polar ya no es ejecutable. Palco exige
muestrear más allá de 3σ, así que el método más rápido en el caso común es el
inutilizable en el caso que el modelo necesita. El criterio que sale del
experimento:

| Situación | Método |
|---|---|
| Sin truncar, o cota cerca de la media | Polar |
| Truncado en la cola | Inversa (1 uniforme, exacta hasta 10σ) |
| Truncado sin poder invertir la CDF | Rechazo con Robert (1995) |

**Experimento 2 — ¿sirve el precio dinámico?** 200 réplicas con números
aleatorios comunes y IC pareado. Sube el ingreso de forma significativa, pero
no cambia *quién* compra: con o sin él, los bots se llevan todo. El precio es
una palanca de recaudación, no de acceso.

**Experimento 4 — ¿cuánto detalle del modelo de atención hace falta?**
Representar las 500 cajas cuesta código y cuesta explicarlo, así que se mide si
aporta: se comparan tres versiones del mismo recorrido secuencial (ordenar por
llegada; ordenar por llegada + checkout; representar además las cajas y la
espera) con las mismas semillas. La respuesta tiene dos mitades: los números
casi no cambian (divergencia máxima ~4 puntos porcentuales, y solo bajo la
lotería), pero las preguntas que se pueden hacer sí — sin cajas no hay espera,
y sin espera la paciencia queda sin efecto y el precio dinámico no es
representable.

**Experimento 3 — ¿qué defensa funciona y a costa de qué?** Cuatro políticas
contra la línea base, con CRN e IC pareados. La métrica principal es la
fracción de boletos que llega a humanos; el ingreso, los abandonos por
impaciencia y los humanos bloqueados por error son el costo. Las cuatro mejoran
el acceso de forma significativa, pero muy desigual: el captcha lleva de 0 % a
78 % a costa de Q 10.7 M de ingreso y 51,558 abandonos, mientras que la lotería
de turno solo llega a 11 %.

Esa cifra de la lotería esconde una **interacción entre políticas** que vale la
pena separar. El experimento 3 corre con precio dinámico, y bajo precio fijo la
misma lotería rinde **45 %**. La diferencia son 17,830 humanos que llegan a la
caja, encuentran el precio en el tope de 3× y se van con excedente negativo
(`salidas_por_precio`; con precio fijo son 1,495). Es decir: el precio dinámico
cancela la mayor parte del beneficio de la lotería, porque expulsa por precio a
los mismos humanos que la lotería acababa de dejar entrar. Las dos políticas se
estorban, y medirlas por separado lo habría ocultado.

El barrido detección/falso positivo da un resultado contraintuitivo que vale la
pena mirar: **la tasa de falso positivo no cuesta acceso hasta el 60 %**. Hay
unas 130,000 llegadas humanas compitiendo por 25,000 boletos, así que bloquear
a uno de cada cinco humanos sigue dejando muchos más humanos que inventario. El
codo aparece en 90 %, cuando la oferta humana por fin cae por debajo del
inventario. Lo que el falso positivo sí cuesta es experiencia de usuario, y por
eso se reporta aparte.

## Reproducibilidad

Todas las corridas usan semillas fijas. Cada réplica de los experimentos separa
cuatro flujos aleatorios derivados de su semilla (normales, llegadas,
generación, pagos) para que el desfase de uno no contamine a los demás, que es
lo que hace válida la comparación con números aleatorios comunes.
