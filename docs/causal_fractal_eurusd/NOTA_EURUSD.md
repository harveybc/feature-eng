# Fractales causales de EURUSD H1 (candidatas de selección)

Estas columnas son la versión para nuestro instrumento, EURUSD horario, de los fractales causales que Retsu calculó para ETHUSDT 4h (`agent-multi/tools/causal_fractal_features.py`, sha256 `703ffb5afb1d76073fdafeeb5f1e7f6ea27916f3b7777ee5e6f12ba72e09cef4`, archivo sin seguimiento en ese checkout al momento del porte). El porte vive en feature-eng como plugin `causal_fractal` (`app/plugins/causal_fractal.py`), con la ventana como parámetro, y reproduce bit a bit los valores del módulo original en ventana 200.

Cada fila usa solo las `w` barras de `close` observadas hasta esa fila, inclusive. No hay un ajuste global, ni sobre validación ni sobre test. La ventana se cuenta en barras observadas, no en tiempo de calendario: 200 barras horarias cubren más de 200 horas de reloj por los fines de semana y por las horas que faltan en el lago.

## Fuente y corte

- Lago admitido: `market_data/forex/g10/eurusd/1h.parquet`, sha256 `c6628375b2872f16dcdbd6b1b90ed8e68bc5bdd852efdef01f2515b8712d4e2e`, solo lectura, filtrado al leer. Ninguna lectura puede alcanzar `2025-01-01`; el cargador rechaza la petición y vuelve a verificar el marco leído.
- Corte del negocio: TRAIN `[2020-01-01, 2024-01-01)`, VALIDATION `[2024-01-01, 2025-01-01)`. TEST (2025) queda CERRADO: no existe archivo de test.
- Calentamiento: cada año calendario se calcula con historia desde el 1 de diciembre anterior (entre 482 y 545 barras, siempre más que la ventana mayor) y esas filas de calentamiento se descartan, nunca se emiten. Por eso TRAIN sale completo desde su primera fila, a diferencia de la nota de ETH, donde las primeras 199 filas quedaban vacías.

| Ventana | Archivo TRAIN | Filas | Archivo VALIDATION | Filas | Filas con las cinco candidatas |
| ---: | --- | ---: | --- | ---: | ---: |
| 200 | `eurusd_h1_causal_fractal_w200_train.parquet` | 24097 | `eurusd_h1_causal_fractal_w200_validation.parquet` | 6243 | todas |
| 96 | `eurusd_h1_causal_fractal_w96_train.parquet` | 24097 | `eurusd_h1_causal_fractal_w96_validation.parquet` | 6243 | todas |
| 48 | `eurusd_h1_causal_fractal_w48_train.parquet` | 24097 | `eurusd_h1_causal_fractal_w48_validation.parquet` | 6243 | todas |

Los arreglos están FUERA del lago y FUERA de Git, en `/var/tmp/features-candidates/eurusd_h1_causal_fractal/` (más `chunks/w<w>/<año>.parquet`, los trozos anuales intermedios). Lo que se versiona es `MANIFEST.json` (hashes de cada parquet, ventana, sha de la fuente y del módulo de Retsu, filas y medianas por split) y este documento. Una copia idéntica del manifiesto queda junto a los arreglos.

## Columnas

Candidatas para una observación: `hurst_rs`, `fractal_dimension_graph` (es `2 - hurst_rs`), `box_counting_dimension`, `variogram_hurst` y `hurst_spectrum_width`. Solo diagnóstico: `correlation_dimension` y `spectral_slope`, causales también pero frágiles en una ventana corta. Intermedias auditables: `variogram_slope`, `hurst_q1..q3`. Cada columna lleva el sufijo `_w<ventana>`.

## Medianas por tramo

| Ventana | Tramo | hurst_rs | variogram_hurst | box_counting_dimension | hurst_spectrum_width |
| ---: | --- | ---: | ---: | ---: | ---: |
| 200 | TRAIN | 0.604 | 0.482 | 1.247 | 0.090 |
| 200 | VALIDATION | 0.601 | 0.472 | 1.240 | 0.116 |
| 96 | TRAIN | 0.597 | 0.466 | 1.100 | 0.095 |
| 96 | VALIDATION | 0.598 | 0.447 | 1.097 | 0.114 |
| 48 | TRAIN | 0.629 | 0.425 | 0.902 | 0.109 |
| 48 | VALIDATION | 0.630 | 0.415 | 0.900 | 0.120 |

Como en ETH, el rango reescalado queda sesgado hacia arriba frente al variograma (0.604 contra 0.482 en 200 barras; la brecha crece al acortar la ventana: 0.629 contra 0.425 en 48). Es una razón para mirar las dos estimaciones juntas, no para declarar persistencia.

## Pruebas (tests/test_causal_fractal.py, 21 pasan con la tienda presente)

- Perturbación futura: alterar las barras posteriores a `i` (salto, ruido, inversión) no cambia la fila `i`, en las tres ventanas; y sí cambia las filas posteriores.
- Vacío de calentamiento: las primeras `w-1` filas son NaN y el resto es finito; ventana por debajo de 33 se rechaza.
- Ventana en barras: la misma serie con un hueco de dos días en el índice produce valores idénticos.
- Réplica al estilo de Retsu sobre una fila real de VALIDATION 2024 por ventana: la fila almacenada se recalcula desde sus propios `w` cierres con una aritmética independiente en Python puro (R/S y variograma) y coincide con el valor guardado.

## Redundancia (solo TRAIN; ver REDUNDANCY.md)

Las correlaciones con rv5, rv24 y ATR/close son pequeñas (|r| ≤ 0.13 para las candidatas salvo `hurst_spectrum_width`, cuya r con rv24 llega a 0.23–0.28). `fractal_dimension_graph` es el negativo exacto de `hurst_rs` (r = −1): una de las dos sobra. `hurst_rs` y `variogram_hurst` comparten r = 0.46 en 200 barras y solo 0.24 en 48. Las columnas varían dentro del día: en 200 barras, entre el 4 % (`variogram_hurst`) y el 30 % (`box_counting_dimension`) de la varianza ocurre dentro del mismo día; en 48 barras, entre el 22 % y el 55 %.

## Límites

- Ventana 48: R/S usa tres escalas (8, 16, 32) en vez de cuatro, y `correlation_dimension` es NaN en toda la ventana (necesita ≥ 60 barras). Retsu exigía ventana ≥ 64; aquí el mínimo es 33 y queda registrado.
- El lago tiene huecos de 2–3 horas repartidos entre febrero y julio de 2023 (5409 barras en 2023 contra ~6230 en los demás años); en esos meses una ventana de `w` barras observadas abarca más tiempo de reloj.
- El reloj de las barras es el de la etiqueta del proveedor (semántica bajo reparación en todo el programa); aquí la etiqueta se usa aritméticamente.
- Los hashes de los parquet dependen de la versión de pyarrow/pandas que los escribió (metadatos); el contenido se verifica por la réplica, no por el hash del archivo.

Estas columnas no seleccionan una característica, no reclaman valor predictivo y no autorizan una estrategia. Sirven para repetir el mismo modelo con y sin ellas.
