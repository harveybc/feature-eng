# Redundancia de los fractales causales EURUSD H1 (solo TRAIN 2020–2023)

Agregados únicamente. Correlaciones de Pearson / Spearman con la volatilidad realizada (rv5, rv24), con la razón ATR(14)/close y entre sí; medianas; y la parte de la varianza que ocurre dentro del día. Nada aquí mide valor predictivo.

## Ventana 200 barras (filas TRAIN: 24097)

| columna | mediana | r rv5 | ρ rv5 | r rv24 | ρ rv24 | r atr_ratio | ρ atr_ratio | varianza intradía | std intradía media | std total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hurst_rs | 0.604 | -0.057 | -0.060 | -0.092 | -0.093 | -0.127 | -0.116 | 0.240 | 0.0244 | 0.0537 |
| fractal_dimension_graph | 1.396 | 0.057 | 0.060 | 0.092 | 0.093 | 0.127 | 0.116 | 0.240 | 0.0244 | 0.0537 |
| box_counting_dimension | 1.247 | 0.024 | 0.051 | -0.000 | 0.023 | 0.037 | 0.046 | 0.300 | 0.0199 | 0.0458 |
| variogram_hurst | 0.482 | -0.087 | -0.086 | -0.094 | -0.125 | -0.082 | -0.114 | 0.035 | 0.0087 | 0.0662 |
| hurst_spectrum_width | 0.090 | 0.106 | 0.059 | 0.227 | 0.187 | 0.154 | 0.175 | 0.051 | 0.0056 | 0.0398 |
| correlation_dimension | 0.970 | -0.021 | -0.011 | -0.019 | 0.019 | -0.046 | -0.003 | 0.085 | 0.0433 | 0.1624 |
| spectral_slope | 0.020 | 0.040 | 0.028 | 0.044 | 0.039 | 0.059 | 0.039 | 0.114 | 0.0459 | 0.1662 |

Pearson entre columnas (ventana 200):

| | hurst_rs | fractal_dimension_graph | box_counting_dimension | variogram_hurst | hurst_spectrum_width | correlation_dimension | spectral_slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| hurst_rs | 1.000 | -1.000 | -0.214 | 0.460 | -0.021 | -0.043 | -0.427 |
| fractal_dimension_graph | -1.000 | 1.000 | 0.214 | -0.460 | 0.021 | 0.043 | 0.427 |
| box_counting_dimension | -0.214 | 0.214 | 1.000 | -0.481 | -0.256 | 0.201 | 0.227 |
| variogram_hurst | 0.460 | -0.460 | -0.481 | 1.000 | 0.027 | -0.537 | -0.527 |
| hurst_spectrum_width | -0.021 | 0.021 | -0.256 | 0.027 | 1.000 | 0.012 | -0.028 |
| correlation_dimension | -0.043 | 0.043 | 0.201 | -0.537 | 0.012 | 1.000 | 0.204 |
| spectral_slope | -0.427 | 0.427 | 0.227 | -0.527 | -0.028 | 0.204 | 1.000 |

## Ventana 96 barras (filas TRAIN: 24097)

| columna | mediana | r rv5 | ρ rv5 | r rv24 | ρ rv24 | r atr_ratio | ρ atr_ratio | varianza intradía | std intradía media | std total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hurst_rs | 0.597 | -0.055 | -0.059 | -0.056 | -0.072 | -0.086 | -0.091 | 0.256 | 0.0439 | 0.0984 |
| fractal_dimension_graph | 1.403 | 0.055 | 0.059 | 0.056 | 0.072 | 0.086 | 0.091 | 0.256 | 0.0439 | 0.0984 |
| box_counting_dimension | 1.100 | 0.032 | 0.046 | -0.008 | -0.023 | 0.032 | -0.003 | 0.475 | 0.0263 | 0.0449 |
| variogram_hurst | 0.466 | -0.083 | -0.065 | -0.091 | -0.100 | -0.050 | -0.058 | 0.070 | 0.0196 | 0.1002 |
| hurst_spectrum_width | 0.095 | 0.119 | 0.051 | 0.229 | 0.171 | 0.124 | 0.139 | 0.101 | 0.0108 | 0.0493 |
| correlation_dimension | 1.054 | 0.009 | 0.015 | -0.004 | 0.032 | 0.005 | 0.030 | 0.225 | 0.0992 | 0.2274 |
| spectral_slope | 0.015 | 0.069 | 0.053 | 0.080 | 0.071 | 0.081 | 0.064 | 0.190 | 0.0896 | 0.2496 |

Pearson entre columnas (ventana 96):

| | hurst_rs | fractal_dimension_graph | box_counting_dimension | variogram_hurst | hurst_spectrum_width | correlation_dimension | spectral_slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| hurst_rs | 1.000 | -1.000 | -0.129 | 0.393 | 0.021 | -0.116 | -0.407 |
| fractal_dimension_graph | -1.000 | 1.000 | 0.129 | -0.393 | -0.021 | 0.116 | 0.407 |
| box_counting_dimension | -0.129 | 0.129 | 1.000 | 0.076 | -0.251 | -0.300 | 0.103 |
| variogram_hurst | 0.393 | -0.393 | 0.076 | 1.000 | 0.165 | -0.603 | -0.424 |
| hurst_spectrum_width | 0.021 | -0.021 | -0.251 | 0.165 | 1.000 | -0.074 | -0.069 |
| correlation_dimension | -0.116 | 0.116 | -0.300 | -0.603 | -0.074 | 1.000 | 0.192 |
| spectral_slope | -0.407 | 0.407 | 0.103 | -0.424 | -0.069 | 0.192 | 1.000 |

## Ventana 48 barras (filas TRAIN: 24097)

| columna | mediana | r rv5 | ρ rv5 | r rv24 | ρ rv24 | r atr_ratio | ρ atr_ratio | varianza intradía | std intradía media | std total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hurst_rs | 0.629 | -0.015 | -0.024 | -0.021 | -0.030 | -0.014 | -0.024 | 0.370 | 0.0880 | 0.1570 |
| fractal_dimension_graph | 1.371 | 0.015 | 0.024 | 0.021 | 0.030 | 0.014 | 0.024 | 0.370 | 0.0880 | 0.1570 |
| box_counting_dimension | 0.902 | -0.017 | -0.011 | -0.029 | -0.042 | -0.030 | -0.030 | 0.546 | 0.0359 | 0.0545 |
| variogram_hurst | 0.425 | -0.069 | -0.051 | -0.025 | -0.012 | -0.008 | 0.004 | 0.219 | 0.0559 | 0.1484 |
| hurst_spectrum_width | 0.109 | 0.103 | 0.038 | 0.276 | 0.210 | 0.108 | 0.104 | 0.229 | 0.0235 | 0.0609 |
| correlation_dimension | — | — | — | — | — | — | — | — | — | — |
| spectral_slope | 0.015 | 0.055 | 0.043 | 0.044 | 0.049 | 0.033 | 0.033 | 0.327 | 0.1792 | 0.3724 |

Pearson entre columnas (ventana 48):

| | hurst_rs | fractal_dimension_graph | box_counting_dimension | variogram_hurst | hurst_spectrum_width | correlation_dimension | spectral_slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| hurst_rs | 1.000 | -1.000 | -0.002 | 0.235 | 0.084 | — | -0.381 |
| fractal_dimension_graph | -1.000 | 1.000 | 0.002 | -0.235 | -0.084 | — | 0.381 |
| box_counting_dimension | -0.002 | 0.002 | 1.000 | 0.532 | 0.204 | — | -0.069 |
| variogram_hurst | 0.235 | -0.235 | 0.532 | 1.000 | 0.415 | — | -0.302 |
| hurst_spectrum_width | 0.084 | -0.084 | 0.204 | 0.415 | 1.000 | — | -0.087 |
| correlation_dimension | — | — | — | — | — | — | — |
| spectral_slope | -0.381 | 0.381 | -0.069 | -0.302 | -0.087 | — | 1.000 |

