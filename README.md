# Quant Portfolio Management

Monorepositorio para las prácticas y el proyecto final de la asignatura.
El paquete Python se llama `quantmgmt`.

## Estructura

```text
quant-management/
├── README.md
├── pyproject.toml
├── uv.lock                  # Versiones exactas de las dependencias
├── .python-version          # Python 3.11 para el entorno del curso
├── .gitignore
├── quantmgmt/
│   ├── __init__.py
│   ├── data/                # Descarga y caché de precios
│   ├── risk/                # Riesgo (P1)
│   ├── analytics/           # Factores y atribución (P2)
│   ├── optimizers/          # MPT, Black-Litterman y HRP (P2-P3)
│   └── backtest/            # Prácticas posteriores
├── notebooks/
│   └── 01_riesgo.ipynb
├── tests/
└── data/
    └── cache/              # Local; excluida de Git
```

Los subpaquetes están preparados para añadir las implementaciones de cada
práctica. `quantmgmt/data/` contiene código; `data/cache/` almacena los datos
descargados.

## Instalación reproducible

Requisitos: Git y `uv` (configuración validada con uv 0.10.0). Si ya tienes
Python y pip, puedes instalar esa versión de uv con:

```bash
python3 -m pip install --user uv==0.10.0
```

Clona el repositorio y ejecuta desde su raíz:

```bash
git clone https://github.com/hectoorperezz/quant-management.git
cd quant-management
uv sync --locked
```

Si ya tienes el repositorio clonado, empieza por `cd quant-management`.
`uv sync --locked` crea `.venv`, instala `quantmgmt` en modo editable y las
dependencias de desarrollo, y respeta las versiones de `uv.lock`. uv utiliza
Python 3.11 según `.python-version` y puede descargarlo si no está disponible.
La carpeta `data/cache/` se crea sola la primera vez que se descargan precios.

## Ejemplo de uso

Comprueba que el paquete se importa correctamente:

```bash
uv run --locked python -c "from quantmgmt import data, risk, analytics, optimizers, backtest; print('quantmgmt listo')"
```

Descarga precios, pásalos a retornos y compara varias carteras:

```python
from quantmgmt import risk
from quantmgmt.data import download_prices

# Precios diarios ajustados de Yahoo Finance, guardados en data/cache/.
prices = download_prices(
    ["SPY", "TLT", "GLD", "HYG", "BRK-B"], start="2007-05-01", end="2026-09-29"
)
returns = risk.to_returns(prices)

risk.sharpe_ratio(returns)       # un valor por activo
risk.max_drawdown(returns)       # caídas en positivo: 0.55 = -55 %
risk.scare_probability(returns)  # métrica propia

# Tabla con todas las métricas, una fila por activo.
risk.risk_summary(returns, benchmark=returns["SPY"])
```

El análisis completo de la práctica 1 está en el notebook:

```bash
uv run --locked jupyter lab notebooks/01_riesgo.ipynb
```

## Módulo de riesgo

`quantmgmt.risk` reúne las métricas de la práctica 1. Todas reciben una
`pd.Series` o un `pd.DataFrame` de retornos simples indexados por fecha y
devuelven un número o un valor por columna, para comparar carteras.

| Familia | Funciones |
| --- | --- |
| Rentabilidad | `to_returns`, `cumulative_returns`, `cagr`, `annualized_return` |
| Dispersión | `annualized_volatility`, `downside_deviation`, `rolling_volatility` |
| Caídas desde máximos | `drawdown_series`, `max_drawdown`, `time_under_water`, `recovery_time` |
| Ratios | `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `tracking_error` |
| Cola | `var_historical`, `var_parametric`, `cvar_historical`, `cvar_parametric`, `skewness`, `kurtosis` |
| Métrica propia | `scare_probability`, `worst_loss_ahead` |
| Resumen | `risk_summary` |

Las caídas, el VaR y el CVaR se expresan en positivo, como pérdidas. Las
convenciones completas están en el docstring de `quantmgmt/risk/__init__.py`.

**Métrica propia: probabilidad de susto.** Si un inversor entra en una fecha
cualquiera, ¿qué probabilidad hay de que llegue a perder más de lo que
soporta (20 % por defecto) durante el año siguiente? Recorre todas las fechas
de entrada posibles y depende del perfil del inversor (`loss_limit` y
`horizon`), por lo que permite comparar carteras para un cliente concreto.
Está documentada en `quantmgmt/risk/custom.py`.

## Desarrollo y pruebas

Implementa la lógica reutilizable en `quantmgmt/`, documenta los experimentos
en `notebooks/` y añade sus pruebas en `tests/`. Para ejecutarlas:

```bash
uv run --locked pytest
```

Para añadir una dependencia, usa `uv add nombre-paquete` o
`uv add --dev nombre-paquete` y versiona juntos `pyproject.toml` y `uv.lock`.
Conserva las semillas de los experimentos y documenta las fuentes y fechas de
los datos para poder reproducir los análisis.

## Uso de Git

Haz commits pequeños por cambio coherente: por ejemplo, una función de riesgo
junto con sus pruebas, o un experimento documentado en el notebook.
Revisa los archivos antes de incluirlos:

```bash
git status
git diff
git add quantmgmt/risk/ tests/
git commit -m "feat(risk): añadir volatilidad anualizada y sus pruebas"
```

La caché de datos, `.venv` y los checkpoints de Jupyter quedan fuera del
historial.
