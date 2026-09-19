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
uv run --locked python -c "from pathlib import Path; Path('data/cache').mkdir(parents=True, exist_ok=True)"
```

Si ya tienes el repositorio clonado, empieza por `cd quant-management`.
`uv sync --locked` crea `.venv`, instala `quantmgmt` en modo editable y las
dependencias de desarrollo, y respeta las versiones de `uv.lock`. uv utiliza
Python 3.11 según `.python-version` y puede descargarlo si no está disponible.
La carpeta de caché debe crearse en cada clon porque Git no la versiona.

## Ejemplo de uso

Comprueba que el paquete se importa correctamente:

```bash
uv run --locked python -c "from quantmgmt import data, risk, analytics, optimizers, backtest; print('quantmgmt listo')"
```

Abre el notebook de la primera práctica:

```bash
uv run --locked jupyter lab notebooks/01_riesgo.ipynb
```

El notebook incluye este ejemplo de rentabilidades sintéticas en formato
decimal, con una semilla fija para repetir los resultados:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
returns = pd.DataFrame(
    rng.normal(loc=0.0005, scale=0.01, size=(252, 3)),
    columns=["Activo A", "Activo B", "Activo C"],
)
print(returns.head())
```

Son datos de demostración: no representan precios ni rentabilidades reales.
Las funciones de riesgo se incorporarán a `quantmgmt/risk/` al desarrollar P1.

## Desarrollo y pruebas

Implementa la lógica reutilizable en `quantmgmt/`, documenta los experimentos
en `notebooks/` y añade sus pruebas en `tests/`. Una vez existan pruebas:

```bash
uv run --locked pytest
```

La estructura inicial todavía no contiene pruebas; en ese estado pytest
informa de que no ha recogido ninguna y devuelve el código 5.

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

Este mensaje es un ejemplo para cuando se implemente esa función. La caché de
datos, `.venv` y los checkpoints de Jupyter quedan fuera del historial.
