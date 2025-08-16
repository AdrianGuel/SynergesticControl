# SynergesticControl

Este repositorio forma parte del curso **Control Sinérgico** disponible en: https://sites.google.com/view/elingedecontrol/cursos-y-talleres/control-sin%C3%A9rgico?authuser=0

La automatización es una de las áreas de la ingeniería moderna con mayor impacto en nuestras vidas, y su avance parece no tener límites. Aunque a primera vista pueda parecer un tema complejo, en este curso te guiaremos paso a paso para que desarrolles un proyecto completo de ingeniería de control, utilizando una metodología que llamamos control sinérgico.

El control sinérgico es una metodología que combina herramientas de análisis geométrico y métodos de cómputo suave para la optimización de modelos y controladores. Así, mientras la teoría de control nos proporciona técnicas para analizar la estabilidad de los sistemas, las metaheurísticas nos permiten refinar los controladores estabilizantes para alcanzar un mejor desempeño.

Para implementar esta metodología, durante el taller exploraremos conceptos fundamentales como la función de transferencia, los controladores PID y los sistemas en lazo cerrado. Posteriormente, abordaremos temas más especializados como la identificación de sistemas, el análisis de estabilidad mediante métodos geométricos, y el diseño de controladores utilizando algoritmos metaheurísticos.

Lo mejor de todo es que las ideas y técnicas que aprenderás aquí tienen aplicación en múltiples áreas de la ingeniería siempre que exista un problema de regulación: motores, convertidores, bioreactores, robots, drones, baterías, entre muchos otros.

A lo largo del taller utilizamos el lenguaje de programación Python para realizar todos nuestros análisis.

## Características

## Instalación y Configuración

Este proyecto utiliza [Poetry](https://python-poetry.org/) para la gestión de dependencias. Asegúrate de tener Poetry instalado en tu sistema.

### Prerrequisitos

- Python 3.11 o superior
- Poetry (para gestión de dependencias)

### Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/AdrianGuel/SynergesticControl.git
   cd SynergesticControl
   ```

2. Instala las dependencias usando Poetry:
   ```bash
   poetry install
   ```

3. Activa el entorno virtual:
   ```bash
   poetry shell
   ```

## Uso

### Ejecutando el Ejemplo

Para ver el ajuste por mínimos cuadrados en acción:

```bash
poetry run python sysIdent/leastsquares.py
```

Esto:
1. Generará datos lineales sintéticos con ruido
2. Ajustará modelos lineales y polinomiales cúbicos
3. Mostrará visualizaciones interactivas de Plotly que muestran:
   - Puntos de datos originales vs. curvas ajustadas
   - Gráficos de residuos para validación del modelo

### Usando como Módulo

Puedes importar y usar las funciones en tus propios scripts:

```python
from sysIdent.leastsquares import solve_least_squares, linear_basis, polynomial_basis, visualize_fit_plotly
import numpy as np

# Tus datos
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2.1, 4.2, 5.8, 8.1, 10.2])

# Ajustar un modelo lineal
theta, Y_hat, residuals = solve_least_squares(X, Y, linear_basis())
print("Parámetros del ajuste lineal:", theta)

# Visualizar los resultados
fig = visualize_fit_plotly(X, Y, linear_basis(), theta, title="Mi Ajuste Lineal")
fig.show()
```

## Desarrollo

### Agregando Dependencias

Para agregar nuevas dependencias:

```bash
poetry add nombre_paquete
```

Para dependencias de desarrollo:

```bash
poetry add --group dev nombre_paquete
```

## Estructura del Proyecto

```
SynergesticControl/
├── pyproject.toml          # Configuración de Poetry y dependencias
├── poetry.lock             # Versiones de dependencias bloqueadas
├── README.md              # Este archivo
└── sysIdent/              # Paquete principal
    └── leastsquares.py    # Utilidades de estimación por mínimos cuadrados
```

## Dependencias

- **NumPy** (>=2.3.2, <3.0.0): Computación numérica y álgebra lineal
- **Plotly** (>=6.3.0, <7.0.0): Visualización interactiva

## Autor

**AdrianGuel** - adrianjguelc@gmail.com

---

*Este repositorio es parte del curso de Control Sinérgico. Para más información sobre el curso, visita: https://sites.google.com/view/elingedecontrol/cursos-y-talleres/control-sin%C3%A9rgico?authuser=0*
