# Tarea — Descomposición aditiva de series meteorológicas multivariadas

**Curso:** Estadística, Detección de Anomalías e Imputación de Series Temporales
**Bloque 3 · Sesión 1**
**Datos:** `ClimaLab_2023-05-31_2025-06-20.parquet` (estación ESOLMET-IER, Temixco)
**Modalidad:** individual · entrega en notebook (`.ipynb`)
**Valor total:** **75 puntos**

---

## Contexto

En la libreta **010** estudiamos los componentes de una serie temporal y la diferencia conceptual entre la descomposición clásica (`seasonal_decompose`) y **STL**. En la libreta **011** aplicamos ambas a la temperatura de bulbo seco (`tdb`) sobre la base de ClimaLab. En esta tarea repetirás el flujo sobre **tres variables**, dos de las cuales esconden trampas conceptuales que sólo se revelan al intentar descomponerlas. La intención no es sólo ejecutar las funciones, sino **defender en texto** lo que ves.

Toda la tarea se realiza con **modelo aditivo**: $y_t = T_t + S_t + R_t$.

---

## Variables a estudiar

| Símbolo | Variable | Unidad | Rango físico esperado |
|---|---|---|---|
| `tdb` | Temperatura de bulbo seco | °C | $-5$ a $50$ |
| `ws` | Velocidad del viento | m/s | $0$ a $40$ |
| `wd` | Dirección del viento | grados | $0$ a $360$ (circular) |

---

## Parte 1 — Preparación común (5 puntos)

1.1. Carga el archivo Parquet desde `../data/ClimaLab_2023-05-31_2025-06-20.parquet`. Reporta forma del DataFrame, período cubierto y frecuencia de muestreo.

1.2. Construye **dos versiones** del DataFrame: una **horaria** (`df.resample("1h").mean()`) y una **diaria** (`df.resample("1D").mean()`). Justifica en 2 renglones por qué cada escala es la adecuada para el ciclo que vas a extraer.

> *Para `wd` no apliques `.mean()` directamente sobre la columna en este punto. La parte 4 explica por qué.*

---

## Parte 2 — Temperatura `tdb` (20 puntos)

2.1. Aplica `seasonal_decompose` (modelo aditivo) y `STL` (modo robusto) para extraer el **ciclo anual** de `tdb`. Decide tú la escala de muestreo y el `period` adecuados, y justifica la elección en una línea. Grafica ambos resultados.

2.2. Aplica `STL` (modo robusto) para extraer el **ciclo diario** de `tdb` sobre una ventana representativa. Igualmente, decide y justifica la escala y el `period`. Grafica el resultado.

2.3. **Interpretación (texto):**

   - ¿Cuál es la amplitud pico-a-pico del ciclo **anual** medida sobre la componente $S_t$ del STL anual?
   - ¿Cuál es la amplitud del ciclo **diario** medida sobre el STL diario?
   - ¿Cuál es mayor? ¿Tiene sentido físico para Temixco? Argumenta.
   - Compara las componentes de tendencia que obtuvieron `seasonal_decompose` y STL en 2.1. ¿Difieren? ¿Dónde y por qué?

---

## Parte 3 — Velocidad de viento `ws` (20 puntos)

3.1. Repite el flujo de la Parte 2 (ciclo anual con clásica + STL, ciclo diario con STL) sobre `ws`.

3.2. **Inspección visual previa.** Antes de descomponer, grafica la serie diaria. ¿Ves un *abanico* (amplitud que crece con el nivel)? ¿Ves valores negativos en alguna componente del STL?

3.3. **Interpretación crítica:**

   - El modelo aditivo asume amplitud estacional **constante**. Mira la componente $S_t$ del STL anual. ¿Cumple ese supuesto? Si no, ¿por qué crees que falla y qué consecuencia tiene sobre el residual $R_t$?
   - La velocidad del viento es **no negativa por construcción**. ¿Tu descomposición aditiva produce algún valor reconstruido $\hat{T}_t + \hat{S}_t < 0$? Verifica numéricamente y comenta. (Pista: imprime el mínimo de `stl.trend + stl.seasonal`.)
   - ¿En qué horas del día se concentra el máximo de viento según $S_t$? ¿Es consistente con la convección térmica diurna típica de un valle?
   - Concluye en 3 renglones: **¿es `ws` un buen candidato para descomposición aditiva sin transformación?**

---

## Parte 4 — Dirección del viento `wd` (22 puntos · la trampa intencional)

4.1. **Intento ingenuo.** Aplica `df.resample("1h").mean()` sobre `wd` directamente, y luego `STL(period=24, robust=True)` sobre el resultado. Grafica.

4.2. **Diagnóstico del fracaso.** Observa el promedio horario de `wd` que produjiste. Considera el siguiente experimento mental:

   > *Si en una hora el viento sopla mitad del tiempo a 350° y mitad a 10°, ¿cuál es la dirección "promedio" físicamente correcta? ¿Qué reporta `mean()` de pandas?*

   Calcula numéricamente ambos resultados y explica en texto por qué el segundo es **incorrecto**. Esa es la razón por la que la descomposición de la celda 4.1 está condenada de antemano.

4.3. **Solución correcta — descomposición vectorial.** La dirección de viento es una **variable circular**. Un tratamiento estándar es descomponer el viento en sus dos componentes cartesianas:

   $$u = -ws \cdot \sin(\mathrm{wd} \cdot \pi/180), \qquad v = -ws \cdot \cos(\mathrm{wd} \cdot \pi/180)$$

   donde $u$ es la componente Este-Oeste y $v$ la componente Norte-Sur (convención meteorológica: el viento *viene de* `wd`).

   - Calcula $u_t$ y $v_t$ a resolución original.
   - Resamplea $u$ y $v$ a horario por `mean` (esto sí es lícito: son cantidades vectoriales lineales).
   - Aplica STL aditiva con `period=24` a $u$ y a $v$ por separado, sobre una ventana representativa.
   - Reconstruye la dirección estacional como $\hat{\mathrm{wd}}_S = \arctan2(-\hat{u}_S, -\hat{v}_S) \cdot 180/\pi \mod 360$.

4.4. **Interpretación:**

   - ¿En qué dirección sopla típicamente el viento de día y de noche en Temixco según tu descomposición vectorial?
   - ¿Coincide con la circulación valle-montaña esperada (vientos térmicos)? Documéntalo.
   - Compara visualmente los residuales $R_t$ del intento ingenuo (4.1) contra los del enfoque vectorial (4.3). ¿Cuál se parece más a ruido blanco?

---

## Parte 5 — Síntesis (8 puntos)

En **media página de texto**, contesta:

1. ¿Para cuál de las tres variables fue más adecuada la descomposición aditiva tal cual? ¿Para cuál fue la más problemática?
2. Identifica un patrón general: ¿qué propiedades de una variable la hacen apta o no para descomposición aditiva directa? Ofrece al menos dos criterios explícitos.
3. Si tuvieras que detectar **anomalías** sobre cada una de estas tres variables (Bloque 4), ¿usarías el residual $R_t$ de la misma manera en las tres? Explica.

---

## Entregables

- Notebook `tarea_03_descomposicion_<apellido>.ipynb` con código ejecutado, gráficas visibles y celdas markdown para las interpretaciones.
- Las interpretaciones cuentan **igual o más** que el código. Una tarea con todas las gráficas y sin texto interpretativo recibe la mitad del puntaje de la parte correspondiente.

## Distribución del puntaje

| Parte | Tema | Puntos |
|---|---|---:|
| 1 | Preparación común | 5 |
| 2 | Descomposición de `tdb` | 20 |
| 3 | Descomposición de `ws` | 20 |
| 4 | `wd`: trampa circular y solución vectorial | 22 |
| 5 | Síntesis | 8 |
| | **Total** | **75** |

## Pistas y advertencias

- Trabaja desde el inicio sólo con segmentos **continuos** de la serie. Si tu ventana elegida tiene un gap mayor a unas pocas horas, escoge otra; STL no maneja `NaN` interiores.
- Para `wd`, `np.arctan2` espera los argumentos en orden `(y, x)`. Revisa la convención meteorológica al implementar.
- No olvides convertir grados a radianes antes de pasar a `sin`/`cos`.
- En la parte 5, el patrón general que se busca es: descomposición aditiva funciona bien cuando la variable es **lineal, sin restricciones de signo, con amplitud estacional aproximadamente constante**. `tdb` cumple las tres; `ws` viola dos; `wd` no es lineal en absoluto.

## Fecha límite

Entregar antes del inicio de la sesión 3. Tareas tardías reciben penalización del 10 % por día.
