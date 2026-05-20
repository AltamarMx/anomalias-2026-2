import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 019a · Teoría · Introducción a SARIMA

    Esta libreta es la **parte teórica**. Aquí desarrollamos:

    - La jerarquía **AR → MA → ARMA → ARIMA → SARIMA** como casos
      particulares de una misma ecuación.
    - **Cómo se construye** la fórmula de SARIMA capa por capa.
    - **Qué hace cada uno** de los siete parámetros `(p, d, q)(P, D, Q, s)`.
    - Por qué los polinomios estacional y no estacional se
      **multiplican** (y de dónde sale el famoso *rezago 25*).

    > El objetivo *no* es ajustar un modelo todavía — es construir la
    > intuición que vas a usar al mirar los datos en
    > **`019b_tdb_exploracion.py`** y al fitear el modelo en `020`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A.1 De AR a SARIMA: una jerarquía, una sola ecuación

    SARIMA no aparece de la nada — es la última escala de una familia de modelos
    que se van *acumulando* uno sobre otro. Cada uno responde a un problema que
    el anterior no podía resolver.

    | Modelo | Notación | Idea central | Limitación que motiva el siguiente |
    |---|---|---|---|
    | **AR(p)** | `(p, 0, 0)` | El valor actual es una combinación lineal de sus `p` últimos valores. | Solo modela memoria del *nivel*, no del *ruido*. |
    | **MA(q)** | `(0, 0, q)` | El valor actual depende de los últimos `q` choques aleatorios (errores). | Memoria limitada y poco flexible para tendencias. |
    | **ARMA(p, q)** | `(p, 0, q)` | Combina memoria de niveles **y** de errores. | Solo funciona en series **estacionarias**. |
    | **ARIMA(p, d, q)** | `(p, d, q)` | Aplica `d` diferencias para *quitar la tendencia* antes de aplicar ARMA. | Ignora ciclos repetitivos (diarios, semanales, anuales). |
    | **SARIMA(p, d, q)(P, D, Q)ₛ** | `(p, d, q)(P, D, Q, s)` | ARIMA **más** una copia estacional del AR/I/MA, que opera en saltos de tamaño `s`. | — |

    **Pista de lectura.** Cada celda de la tabla es un caso particular de
    SARIMA con algunos órdenes en cero. Cuando trabajemos con la serie `tdb`,
    todos esos modelos son alcanzables moviendo perillas en un único widget.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A.2 ¿Cómo se construyó SARIMA? Una receta por capas

    La fórmula final intimida, pero **se construye agregando ideas, una a la
    vez**. Aquí está la receta — cada paso resuelve un problema concreto del
    paso anterior.

    ### Paso 1 · AR(p): "lo que pasa hoy se parece a lo que pasó ayer"

    $$
    y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \dots + \phi_p y_{t-p} + \varepsilon_t
    $$

    ### Paso 2 · MA(q): "lo que pasa hoy arrastra choques recientes"

    $$
    y_t = \varepsilon_t + \theta_1 \varepsilon_{t-1} + \dots + \theta_q \varepsilon_{t-q}
    $$

    ### Paso 3 · ARMA(p, q): junta las dos memorias

    $$
    y_t = \underbrace{\phi_1 y_{t-1} + \dots + \phi_p y_{t-p}}_{\text{AR}}
        \;+\;
        \varepsilon_t + \underbrace{\theta_1 \varepsilon_{t-1} + \dots + \theta_q \varepsilon_{t-q}}_{\text{MA}}
    $$

    ARMA es **denso y compacto**, pero exige una serie **estacionaria** (media
    y varianza estables). Casi ninguna serie real lo es de entrada.

    ### Paso 4 · I(d): diferencia para quitar la tendencia

    Si la serie tiene tendencia, ARMA no funciona — la media no es constante.
    Trabajamos con la *diferencia*:

    $$
    \Delta y_t \;=\; y_t - y_{t-1}
    $$

    Si la tendencia es lineal, `d = 1` basta; si es cuadrática, `d = 2`.
    Después aplicamos ARMA sobre $\Delta^d y_t$. Eso es **ARIMA(p, d, q)**.

    ### Paso 5 · Una notación más corta: el operador `L`

    Las ecuaciones de arriba funcionan, pero son largas. Vamos a darles una
    **versión taquigráfica** — la misma idea escrita más corto.

    #### 5a · `L` es solo "un paso atrás"

    $$
    L \, y_t \;\equiv\; y_{t-1}
    \qquad
    L^2 \, y_t \;=\; y_{t-2}
    \qquad
    L^k \, y_t \;=\; y_{t-k}
    $$

    Es decir: aplicar `L` a una observación te da la observación anterior.
    Nada más.

    #### 5b · Reescribimos un ARMA(1, 1) usando `L`

    Vamos paso a paso, sin saltos. Partimos del ARMA(1, 1) en forma normal:

    $$
    y_t \;=\; \phi_1\, y_{t-1} \;+\; \varepsilon_t \;+\; \theta_1\, \varepsilon_{t-1}
    $$

    **(1) Reemplazamos** $y_{t-1}$ por $L\, y_t$ y $\varepsilon_{t-1}$ por
    $L\, \varepsilon_t$ (es solo notación nueva, no hay matemática nueva):

    $$
    y_t \;=\; \phi_1\, L\, y_t \;+\; \varepsilon_t \;+\; \theta_1\, L\, \varepsilon_t
    $$

    **(2) Movemos** todas las $y$ a la izquierda y todas las $\varepsilon$ a
    la derecha:

    $$
    y_t \;-\; \phi_1\, L\, y_t \;=\; \varepsilon_t \;+\; \theta_1\, L\, \varepsilon_t
    $$

    **(3) Factorizamos** $y_t$ a la izquierda y $\varepsilon_t$ a la derecha:

    $$
    (1 - \phi_1 L)\, y_t \;=\; (1 + \theta_1 L)\, \varepsilon_t
    $$

    **(4) Le ponemos nombre** a cada paréntesis — pero el contenido es lo de
    antes:

    $$
    \underbrace{(1 - \phi_1 L)}_{\phi_1(L)}\, y_t
    \;=\;
    \underbrace{(1 + \theta_1 L)}_{\theta_1(L)}\, \varepsilon_t
    $$

    Esa es la **forma compacta** de un ARMA(1, 1). Los paréntesis se llaman
    *polinomios* en `L`, pero **son solo los coeficientes del modelo, escritos
    juntos**.

    #### 5c · Para `(p, q)` general

    Los paréntesis crecen incluyendo más rezagos, pero la estructura es la
    misma:

    $$
    \phi_p(L) \;=\; 1 - \phi_1 L - \phi_2 L^2 - \dots - \phi_p L^p
    $$

    $$
    \theta_q(L) \;=\; 1 + \theta_1 L + \theta_2 L^2 + \dots + \theta_q L^q
    $$

    Y un **ARMA(p, q) cualquiera** queda:

    $$
    \phi_p(L)\, y_t \;=\; \theta_q(L)\, \varepsilon_t
    $$

    > Misma ecuación que conocías, solo *empaquetada*. La parte de la izquierda
    > dice cómo $y_t$ depende de su propio pasado; la de la derecha dice cómo
    > el ruido $\varepsilon_t$ aporta sus choques pasados.

    #### 5d · La diferencia, como caso particular

    Recuerda $\Delta y_t = y_t - y_{t-1}$. En la nueva notación:

    $$
    \Delta y_t \;=\; y_t - L\, y_t \;=\; (1 - L)\, y_t
    $$

    Es decir, **diferenciar = multiplicar por $(1 - L)$**. Diferenciar `d`
    veces es multiplicar por $(1 - L)^d$. Con eso, **ARIMA(p, d, q)** completo
    cabe en una línea:

    $$
    \phi_p(L)\,(1-L)^{d}\, y_t \;=\; \theta_q(L)\,\varepsilon_t
    $$

    ### Paso 6 · El ciclo: una segunda capa AR/I/MA en saltos de `s`

    ARIMA ya quita la tendencia, pero ignora **patrones que se repiten cada `s`
    pasos** — el ciclo diario en la temperatura, por ejemplo. La idea es
    agregar una *segunda* familia de polinomios que opera en pasos de `s` en
    vez de pasos de `1`:

    - AR estacional: $\Phi_P(L^s) = 1 - \Phi_1 L^s - \dots - \Phi_P L^{Ps}$
    - MA estacional: $\Theta_Q(L^s) = 1 + \Theta_1 L^s + \dots + \Theta_Q L^{Qs}$
    - Diferencia estacional: $(1 - L^s)^D$ — resta "mismo punto del ciclo previo".

    Multiplicamos los polinomios *no estacionales* por los *estacionales* a
    ambos lados de la ecuación. Ese **producto** es lo que da a SARIMA su
    poder.

    ### Paso 7 · La fórmula final

    Juntando todo:

    $$
    \underbrace{\Phi_P(L^{s})}_{\text{AR estacional}}\,
    \underbrace{\phi_p(L)}_{\text{AR}}\,
    \underbrace{(1-L^{s})^{D}}_{\text{dif. estacional}}\,
    \underbrace{(1-L)^{d}}_{\text{dif. regular}}\,
    y_t
    \;=\;
    \underbrace{\Theta_Q(L^{s})}_{\text{MA estacional}}\,
    \underbrace{\theta_q(L)}_{\text{MA}}\,
    \varepsilon_t
    $$

    **Léelo de derecha a izquierda sobre $y_t$:**

    1. Aplica `d` diferencias regulares → quita tendencia.
    2. Aplica `D` diferencias estacionales → quita el ciclo de periodo `s`.
    3. Aplica el AR no estacional y el AR estacional **al mismo tiempo** (son
       un producto, ven la serie ya diferenciada).
    4. El lado derecho hace lo mismo con MA, pero sobre los errores
       $\varepsilon_t$.

    > **Idea clave.** SARIMA es **ARMA aplicado dos veces a la vez** — uno en
    > rezagos consecutivos `(p, q)` y otro en rezagos estacionales `(P, Q)` —
    > sobre una serie a la que primero le quitamos la tendencia (`d`) y el
    > ciclo (`D, s`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A.2 bis · ¿Entonces "la suma de los dos polinomios es cero"?

    Es una duda que aparece naturalmente al ver la forma compacta:

    $$
    \phi_p(L)\, y_t \;=\; \theta_q(L)\, \varepsilon_t
    $$

    Es tentador pensar que, pasando todo a un lado,
    $\phi_p(L)\,y_t - \theta_q(L)\,\varepsilon_t = 0$ significa que "los
    polinomios se cancelan". **No.**

    | Lado izquierdo | Lado derecho |
    |---|---|
    | $\phi_p(L)\, y_t$ | $\theta_q(L)\, \varepsilon_t$ |
    | Opera sobre $y$ (la temperatura). | Opera sobre $\varepsilon$ (el ruido aleatorio). |
    | Es lo que queda de $y_t$ **tras restarle su parte AR**. | Es el choque actual + parte del choque pasado (la parte MA). |

    La ecuación dice algo más sutil que "dos polinomios suman cero":

    > **En cada instante $t$**, el número que sale de aplicar $\phi_p(L)$ a la
    > serie de temperaturas coincide con el número que sale de aplicar
    > $\theta_q(L)$ a la serie de ruidos.

    Son **dos transformaciones distintas** (una sobre $y$, otra sobre
    $\varepsilon$) que dan **el mismo número** en cada $t$. Esa es la
    *restricción* que define un proceso ARMA.

    **Analogía rápida.** *"500 g de harina = 4 tazas"* no significa que harina
    y tazas sumen cero, ni que sean lo mismo. Significa que ambas miden la
    misma porción, expresada de dos formas distintas. Igual aquí: lo que queda
    de $y_t$ tras quitarle su pasado (izquierda) equivale al choque
    estructurado (derecha).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A.3 Parámetro por parámetro, en clave de `tdb`

    Siete números — `(p, d, q)(P, D, Q, s)` — controlan toda la flexibilidad
    del modelo. Esta es la guía rápida, anclada a la temperatura del ClimaLab
    (remuestreada a hora, así que un "paso" es 1 hora).

    ### Parte no estacional

    | Símbolo | Qué controla | Imagen para `tdb` |
    |---|---|---|
    | **`p`** | AR de corto plazo. Usa $y_{t-1}, \dots, y_{t-p}$. | "La temperatura *de hace 1–2 horas* es predictiva de la temperatura *de ahora*." |
    | **`d`** | Diferenciación regular. Quita tendencia lineal/cuadrática. | ¿Sube la temperatura promedio año tras año? Si sí, `d = 1` puede ayudar. |
    | **`q`** | MA de corto plazo. Usa errores recientes. | "Si me equivoqué hace una hora prediciendo 25 °C, parte de esa sorpresa sigue." |

    ### Parte estacional

    | Símbolo | Qué controla | Imagen para `tdb` (`s = 24`) |
    |---|---|---|
    | **`P`** | AR estacional. Usa $y_{t-s}, y_{t-2s}, \dots$. | "La temperatura *ayer a esta misma hora* y *antier a esta hora* anticipan la de hoy." |
    | **`D`** | Diferenciación estacional. Calcula $y_t - y_{t-s}$. | "Trabajo no con la temperatura cruda sino con *cuánto difiere* de la misma hora ayer." |
    | **`Q`** | MA estacional. Usa errores en rezagos de `s`. | "Si me equivoqué *ayer a esta misma hora*, arrastro parte de esa corrección hoy." |
    | **`s`** | Periodo del ciclo. | `s = 24` para datos horarios con ciclo diario. |

    ### La regla práctica más importante

    > **`d + D ≤ 2`.** Diferenciar de más amplifica el ruido y empeora el
    > pronóstico aunque el AIC baje. Para `tdb`, la combinación más natural a
    > probar primero es `d = 0, D = 1` con `s = 24`: la serie no tiene
    > tendencia fuerte año a año, pero sí un ciclo diario potente.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A.4 Una aclaración que evita un malentendido común

    Es tentador pensar que `D = 1` con `s = 24` calcula un **perfil promedio
    horario** — un vector de 24 promedios (uno por hora del día) que después
    se le resta a cada punto. **No es así.**

    Lo que hace `D = 1` es una **diferencia rezagada**:

    $$
    y'_t \;=\; y_t - y_{t-24}
    $$

    Compara cada hora con **esa misma hora del día anterior, exactamente uno**,
    no con el promedio de todos los días previos.

    ### Entonces, ¿de dónde sale el "perfil diario" del modelo?

    Sale **implícitamente**:

    - Si el ciclo diario es estable, $y_t$ y $y_{t-24}$ contienen
      aproximadamente la misma componente cíclica y al restarlos **se cancela**.
    - En $y'_t$ solo quedan las **desviaciones día-a-día** respecto al ciclo.
    - El modelo **nunca guarda** un vector de 24 promedios — eso sería otra
      cosa (descomposición estacional clásica, tipo `seasonal_decompose` o
      STL).

    | Enfoque | Cómo trata la estacionalidad |
    |---|---|
    | **STL / `seasonal_decompose`** | Calcula un perfil explícito (promedio por hora del día sobre todo el train) y lo resta. |
    | **SARIMA con `D = 1`** | Diferencia rezagada $y_t - y_{t-s}$. El perfil queda **implícito**. |
    | **SARIMA con `P, Q` (sin `D`)** | Usa el valor (`P`) o el error (`Q`) del mismo instante de **uno o dos ciclos** anteriores — no de todo el histórico. |

    > Cuando decimos *"el modelo aprende su propio perfil diario"* nos referimos
    > al **comportamiento efectivo**, no a la mecánica interna. Por eso `D = 1`
    > funciona bien incluso con poca historia: no necesita muchas vueltas del
    > ciclo para estimar un promedio robusto, **solo necesita que el ciclo de
    > ayer y el de hoy se parezcan**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A.5 ¿Por qué se **multiplican** las dos partes? Rezagos cruzados gratis

    SARIMA no es "ARMA + ARMA estacional" sumados, sino el **producto** de los
    polinomios no estacional y estacional. Esto hace aparecer **rezagos
    cruzados** sin gastar parámetros extra.

    ### Ejemplo concreto: `p = 1`, `P = 1`, `s = 24`

    Los dos polinomios AR son:

    $$
    \phi_1(L) \;=\; 1 - \phi_1 L
    \qquad
    \Phi_1(L^{24}) \;=\; 1 - \Phi_1 L^{24}
    $$

    #### Expandiendo el producto — paso a paso

    Vamos a multiplicarlos exactamente como $(a + b)(c + d) = ac + ad + bc + bd$.
    Aquí $a = 1$, $b = -\phi_1 L$, $c = 1$, $d = -\Phi_1 L^{24}$. Salen
    **cuatro términos**:

    | Producto | Resultado | De dónde viene |
    |---|---|---|
    | $1 \cdot 1$ | $1$ | el "1" trivial. |
    | $1 \cdot (-\Phi_1 L^{24})$ | $-\Phi_1\, L^{24}$ | rezago **24** (estacional). |
    | $(-\phi_1 L) \cdot 1$ | $-\phi_1\, L$ | rezago **1** (no estacional). |
    | $(-\phi_1 L) \cdot (-\Phi_1 L^{24})$ | $+\phi_1\Phi_1\, L \cdot L^{24} \;=\; +\phi_1\Phi_1\, L^{25}$ | rezago **25** — el cruzado. |

    > **El rezago 25 sale de multiplicar $L \cdot L^{24}$**, igual que
    > $x \cdot x^{24} = x^{25}$. Aplicar `L` 1 vez y luego `L` 24 veces más es
    > retroceder 25 pasos en total.

    Sumando los cuatro términos:

    $$
    (1 - \phi_1 L)(1 - \Phi_1 L^{24})
    \;=\;
    1 - \phi_1 L - \Phi_1 L^{24} + \phi_1\Phi_1\, L^{25}
    $$

    Al aplicar ese polinomio a $y_t$, cada término "alcanza" un rezago
    distinto:

    | Término del polinomio | Sobre $y_t$ produce | Significado |
    |---|---|---|
    | $1$ | $y_t$ | el valor actual. |
    | $-\phi_1 L$ | $-\phi_1\, y_{t-1}$ | una hora atrás. |
    | $-\Phi_1 L^{24}$ | $-\Phi_1\, y_{t-24}$ | ayer a esta misma hora. |
    | $+\phi_1\Phi_1 L^{25}$ | $+\phi_1\Phi_1\, y_{t-25}$ | **hace 25 horas** = una hora antes de ayer a esta hora. |

    ¡El **rezago 25** aparece **solo**, sin que el modelo tenga un coeficiente
    extra para él! Su peso es el producto de los dos coeficientes ya estimados,
    $\phi_1\Phi_1$. Y tiene sentido físico: *"hace 25 horas"* es *"hace una
    hora del valor de ayer a esta misma hora"*.

    > **Esa es la razón por la que SARIMA logra capturar dinámicas en
    > múltiples escalas con muy pocos coeficientes.** Con solo dos perillas
    > (`p = 1`, `P = 1`) ya estás usando los rezagos 1, 24 y 25
    > *simultáneamente*.

    ### Cómo se ve en los residuos

    El veredicto rápido sobre `(P, D, Q, s)` **no está en el AIC** — está en la
    **ACF de los residuos** (la vamos a ver en la libreta 020):

    | Lo que ves en la ACF de residuos | Qué mover |
    |---|---|
    | Pico aislado grande en lag **24** | Te falta `D = 1` o `Q = 1`. |
    | Decaimiento lento en múltiplos de 24 (24, 48, 72) | Sube `P` a 1 o 2. |
    | Picos en lags pequeños (1–5) | Eso es la parte no estacional — ajusta `p` o `q`. |

    ---

    Con eso tienes la fórmula completa, su construcción capa por capa y la
    intuición de cada parámetro. **Ahora vamos a los datos.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Siguiente paso

    Pasa a **`019b_tdb_exploracion.py`** para ver la serie `tdb` del
    ClimaLab y conectar lo que acabas de leer con datos reales:
    estacionalidades, ACF/PACF, pruebas de estacionariedad, y la elección
    inicial de `(p, d, q)(P, D, Q, s)` para `020`.
    """)
    return


if __name__ == "__main__":
    app.run()
