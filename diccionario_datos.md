# 📖 Diccionario de Datos: Proyecto Nacimientos 2018

Variables del dataset de microdatos del DANE (Estadísticas Vitales, nacimientos 2018)
usadas para predecir **cesárea** y **bajo peso al nacer**. Las etiquetas se verificaron
contra el [diccionario oficial del DANE](https://microdatos.dane.gov.co/index.php/catalog/652/data-dictionary/F41?file_name=nac2018).

> ⚠️ Casi todas las variables son **códigos categóricos**, no medidas: `EDAD_MADRE = 6`
> significa "35-39 años", no 6 años; `PESO_NAC = 4` significa "2.000-2.499 g".
> El código **-1** no es del DANE: lo asigna `src/preprocessing.py` al imputar nulos.

## 📊 Variables del modelo (10)

| Variable | Descripción DANE | Códigos |
| :--- | :--- | :--- |
| `EDAD_MADRE` | Edad de la madre a la fecha del parto | 1 = 10-14 · 2 = 15-19 · 3 = 20-24 · 4 = 25-29 · 5 = 30-34 · 6 = 35-39 · 7 = 40-44 · 8 = 45-49 · 9 = 50-54 años (99 = sin información, excluido) |
| `T_GES` | Tiempo de gestación del nacido vivo | 1 = <22 · 2 = 22-27 · 3 = **28-37** · 4 = **38-41** · 5 = 42 y más semanas · 6 = ignorado · 9 = sin información |
| `NUMCONSUL` | Número de consultas prenatales | 0-25 (00 = ninguna; 99 = sin información → faltante) |
| `N_HIJOSV` | Hijos nacidos vivos que ha tenido la madre, incluido el presente | Número (99 = sin información → faltante) |
| `N_EMB` | Número de embarazos, incluido el presente | Número (99 = sin información, excluido) |
| `AREA_RES` | Área de residencia habitual de la madre | 1 = cabecera municipal · 2 = centro poblado · 3 = rural disperso · 9 = sin información · -1 = nulo |
| `IDCLASADMI` | Entidad administradora en salud de la madre | 1 = EPS (contributivo) · 2 = EPS subsidiado · 3 = entidad adaptada · 4 = entidad especial · 5 = entidad exceptuada · 9 = sin información · -1 = nulo (= no asegurada, ver nota) |
| `SEXO` | Sexo del nacido vivo | 1 = masculino · 2 = femenino (3 = indeterminado, excluido) |
| `MUL_PARTO` | Multiplicidad del embarazo | 1 = simple · 2 = doble · 3 = triple · 4 = cuádruple o más · 9 = sin información |
| `CODPTORE` | Departamento de residencia habitual de la madre | Código DIVIPOLA (5 = Antioquia, 11 = Bogotá, 76 = Valle del Cauca, …) · -1 = nulo |

**Nota sobre `IDCLASADMI`:** en el dataset crudo, los nulos de `IDCLASADMI` coinciden uno a
uno con `SEG_SOCIAL = 5` (no asegurado), y los códigos 1 y 3 corresponden al régimen
contributivo. El código 5 es **entidad exceptuada**, no "sin afiliación".

## 🎯 Variables objetivo y derivadas

| Variable | Lógica de creación | Observación |
| :--- | :--- | :--- |
| `CESAREA` | `1` si `TIPO_PARTO == 2` | `TIPO_PARTO`: 1 = espontáneo · 2 = cesárea · 3 = instrumentado · 4 = ignorado (excluido) |
| `BAJO_PESO` | `1` si `PESO_NAC ∈ {1, 2, 3, 4}` (< 2.500 g), `0` si `PESO_NAC ∈ {5, …, 8}` | `PESO_NAC = 9` es "sin información" (4.238 registros tras la limpieza): queda sin etiqueta y no entra al entrenamiento |
| `PREMATURO` | `1` si `T_GES_AGRU_CIE ∈ {1, 2, 3}` (< 37 semanas) | Se usa la agrupación CIE porque el código 3 de `T_GES` incluye la semana 37 |
| `EDAD_RIESGO` | `1` si `EDAD_MADRE ∈ {1, 7, 8, 9}` | 10-14 años o 40 años y más |
| `PRIMIGESTA` | `1` si `N_EMB == 1` | Primer embarazo |

`PESO_NAC`: 1 = <1.000 · 2 = 1.000-1.499 · 3 = 1.500-1.999 · 4 = 2.000-2.499 · 5 = 2.500-2.999 ·
6 = 3.000-3.499 · 7 = 3.500-3.999 · 8 = 4.000 y más gramos · 9 = sin información.

`T_GES_AGRU_CIE` (se conserva en el dataset procesado, no es variable del modelo):
1 = <22 · 2 = 22-27 · 3 = 28-36 · 4 = 37-41 · 5 = 42 y más semanas · 6 = ignorado · 9 = sin información.

---

## 🔗 Fuentes y Referencias
* **Dataset Original:** [DANE - Nacimientos 2018](https://microdatos.dane.gov.co/index.php/catalog/652/data-dictionary)
* **Catálogo usado por el agente:** `agent/catalog.py`
* **Estado:** Datos procesados y limpios en `data/processed/`.
