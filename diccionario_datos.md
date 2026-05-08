# 📖 Diccionario de Datos: Proyecto Nacimientos 2018

Este diccionario describe las variables seleccionadas del dataset de Microdatos del DANE para predecir el **Bajo Peso al Nacer** y el **Tipo de Parto**.

## 📊 Variables del Modelo

| Variable | Nombre en Dataset | Descripción | Valores / Categorías |
| :--- | :--- | :--- | :--- |
| **Edad Madre** | `EDAD_MADRE` | Edad cronológica de la madre. | Años (Ej: 25) |
| **Semanas de Gestación** | `SEMANAS` | Tiempo de embarazo al nacer. | Semanas (Ej: 38) |
| **Peso al Nacer** | `PESO_NAC` | Peso del neonato en gramos. | < 2500g = Bajo peso |
| **Talla al Nacer** | `TALLA_NAC` | Longitud del neonato en cm. | Centímetros (Ej: 49) |
| **Departamento** | `CODPTORE` | Depto. de residencia de la madre. | 11: Bogotá, 05: Antioquia, etc. |
| **Área de Residencia** | `AREA_RES` | Tipo de zona de vivienda. | 1: Cabecera, 2: Centro, 3: Rural |
| **Régimen Salud** | `IDCLASADMI` | Tipo de afiliación a salud. | 1: Contributivo, 2: Subsidiado |
| **Tipo de Parto** | `TIPO_PARTO` | Forma en que terminó el parto. | 1: Espontáneo, 2: Cesárea |

---

## 🛠️ Variables Generadas (Feature Engineering)

| Variable | Lógica de Creación | Objetivo |
| :--- | :--- | :--- |
| **BAJO_PESO** | `1` si `PESO_NAC` < 2500, else `0` | **Variable Objetivo (Target)** |

---

## 🔗 Fuentes y Referencias
* **Dataset Original:** [DANE - Nacimientos 2018](https://microdatos.dane.gov.co/index.php/catalog/652/data-dictionary)
* **Estado:** Datos procesados y limpios en `data/processed/`.