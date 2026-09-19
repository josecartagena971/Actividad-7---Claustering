# Actividad 07 (Unidad II)  Clustering lineal y no lineal, métricas de validación - Resumen Ejecutivo

**Universidad Nacional del Altiplano Puno · Maestría en Ciencia de Datos · Machine Learning I, Grupo A**

**Integrante(s):** 
- Jose Jhonatan Quispe Cartagena
- Oscar Edy Vilca Quis

## Objetivo

Comparar tres algoritmos de clustering: **K-means**, **DBSCAN** y **SpectralClustering** sobre tres datasets con geometrías distintas (**Iris**, **Moons**, **Circles**), evaluando cada resultado con:

- **Métricas de validación externa** (necesitan etiquetas reales): Adjusted Rand Index (ARI), V-measure.
- **Métricas de validación interna** (no necesitan etiquetas reales): Silhouette Score, Davies-Bouldin Index, método del codo (Elbow).

Y determinando la **naturaleza lineal o no lineal** de cada modelo.

**Notebook completo con todo el código y las visualizaciones:** [Abrir Notebook: Actividad_07_Clustering (1).ipynb](./Actividad_07_Clustering%20%281%29.ipynb)

## Metodología general

En cada dataset:
1. Se estandarizaron las variables con `StandardScaler`.
2. Se aplicó el **método del codo** (inercia de K-means) para explorar el número de clusters.
3. Se entrenaron los 3 modelos: K-means y SpectralClustering con `n_clusters` = número real de clases; DBSCAN con `eps`/`min_samples` ajustados por búsqueda.
4. Se calcularon las métricas externas (contra las etiquetas reales) e internas (solo con los datos y las etiquetas predichas).

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import adjusted_rand_score, v_measure_score, silhouette_score, davies_bouldin_score

def entrenar_modelos(X, n_clusters, dbscan_eps, dbscan_min_samples):
    resultados = {}
    resultados["KMeans"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X)
    resultados["DBSCAN"] = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(X)
    resultados["SpectralClustering"] = SpectralClustering(
        n_clusters=n_clusters, affinity="nearest_neighbors", n_neighbors=10, random_state=42
    ).fit_predict(X)
    return resultados
```

---

## Caso A — Iris (150 muestras, 4 variables, 3 especies)

```python
from sklearn.datasets import load_iris
iris = load_iris()
X_iris_s = StandardScaler().fit_transform(iris.data)
resultados_iris = entrenar_modelos(X_iris_s, n_clusters=3, dbscan_eps=0.65, dbscan_min_samples=3)
```

| Modelo | Clusters | Ruido | ARI | V-measure | Silhouette | Davies-Bouldin |
|---|---|---|---|---|---|---|
| KMeans | 3 | 0 | 0.620 | 0.659 | 0.460 | 0.834 |
| DBSCAN (eps=0.65) | 3 | 7 | 0.537 | 0.645 | 0.353 | 14.321 |
| SpectralClustering | 3 | 0 | **0.646** | **0.684** | 0.459 | 0.822 |

### ¿Qué significan estas métricas? 

El método del codo sugiere 2–3 grupos, coherente con las 3 especies, aunque *versicolor* y *virginica* se solapan bastante en el espacio de características. KMeans y SpectralClustering logran ARI/V-measure moderados-altos (~0.55–0.68) separando bien *setosa* pero confundiendo parte de las otras dos especies. DBSCAN obtiene un ARI algo menor porque, al basarse en densidad, fusiona o deja como ruido puntos de la zona solapada, esos 7 puntos con etiqueta `-1` corresponden a observaciones donde la densidad local no alcanza el umbral `min_samples`. Su Davies-Bouldin es mucho peor (14.321 vs ~0.83) porque genera un cluster pequeño y disperso que distorsiona esta métrica interna; ARI/V-measure no penalizan tanto este efecto porque esos puntos sí pertenecían a su clase real.

---

## Caso B — Moons (300 muestras sintéticas, 2 medialunas entrelazadas)

```python
from sklearn.datasets import make_moons
X_moons, y_moons = make_moons(n_samples=300, noise=0.07, random_state=42)
X_moons_s = StandardScaler().fit_transform(X_moons)
resultados_moons = entrenar_modelos(X_moons_s, n_clusters=2, dbscan_eps=0.24, dbscan_min_samples=3)
```

| Modelo | Clusters | Ruido | ARI | V-measure | Silhouette | Davies-Bouldin |
|---|---|---|---|---|---|---|
| KMeans | 2 | 0 | 0.479 | 0.382 | 0.495 | 0.806 |
| DBSCAN (eps=0.24) | 2 | 0 | **1.000** | **1.000** | 0.382 | 1.023 |
| SpectralClustering | 2 | 0 | **1.000** | **1.000** | 0.382 | 1.023 |

### ¿Qué significan estas métricas?

El método del codo no muestra un codo claro en k=2 al basarse en distancia a un centroide, no puede reflejar la estructura no convexa de los datos. KMeans, limitado a fronteras lineales, corta ambas medialunas por la mitad (ARI≈0.48). DBSCAN y SpectralClustering, al permitir separación no lineal, alcanzan ARI/V-measure **perfectos** (1.000). Aun así, su Silhouette (~0.38) no es cercano a 1, porque esta métrica interna también asume forma convexa: un resultado externamente perfecto puede verse "mediocre" según una métrica interna, cuando en realidad el cluster real simplemente no es compacto.

---

## Caso C — Circles (300 muestras sintéticas, 2 círculos concéntricos)

```python
from sklearn.datasets import make_circles
X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)
X_circles_s = StandardScaler().fit_transform(X_circles)
resultados_circles = entrenar_modelos(X_circles_s, n_clusters=2, dbscan_eps=0.32, dbscan_min_samples=3)
```

| Modelo | Clusters | Ruido | ARI | V-measure | Silhouette | Davies-Bouldin |
|---|---|---|---|---|---|---|
| KMeans | 2 | 0 | -0.003 | 0.000 | 0.353 | 1.175 |
| DBSCAN (eps=0.32) | 2 | 0 | **1.000** | **1.000** | 0.110 | 163.270 |
| SpectralClustering | 2 | 0 | **1.000** | **1.000** | 0.110 | 163.270 |

### ¿Qué significan estas métricas?

Este es el caso más extremo. KMeans obtiene ARI≈0 (equivalente a una asignación aleatoria), porque ambos círculos comparten el mismo centro y no existe una línea recta que los separe. DBSCAN y SpectralClustering alcanzan de nuevo ARI/V-measure perfectos (1.000). Sin embargo, su Silhouette es muy bajo (0.110) y su Davies-Bouldin es extremadamente alto/malo (163.270), mucho peor que el de KMeans, que en realidad falló. Esto ocurre porque ambos círculos comparten centro: cualquier métrica interna basada en distancia a un centroide interpreta que están "encimados", aunque estén perfectamente separados según la verdad fundamental.

> **Conclusión clave:** cuando la estructura real no es convexa, las métricas internas pueden contradecir directamente a las externas, y hay que priorizar estas últimas si se dispone de etiquetas reales.

---

## Naturaleza lineal / no lineal e hiperparámetros críticos

| Modelo | Naturaleza | Hiperparámetros críticos |
|---|---|---|
| **KMeans** | Lineal (fronteras convexas) | `n_clusters`; inicialización de centroides |
| **DBSCAN** | Lineal y no lineal (basado en densidad) | `eps` (radio de vecindad) y `min_samples` |
| **SpectralClustering** | No lineal (grafo de similitud) | `n_clusters`; `affinity` (`n_neighbors` o `gamma`) |

## Resumen — Adjusted Rand Index por modelo y dataset

| Dataset | KMeans | DBSCAN | SpectralClustering |
|---|---|---|---|
| Iris | 0.620 | 0.537 | 0.646 |
| Moons | 0.479 | 1.000 | 1.000 |
| Circles | -0.003 | 1.000 | 1.000 |

## Conclusiones generales

- El factor decisivo no es qué algoritmo es "mejor" en general, sino **si la geometría real de los datos es convexa o no**: KMeans funciona razonablemente en Iris pero falla progresivamente en Moons y por completo en Circles, exactamente en la medida en que la estructura deja de ser linealmente separable.
- DBSCAN y SpectralClustering, ambos capaces de capturar estructuras no lineales, mantienen un desempeño alto (ARI=1.000) en Moons y Circles, siempre que sus hiperparámetros críticos (`eps`/`min_samples`; `affinity`/`n_neighbors`) estén bien ajustados a la escala de los datos.
- Las métricas externas e internas **no siempre coinciden**: en Circles, las internas califican como deficiente un resultado que las externas confirman como perfecto. Esto ocurre porque Silhouette y Davies-Bouldin asumen clusters convexos/compactos, un supuesto que no se cumple en datos con forma de anillo o medialuna.
- El método del codo, al basarse en inercia (distancia euclidiana a centroides), solo resulta informativo cuando los clusters subyacentes son razonablemente convexos (Iris); en Moons y Circles no muestra un codo claro, aun conociendo de antemano que el número real de grupos es 2.
- **Recomendación práctica:** si se dispone de etiquetas reales, usar validación externa (ARI, V-measure) como criterio principal; si no se dispone de ellas, complementar Silhouette/Davies-Bouldin con una inspección visual de los datos, ya que estas métricas internas pueden inducir a error en estructuras no convexas.
