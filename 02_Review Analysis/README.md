# Análisis de Sentimientos en Reseñas de Películas

Este proyecto tiene como objetivo construir un modelo de clasificación de texto que pueda predecir el sentimiento (positivo o negativo) de reseñas de películas. Utilizamos técnicas de Procesamiento de Lenguaje Natural (NLP) y algoritmos de machine learning para analizar y clasificar las reseñas.

## Descripción

El objetivo de este proyecto es realizar un análisis de sentimientos en reseñas de películas utilizando técnicas de NLP y machine learning. El modelo entrenado será capaz de clasificar las reseñas en positivas o negativas.

## Conjunto de Datos

El conjunto de datos utilizado es `movie_reviews.csv`, que contiene las siguientes columnas:

- `review`: Texto de la reseña de la película.
- `sentiment`: Etiqueta de sentimiento (positivo o negativo).

## Entrenamiento del Modelo

Se utiliza el algoritmo Naive Bayes para entrenar el modelo de clasificación de texto. El flujo general incluye:

1. Carga del conjunto de datos.
2. Preprocesamiento del texto.
3. División de los datos en conjuntos de entrenamiento y prueba.
4. Entrenamiento del modelo.
5. Evaluación del modelo.

## Evaluación del Modelo

El rendimiento del modelo se evalúa utilizando métricas como precisión, recall, F1 score y la matriz de confusión.

## Ejemplos de Uso

Aquí hay algunos ejemplos de cómo se pueden clasificar nuevas reseñas usando el modelo entrenado:

```python
ejemplos = ["This movie is fantastic!", "I did not enjoy this film at all."]
ejemplos_features = vectorizer.transform(ejemplos)
predicciones = model.predict(ejemplos_features)
print(predicciones)  # Salida esperada: ['positive', 'negative']
