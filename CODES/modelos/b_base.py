import json
import shutil
import sys
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, dayofweek, month, upper, count as spark_count
)
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

# Configuracion de la sesion de Spark asignando memoria suficiente para el driver y ejecutores
spark = SparkSession.builder \
    .appName("Traffic_RF_Reporte_Completo") \
    .master("local[*]") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "6g") \
    .config("spark.memory.fraction", "0.8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("="*70)
print("RANDOM FOREST - REPORTE COMPLETO")
print("="*70)

try:
    df = spark.read.csv("datasetLimpio", header=True, inferSchema=False)
    print(f"Dataset cargado: {df.count():,} registros")
except:
    print("Error cargando dataset.")
    sys.exit()

# Definicion de columnas numericas para castear y limpiar nulos
num_cols = ["all_motor_vehicles", "latitude", "longitude", "hour", "all_hgvs"]

for c in num_cols:
    df = df.withColumn(c, when(col(c) == "NULL", lit(None)).otherwise(col(c)).cast("double"))

# Estandarizacion de direcciones y filtrado de valores validos unicamente
df = df.withColumn("direction_of_travel", upper(col("direction_of_travel")))
df = df.filter(col("direction_of_travel").isin(["N", "S", "E", "W"]))

# Eliminacion de filas que no tengan datos en variables criticas
df = df.filter(
    col("all_motor_vehicles").isNotNull() &
    col("latitude").isNotNull()
)

print(f"Después de limpieza: {df.count():,} registros")

# Calculo de quintiles aproximados para dividir el dataset en 5 partes iguales segun el volumen de trafico
print("\nCalculando umbrales")
quantiles = df.stat.approxQuantile("all_motor_vehicles", [0.20, 0.40, 0.60, 0.80], 0.01)
L1, L2, L3, L4 = map(int, quantiles)

print(f"   Clase 0: 0-{L1} | Clase 1: {L1+1}-{L2} | Clase 2: {L2+1}-{L3}")
print(f"   Clase 3: {L3+1}-{L4} | Clase 4: {L4+1}+")

# Creacion de la columna objetivo 'traffic_class' basada en los umbrales calculados
df = df.withColumn(
    "traffic_class",
    when(col("all_motor_vehicles") <= L1, 0.0)
    .when(col("all_motor_vehicles") <= L2, 1.0)
    .when(col("all_motor_vehicles") <= L3, 2.0)
    .when(col("all_motor_vehicles") <= L4, 3.0)
    .otherwise(4.0)
)

print("\nBalance de clases:")
class_dist = df.groupBy("traffic_class").agg(spark_count("*").alias("count")).orderBy("traffic_class")
total = df.count()
for row in class_dist.collect():
    pct = (row['count'] / total) * 100
    print(f"   Clase {int(row['traffic_class'])}: {row['count']:,} ({pct:.1f}%)")

# Ingenieria de caracteristicas para extraer dia y mes de la fecha
print("\nPreparando las features")
df = df.withColumn("day_of_week", dayofweek(col("count_date"))) \
       .withColumn("month", month(col("count_date")))

df = df.na.fill({"day_of_week": 1, "month": 1, "all_hgvs": 0})

# Transformacion de variables categoricas (texto) a indices numericos para el modelo
index_dir = StringIndexer(inputCol="direction_of_travel", outputCol="direction_idx", handleInvalid="skip")
index_region = StringIndexer(inputCol="region_name", outputCol="region_idx", handleInvalid="skip")

features_col = [
    "latitude",
    "longitude",
    "hour",          
    "day_of_week",
    "month",
    "all_hgvs",      # IMPORTANTISIMA ESTA
    "direction_idx",
    "region_idx"
]

# VectorAssembler consolida todas las features en un unico vector denso, requerido por Spark ML
assembler = VectorAssembler(inputCols=features_col, outputCol="features", handleInvalid="skip")

# Configuracion del clasificador Random Forest con sus hiperparametros clave
rf = RandomForestClassifier(
    labelCol="traffic_class",
    featuresCol="features",
    numTrees=40,         
    maxDepth=10,         
    maxBins=128,         
    seed=42
)

# Creacion del Pipeline que ejecuta indexadores, ensamblador y modelo en secuencia
pipeline = Pipeline(stages=[index_dir, index_region, assembler, rf])

print("\nEntrenando Modelo Random Forest")
train, test = df.randomSplit([0.8, 0.2], seed=42)
print(f"   Train: {train.count():,} | Test: {test.count():,}")

try:
    model = pipeline.fit(train)
    print("Entrenamiento exitoso")
except Exception as e:
    print(f"Error: {e}")
    sys.exit()

print("\n" + "="*70)
print("EVALUACIÓN DEL MODELO")
print("="*70)

predictions = model.transform(test)
evaluator = MulticlassClassificationEvaluator(labelCol="traffic_class", predictionCol="prediction", metricName="accuracy")
acc = evaluator.evaluate(predictions)

print(f"\nEXACTITUD GLOBAL: {acc:.2%}")

# Conversion a RDD para usar MulticlassMetrics que permite calculos detallados como matriz de confusion
pred_and_labels = predictions.select("prediction", "traffic_class") \
    .rdd.map(lambda x: (float(x[0]), float(x[1])))
metrics = MulticlassMetrics(pred_and_labels)

# Obtencion de la matriz de confusion como array numpy
confusion = metrics.confusionMatrix().toArray()
class_names = ["MUY BAJO", "BAJO", "MEDIO", "ALTO", "MUY ALTO"]

print("\nMATRIZ DE CONFUSIÓN:")
print("        Predicho →")
print("Real ↓  |  C0  |  C1  |  C2  |  C3  |  C4  |")
print("--------|------|------|------|------|------|")
for i, row in enumerate(confusion):
    row_str = " | ".join([f"{int(val):4d}" for val in row])
    print(f"{class_names[i]:8s}| {row_str} |")

print("\nMÉTRICAS POR CLASE:")
print(f"{'Clase':<12} {'Precisión':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 60)

class_metrics = {}
for i, name in enumerate(class_names):
    precision = metrics.precision(float(i))
    recall = metrics.recall(float(i))
    f1 = metrics.fMeasure(float(i))
    print(f"{name:<12} {precision:>10.2%}  {recall:>10.2%}  {f1:>10.2%}")
    class_metrics[name] = {"precision": precision, "recall": recall, "f1": f1}

# Extraccion de la importancia de variables desde el modelo entrenado (ultimo paso del pipeline)
print("\nIMPORTANCIA DE VARIABLES:")
rf_model = model.stages[-1] 
importances = rf_model.featureImportances.toArray()

importance_pairs = sorted(zip(features_col, importances), key=lambda x: x[1], reverse=True)
feature_dict = {}

for i, (feat, imp) in enumerate(importance_pairs, 1):
    print(f"   {i:2d}. {feat:<25s} {imp:.4f}")
    feature_dict[feat] = float(imp)

# Guardado del modelo y creacion del JSON con metadatos para reproducir la configuracion
model_path = "model_traffic_RF_Final"
shutil.rmtree(model_path, ignore_errors=True)
model.save(model_path)

config = {
    "model_type": "RandomForest",
    "L1": L1, "L2": L2, "L3": L3, "L4": L4,
    "direction_labels": model.stages[0].labels,
    "region_labels": model.stages[1].labels,
    "accuracy": acc,
    "class_metrics": class_metrics,
    "feature_importance": feature_dict,
    "hyperparameters": {
        "numTrees": 40,
        "maxDepth": 10
    }
}

with open("model_config_RF.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"Modelo guardado en: {model_path}")
print("="*70)
spark.stop()