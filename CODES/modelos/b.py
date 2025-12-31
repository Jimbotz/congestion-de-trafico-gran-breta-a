import json
import shutil
import sys
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, sin, cos, dayofweek, month, upper
)
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ---------------------------------------------------------
# 1. CONFIGURACION (Optimizada para estabilidad)
# ---------------------------------------------------------
spark = SparkSession.builder \
    .appName("Traffic_Classifier_Final_Stable") \
    .master("local[*]") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "6g") \
    .config("spark.memory.fraction", "0.8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("Cargando datos...")
try:
    df = spark.read.csv("datasetLimpio", header=True, inferSchema=False)
except:
    print("Error cargando dataset.")
    sys.exit()

# ---------------------------------------------------------
# 2. LIMPIEZA
# ---------------------------------------------------------
# Solo usamos variables que no tengan miles de categorias únicas
num_cols = ["all_motor_vehicles", "latitude", "longitude", "hour", "all_hgvs"]

for c in num_cols:
    df = df.withColumn(c, when(col(c) == "NULL", lit(None)).otherwise(col(c)).cast("double"))

# Direccion limpia
df = df.withColumn("direction_of_travel", upper(col("direction_of_travel")))
df = df.filter(col("direction_of_travel").isin(["N", "S", "E", "W"]))

df = df.filter(
    col("all_motor_vehicles").isNotNull() &
    col("latitude").isNotNull()
)

# ---------------------------------------------------------
# 3. QUINTILES (Clases)
# ---------------------------------------------------------
quantiles = df.stat.approxQuantile("all_motor_vehicles", [0.20, 0.40, 0.60, 0.80], 0.01)
L1, L2, L3, L4 = map(int, quantiles)

print(f"Umbrales: {L1}, {L2}, {L3}, {L4}")

df = df.withColumn(
    "traffic_class",
    when(col("all_motor_vehicles") <= L1, 0.0)
    .when(col("all_motor_vehicles") <= L2, 1.0)
    .when(col("all_motor_vehicles") <= L3, 2.0)
    .when(col("all_motor_vehicles") <= L4, 3.0)
    .otherwise(4.0)
)

# ---------------------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------------------
df = df.withColumn("day_of_week", dayofweek(col("count_date"))) \
       .withColumn("month", month(col("count_date"))) \
       .withColumn("hour_sin", sin(2 * np.pi * col("hour") / 24)) \
       .withColumn("hour_cos", cos(2 * np.pi * col("hour") / 24))

df = df.na.fill({"day_of_week": 1, "month": 1, "all_hgvs": 0})

# ---------------------------------------------------------
# 5. INDEXADORES (Sin Road Name para ahorrar RAM)
# ---------------------------------------------------------
index_dir = StringIndexer(inputCol="direction_of_travel", outputCol="direction_idx", handleInvalid="skip")
index_region = StringIndexer(inputCol="region_name", outputCol="region_idx", handleInvalid="skip")

# ---------------------------------------------------------
# 6. FEATURES
# ---------------------------------------------------------
# Eliminamos 'road_idx' porque consume demasiada memoria
features_col = [
    "latitude",
    "longitude",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "month",
    "all_hgvs",
    "direction_idx",
    "region_idx"
]

assembler = VectorAssembler(inputCols=features_col, outputCol="features", handleInvalid="skip")

# ---------------------------------------------------------
# 7. MODELO EQUILIBRADO
# ---------------------------------------------------------
rf = RandomForestClassifier(
    labelCol="traffic_class",
    featuresCol="features",
    numTrees=40,         # 40 es un numero solido
    maxDepth=10,         # Profundidad suficiente
    maxBins=128,         # 128 es seguro para memoria
    seed=42
)

pipeline = Pipeline(stages=[index_dir, index_region, assembler, rf])

# ---------------------------------------------------------
# 8. ENTRENAMIENTO
# ---------------------------------------------------------
print("Entrenando Modelo Optimizado...")
train, test = df.randomSplit([0.8, 0.2], seed=42)

try:
    model = pipeline.fit(train)
    print("Entrenamiento exitoso")
except Exception as e:
    print(f"Error: {e}")
    sys.exit()

# ---------------------------------------------------------
# 9. EVALUACION Y GUARDADO
# ---------------------------------------------------------
predictions = model.transform(test)
evaluator = MulticlassClassificationEvaluator(labelCol="traffic_class", predictionCol="prediction", metricName="accuracy")
acc = evaluator.evaluate(predictions)

print(f"\nEXACTITUD: {acc:.2%}")

# Guardar
shutil.rmtree("model_traffic_classifier_v4", ignore_errors=True)
model.save("model_traffic_classifier_v4")

# Configuracion para la App
# Extraemos labels de direction (stage 0)
dir_labels = model.stages[0].labels

config = {
    "L1": L1, "L2": L2, "L3": L3, "L4": L4,
    "direction_labels": dir_labels,
    "accuracy": acc
}

with open("model_config_v2.json", "w") as f:
    json.dump(config, f)

print("Modelo y configuración guardados. Listo para usar la App.")
spark.stop()