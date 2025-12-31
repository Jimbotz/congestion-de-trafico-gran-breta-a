from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("TrafficDataCleaning") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Carga del archivo CSV infiriendo automaticamente los tipos de datos de cada columna
path = "./archive/dft_traffic_counts_raw_counts/dft_traffic_counts_raw_counts.csv" # aca meti todos los codigos a la carpeta CODES entonces para que todos funcionnen de manera correcta se tienen que sacar a la carpeta raiz o ajustar para que asi tambien funcionen
df = spark.read.csv(path, header=True, inferSchema=True)

# Calculo de dimensiones iniciales para comparar despues de la limpieza
total_rows_orig = df.count()
total_cols_orig = len(df.columns)

print(f"ESTADO ORIGINAL:")
print(f"Total de registros (filas): {total_rows_orig}")
print(f"Total de columnas: {total_cols_orig}")
print("-" * 30)

# Definicion explicita de las variables que seran utiles para el modelo
columnas_deseadas = [
    "region_name", "local_authority_name",              
    "year", "count_date", "hour",                       
    "direction_of_travel",                              
    "latitude", "longitude", "road_name", "road_type",  
    "start_junction_road_name", "end_junction_road_name",
    "pedal_cycles", "two_wheeled_motor_vehicles",       
    "cars_and_taxis", "buses_and_coaches", 
    "lgvs", "all_hgvs", "all_motor_vehicles"
]

# Creacion de un nuevo DataFrame reducido verticalmente
df_seleccionado = df.select(columnas_deseadas)

# Eliminacion estricta de filas con nulos, esto descarta registros incompletos como los que no tienen cruces definidos
df_limpio = df_seleccionado.dropna()

total_rows_final = df_limpio.count()
total_cols_final = len(df_limpio.columns)

print(f"ESTADO FINAL (Limpio):")
print(f"Total de registros conservados: {total_rows_final}")
print(f"Total de columnas conservadas: {total_cols_final}")
print(f"Registros eliminados: {total_rows_orig - total_rows_final}")
print("-" * 30)

# Guardado en formato CSV sobreescribiendo si ya existe la carpeta de destino
df_limpio.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("datasetLimpio")

print("\nProceso terminado. Dataset guardado en la carpeta: datasetLimpio")

spark.stop()