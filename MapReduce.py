from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, LongType
import time

#Initialize Spark Session and Context
spark = SparkSession.builder.master("local[*]").appName("Task4").getOrCreate()
sc = spark.sparkContext

#Define schema to properly read the JSON metadata
schema = StructType([
    StructField("app_id", LongType(), nullable=True),
    StructField("description", StringType(), nullable=True),
    StructField("tags", ArrayType(StringType()), nullable=True)
])

#Read metadata JSON into a DataFrame and convert to RDD with app_id and tags
meta_data_df = spark.read.option("multiline", "true").schema(schema).json("fixed_games_metadata.json")
meta_data_rdd = meta_data_df.rdd.map(lambda row: (str(row.app_id), row.tags))

#Read recommendations, filter out header and malformed lines, then map to app_id and is_recommended
df_recs_rdd = sc.textFile("recommendations.csv") \
    .map(lambda line: line.split(",")) \
    .filter(lambda x: len(x) > 1 and x[0] != "app_id") \
    .map(lambda x: (x[0], x[4].strip().lower()))


start = time.time()

#Join metadata and recommendation RDDs on app_id
joined_rdd = meta_data_rdd.join(df_recs_rdd)

#Filter only records that are recommended
recommended_only_rdd = joined_rdd.filter(lambda x: x[1][1] == "true")

#Extract (tag, 1) for each tag in recommended games using flatMap
tag_recommend_counts_rdd = recommended_only_rdd.flatMap(lambda x: [
    (tag.strip(), 1) for tag in x[1][0] if tag is not None
])

#Reduce by key to count total recommendations per tag
tag_recommend_counts_rdd = tag_recommend_counts_rdd.reduceByKey(lambda a, b: a + b)

#Take top 10 tags with highest recommendation counts
top_tags = tag_recommend_counts_rdd.takeOrdered(10, key=lambda x: -x[1])

end = time.time()

print("Top 10 Most Recommended Tags:")
for tag, count in top_tags:
    print(f"{tag}: {count}")

print(f"Task 4 - Time: {end - start:.2f} sec")

spark.stop()
