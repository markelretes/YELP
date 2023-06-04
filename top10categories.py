from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, explode, col, desc

conf = SparkConf()
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

#   Read the data from business dataset
business = spark.read.json('/YELP/input/yelp_academic_dataset_business.json')

#   Column selection
data = business['categories', 'stars']
topCategories = data.withColumn("categories", split(data["categories"], ", "))

#   Calculation of the mean rating for each category
topCategories = topCategories.select(explode(topCategories.categories).alias("category"), topCategories.stars)
topCategories = topCategories.groupBy("category").agg({"stars": "avg"})
topCategories = topCategories.sort(desc("avg(stars)")).limit(10)

#   Save the output on HDFS
topCategories.rdd.map(lambda row: str((row))).saveAsTextFile('/YELP/output-top10categories')