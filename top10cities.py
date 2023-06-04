from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import desc

conf = SparkConf()
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

# Read the json file of business dataset
business = spark.read.json('/yelp/input/yelp_academic_dataset_business.json')

# Select the columns city and stars
topCities = business["city", "stars"]

# Get the average rating per city
topCities = topCities.groupBy("city").agg({"stars": "avg"})

# Get the top 10 with the highest average rating
topCities = topCities.sort(desc("avg(stars)")).limit(10)