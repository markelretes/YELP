from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

conf = SparkConf()
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

#   Read the data from business dataset
business = spark.read.json('/yelp/input/yelp_academic_dataset_business.json')

#   Get the top 10 most rated businesses
top10reviews = business.orderBy(col('review_count').desc()).limit(10)

#   Select the reviews to store in the text file
selected_columns = ['business_id', 'name', 'review_count']
selected_reviews = top10reviews.select(*selected_columns)

#   Save the output in directory /output-top10business in the HDFS
selected_reviews.rdd.map(lambda row: ",".join([str(row[column]) for column in selected_columns])).saveAsTextFile('/yelp/output-top10business')