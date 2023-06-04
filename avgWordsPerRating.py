from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.types import IntegerType

conf = SparkConf()
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

# Load the data from the reviews dataset
reviews = spark.read.json('/yelp/input/yelp_academic_dataset_review.json')

# Select columns stars and text
starsText = reviews['stars', 'text']

# Use tokenizer to split the text from user reviews into words
tokenizer = Tokenizer(inputCol="text", outputCol="words")

regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")

countTokens = udf(lambda words: len(words), IntegerType())

tokenized = tokenizer.transform(starsText)
tokenized.select("text", "words")\
    .withColumn("tokens", countTokens(col("words")))

regexTokenized = regexTokenizer.transform(starsText)
regexTokenized.select("text", "words") \
    .withColumn("tokens", countTokens(col("words")))

starsText = regexTokenized['stars', 'words']

# Compute the average number of words per rating (1-5)
starsTextRDD = starsText.rdd
starsTextRDD = starsTextRDD.map(lambda x: (x[0], len(x[1])))
starsText = spark.createDataFrame(starsTextRDD)
starsText = starsText.groupBy("_1").agg({"_2": "avg"})

# Save the output in a textfile in the directory /yelp/output-avgWordsPerRating in the HDFS
starsText.rdd.map(lambda row: str((row))).saveAsTextFile('/yelp/output-avgWordsPerRating')
