from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import StopWordsRemover, RegexTokenizer, Tokenizer
from pyspark.sql.functions import when, col, udf, concat, lit
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import HashingTF, IDF
from scipy.spatial import distance

conf = SparkConf()
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

#   Read the data from business dataset
business = spark.read.json('/YELP/input/yelp_academic_dataset_business.json')
#   As the exectuion with the full YELP dataset is too long, we have selected the 1% of the dataset
business = business.sample(fraction=0.01, seed=42)

business = business.drop("address")
business = business.drop("attributes")
business = business.drop("hours")
business = business.drop("latitude")
business = business.drop("longitude")
business = business.na.drop()

business = business.withColumn("is_open", when(col("is_open") == 1, "open").otherwise("closed"))
business = business.withColumn("review_count", col("review_count").cast("string"))
business = business.withColumn("stars", (((2*col("stars")).cast("integer"))).cast("string"))

business = business.withColumn("text", concat(col("city"), lit(" "), col("is_open"), lit(" "), col("stars"), lit(" "), col("state"), lit(" "), col("categories")))

# Create the tokenizer
tokenizer = Tokenizer(inputCol="text", outputCol="words")
# Create the regexTokenizer
regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")

# Count the tokens
countTokens = udf(lambda words: len(words), IntegerType())

# Apply the tokenizers
tokenized = tokenizer.transform(business)
tokenized.select("text", "words")\
    .withColumn("tokens", countTokens(col("words")))

regexTokenized = regexTokenizer.transform(business)
regexTokenized.select("text", "words") \
    .withColumn("tokens", countTokens(col("words")))

business = regexTokenized
# Apply the StopWordsRemover
remover = StopWordsRemover(inputCol="words", outputCol="filtered_text")
business = remover.transform(regexTokenized)

# Creation of the hashing and apply it
hashingTF = HashingTF(inputCol="filtered_text", outputCol="rawFeatures")
business = hashingTF.transform(business)
# Creation of the IDF and apply it
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(business)
business = idfModel.transform(business)

# Set the id of the text to compare
businessRDD = business.rdd
vectors = businessRDD.map(lambda x: (x["business_id"], x["features"]))
# Get the featureVector of the id to compare
# For comparation, I have selected the first business
vectorComparacion = vectors.first()[1]
# Compare each id with the selected one
results = vectors.map(lambda x: (x[0], distance.cosine(x[1].toArray(), vectorComparacion.toArray())))
# Order the results and take the top 5
results = results.sortBy(lambda x: x[1], ascending = True).take(6)
# Convert results into a RDD
results_rdd = spark.sparkContext.parallelize(results)

# Convert elements of RDD into strings
results_string_rdd = results_rdd.map(lambda row: str(row))

# Save txt in HDFS
# The business id's and distance is saved 
results_string_rdd.saveAsTextFile('/YELP/output-businessRecommendation')
