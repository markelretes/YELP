from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import HashingTF, IDF, StopWordsRemover, RegexTokenizer, Tokenizer
from pyspark.sql.functions import col, udf, size, when


conf = SparkConf()
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

# Load the data from the reviews dataset
reviews = spark.read.json('/yelp/input/yelp_academic_dataset_review.json')

# Limit the dataset to a 1% of it
data = reviews.sample(fraction=0.01, seed=42)

# Keep the 'stars' and 'text' columns
selected_columns = ['stars', 'text']
selected_reviews = data.select(*selected_columns)

# The reviews with 4 and 5 stars are positive and 1, 2 and 3 negative
data = data.withColumn("StarsLabel", when((data.stars == 4)|(data.stars == 5), "Positive").otherwise("Negative"))
data = data.withColumn("label", when(data.StarsLabel == "Positive", 1).otherwise(0))

# Create the tokenizer
tokenizer = Tokenizer(inputCol="text", outputCol="words")

# Create the regexTokenizer
regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")

# Count the tokens
countTokens = udf(lambda words: len(words), IntegerType())

# Apply the tokenizers
tokenized = tokenizer.transform(data)
tokenized.select("text", "words")\
    .withColumn("tokens", countTokens(col("words")))

regexTokenized = regexTokenizer.transform(data)
regexTokenized.select("text", "words") \
    .withColumn("tokens", countTokens(col("words")))

# Apply the StopWordsRemover
remover = StopWordsRemover(inputCol="words", outputCol="filtered_text")
data = remover.transform(regexTokenized)
data = data.filter(size("filtered_text") > 0)

# Create the hashing and apply it
hashingTF = HashingTF(inputCol="filtered_text", outputCol="rawFeatures")
df1 = hashingTF.transform(data)

# Create the IDF and apply it
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(df1)
df1 = idfModel.transform(df1)

df1 = df1.select("features", "label")
train_data, test_data = df1.randomSplit([0.8, 0.2], seed=42)

# Specify the input column and output column for LogisticRegression
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Fit the model
lrModel = lr.fit(train_data)

predictions = lrModel.transform(test_data)

predictions.rdd.map(lambda row: str(row)).saveAsTextFile('/yelp/output-predictions')

