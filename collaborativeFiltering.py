from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import udf, col
from pyspark.ml.evaluation import RegressionEvaluator

conf = SparkConf()
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

# Load the data from the reviews dataset in the HDFS
reviews = spark.read.json('/yelp/input/yelp_academic_dataset_review.json')

# Select columns business_id, stars and user_id
data = reviews["business_id", "stars", "user_id"]

# Create a column for unique integer values for business_id
business_indexer = StringIndexer(inputCol="business_id", outputCol="business_idx")
data = business_indexer.fit(data).transform(data)
data = data.withColumn("business_idx", col("business_idx").cast("int"))

# For converting user_id to integer values, we will be using two dictionaries:
# user_id_to_int and int_to_user_id
user_id_to_int = {}
int_to_user_id = {}

# Retrive all user IDs from the column in the dataframe
user_ids = data.select("user_id").distinct().rdd.map(lambda row: row[0]).collect()

# Assign integer values to user IDs
counter = 0
for user_id in user_ids:
    if user_id not in user_id_to_int:
        user_id_to_int[user_id] = counter
        int_to_user_id[counter] = user_id
        counter += 1

# We define a mapper to to convert strings to integers using the dictionary
user_id_mapper = udf(lambda user_id: user_id_to_int[user_id], IntegerType())

# We create a new column 'user_idx' from the mapper
data = data.withColumn('user_idx', user_id_mapper(data['user_id']))

# We split the data into training (80%) and testing (20%) sets
(training, test) = data.randomSplit([0.8, 0.2])

# We create an ALS model and train it
als = ALS(rank=15, maxIter=20, regParam=0.2, userCol="user_idx", itemCol="business_idx", ratingCol="stars", coldStartStrategy="drop")
model = als.fit(training)

# Once trained, we make predictions and evaluate them
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="stars", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)

# We generate top 10 business recommendations for users: 0, 1, 2, 3, 4.
userSubset = [0, 1, 2, 3, 4]

users = spark.createDataFrame([(user,) for user in userSubset], ["user_idx"])
userRecommendations = model.recommendForUserSubset(users, 10)

# Save the output recommendations for 5 users in directory /yelp/output-collaborativeFiltering in the HDFS
userRecommendations.rdd.map(lambda row: str((row))).saveAsTextFile('/yelp/output-collaborativeFiltering')