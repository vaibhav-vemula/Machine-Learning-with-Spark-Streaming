from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import length, regexp_replace
from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans
from pyspark.sql.types import DoubleType

sc = SparkContext.getOrCreate()
ssc = StreamingContext(sc,5)
spark = SparkSession.builder.getOrCreate()

lines = ssc.socketTextStream("localhost", 6100)
def extract(rd):
    x = spark.read.json(rd)
    if(len(x.head(1)) > 0):
        f = x.collect()
        data = list(f[0].asDict().values())
        df=spark.createDataFrame(data)
        data = x.withColumn('length',length(x['Message'])).withColumnRenamed("Spam/Ham","class").withColumnRenamed("Message","text")
        data = data.drop('Subject')
        data = data.withColumn("text", regexp_replace("text", ",", ""))\
            .withColumn("text", regexp_replace("text", ",", ""))\
            .withColumn("text", regexp_replace("text", ":", ""))\
            .withColumn("text", regexp_replace("text", "-", ""))\
            .withColumn("text", regexp_replace("text", "/", ""))\
            .withColumn("text", regexp_replace("text", "\n", ""))\
            .withColumn("text", regexp_replace("text", "  ", ""))\
            .withColumn("text", regexp_replace("text", ";", ""))
        data= data.where(data['length']<20000)
        tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
        stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
        count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec')
        idf = IDF(inputCol="c_vec", outputCol="tf_idf")
        ham_spam_to_num = StringIndexer(inputCol='class',outputCol='label')
        clean_up = VectorAssembler(inputCols=['tf_idf','length'],outputCol='features')
        data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,count_vec,idf,clean_up])
        cleaner = data_prep_pipe.fit(data)
        clean_data = cleaner.transform(data)
        clean_data = clean_data.select(['label','features'])
        (training,testing) = clean_data.randomSplit([0.7,0.3])
        # Naive Bayes
        nb = NaiveBayes()
        spam_predictor = nb.fit(training)
        test_results = spam_predictor.transform(testing)
        acc_eval = MulticlassClassificationEvaluator()
        acc = acc_eval.evaluate(test_results)
        print("Accuracy of model at predicting spam was: {}".format(acc))
        # Linear Regression
        lr = LinearRegression(featuresCol = 'features', labelCol='label', maxIter=10, regParam=0.3, elasticNetParam=0.8)
        lr_model = lr.fit(training)
        trainingSummary = lr_model.summary
        print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
        print("r2: %f" % trainingSummary.r2)
        # K Means Clustering
        kmeans = KMeans().setK(4).setSeed(1)
        kmodel = kmeans.fit(training)
        predictions=kmodel.transform(testing)
        predictions = predictions.withColumn("prediction", predictions["prediction"].cast(DoubleType()))
        evaluatorsvm = MulticlassClassificationEvaluator(predictionCol="prediction")
        print(evaluatorsvm.evaluate(predictions))
        

lines.foreachRDD(extract)
ssc.start()
ssc.awaitTermination()



