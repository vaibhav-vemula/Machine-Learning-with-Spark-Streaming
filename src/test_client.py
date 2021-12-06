from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession

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
        df.show()

        
lines.foreachRDD(extract)

ssc.start()
ssc.awaitTermination()     