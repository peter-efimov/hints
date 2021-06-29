# Запустить спарк на рабочем компе в локальном режиме
import findspark
findspark.init('C:\spark\spark-3.0.1-bin-hadoop2.7')
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('test_app').getOrCreate()