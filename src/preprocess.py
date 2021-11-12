import pyspark
import socket

c = socket.socket()
c.connect(('localhost',6100))