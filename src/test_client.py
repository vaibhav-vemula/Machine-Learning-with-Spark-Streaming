import socket

c = socket.socket()
c.connect(('localhost',6100))

while True:
    print(c.recv(500000).decode())