import os
import socket
import time
host,port = os.environ['NDS_MASTER'].split(':')
for _ in range(300):
    try:
        with socket.create_connection((host,int(port)),timeout=.2): break
    except OSError: time.sleep(.1)
else: raise RuntimeError('Master unavailable')
