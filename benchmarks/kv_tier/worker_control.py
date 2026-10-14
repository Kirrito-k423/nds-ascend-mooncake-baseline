"""Clear replicas through the exact SDK client which successfully published them.

Task-owned Unix sockets stay inside the model container's /evidence bind mount.
No device operation, remote command execution, or new SDK client is involved.
"""
import json
import os
from pathlib import Path
import socket
import threading
import traceback

MAX_MESSAGE=1024*1024


def read_message(connection):
    chunks=[];size=0
    while True:
        data=connection.recv(65536)
        if not data:break
        chunks.append(data);size+=len(data)
        if size>MAX_MESSAGE:raise ValueError('Worker control message exceeds 1MiB')
    return json.loads(b''.join(chunks))


class WorkerControl:
    def __init__(self,store,source_host):
        self.store=store;self.source_host=source_host;self.owned=set();self.lock=threading.Lock()
        self.hostname=store.get_hostname()
        assert self.hostname.startswith(source_host+':')

    def remember(self,keys,returns):
        assert len(keys)==len(returns)
        with self.lock:self.owned.update(k for k,rc in zip(keys,returns) if rc==0)

    def clear(self,requested,segments):
        assert segments and all(isinstance(s,str) and s.startswith(self.source_host+':') for s in segments)
        with self.lock:keys=sorted(set(requested)&self.owned)
        result={s:list(self.store.batch_replica_clear(keys,s)) for s in sorted(set(segments))} if keys else {}
        return {'client_hostname':self.hostname,'pid':os.getpid(),'owned_keys':keys,'clear_results':result}

    def start(self,directory):
        path=Path(directory)/f'worker-clear-{os.getpid()}-{self.hostname.rsplit(":",1)[1]}.sock'
        assert not path.exists(),'Refuse an existing worker socket'
        server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        server.bind(str(path));os.chmod(path,0o600);server.listen(4);server.settimeout(.5)
        self.stopped=threading.Event();self.server=server
        def serve():
            while not self.stopped.is_set():
                try:connection,_=server.accept()
                except socket.timeout:continue
                except OSError:
                    if self.stopped.is_set():break
                    raise
                with connection:
                    connection.settimeout(30)
                    try:
                        body=read_message(connection);result=self.clear(body['keys'],body['segments'])
                    except BaseException:result={'error':traceback.format_exc()}
                    connection.sendall(json.dumps(result).encode())
        self.thread=threading.Thread(target=serve,daemon=True);self.thread.start()
        self.path=str(path);return self.path

    def close(self):
        self.stopped.set();self.server.close();self.thread.join(timeout=2)
        assert not self.thread.is_alive(),'Worker control thread did not stop'
        Path(self.path).unlink()


def request_clear(path,keys,segments):
    resolved=Path(path)
    assert resolved.is_absolute() and resolved.parent==Path('/evidence/disk'),'Unexpected worker socket directory'
    assert resolved.name.startswith('worker-clear-') and resolved.suffix=='.sock'
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
        connection.settimeout(30);connection.connect(path)
        connection.sendall(json.dumps({'keys':keys,'segments':segments}).encode());connection.shutdown(socket.SHUT_WR)
        result=read_message(connection)
    assert 'error' not in result,result.get('error')
    return result
