"""Check reusable preflight sockets without permitting an active listener."""
import socket
import unittest


def preflight(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind(('127.0.0.1',port))


class PortTest(unittest.TestCase):
    def test_active_listener_is_rejected(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.bind(('127.0.0.1',0));server.listen(1)
            with self.assertRaises(OSError):preflight(server.getsockname()[1])

    def test_restart_after_server_active_close(self):
        server=socket.socket();server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        server.bind(('127.0.0.1',0));port=server.getsockname()[1];server.listen(1)
        client=socket.create_connection(('127.0.0.1',port));accepted,_=server.accept()
        accepted.shutdown(socket.SHUT_WR);accepted.close()
        client.recv(1);client.close();server.close()
        preflight(port)


if __name__=='__main__':unittest.main()
