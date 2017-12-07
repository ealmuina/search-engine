def receive_string(sock):
    result = ''
    while True:
        current = sock.recv(4096)
        if not current:
            return result
        result += current.decode()
