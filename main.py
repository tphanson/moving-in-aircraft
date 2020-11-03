import sys
import socket
from test import floorNet

# Init botshell
def gen_botshell():
    botshell = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    botshell.connect("/app/bot_shell.sock")
    botshell.sendall(b"wake_head\n")
    return botshell

if __name__ == "__main__":

    if sys.argv[1] == '--test':
        if sys.argv[2] == 'debug':
            floorNet.infer(gen_botshell(), debug=True)
        if sys.argv[2] == 'infer':
            floorNet.infer(gen_botshell(), debug=False)
        if sys.argv[2] == 'cluster':
            floorNet.cluster()
    else:
        print("Error: Invalid option!")
