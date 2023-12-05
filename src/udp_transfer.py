# class for the client to send and receive data from the server

from socket import *
import sys
import traceback
import logging

class UdpTransfer:
    def __init__(self,in_ip,in_port,in_buffer=1024):
    
        self.ip=in_ip
        self.port=int(in_port)
        self.addr = (self.ip,self.port)
        self.sock = socket(AF_INET,SOCK_DGRAM)
        self.buffer = in_buffer
        
    def send_data(self, execution_mode, file_path, other_message = ""):
        
        file_name = file_path[file_path.rfind("/")+1:]
        try:
            if other_message == "":
                self.sock.sendto((execution_mode+"@"+file_name).encode(),self.addr)
            else:
                self.sock.sendto((execution_mode+"@"+file_name+"@"+other_message).encode(),self.addr)
            f=open(file_path,"rb")
            data = f.read(self.buffer)
            while (data):
                #if(self.sock.sendto(data,self.addr)):
                    self.sock.sendto(data,self.addr) # new
                    data = f.read(self.buffer)
            f.close()
        except Exception as e:
            logging.error(traceback.format_exc())
            # print("doc:",e.__doc__)
            # print("message:",e.message)
        
    def recieve_data(self):

        message, ip = self.sock.recvfrom(self.buffer)
        return message.decode()
