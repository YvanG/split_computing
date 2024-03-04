import socket
import pickle
import json
import bson
from pydoc import locate
import time
import argparse
import numpy as np
from sccommon import ndarray_list_to_bson, bson_to_ndarray_list
import numpy as np
import sys
import requests

class SCClient:

    def __init__(self, host, ip):
        self.server_address = (host, ip)
        self.sc_class = None
        self.sc_process = None
        self.sc_loader_class = None
        self.sc_loader = None

    def registerSCLoader(self, class_name = "scloader.SCLoader"):
        try:
            self.sc_loader_class = locate(class_name)
            self.sc_loader = self.sc_loader_class()
            print(f"Class {class_name} registered as SCLoader")
            self.sc_loader.initialize()
        except: 
            print("Class does not exist")       

    def registerSCProcess(self, class_name = "scprocess.SCProcess"):
        try:
            self.sc_class = locate(class_name)
            print(self.sc_class)
            self.sc_process = self.sc_class()
            print(self.sc_process)
            print(f"Class {class_name} registered as SCProcess")
            self.sc_process.initialize()
        except: 
            print("Class does not exist")       

    def run(self):  
        while True:
            input_data = self.sc_loader.next()
            if isinstance(input_data, bool):
                if input_data == False:
                    break
            #data = {}
            start_time = time.time()
            print(self.sc_process)
            output = self.sc_process.process_head(input_data)
            end_time = time.time()
            #data["data"] = output[0].__dict__
            #data["head_duration"] = end_time - start_time
            #data["timestamp"] = time.time()
            #print(type(data))     
            #s_data = json.dumps(data).encode('utf-8')
            output.append(np.array([end_time - start_time, time.time()]))

            s_data = ndarray_list_to_bson(output)
            #s_data = bson.dumps(data).encode('utf-8')
            #print(sys.getsizeof(output), len(output), output[1])
            print(sys.getsizeof(s_data))
            #print(s_data)
            #data2 = bson_to_ndarray_list(s_data)
            #print(data2)
      
            #s_data = json.dumps(data, default=lambda o: o.__dict__, sort_keys=True, indent=4).encode('utf-8')
            url = f"http://{self.server_address[0]}:{self.server_address[1]}/upload_data"
            response = requests.post(url, data=s_data)
            print(response.text)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog="Scplit Computing HEAD", description="Application runs HEAD of the network within split computing framework")

    parser.add_argument('-ho', '--host', default="localhost")
    parser.add_argument('-p', '--port', default=9999)
    parser.add_argument('-P', '--process', default="scprocess.SCProcess")
    parser.add_argument('-L', '--loader', default="scloader.SCLoader")

    args = parser.parse_args()

    HOST = args.host
    PORT = args.port 
    print(args.process)

    # HOST, PORT = "localhost", 9999
    client = SCClient(HOST, PORT)
    client.registerSCLoader(args.loader)
    client.registerSCProcess(args.process)
    client.run()

