import socket
import sys
import json
import bson
from pydoc import locate
import time
import argparse 
from sccommon import json_to_ndarray_list, bson_to_ndarray_list
from flask import Flask, request, jsonify
from flask.views import MethodView


app = Flask(__name__)


class SCServer(MethodView): 

    def __init__(self, process_name):

        self.sc_class = None
        self.sc_process = None
        self.registerSCProcess(process_name)
        self.data = None

    def post(self):
        receive_time = time.time()
        if request.data:
            data = json_to_ndarray_list(request.data)
            temp = data.pop()
            head_duration = temp[0]
            head_timestamp = temp[1]
            print(head_duration, head_timestamp)
            self.sc_process.initialize()
            start_time = time.time()
            print(self.sc_process)
            output = self.sc_process.process_tail(data)
            end_time = time.time()
            tail_duration = end_time - start_time
            transmission_duration = receive_time - head_timestamp
            timesum = transmission_duration + tail_duration + head_duration
            print("SPEED RESULTS:")
            print("--------------")
            print(f"Head duration: {head_duration}, Tail duration {tail_duration}, Transmission Duration {transmission_duration}, SUM: {timesum}")
            print("-------------------------------------------------------")
            print("Results")
            print("-------")
            return jsonify({"message": "Data received successfully"}), 200
        else:
            return jsonify({"message": "No data received"}), 400        

    def registerSCProcess(self, class_name = "scprocess.SCProcess"):
        try:
            self.sc_class = locate(class_name)
            self.sc_process = self.sc_class()
            print(f"Class {class_name} registered as SCProcess")
        except: 
            print("Class does not exist")       



if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(prog="Scplit Computing HEAD", description="Application runs HEAD of the network within split computing framework")

    parser.add_argument('-ho', '--host', default="0.0.0.0")
    parser.add_argument('-p', '--port', default=9999, type=int)
    parser.add_argument('-P', '--process', default="scprocess.SCProcess")

    args = parser.parse_args()
    
    process_name = args.process
    print(process_name)
    matrix_upload_view = SCServer.as_view('upload_data', process_name=process_name)

    # Adding the URL rule for the class-based view
    app.add_url_rule('/upload_data', view_func=matrix_upload_view)    
    
    server_address = (args.host, args.port)
    print('starting up on %s port %s' % server_address)
    app.run(debug=False, host=args.host, port=args.port)