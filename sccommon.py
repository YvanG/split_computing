import json
from json import JSONEncoder
import numpy as np 
import bson

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def ndarray_to_json(data):
    encodedNumpyData = json.dumps({"data":data.tolist()})
    return encodedNumpyData

def json_to_ndarray(data):
    decodedArrays = json.loads(data)
    finalNumpyArray = np.asarray(decodedArrays["array"])
    return finalNumpyArray

def ndarray_to_bson(data):
    encodedNumpyData = bson.dumps({"data":data.tolist()})
    return encodedNumpyData

def bson_to_ndarray(data):
    decodedArrays = bson.loads(data)
    finalNumpyArray = np.asarray(decodedArrays["array"])
    return finalNumpyArray

def ndarray_list_to_json(data):
    list_of_lists = []
    for item in data:
        list_of_lists.append({"item":item.tolist()})
    encoded_list = json.dumps(list_of_lists)
    return encoded_list

def json_to_ndarray_list(data):
    decodeddata = json.loads(data)
    final_data = []
    for item in decodeddata:
        final_data.append(np.asarray(item["item"]))
    return final_data

def ndarray_list_to_bson(data):
    list_of_lists = []
    for item in data:
        #list_of_lists.append({"item":item.tolist()})
        print(type(item))
        list_of_lists.append(item.tolist())
    #print(list_of_lists)
    encoded_list = bson.dumps({"data": list_of_lists})
    return encoded_list

def bson_to_ndarray_list(data):
    #decodeddata = bson.loads(data)
    decodeddata = bson.loads(data)

    final_data = []
    for item in decodeddata["data"]:
        final_data.append(np.asarray(item))
    return final_data    

if __name__ == "__main__":

    numpyArrayOne = np.array([[11, 22, 33], [44, 55, 66], [77, 88, 99]])
    numpyArrayTwo = np.array([[1, 2, 3],[11, 22, 33], [44, 55, 66], [77, 88, 99]])

    l = [numpyArrayOne, numpyArrayTwo]

    print(l)

    print()

    ll = ndarray_list_to_bson(l)

    print(ll)

    print()

    lll = bson_to_ndarray_list(ll)

    print(lll)


    """
    # Serialization
    numpyData = {"array": numpyArrayOne}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    print("Printing JSON serialized NumPy array")
    print(encodedNumpyData)

    # Deserialization
    print("Decode JSON serialized NumPy array")
    decodedArrays = json.loads(encodedNumpyData)

    finalNumpyArray = np.asarray(decodedArrays["array"])
    print("NumPy Array")
    print(finalNumpyArray)
    """