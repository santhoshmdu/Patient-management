
import json

AdminFilename = "Admintable2.json"
PatientFilename = "PatientFileJSON.json"
file1 = "start.html"

users = {
    "Seshadhri":"Hi",
    "santhosh": "Hey",
    "Gayathri": "You"
}
fig = {
    "P1":{
        "management": "P1.jpg",
        "measure": "P1.jpg",
        "model": "P1.jpg"
        }
    }


users1 ={
    "Patient1": "P1",
    "Patient2": "P2",
    "Patient3": "P3",
    "Patient4": "P4",
    "Patient5": "P5",
    "Patient6": "P6",
    "Patient7": "P7",
    "Patient8": "P8",
    "Patient9": "P9",
    "Patient10": "P10",
    "Patient11": "P11",
    "Patient12": "P12",
    "Patient13": "P13",
    "Patient14": "P14",
    "Patient15": "P15",

}


port = "http://127.0.0.1:5000"
pathtoStartImage = "./assets/img/kare.jpg"


def load_data():
    with open(data_file, 'r') as file:
        data = json.load(file)
        return data




#def load_data():
    #   global datas
    #try:
        #   with open(data_file, 'r') as file:
    #   datas = json.load(file)
    #except FileNotFoundError:
#   datas = []


#def save_data():
#    with open(data_file, 'w') as file:
#        json.dump(datas, file)