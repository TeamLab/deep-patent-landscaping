from multiprocessing import Process, Queue
import os
from datetime import datetime

data_path = "data/origin_{}.csv"
#target = "refs"

data_list = ['1MW', 'orbit_satellite', 'underwater_glider', 'nighttime_driver_visibility', 'offshore_plant', "radar_rain_gauge"]
target_list =  ["refs", "ipc", 'cpc', 'uspc']

base_command = "python gcn/get_graph_data.py --data {} --target {}"

def f(path, target):
    command = "python gcn/get_graph_data.py --data {} --target {}".format(path, target)
    print(command)
    os.system(command)
    print("END : {} / {} / {}".format(datetime.now(), target, path))

print("START :", datetime.now())
for file in data_list:
    path = data_path.format(file, file)
    for target in target_list:
        proc = Process(target=f, args=(path, target,)) 
        proc.start()

