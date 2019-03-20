#data_list = ['1MW', 'orbit_satellite', 'underwater_glider', 'nighttime_driver_visibility', 'offshore_plant', "radar_rain_gauge"]
data_list = ['1MW', 'orbit_satellite', 'nighttime_driver_visibility']
target_list =  ["refs"]#, "ipc", 'cpc', 'uspc']

gcn_command = "python gcn/gcn_train.py --data_path data/{} --model_path models/{} --data {}.pkl"
transformer_command = "python transformer/transformer_train.py --data_path data/{} --model_path models/{} --data data/{}.csv"
train_command = "python train.py --data_path data/{} --model_path models/{} --data data/{}.csv --graph_model {}"
import os
import itertools

target_value = [" ".join(target) for i in range(1,len(target_list)+1) for target in list(itertools.combinations(target_list,i))]

for data in data_list:
    #for target in target_list:
    #    command = gcn_command.format(data, data, target)
    #    print(command)
    #    os.system(command)
 
    #command = transformer_command.format(data, data, data)
#    print(command)
#    os.system(command)
    for t in target_value:
        command = train_command.format(data, data, data, t)
        print(command)
#:        os.system(command)
    


