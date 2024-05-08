import configparser
from sumolib import checkBinary
import os
import sys


def import_train_configuration(config_file):
    """
    从config文件中读取在训练过程需要使用的参数
    parameter：config_file：config文件的路径
    return：config：读取的参数
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}
    config['gui'] = content['simulation'].getboolean('gui')
    config['total_episodes'] = content['simulation'].getint('total_episodes')
    config['max_steps'] = content['simulation'].getint('max_steps')
    config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
    config['green_duration'] = content['simulation'].getint('green_duration')
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')
    config['num_layers'] = content['model'].getint('num_layers')
    config['width_layers'] = content['model'].getint('width_layers')
    config['width_layers_decrease'] = content['model'].getint('width_layers_decrease')
    config['batch_size'] = content['model'].getint('batch_size')
    config['learning_rate'] = content['model'].getfloat('learning_rate')
    config['training_epochs'] = content['model'].getint('training_epochs')
    config['memory_size_min'] = content['memory'].getint('memory_size_min')
    config['memory_size_max'] = content['memory'].getint('memory_size_max')
    config['num_states_width'] = content['agent'].getint('num_states_width')
    config['num_states_height'] = content['agent'].getint('num_states_height')
    config['num_actions'] = content['agent'].getint('num_actions')
    config['gamma'] = content['agent'].getfloat('gamma')
    config['models_path_name'] = content['dir']['models_path_name']
    config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
    return config


def import_test_configuration(config_file):
    """
    从config文件中读取在测试过程需要使用的参数
    parameter：config_file：config文件的路径
    return：config：读取的参数
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}
    config['gui'] = content['simulation'].getboolean('gui')
    config['max_steps'] = content['simulation'].getint('max_steps')
    config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
    config['episode_seed'] = content['simulation'].getint('episode_seed')
    config['green_duration'] = content['simulation'].getint('green_duration')
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')
    config['num_states_width'] = content['agent'].getint('num_states_width')
    config['num_states_height'] = content['agent'].getint('num_states_height')
    config['num_actions'] = content['agent'].getint('num_actions')
    config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
    config['models_path_name'] = content['dir']['models_path_name']
    config['model_to_test'] = content['dir'].getint('model_to_test') 
    return config

def import_prune_configuration(config_file):
    """
    Read the config file regarding the pruning and import its content
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}
    config['gui'] = content['simulation'].getboolean('gui')
    config['total_episodes'] = content['simulation'].getint('total_episodes')
    config['max_steps'] = content['simulation'].getint('max_steps')
    config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
    config['green_duration'] = content['simulation'].getint('green_duration')
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')
    config['model_folder_path_0'] = content['model']['model_folder_path_0']
    config['model_folder_path_1'] = content['model']['model_folder_path_1']
    config['model_folder_path_2'] = content['model']['model_folder_path_2']
    config['model_folder_path_3'] = content['model']['model_folder_path_3']
    config['model_folder_path_4'] = content['model']['model_folder_path_4']
    config['batch_size'] = content['model'].getint('batch_size')
    config['learning_rate'] = content['model'].getfloat('learning_rate')
    config['training_epochs'] = content['model'].getint('training_epochs')
    config['memory_size_min'] = content['memory'].getint('memory_size_min')
    config['memory_size_max'] = content['memory'].getint('memory_size_max')
    config['num_states_width'] = content['agent'].getint('num_states_width')
    config['num_states_height'] = content['agent'].getint('num_states_height')
    config['num_actions'] = content['agent'].getint('num_actions')
    config['gamma'] = content['agent'].getfloat('gamma')
    config['models_path_name'] = content['dir']['models_path_name']
    config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
    return config

def set_sumo(gui, sumocfg_file_name, max_steps):
    """
    配置SUMO的各项参数
    gui：布尔值Ture或False
    sumocfg_file_name：执行的.sumocfg文件
    max_steps:SUMO模拟运行的时间
    """
    # 从$SUMO_HOME/tools 目录中导入python模块
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # 设置cmd模式或可视模式
    if gui == False:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
 
    # 设置在仿真时cmd可以控制运行sumo
    sumo_cmd = [sumoBinary, "-c", os.path.join('intersection', sumocfg_file_name), "--no-step-log", "true", "--waiting-time-memory", str(max_steps)]

    return sumo_cmd


def set_train_path(models_path_name):
    """
    Create a new model path with an incremental integer, also considering previously created model paths
    """
    models_path = os.path.join(os.getcwd(), models_path_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    dir_content = os.listdir(models_path)
    if dir_content:
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(models_path, 'model_'+new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path 


def set_test_path(models_path_name, model_n, episode_seed):
    """
    Returns a model path that identifies the model number provided as argument and a newly created 'test' path
    """
    model_folder_path = os.path.join(os.getcwd(), models_path_name, 'model_'+str(model_n), '')

    if os.path.isdir(model_folder_path):    
        plot_path = os.path.join(model_folder_path, 'test_'+str(episode_seed), '')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        return model_folder_path, plot_path
    else: 
        sys.exit('The model number specified does not exist in the models folder')