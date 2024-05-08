from utils import import_test_configuration, set_sumo, set_test_path
from generator import TrafficGenerator


config = import_test_configuration(config_file='attack_testing_settings.ini')
TrafficGen = TrafficGenerator(
            config['max_steps'], 
            config['n_cars_generated'],
        )

TrafficGen.generate_routefile(seed=config['episode_seed'])