import yaml

import model_binary
import network_watts


class ModelWatts(model_binary.ModelBinary):

    def __init__(self, config_file, logger_name):
        with open(config_file, 'r') as config_yaml:
            self.config = yaml.load(config_yaml)

        self.setup_model_meta(self.config)

        self.logger = self.setup_logging(self.config, logger_name)

        self.logger.debug('Model created with configurations and logging.')
        self.logger.info('Model Description: {}'.format(self.description))

    def setup_model_graph(self):
        self.logger.info('Setting up Model Graph')
        self.network = network_watts.NetworkWatts(self.config, self.logger)
        self.network.create_agents_in_graph().generate_graph_network()
        return(self)

    def setup_model_run(self):
        self.logger.info('Setting up Model Run')

    def run_model(self):
        self.logger.info('running model')

    def step(self):
        self.logger.debug('stepping')

    def run_batch_sweep_model(self):
        pass
