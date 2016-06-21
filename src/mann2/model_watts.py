import yaml
import sys
import os

from tqdm import tqdm

from mann2 import model_binary
from mann2 import network_watts
from mann2.helper_tqdm import HelperTqdm


class ModelWatts(model_binary.ModelBinary, HelperTqdm):

    def __init__(self, config_file, logger_name):
        super(ModelWatts, self).__init__()

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
        self.network.seed_agents()
        return(self)

    def run_model(self, output_f_agent_step_info):
        self.logger.info('Running model')

        total_ticks = range(self.config['single_sim']['num_time_ticks'])

        tqdm_sim_num, tqdm_show, tqdm_position = self.parse_tqdm_config(
            self.config)
        for time_tick in tqdm(total_ticks,
                              desc='Sim:{} Steps'.format(tqdm_sim_num),
                              disable=tqdm_show,
                              position=tqdm_position,
                              leave=False):
            self.step(time_tick, output_f_agent_step_info)
        self.logger.info('Model Run Finished')

    def step(self, time_tick, output_f_agent_step_info):
        self.logger.debug('Model step {}'.format(time_tick))
        self.network.step(time_tick, output_f_agent_step_info)
        self.logger.debug("Size of network: {}"
                          .format(sys.getsizeof(self.network)))
        self.logger.debug('Finished Model step {}'.format(time_tick))

    def clean_up(self, output_f_agent_step_info):
        self.logger.info("Cleaning up simulation")

        self.logger.info("Writing remaining states")
        for agent_id in self.network.graph.nodes():
            agent = self.network.graph.node[agent_id]['agent']

            for time, state in agent.past_states_for_write.items():
                output_f_agent_step_info.write(
                    "{},{},{},{}\n"
                    .format(time, agent_id, agent.agent_id, state))

    def run_batch_sweep_model(self):
        pass
