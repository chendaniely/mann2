import yaml

from tqdm import tqdm

from mann2 import model_lens
from mann2 import network_lens_attitudeDiffusion

from mann2.helper_tqdm import HelperTqdm

class ModelLensAttitudeDiffusion(model_lens.ModelLens, HelperTqdm):

    def __init__(self, config_file, logger_name):
        super(ModelLensAttitudeDiffusion, self).__init__()

        with open(config_file, 'r') as config_yaml:
            self.config = yaml.load(config_yaml)

        self.setup_model_meta(self.config)

        self.logger = self.setup_logging(self.config, logger_name)

        self.logger.debug('Model created with configurations and logging.')
        self.logger.info('Model Description: {}'.format(self.description))

    def run_model(self, output_f_agent_step_info):
        self.logger.info('Running model')
        total_ticks = range(self.config['single_sim']['num_time_ticks'])
        tqdm_sim_num, tqdm_show, tqdm_position = self.parse_tqdm_config(self.config)
        for time_tick in tqdm(total_ticks,
                              desc='Sim:{} Steps'.format(tqdm_sim_num),
                              disable=tqdm_show,
                              position=tqdm_position,
                              leave=False):
            self.step(time_tick, output_f_agent_step_info)
        self.logger.info('Model Run Finished')

    def run_batch_sweep_model(self): pass

    def setup_model_graph(self):
        self.logger.info('Setting up Model Graph')
        self.network = network_lens_attitudeDiffusion.\
                       NetworkLensAttitudeDiffusion(self.config, self.logger)
        self.network.\
            create_agents_in_graph().\
            generate_graph_network()
        return(self)

    def setup_model_run(self):
        self.logger.info('Setting up Model Run')
        self.network.seed_agents()
        self.logger.info('Finished setup_model_run')
        return(self)

    def setup_output_file(self, output_f_agent_step_info):
        s = '{},'*4 + '{}\n'
        ps = ','.join(['pos_{}'.format(i) for i in range(10)])
        ns = ','.join(['neg_{}'.format(i) for i in range(10)])
        output_f_agent_step_info.write(s.format('time', 'gid', 'aid', ps, ns))

    def step(self, time_tick, output_f_agent_step_info):
        self.logger.debug('Model step {}'.format(time_tick))
        self.network.step(time_tick, output_f_agent_step_info)

    def clean_up(self, output_f_agent_step_info):
        self.logger.info("Cleaning up simulation")

        self.logger.info("Writing remaining states")
        for agent_id in self.network.graph.nodes():
            agent = self.network.graph.node[agent_id]['agent']
            for time, state in agent.past_states_for_write.items():
                agent.write_state_to_f(time, state, output_f_agent_step_info)
        self.logger.info("Finished cleaning up simulation")
