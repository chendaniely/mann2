import random

import networkx as nx
import matplotlib.pyplot as plt
from pandas import DataFrame

from tqdm import tqdm

from mann2 import network
from mann2.helper_tqdm import HelperTqdm
from mann2 import agent_lens_attitudeDiffusion


class NetworkLensAttitudeDiffusion(network.Network, HelperTqdm):

    def __init__(self, config, logger):
        super(NetworkLensAttitudeDiffusion, self).__init__(config, logger)

    def create_agents_in_graph(self):
        self.logger.info('Addings agents into graph')
        num_agents = int(self.config['single_sim']['num_agents'])

        tqdm_sim_number, tqdm_show, tqdm_position = \
            self.parse_tqdm_config(self.config)
        for i in tqdm(range(num_agents),
                      desc='Sim:{} Creating Agents'.format(tqdm_sim_number),
                      disable=tqdm_show,
                      position=tqdm_position,
                      leave=False):
            self.logger.info('Creating agent: {}'.format(i))
            new_agent = agent_lens_attitudeDiffusion.AgentLensAttitudeDiffusion(
                self.config, self.logger)
            new_agent.train(self.config['lens']['in_training'])

            self.graph.add_node(new_agent.agent_id, agent=new_agent)

            self.logger.debug('Agent {} added to graph'.format(new_agent.agent_id))
        return(self)

    def generate_graph_network(self):
        self.logger.debug('Connecting agents into network')
        tqdm_sim_number, tqdm_show, tqdm_position = \
            self.parse_tqdm_config(self.config)
        for u, v in tqdm(self.nx_graph.edges_iter(),
                         desc='Sim:{} Gen graph network'.format(
                             tqdm_sim_number),
                         disable=tqdm_show,
                         position=tqdm_position,
                         leave=False):
            self.graph.add_edge(u, v)
            if bool(self.config['graph']['force_directed']):
                self.logger.info('Forcing direction by adding reverse edges')
                self.graph.add_edge(v, u)
        if self.config['graph']['show']:
            nx.draw_circular(self.graph)
            plt.show()
        self.logger.info("Network Graph of agents created.")
        return(self)

    def seed_agents(self):
        num_agents_seed = int(eval(
            self.config['single_sim']['seed_agents']['num_seed']))
        self.logger.info("Seeding {} agents in Network Lens Attitude Diffusion".\
                         format(num_agents_seed))
        seed_method = self.config['single_sim']['seed_agents']['seed_method']
        self.logger.info('Seed Method: {}'.format(seed_method))

        tqdm_sim_number, tqdm_show, tqdm_position = self.parse_tqdm_config(
            self.config)

        if seed_method == 'random':
            self.seed_agents_random(num_agents_seed,
                                    tqdm_sim_number, tqdm_show, tqdm_position)
        else: raise ValueError('Unknown seeding method passed')
        return(self)

    def seed_agents_random(self, num_agents_seed,
                           tqdm_sim_number, tqdm_show, tqdm_position):
        self.logger.info('Seeding Randomly')

        pool_of_agents = self.graph.node.keys()
        self.logger.debug('Pool of agents: {}'.format(pool_of_agents))

        agents_to_seed = random.sample(pool_of_agents, num_agents_seed)
        self.logger.debug('Agents picked for seed: {}'.format(agents_to_seed))

        seed_value_method = self.config['single_sim']['seed_agents']['seed_value']['seed_value_method']
        self.logger.info('Picking {} seed_value_options'.format(seed_value_method))

        for agent_id in tqdm(agents_to_seed,
                             desc='Sim:{} Seeding Agents'.format(tqdm_sim_number),
                             disable=tqdm_show,
                             position=tqdm_position,
                             leave=False):
            agent_selected = self.graph.node[agent_id]['agent']
            self.logger.info("Seeding Agent: {}".format(agent_selected))
            if seed_value_method == 'random':
                agent_selected.state = self.get_seed_agent_value_random()
            else:
                raise ValueError('Unknown seed value method: {}'.format(seed_value_method))

            if self.config['single_sim']['seed_agents']['cycle_after_seed']:
                self.logger.info('Cycling neural network after seed')
                in_file = self.config['lens']['seed']['in_seed_cycle']
                ex_file = self.config['lens']['seed']['ex_file']['fn']
                ex_file_append_id = self.config['lens']['seed']['ex_file']['append_agent_id']
                out_file = self.config['lens']['seed']['out_file']['fn']
                out_file_append_id = self.config['lens']['seed']['out_file']['append_agent_id']
                agent_selected.update('seed', -1, in_file,
                                      ex_file, ex_file_append_id,
                                      out_file, out_file_append_id)
            self.logger.debug('Finished Seeding Agent: {}'.format(agent_selected))
        return(self)

    def get_seed_agent_value_random(self):
        seed_options = self.config['single_sim']['seed_agents']['seed_value']['seed_value_options']
        selected_option = random.sample(seed_options.keys(), 1)[0]
        seed_values = seed_options[selected_option]
        new_state = DataFrame(seed_values)
        return(new_state)

    def scheduler_random_sequential_update(self, time_tick, output_f_agent_step_info):
        self.logger.debug('Schedule: Random Sequential Update')

        num_agents_update = eval(self.config['single_sim']['scheduler']['num_agents_update'])
        self.logger.debug('Num agents to update: {}'.format(num_agents_update))

        shuffled_agents = random.sample(self.graph.nodes(), num_agents_update)
        self.logger.debug("Shuffled agents: {}".format(shuffled_agents))

        for idx, agent_id in enumerate(shuffled_agents):
            agent = self.graph.node[agent_id]['agent']
            self.logger.debug('Updating Agent: {}'.format(agent))
            agent.update(mode=self.config['single_sim']['update_sim_mode'],
                         tick=time_tick,
                         output_f_agent_step_info=output_f_agent_step_info,
                         in_file=self.config['lens']['sim']['in_file'],
                         ex_file=self.config['lens']['sim']['ex_file']['fn'],
                         ex_file_append_id=self.config['lens']['sim']['ex_file']['append_agent_id'],
                         out_file=self.config['lens']['sim']['out_file']['fn'],
                         out_file_append_id=self.config['lens']['sim']['out_file']['append_agent_id'],
                         graph=self.graph)

    def step(self, time_tick, output_f_agent_step_info):
        self.logger.debug('Network Step')
        scheduler_type = self.config['single_sim']['scheduler']['type']
        self.logger.debug("Scheduler type: {}".format(scheduler_type))

        if scheduler_type == 'random_sequential_update':
            self.scheduler_random_sequential_update(time_tick, output_f_agent_step_info)
