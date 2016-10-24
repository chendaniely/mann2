import networkx as nx
import matplotlib.pyplot as plt
import random
import sys
# from time import sleep  # used for testing

from tqdm import tqdm

from mann2 import network
from mann2 import agent_watts
from mann2.helper_tqdm import HelperTqdm


class NetworkWatts(network.Network, HelperTqdm):

    def __init__(self, config, logger):
        super(NetworkWatts, self).__init__(config, logger)

    def create_agents_in_graph(self):
        self.logger.info('Adding agents into graph')
        num_agents = int(self.config['single_sim']['num_agents'])

        tqdm_sim_number, tqdm_show, tqdm_position = \
            self.parse_tqdm_config(self.config)
        for _ in tqdm(range(num_agents),
                      desc='Sim:{} Creating Agents'.format(tqdm_sim_number),
                      disable=tqdm_show,
                      position=tqdm_position,
                      leave=False):
            new_agent = agent_watts.AgentWatts(self.config, self.logger)

            self.graph.add_node(new_agent.agent_id, agent=new_agent)
            self.logger.debug(
                'Agent {} added to graph'.format(new_agent.agent_id))
        return(self)

    def generate_graph_network(self):
        self.logger.debug("Connecting agents into network")
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
        self.logger.info("Seeding {} Agents in Network Watts".
                         format(num_agents_seed))

        seed_method = self.config['single_sim']['seed_agents']['seed_method']
        self.logger.info('Seed Method: {}'.format(seed_method))

        tqdm_sim_number, tqdm_show, tqdm_position = self.parse_tqdm_config(
            self.config)

        if seed_method == 'random':
            self.logger.info('Seeding Randomly')
            pool_of_agents = self.graph.node.keys()
            self.logger.debug('Pool of agents: {}'.format(pool_of_agents))

            agents_to_seed = random.sample(pool_of_agents, num_agents_seed)
            self.logger.debug(
                'Agents picked for seed: {}'.format(agents_to_seed))

            for agent_id in tqdm(agents_to_seed,
                                 desc='Sim:{} Seeding Agents'.format(
                                     tqdm_sim_number),
                                 disable=tqdm_show,
                                 position=tqdm_position,
                                 leave=False):
                agent_selected = self.graph.node[agent_id]['agent']
                self.logger.info("Seeding Agent: {}".format(agent_selected))

                agent_selected.state = self.config['single_sim']['seed_agents']['seed_value']
                self.logger.debug("Seed values added to past states t=-1")

                agent_selected.past_states_for_write = -1
                self.logger.debug("Finished Seeding Agent: {}".format(agent_selected))
        else:
            raise ValueError('Unknown seeding method passed')

    def scheduler_random_sequential_update(self, time_tick,
                                           output_f_agent_step_info):
        self.logger.debug("Scheduler: Random Sequential Update")
        num_agents_update = eval(self.config['single_sim']['scheduler']['num_agents_update'])
        self.logger.debug("Num agents to update: {}".
                          format(num_agents_update))
        shuffled_agents = random.sample(self.graph.nodes(), num_agents_update)
        self.logger.debug("Shuffled agents: {}".format(shuffled_agents))
        for idx, agent_id in enumerate(shuffled_agents):
            # TODO this update code should be moved into the AgentWatts code
            # so the call becomes something like: agent.update('')
            agent = self.graph.node[agent_id]['agent']
            self.logger.debug('Updating Agent: {}'.format(agent))
            predecessors = self.graph.predecessors(agent_id)
            len_predecessors = len(predecessors)
            if len_predecessors == 0:
                self.logger.debug("No predecessors found.")
            else:
                sum_pred_state = sum([self.graph.node[x]['agent'].state
                                      for x in predecessors])
                prop_1 = sum_pred_state / len_predecessors
                self.logger.debug("sum:{} / pred:{} = prop_1:{}".
                                  format(sum_pred_state,
                                         len_predecessors,
                                         prop_1))

                agent_threshold = agent.threshold
                agent_state = agent.state

                if prop_1 >= agent_threshold and agent_state == 0:
                    self.logger.debug("Updating Agent: {}".format(agent))
                    self.graph.node[agent_id]['agent'].state = 1
                    self.logger.debug("Updated Agent: {}".format(agent))
                elif prop_1 >= agent_threshold and agent_state == 1:
                    self.logger.debug("{} already has state 1".format(agent))
                elif prop_1 < agent_threshold:
                    self.logger.debug(
                        "Agent {} did not meet threshold".format(agent))
                else:
                    raise ValueError(
                        "Unexpected prop/threshold/state combination")

            agent.past_states_for_write = time_tick
            size_of_past_states = sys.getsizeof(agent.past_states_for_write)
            self.logger.debug("size of past states: {} len of past states: {}".format(
                size_of_past_states,
                (agent.past_states_for_write)))
            agent_write_size_threshold = self.config['single_sim']['agent_write_size']
            if size_of_past_states >= agent_write_size_threshold:
                self.logger.debug('State size threshold limit reached, writing to output...')
                output_f_agent_step_info.write(
                    "{},{}\n".format(time_tick, agent.state_string))
                agent.past_states_for_write = {}

    def step(self, time_tick, output_f_agent_step_info):
        scheduler_type = self.config['single_sim']['scheduler']['type']
        self.logger.debug("Scheduler type: {}".format(scheduler_type))
        if scheduler_type == 'random_sequential_update':
            self.scheduler_random_sequential_update(
                time_tick, output_f_agent_step_info)
