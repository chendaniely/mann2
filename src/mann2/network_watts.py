import networkx as nx
import matplotlib.pyplot as plt

import network
import agent_watts


class NetworkWatts(network.Network):

    def __init__(self, config, logger):
        super(NetworkWatts, self).__init__(config, logger)

    def create_agents_in_graph(self):
        self.logger.info('Adding agents into graph')
        num_agents = int(self.config['single_sim']['num_agents'])
        # self.all_agents = {}
        for _ in range(num_agents):
            new_agent = agent_watts.AgentWatts(self.config, self.logger)

            self.graph.add_node(new_agent.agent_id, agent=new_agent)
            self.logger.debug(
                'Agent {} added to graph'.format(new_agent.agent_id))

            # self.all_agents[new_agent.agent_id] = new_agent
            # self.logger.debug(
            #     'Agent {} added to dict of all_agents'.format(new_agent.agent_id))
        return(self)

    def generate_graph_network(self):
        self.logger.debug("Connecting agents into network")
        for u, v in self.nx_graph.edges_iter():
            self.graph.add_edge(u, v)
            if bool(self.config['graph']['force_directed']):
                self.logger.info('Forcing direction by adding reverse edges')
                self.graph.add_edge(v, u)
        # nx.draw_circular(self.graph)
        # plt.show()
        self.logger.info("Network Graph of agents created.")
        return(self)
