import networkx as nx
import matplotlib.pyplot as plt

import network
import agent_watts


class NetworkWatts(network.Network):

    def __init__(self, config, logger):
        super(NetworkWatts, self).__init__(config, logger)

        print(self.config)

    def generate_network(self):
        self.logger.info('Adding agents into graph')
        num_agents = int(self.config['single_sim']['num_agents'])
        all_agents = {}
        for _ in range(num_agents):
            new_agent = agent_watts.AgentWatts(self.config, self.logger)
            self.graph.add_node(new_agent)
            self.logger.debug(
                'Agent {} added to graph'.format(new_agent.agent_id))

            all_agents[new_agent.agent_id] = new_agent
            self.logger.debug(
                'Agent {} added to dict of all_agents'.format(new_agent.agent_id))

        self.logger.debug("Connecting agents into network")
            self.graph.add_edge(all_agents[u], all_agents[v])
        nx.draw_circular(self.graph)
        plt.show()
        for u, v in self.nx_graph.edges_iter():
        self.logger.info("Network Graph of agents created.")
