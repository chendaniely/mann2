import abc
import networkx as nx


class Network(abc.ABC):

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.logger.info('Creating Network')

        graph_type = self.config['graph']['type']
        self.logger.info('Creating {}'.format(graph_type))
        eval_str = '{}'.format(graph_type)
        self.logger.debug('eval: {}'. format(eval_str))
        self.graph = eval(eval_str)
        self.logger.debug('Type of self.graph: {}'.format(type(self.graph)))

        self.logger.info('Generating Network')
        graph_generator = self.config['graph']['generator']
        self.logger.info('Graph Generator: {}'.format(graph_generator))
        eval_str = '{}'.format(graph_generator)
        self.logger.debug('eval: {}'.format(eval_str))
        network = eval(eval_str)
        self.logger.debug('Type of network: {}'.format(type(network)))
        self.network = network  # used to generate network using agent as nodes

        nx_edge_list_filename = self.config['graph']['nx_edge_list_filename']
        self.logger.info(
            'Writing networkx edge list: {}'.format(nx_edge_list_filename))
        nx.write_edgelist(network, nx_edge_list_filename)

    @abc.abstractmethod
    def generate_network(self): pass
