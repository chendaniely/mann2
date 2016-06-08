from random import choice
import networkx as nx

import model_watts


def main():
    # initialize the model
    model = model_watts.ModelWatts('config_model_watts.yaml', 'model_watts')

    # setup model
    model.setup_model_graph()

    # TODO unit test this check
    # selected_agent_id = choice(model.network.graph.nodes())
    # print(model.network.graph.nodes())
    # print('selected_agent_id: {}'.format(selected_agent_id))
    # print(nx.info(model.network.graph))
    # print(model.network.graph.node[selected_agent_id])
    # print(type(model.network.graph.node[selected_agent_id]['agent']))

    # model.setup_model_run()

if __name__ == '__main__':
    main()
