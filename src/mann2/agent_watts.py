import agent_binary


class AgentWatts(agent_binary.AgentBinary):
    num_class_insances = 0

    def __init__(self, config, logger):
        super(AgentWatts, self).__init__()

        self._agent_id = AgentWatts.num_class_insances
        AgentWatts.num_class_insances += 1

        self.config = config
        self.logger = logger

        self.state = self.config['agent']['init_value']
        self.logger.debug("Created {}".format(self))

    def __hash__(self):
        return hash(self.agent_id)

    def __repr__(self):
        return '{}-{} STATE:{}'.\
            format(self.__class__.__name__, self.agent_id, self.state)

    @property
    def agent_id(self):
        return self._agent_id
