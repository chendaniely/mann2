import agent_binary


class AgentWatts(agent_binary.AgentBinary):
    num_class_insances = 0

    def __init__(self, config, logger):
        self._agent_id = AgentWatts.num_class_insances
        AgentWatts.num_class_insances += 1

        self.config = config
        self.config_agent = config['single_sim']['agents']  # convenience

        self.logger = logger

        self._state = self.config_agent['init_value']

        self.logger.debug("Agent {} created.  State: {}".format(self.agent_id,
                                                                self.state))

    def __hash__(self):
        return hash(self.agent_id)

    def __repr__(self):
        return '{}-{} STATE:{}'.\
            format(self.__class__.__name__, self.agent_id, self.state)

