import numpy as np

from mann2 import agent_binary


class AgentWatts(agent_binary.AgentBinary):
    num_class_insances = 0

    def __init__(self, config, logger):
        super(AgentWatts, self).__init__()

        self._agent_id = AgentWatts.num_class_insances
        AgentWatts.num_class_insances += 1

        self.config = config
        self.logger = logger

        self.state = self.config['agent']['init_value']
        self.threshold = eval(self.config['agent']['threshold'])
        self._past_states_for_write = {}
        self.logger.debug("Created {}".format(self))

    def __hash__(self):
        return hash(self.agent_id)

    def __repr__(self):
        return '{}-{} STATE:{} TH:{}'.\
            format(self.__class__.__name__,
                   self.agent_id, self.state,
                   self.threshold)

    def state_string(self):
        return self.state

    @property
    def agent_id(self):
        return self._agent_id

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        assert 0 <= value <= 1
        self._threshold = value

    @property
    def past_states_for_write(self):
        return self._past_states_for_write

    @past_states_for_write.setter
    def past_states_for_write(self, time):
        assert self._past_states_for_write.get(time) is None,\
            "Time already exists"
        self._past_states_for_write[time] = self.state
