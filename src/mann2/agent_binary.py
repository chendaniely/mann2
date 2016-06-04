import agent


class AgentBinary(agent.Agent):

    def __init__(self):
        super(AgentBinary, self).__init__()

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        assert new_state in [0, 1]
        self._state = new_state
