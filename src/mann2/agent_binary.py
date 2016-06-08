import agent


class AgentBinary(agent.Agent):

    def __init__(self):
        super(AgentBinary, self).__init__()

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        assert new_state in [0, 1],\
            "Expected a 0 or 1, got {} instead".format(new_state)
        self._state = new_state
