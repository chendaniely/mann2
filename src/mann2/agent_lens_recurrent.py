from mann2 import agent_lens


class AgentLensRecurrent(agent_lens.AgentLens):

    def __init__(self):
        super(AgentLensRecurrent, self).__init__()

    @property
    def state(self):
        return self._state
