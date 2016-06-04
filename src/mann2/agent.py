import abc


class Agent(abc.ABC):

    @property
    def agent_id(self):
        return self._agent_id

    @property
    def state(self):
        return self._state

    @state.setter
    @abc.abstractmethod
    def state(self, new_state):
        self._state = new_state
