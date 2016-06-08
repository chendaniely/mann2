import abc


class Agent(abc.ABC):

    @abc.abstractmethod
    def __repr__(self):
        pass

    @property
    @abc.abstractmethod
    def agent_id(self):
        pass

    @property
    def state(self):
        return self._state

    @state.setter
    @abc.abstractmethod
    def state(self, new_state):
        self._state = new_state
