from abc import ABCMeta, abstractmethod, ABC


class Model(ABC):

    @abstractmethod
    def setup_model_meta(self): raise NotImplementedError()

    @abstractmethod
    def setup_logging(self): raise NotImplementedError()

    @abstractmethod
    def setup_model_run(self): raise NotImplementedError()

    @abstractmethod
    def run_model(self): raise NotImplementedError()

    @abstractmethod
    def run_batch_sweep_model(self): raise NotImplementedError()

    @abstractmethod
    def step(self): raise NotImplementedError()
