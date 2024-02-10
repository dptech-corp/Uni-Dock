from abc import ABC, abstractmethod


class ConfGeneratorBase(ABC):
    @staticmethod
    @abstractmethod
    def check_env():
        raise NotImplementedError

    @abstractmethod
    def generate_conformation(self, *args, **kwargs):
        raise NotImplementedError
