from abc import ABC, abstractmethod


class Base(ABC):
    @abstractmethod
    def check_dependencies(self):
        """ Check ligand type by file path """
        pass
