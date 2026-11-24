from abc import ABC, abstractmethod

class PipelineStep[T](ABC):
    @abstractmethod
    def run(self) -> T:
        pass
