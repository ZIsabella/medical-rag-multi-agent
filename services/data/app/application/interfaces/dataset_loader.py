from abc import ABC, abstractmethod
from typing import Iterable
from app.domain.documents import QADocument

class DatasetLoader(ABC):
    @abstractmethod
    def load(self) -> Iterable[QADocument]:
        pass
