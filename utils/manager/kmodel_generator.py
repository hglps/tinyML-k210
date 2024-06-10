from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

class KModelGenerator():

    def __init__(self, strategy: Strategy, model_file: str, output_filename: str) -> None:
        self._strategy = strategy
        self.model_file = model_file
        self.output_filename = output_filename

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        self._strategy = strategy      

    def run(self) -> None:
        print("Started KModel Generation...")
        result = self._strategy.generate_kmodel(self.model_file, self.output_filename)
        print("Finished KModel Generation...")


class Strategy(ABC):

    @abstractmethod
    def generate_kmodel(self, model_file: str, output_filename: str) -> None:
        pass
