from abc import ABC, abstractmethod

class BrokerABC(ABC):
    @abstractmethod
    def get_data(symbol, n):
        pass

    @abstractmethod
    def orders(symbol, lot, buy=True, id_position=None):
        pass

    @abstractmethod
    def resume():
        pass

    @abstractmethod
    def run(symbol, long, short, lot):
        pass

    @abstractmethod
    def close_all_night():
        pass

