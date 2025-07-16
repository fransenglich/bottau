
import BrokerABC

class YFinanceInterface(BrokerABC.BrokerABC):
    def get_data(symbol, n, timeframe):
        pass

    def orders(symbol, lot, buy=True, id_position=None):
        pass

    def resume():
        pass

    def run(symbol, long, short, lot):
        pass

    def close_all_night():
        pass