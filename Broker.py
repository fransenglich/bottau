
import os

import BrokerABC

def init() -> BrokerABC.BrokerABC:
    if os.name == "nt":
        import MT5
        return MT5.MT5()

    elif os.name == "posix":
        import YFinanceInterface
        return YFinanceInterface.YFinanceInterface()

    else:
        raise Exception("Bug: unhandled OS.")
