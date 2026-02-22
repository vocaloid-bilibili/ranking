# common/proxy.py
from abc import ABC, abstractmethod


class Proxy(ABC):
    proxy_server: str

    @abstractmethod
    def random_proxy(self):
        pass
