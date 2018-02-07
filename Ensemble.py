from Devices import AirConditioningSystem
from threading import Lock

class AirConditioningEnsemble:

    def __init__(self, number_of_devices, tick=0.1, default_policy = None, random_seed = None):
        self.__devices = []
        self.__log = {"energy" : 0, "working" : 0, "ON" : 1}
        self.__log_lock = Lock()
        self.last_check = 0

        for i in range(number_of_devices):
            self.__devices.append(AirConditioningSystem(id=i, log=self.__log, log_lock=self.__log_lock, tick=tick,
                                                        default_policy=default_policy, random_seed = random_seed))

    def run(self):
        if len(self.__devices) == 0:
            raise Exception("No devices to launch")

        for device in self.__devices:
            device.start()

    def stop(self):
        with self.__log_lock:
            self.__log["ON"] = 0

        for device in self.__devices:
            device.join()

        self.__devices = []

    def change_policy(self, P):
        for device in self.__devices:
            device.change_policy(P)

    def get_consumption(self):
        with self.__log_lock:
            return self.__log["energy"]

    def get_number_of_working_devices(self):
        with self.__log_lock:
            return self.__log["working"]

    def get_temperature(self, name):
        with self.__log_lock:
            return self.__log["%s_temp"%name]