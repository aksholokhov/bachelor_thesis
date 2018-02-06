import numpy as np
from threading import Thread, Lock
from time import sleep

class GeneralDevice(Thread):

    def __init__(self):
        Thread.__init__(self)
        pass


class AirConditioningSystem(GeneralDevice):

    def __init__(self, name, log, log_lock, continuous = False, tick = 0.1):
        GeneralDevice.__init__(self)
        self.__MAX_COMFORT_TEMP = 30
        self.__MIN_COMFORT_TEMP = 14
        self.__MEAN_SENSITIVITY = 1
        self.__SENSITIVITY_VARIANCE = 2
        self.__ENERGY_CONSUMPTION = 1
        self.__DEFAULT_POLICY = [[0.5, 0.5], [0.5, 0.5]]
        self.__P = self.__DEFAULT_POLICY

        self.__p_lock = Lock()
        self.__tick = tick
        self.__log = log
        self.__log_lock = log_lock
        self.__name = name
        self.__optimal_temperature = np.random.randint(self.__MIN_COMFORT_TEMP, self.__MAX_COMFORT_TEMP)
        self.__temperature = np.random.randint(self.__MIN_COMFORT_TEMP, self.__MAX_COMFORT_TEMP)
        self.__state = np.random.randint(0, 2)

        with self.__log_lock:
            self.__log["working"] += self.__state
            self.__log["%s_temp" % self.__name] = self.__temperature

        if continuous:
            raise Exception("Continuous model hasn't implemented yet :(")
            #self.__sensitivity = np.abs(np.random.normal(self.__MEAN_SENSITIVITY, self.__SENSITIVITY_VARIANCE))
            #self.__p = lambda t: 1 - 1./np.cosh(self.__sensitivity*(t - self.__optimal_temperature))
        else:
            self.__sensitivity = np.abs(np.random.normal(0, self.__SENSITIVITY_VARIANCE))

            def p(t):
                with self.__p_lock:
                    if np.abs(t - self.__optimal_temperature) < self.__sensitivity:
                        return 0
                    if self.__state == 0:
                        return self.__P[0][1]
                    elif self.__state == 1:
                        return self.__P[1][0]
                    else:
                        raise Exception("wrong state")

            self.__p = p

    def run(self):
        power_off = 0
        while not power_off:
            sleep(self.__tick)
            last_state = self.__state

            if np.random.random() < self.__p(self.__temperature):
                self.__state ^= 1
            else:
                self.__temperature += 1 if self.__state == 0 else -1

            with self.__log_lock:
                self.__log["working"] += self.__state - last_state
                self.__log["energy"] += last_state*self.__ENERGY_CONSUMPTION
                self.__log["%s_temp"%self.__name] = self.__temperature
                if self.__log["ON"] == 0:
                    power_off = 1


    def change_policy(self, P):
        with self.__p_lock:
            self.__P = P