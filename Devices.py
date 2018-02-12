import numpy as np
from random import Random
from threading import Thread, Lock
from time import sleep

class GeneralDevice(Thread):

    def __init__(self):
        Thread.__init__(self)


class AirConditioningSystem(GeneralDevice):

    def __init__(self, id, log, log_lock, continuous = False, tick = 0.1, default_policy = None, random_seed = None):
        GeneralDevice.__init__(self)

        rnd = Random()
        if random_seed is not None:
            rnd.seed(random_seed + int(id))

        self.__rnd = rnd

        if default_policy is None:
            default_policy = [[0.5, 0.5], [0.5, 0.5]]

        self.__MAX_COMFORT_TEMP = 30
        self.__MIN_COMFORT_TEMP = 14
        self.__SENSITIVITY_VARIANCE = 3
        self.__ENERGY_CONSUMPTION = 1
        self.__MIN_TEMP = 10
        self.__MAX_TEMP = 40
        self.__DEFAULT_POLICY = default_policy
        self.__P = self.__DEFAULT_POLICY

        self.__p_lock = Lock()
        self.__tick = tick
        self.__log = log
        self.__log_lock = log_lock
        self.__name = id
        self.__optimal_temperature = rnd.randint(self.__MIN_COMFORT_TEMP, self.__MAX_COMFORT_TEMP)
        self.__temperature = rnd.randint(self.__MIN_COMFORT_TEMP, self.__MAX_COMFORT_TEMP)
        self.__state = 1#rnd.randint(0, 1)

        with self.__log_lock:
            self.__log["working"] += self.__state
            self.__log["%s_temp" % self.__name] = self.__temperature

        if continuous:
            raise Exception("Continuous model hasn't implemented yet :(")
            #self.__sensitivity = np.abs(np.random.normal(self.__MEAN_SENSITIVITY, self.__SENSITIVITY_VARIANCE))
            #self.__p = lambda t: 1 - 1./np.cosh(self.__sensitivity*(t - self.__optimal_temperature))
        else:
            self.__sensitivity = rnd.randint(1, self.__SENSITIVITY_VARIANCE)

            def p(t):
                with self.__p_lock:
                    if np.abs(t - self.__optimal_temperature) < self.__sensitivity:
                        return 0
                    if self.__state == 0:
                        if t < self.__optimal_temperature:
                            return 0
                        return self.__P[0][1]
                    elif self.__state == 1:
                        if t > self.__optimal_temperature:
                            return 0
                        return self.__P[1][0]
                    else:
                        raise Exception("wrong state")

            self.__p = p

    def run(self):
        power_off = 0
        while not power_off:
            sleep(self.__tick)
            last_state = self.__state

            if self.__rnd.random() < self.__p(self.__temperature):
                self.__state ^= 1
                #print(self.__state, self.__temperature, self.__p(self.__temperature), self.__optimal_temperature, self.__sensitivity)
            else:
                self.__temperature += 1 if self.__state == 0 else -1
                self.__temperature = max(min(self.__temperature, self.__MAX_TEMP), self.__MIN_TEMP)

            with self.__log_lock:
                self.__log["working"] += self.__state - last_state
                self.__log["energy"] += last_state*self.__ENERGY_CONSUMPTION
                self.__log["%s_temp"%self.__name] = self.__temperature
                if self.__log["ON"] == 0:
                    power_off = 1


    def change_policy(self, P):
        with self.__p_lock:
            self.__P = P