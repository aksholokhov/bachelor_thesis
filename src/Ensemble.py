from src.Devices import AirConditioningSystem, AbstractDevice
from threading import Lock
from random import Random
import numpy as np
from scipy.optimize import minimize
from numpy.linalg import matrix_power

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

class AbstractEnsemble:

    def __init__(self, number_of_devices, tick=0.1, default_policy = None, random_seed = None):
        self.__devices = []
        self.__N = default_policy.shape[0]
        self.__log = {"working" : np.zeros(self.__N), "ON" : 1}
        self.__log_lock = Lock()

        for i in range(number_of_devices):
            self.__devices.append(AbstractDevice(id=i,
                                                 log=self.__log,
                                                 log_lock=self.__log_lock,
                                                 default_policy=default_policy,
                                                 tick=tick,
                                                 random_seed = random_seed))

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

    def get_state_distribution(self):
        with self.__log_lock:
            return self.__log["working"]

class FastAbstractEnsemble:

    def __init__(self, number_of_devices, N, m, device_generator):
        self.__devices = []
        self.__n = number_of_devices
        self.__state_distribution = np.zeros(N)  # for debug purpose only: ensemble can not know this
        self.__control_distribution = np.zeros(m)   # as well as this
        self.log = {"accepted" : [],
                    "state_distribution" : self.__state_distribution,
                    "total_consumption": [],
                    "control_distribution" : self.__control_distribution
                    }
        self.__time = 0
        for i in range(self.__n):
            device = device_generator.generate_device(i, self.log, self.__time)
            self.__devices.append(device)

    def run(self, control):
        accepted = 0
        self.log["total_consumption"].append(0)
        for device in self.__devices:
            accepted += device.run(control)
        self.log["accepted"].append(accepted)
        self.__time = (self.__time + 1) % 24


class DeviceGenerator:

    def __init__(self, random_seed, room_model):
        rnd = Random()
        self.__rnd = rnd
        self.__rs = random_seed
        self.__room_model = room_model

    def generate_device(self, number, logger, time):
        rnd = self.__rnd
        params = self.__room_model.user_profile
        rnd.seed(self.__rs + number)
        temps = np.zeros(24)
        morning_time = int(rnd.normalvariate(params["morning_time_mean"], params["time_variance"])) % 24
        day_time = int(rnd.normalvariate(params["day_time_mean"], params["time_variance"])) % 24
        evening_time = int(rnd.normalvariate(params["evening_time_mean"], params["time_variance"])) % 24
        night_time = int(rnd.normalvariate(params["night_time_mean"], params["time_variance"])) % 24
        morning_time = min(morning_time, day_time-1)
        day_time = min(day_time, evening_time-1)
        evening_time = min(evening_time, night_time-1)
        if night_time <= morning_time:
            night_time = min(night_time, morning_time-1)

        temps[morning_time:day_time] = rnd.normalvariate(params["morning_temp_mean"], params["temp_variance"])
        temps[day_time:evening_time] = rnd.normalvariate(params["day_temp_mean"], params["temp_variance"])
        if night_time >= morning_time:
            temps[evening_time:night_time] = rnd.normalvariate(params["evening_temp_mean"], params["temp_variance"])
            night_temp = rnd.normalvariate(params["night_temp_mean"], params["temp_variance"])
            temps[night_time:] = night_temp
            temps[:morning_time] = night_temp
        else:
            evening_temp = rnd.normalvariate(params["evening_temp_mean"], params["temp_variance"])
            temps[evening_time:] = evening_temp
            temps[:night_time] = evening_temp
            temps[night_time:morning_time] = rnd.normalvariate(params["night_temp_mean"], params["temp_variance"])
        return ThermalDevice(temps, self.__room_model, logger, self.__rs + number, time)


class ThermalDevice:

    def __init__(self, temps, room_model, logger, seed, time):
        self.__temps = temps
        self.__room_model = room_model
        self.__logger = logger
        self.__rnd = Random()
        self.__rnd.seed(seed)
        state = self.__rnd.randint(0, self.__room_model.N-1)
        logger["state_distribution"][state] += 1
        self.__state = state
        self.__time = time
        self.__policy = self.choose_policy()
        self.__logger["control_distribution"][self.__policy] += 1

    # for debug purposes only: to be removed
    def get_temps(self):
        return self.__temps

    def run(self, policy):
        self.__logger["control_distribution"][self.__policy] -= 1
        self.__policy = self.choose_policy()
        if policy == 0:
            is_accept = 0
        else:
            is_accept = self.__room_model.accept_func(policy-1, self.__policy, self.__time)
            if is_accept:
                self.__policy = policy-1


        self.__logger["control_distribution"][self.__policy] += 1

        for _ in range(self.__room_model.tau):
            transition_probabilities = self.__room_model.controls[self.__policy][self.__state]
            next_state = self.__rnd.choices(range(self.__room_model.N), weights=transition_probabilities)[0]
            self.__logger["state_distribution"][self.__state] -= 1
            self.__logger["state_distribution"][next_state] += 1
            self.__state = next_state
            self.__logger["total_consumption"][-1] += self.__room_model.q[self.__state]
        self.__time = (self.__time + 1) % 24
        return is_accept

    def choose_policy(self):
        t = self.__time
        Q = self.__temps[t]
        return np.argmax(np.abs(self.__room_model.temps(self.__time) - Q))



def eiv(P):
    S, U = np.linalg.eig(P.T)
    stationary = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
    stationary = stationary / np.sum(stationary)
    return np.real(stationary)

def generate_features(pi, room_model, arm, t):
    if arm == 0:
        return [sum([room_model.q.T.dot(matrix_power(control.T, i).dot(pi)) for i in range(1, room_model.tau+1)])
                for control in room_model.controls]
    else:
        features = []
        for j, control in enumerate(room_model.controls):
            P = room_model.controls[arm-1] if room_model.accept_func(arm-1, j, t) else control
            k = sum([room_model.q.T.dot(matrix_power(P.T, i).dot(pi)) for i in range(1, room_model.tau+1)])
            features.append(k)
        return features

def generate_omega_features(pi, room_model, n, arm):
    P_i = room_model.controls[arm-1]
    features = []
    for P_j, n_j in zip(room_model.controls, n):
        k = sum([n_j*room_model.q.T.dot((matrix_power(P_i.T, xi) - matrix_power(P_j.T, xi)).dot(pi)) for xi in range(1, room_model.tau + 1)])
        features.append(k)
    return np.array(features)


def learn_regression(X, y, n):
    m = X.shape[1]
    f = lambda w: np.linalg.norm(X.dot(w) - y)**2
    cons = {'type' : 'eq', 'fun' : lambda x: sum(x) - n}
    bounds = tuple([(0, None) for _ in range(m)])
    res = minimize(f, np.zeros(m).reshape(-1, 1), constraints=cons, bounds=bounds)
    return res.x

def learn_omega(X, y):
    m = X.shape[1]
    f = lambda w: np.linalg.norm(X.dot(w) - y) ** 2
    bounds = tuple([(0, 1) for _ in range(m)])
    res = minimize(f, np.zeros(m).reshape(-1, 1), bounds=bounds)
    return res.x

class RoomModel:

    def __init__(self, environment_temp, controls, q, tau, user_profile, omega_is_known):
        self.__c = environment_temp
        self.controls = controls
        self.N = controls[0].shape[0]
        self.user_profile = user_profile
        self.q = q
        self.tau = tau # thermal_coefficient * tau ~= 4 -- works good
        self.__thermal_coefficient = lambda t: 1.2
        self.temps = lambda t: [environment_temp(t) + self.tau*self.__thermal_coefficient(t) * q.T.dot(eiv(x)) for x in controls]
        self.accept_func = lambda i, j, t: np.abs(self.temps(t)[i] - self.temps(t)[j]) < 3 and self.temps(t)[i] >= 17 and self.temps(t)[i] <= 26
        if omega_is_known:
            self.accept = np.array([[[self.accept_func(i, j, t)
                                for t in range(24)]
                                     for j in range(len(controls))]
                                        for i in range(len(controls))])


class SystemOperator:

    def __init__(self, generation_price, consumption_threshold):
        self.__gen_price = generation_price
        self.__time = 0
        self.__target = 0
        self.__target_delay = 0
        self.__consumption_threshold = consumption_threshold

    def send_consumption(self, consumption):
        self.__last_consumption = consumption
        next_hour = (self.__time + 1) % 24
        self.__target = self.__consumption_threshold #self.__consumption_threshold/self.__gen_price(next_hour)
        self.__time = next_hour

    def get_target(self):
        return self.__target

