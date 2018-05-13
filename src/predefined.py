import numpy as np
from numpy import array

default_controls = [
    array([[0.8, 0.2, 0.], [0.8, 0.2, 0.], [0.8, 0.2, 0]]),
    array([[0.2, 0.8, 0.], [0.2, 0.8, 0.], [0.2, 0.8, 0]]),
    array([[0.2, 0., 0.8], [0.5, 0., 0.5], [0.2, 0., 0.8]]),
    array([[0.5, 0.25, 0.25], [0., 0.3, 0.7], [0.3, 0.0, 0.7]]),
    array([[0.9, 0.1, 0.], [0.1, 0.9, 0], [0.5, 0.5, 0]]),
    array([[0.7, 0.3, 0.], [0., 0.7, 0.3], [0.3, 0., 0.7]])
]

q = array([0, 1, 2])

environment_temp = lambda t: 20 + 3*np.sin(2*np.pi/24*(t+15))

default_preference_profile = {
    "morning_time_mean" : 7,
    "day_time_mean" : 13,
    "evening_time_mean" : 18,
    "night_time_mean" : 23,
    "morning_temp_mean" : 24,
    "day_temp_mean" : 19,
    "evening_temp_mean" : 23,
    "night_temp_mean" : 23,
    "temp_variance" : 1,
    "time_variance" : 1
}

generation_price = lambda t: 1 + 0.5*np.sin(2*np.pi*(t+13)/(24))
