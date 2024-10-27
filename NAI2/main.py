import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

temperature = ctrl.Antecedent(np.arange(15, 36, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
number_of_people = ctrl.Antecedent(np.arange(0, 13, 1), 'number_of_people')
fan_power = ctrl.Consequent(np.arange(0, 11, 1), 'fan_power')


temperature['low'] = fuzz.trimf(temperature.universe, [15, 15, 20])
temperature['medium'] = fuzz.trimf(temperature.universe, [18, 23, 28])
temperature['high'] = fuzz.trimf(temperature.universe, [27, 35, 35])

humidity.automf(3) #poor average good

number_of_people['low'] = fuzz.trimf(number_of_people.universe, [0, 0, 5])
number_of_people['medium'] = fuzz.trimf(number_of_people.universe, [3, 6, 10])
number_of_people['high'] = fuzz.trimf(number_of_people.universe, [8, 10, 12])

fan_power['low'] = fuzz.trimf(fan_power.universe, [0, 0, 4])
fan_power['medium'] = fuzz.trimf(fan_power.universe, [3, 5, 7])
fan_power['high'] = fuzz.trimf(fan_power.universe, [6, 8, 10])

temperature['low'].view()
humidity.view()
number_of_people.view()
fan_power.view()


rule1 = ctrl.Rule(temperature['low'] | humidity['poor'] | number_of_people['low'], fan_power['low'])
rule2 = ctrl.Rule(temperature['medium'] | humidity['average'] | number_of_people['high'], fan_power['medium'])
rule3 = ctrl.Rule(temperature['high'] | humidity['good'] | number_of_people['high'], fan_power['high'])
rule4 = ctrl.Rule(temperature['high'] | humidity['good'] | number_of_people['high'], fan_power['high'])


power_of_fan_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])

power_of_fan = ctrl.ControlSystemSimulation(power_of_fan_ctrl)

power_of_fan.input['temperature'] = 35
power_of_fan.input['humidity'] = 100
power_of_fan.input['number_of_people'] = 12


power_of_fan.compute()

print (power_of_fan.output['fan_power'])
fan_power.view(sim=power_of_fan)


plt.show()
