#Autorzy kodu
#Paulina Debis s25067
#Krystian Jank s24586

#Cel projektu:
#Stworzenia systemu który będzie polegał na użyciu logiki rozmytej.
#aplikacja powinna poprawnie dostosować moc wentylatora do warunków panujących w pomieszczeniu,
#takich jak temperatura, wilgotność i liczba osób

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

"""
Definicja zmiennych:
Zmienne wejściowe w raz z zakresem (Antecedent)=> Temperatura,Wilgotność,Liczba osób w pokoju
Zmienne wyjściwoe w raz z zakresem (Consequent) => Moc wiatraków
"""
temperature = ctrl.Antecedent(np.arange(15, 36, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
number_of_people = ctrl.Antecedent(np.arange(0, 13, 1), 'number_of_people')
fan_power = ctrl.Consequent(np.arange(0, 11, 1), 'fan_power')

"""
Kategoryzujemy/Definujemy wartości odpowiednio dla temepatury,wilogtności (aut.),liczby osób.
fuzz.trimf tworzy trójkątną funkcję przynależności dla low,medium,high
"""
temperature['low'] = fuzz.trimf(temperature.universe, [15, 15, 20])
temperature['medium'] = fuzz.trimf(temperature.universe, [18, 23, 28])
temperature['high'] = fuzz.trimf(temperature.universe, [27, 35, 35])

humidity.automf(3) # aut. poor average good

number_of_people['low'] = fuzz.trimf(number_of_people.universe, [0, 0, 5])
number_of_people['medium'] = fuzz.trimf(number_of_people.universe, [3, 6, 10])
number_of_people['high'] = fuzz.trimf(number_of_people.universe, [8, 10, 12])

fan_power['low'] = fuzz.trimf(fan_power.universe, [0, 0, 4])
fan_power['medium'] = fuzz.trimf(fan_power.universe, [3, 5, 7])
fan_power['high'] = fuzz.trimf(fan_power.universe, [6, 8, 10])

"""Tworzenie wykresów"""
temperature['low'].view()
humidity.view()
number_of_people.view()
fan_power.view()

"""
Definicja reguł logiki rozmytej:
Definiujemy jaką moc powinien posiadać wiatrak w zależności od zmiennych wejsciowych.
Znak | (or) służy do łączenia warunków  
"""
rule1 = ctrl.Rule(temperature['low'] & humidity['poor'] & number_of_people['low'], fan_power['low'])
rule2 = ctrl.Rule(temperature['low'] & humidity['poor'] & number_of_people['high'], fan_power['medium'])
rule3 = ctrl.Rule(temperature['low'] & humidity['good'] & number_of_people['low'], fan_power['medium'])
rule4 = ctrl.Rule(temperature['medium'] & humidity['poor'] & number_of_people['low'], fan_power['medium'])
rule5 = ctrl.Rule(temperature['medium'] & humidity['good'] & number_of_people['low'], fan_power['medium'])
rule6 = ctrl.Rule(temperature['medium'] & humidity['poor'] & number_of_people['high'], fan_power['high'])
rule7 = ctrl.Rule(temperature['high'] & number_of_people['high'], fan_power['high'])
rule8 = ctrl.Rule(temperature['high'] & humidity['poor'] & number_of_people['low'], fan_power['medium'])
rule9 = ctrl.Rule(temperature['high'] & humidity['good'] & number_of_people['low'], fan_power['high'])
rule10 = ctrl.Rule(temperature['low'] & humidity['good'] & number_of_people['high'], fan_power['medium'])
rule11 = ctrl.Rule(temperature['medium'] & humidity['average'] & number_of_people['medium'], fan_power['medium'])

"""
Tworzenie systemu kontrolnego poprzez zdefiniowanie wcześniej reguł (power_of_fan_ctrl)
Inicjalizacja symulacji zdefinowanego systemu (power_of_fan)  

"""
power_of_fan_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11])
power_of_fan = ctrl.ControlSystemSimulation(power_of_fan_ctrl)


"""
Podane wartości wejsciowe => temperatura,wilgotnosc i liczba osob w pokoju
Nasze zakresy podane na początku projektu:
Temperatura => 15-35
Wilgotność  => 0-100
Liczba osób w pokoju => 0-12

"""
power_of_fan.input['temperature'] = 30
power_of_fan.input['humidity'] = 99
power_of_fan.input['number_of_people'] = 11

"""
Wykonywanie obiczen dla podanych wartosci
Wyświetlenie mocy w konsoli
Wyświetlenie mocy wentylatora do zdefiniowanych kategorii.
Wyświetla wszystkie wykresy na ekranie. 
"""
power_of_fan.compute()
print (power_of_fan.output['fan_power'])
fan_power.view(sim=power_of_fan)
plt.show()
