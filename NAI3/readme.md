Cel projektu:
Celem projektu jest stworzenie systemu rekomendacji filmów, który wykorzystuje metody takie jak 
współczynnik korelacji Pearsona oraz odległość Euklidesowa między użytkownikami w celu przewidywania filmów,
które mogą ich zainteresować. Projekt działałby efektywniej w przypadku większej ilości danych, jednak nie
wszyscy koledzy wypełnili tabele ze swoimi obejrzanymi filmami. System mógłby być również bardziej funkcjonalny,
gdyby został połączony z odpowiednim API, jednak ze względu na ograniczony czas nie udało się
zrealizować tej funkcjonalności.

Autorzy projektu:
Paulina Debis s25067
Krystian Jank s24586

Aby uruchomić projekt należy wpisać w terminalu:
python main.py --user1 "{IMIE NAZWISKO Z JSON}" --score-type "{Pearson || Euclidean}" --recommend

np
python main.py --user1 "Pawel Czapiewski"  --score-type "Pearson" --recommend


Jeśli posiadamy zainstalowanego pythona wystarczy zainstalować biblioteki użyte w projekcie za pomocą:
pip install numpy



