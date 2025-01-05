import cv2
import dlib
from imutils import face_utils

# Detektor twarzy dlib i klasyfikator punktów charakterystycznych
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/debis/Downloads/shape_predictor_68_face_landmarks.dat")

# Inicjalizacja kamery
cap = cv2.VideoCapture(0)


# Funkcja do obliczania współczynnika "EAR" (Eye Aspect Ratio)
def eye_aspect_ratio(eye):
    # Odległości między punktami na oku
    A = dist(eye[1], eye[5]) #odleglosc w pionie 1
    B = dist(eye[2], eye[4]) #odleglosc w pionie 2
    C = dist(eye[0], eye[3]) #odleglosc z poziomie

    ear = (A + B) / (2.0 * C)
    return ear


# Funkcja obliczająca odległość między dwoma punktami
def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


# Próg EAR do wykrywania zamknięcia oczu
EAR_THRESHOLD = 0.3

# Zmienna do śledzenia stanu oczu (otwarte/zamknięte)
eyes_closed = False

while True:
    # Wczytaj klatkę z kamery
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Wykrywanie twarzy
    faces = detector(gray)

    for face in faces:
        # Określenie punktów charakterystycznych twarzy
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Wycinanie punktów odpowiadających oczom (lewe i prawe oko)
        left_eye = shape[42:48]
        right_eye = shape[36:42]

        # Oblicz EAR dla obu oczu
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Średni EAR
        ear = (left_ear + right_ear) / 2.0

        # Sprawdzenie, czy oczy są zamknięte
        if ear < EAR_THRESHOLD:
            if not eyes_closed:
                eyes_closed = True
                print("Oczy zamknięte!")
        else:
            if eyes_closed:
                eyes_closed = False
                print("Oczy otwarte!")

        # Rysowanie obwódki wokół oczu
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

    # Jeśli oczy są zamknięte, wyświetl napis na ekranie
    if eyes_closed:
        cv2.putText(frame, "Otworz oczy!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Wyświetlanie obrazu
    cv2.imshow("Eye Blink Detection", frame)

    # Zakończenie programu po naciśnięciu 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:  # Esc
        break

# Zwolnienie zasobów
cap.release()
cv2.destroyAllWindows()
