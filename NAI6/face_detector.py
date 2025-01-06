import cv2
import dlib
from imutils import face_utils

# do dlib potrzebna jest instalacja CMake i dodatnie CMake/bin do zmiennych środowiskowych

# Detektor twarzy dlib i klasyfikator punktów charakterystycznych
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ścieżka do pliku

# Uruchomienie kamery
cap = cv2.VideoCapture(0)

# Otwórz plik wideo MP4 jako reklamę
ad_video = cv2.VideoCapture("video.mp4")


# Funkcja do obliczania współczynnika "EAR" (Eye Aspect Ratio)
def eye_aspect_ratio(eye):
    A = dist(eye[1], eye[5])  # pomiar pionowy oka 1
    B = dist(eye[2], eye[4])  # pomiar pionowy oka 2
    C = dist(eye[0], eye[3])  # pomiar poziomy oka
    return (A + B) / (2.0 * C)


# Funkcja obliczająca odległość między dwoma punktami - pitagoras
def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


# Próg EAR i liczniki na detekcję oczu
EAR_THRESHOLD = 0.3  # Próg od ktorego program stwierdzi czy oczy są zamknięte
MAX_NOT_LOOKING = 30  # Liczba klatek przy oku zamkniętym, po których reklama się zatrzyma
MIN_LOOKING = 15  # Liczba klatek potrzebna do wznowienia reklamy jak juz się widz patrzy

not_looking_frames = 0
looking_frames = 0
ad_paused = False

while True:
    # Wczytaj klatkę z kamery
    _, frame = cap.read()

    # Zamiana na skalę szarości
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Wykrywanie twarzy
    faces = detector(gray)

    eyes_detected = False  # Flaga na detekcję oczu
    ear = 0  # EAR jako wartość domyślna

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

        # Sprawdzenie, czy oczy są otwarte
        if ear > EAR_THRESHOLD:
            eyes_detected = True

        # Rysowanie obramowania wokół oczu
        # left_eye_hull = cv2.convexHull(left_eye)
        # right_eye_hull = cv2.convexHull(right_eye)
        # cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

    # Aktualizacja liczników w zależności od stanu oczu
    if eyes_detected:
        looking_frames += 1
        not_looking_frames = 0
    else:
        not_looking_frames += 1
        looking_frames = 0

    # Zatrzymanie lub wznowienie reklamy
    if not_looking_frames > MAX_NOT_LOOKING:
        ad_paused = True
    elif looking_frames > MIN_LOOKING:
        ad_paused = False

    # Wyświetlanie reklamy lub komunikatu
    if ad_paused:
        cv2.putText(frame, "Wroc do reklamy!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Odtwarzanie klatki z reklamy
        ret, ad_frame = ad_video.read()
        if not ret:
            ad_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Jeśli wideo się skończyło, zrestartuj
            ret, ad_frame = ad_video.read()

        if ret:
            # Dopasuj rozmiar reklamy do części okna
            ad_frame_resized = cv2.resize(ad_frame, (400, 200))
            # Dopasuj wideo do prawidłowego fragmentu okna
            h, w, _ = ad_frame_resized.shape
            frame[:h, :w] = ad_frame_resized

    # Wyświetlanie obrazu
    cv2.imshow("Reklama z detekcja oczu", frame)

    # Zakończenie programu po naciśnięciu 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:  # Esc
        break

# Zwolnienie zasobów
cap.release()
ad_video.release()
cv2.destroyAllWindows()
