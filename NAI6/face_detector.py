import cv2
import dlib
from imutils import face_utils

""" Detektor twarzy (dlib) i plik ktory zawiera model 68 punktów charakterystycznych twarzy (np. oczy, nos, usta)"""
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

"""" Otwiera kamerę komputera (0 oznacza domyślną kamerę) i przypisujemy video/reklame do zmiennej."""
cap = cv2.VideoCapture(0)
ad_video = cv2.VideoCapture("video.mp4")


def eye_aspect_ratio(eye):
    """
    Funckja przyjmujaca dane wejsciowe oka,(rysunek znaleziony pod postem na stackoverflow)
    mierzymy odleglosc miedzy gorna krawędza oka po lewej stronie a dolna krawedzia(pion) a nastepnie poziom.
    Aby uzyskać odpowiednią równowagę między tymi dwoma wymiarami (wysokość i szerokość) mnozymy * 2 szerokosc oka

        eye[1]    eye[2]
         *-------*
        /         \
  eye[0]          eye[3]
        \         /
         *-------*
      eye[5]    eye[4]
"""
    A = dist(eye[1], eye[5])
    B = dist(eye[2], eye[4])
    C = dist(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def dist(p1, p2):
    """"Funckja uzyta do obliczenia pitagorasa,uzywana do obliczen odlegosci miedzy krawedziami oka """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


""""
USTAWIENIA:
EAR_THRESHOLD - Próg od ktorego program stwierdzi czy oczy są zamknięte
MAX_NOT_LOOKING- Jesli przez 30 klatek aplikacja uzna ze oczy są zamknięte,wyswietni sie komuniat 
MIN_LOOKING - Jesli przez 15 klatek aplikacja uzna ze oczy są otwarte,usunie komuniat i wznowi reklame 
not_looking_frames - Ile klatek minęło od ostatniego wykrycia, kiedy oczy były zamknięte < EAR_THRESHOLD = komunikat .
looking_frames- Ile klatek minęło od ostatniego wykrycia, kiedy oczy były otwarte > MIN_LOOKING = reklama .
ad_paused- Flaga czy reklama jest wstrzymana, True/False
"""
EAR_THRESHOLD = 0.3
MAX_NOT_LOOKING = 30
MIN_LOOKING = 15

not_looking_frames = 0
looking_frames = 0
ad_paused = False

while True:
    """"
    Nieskonczona petla ktora:
    - Wczytuje obraz z kamery->Zmienia dostarczony obraz na skale szarosci z (3 kanalow rgb na 1 kanal) -> wykrywa twarz
    -> liczone sa wartosci dla prawego i lewego oka -> wyswietlenie kolejnej klatki wideo/komunikatu 
    jesli nie patrzy+zwiekszenie odpowiednich licznikow-> resize okienka i filmu ->jesli filmik sie skonczyl wracamy do 1 klatki
    ->opcjonalne wyjscie z ESC
    """

    # tutaj pierwszy argument bedzie niepotrzebny bo to flaga czy udalo sie wczytac z kamery, _ sluzy do pomijania zwracanego argumentu!
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    # print(faces)
    #twarz sklada sie rectangles[[(114, 254) (293, 433)]]

    eyes_detected = False
    ear = 0

    for face in faces:
        """" Wypisanie punktow charakterstycznych dla twarzy (oczy,brwi..) 
             Zmiana punktow na numpa,w celu optymalizacji
             Wycinanie lewego i prawego oka z twarzy
             Obliczenie eye_aspect_ratio dla oka i sredniej wartosci i
             sprawdzenie czy jest wieksza od wartosci z  naszych ustawien
             """
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[42:48]
        right_eye = shape[36:42]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        if ear > EAR_THRESHOLD:
            eyes_detected = True

        # To nam nie bedzie potrzebne,wyswietla obrys oka
        # left_eye_hull = cv2.convexHull(left_eye)
        # right_eye_hull = cv2.convexHull(right_eye)
        # cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)


    if eyes_detected:
        looking_frames += 1
        not_looking_frames = 0
    else:
        not_looking_frames += 1
        looking_frames = 0

    if not_looking_frames > MAX_NOT_LOOKING:
        ad_paused = True
    elif looking_frames > MIN_LOOKING:
        ad_paused = False

    if ad_paused:
        """" Jesli nasza flaga z ustawien jest ustawiona na true wyswietlamy komunikat,jesli nie to:
             wyswietlamy kolejna klatke z reklamy,jesli nie ma kolejnej klatki resetujemy wideo .set
             reklama powinna sie wyswietlic w okienku,400, 200, tak samo wideo po przekalowaniu
             """
        cv2.putText(frame, "Wroc do reklamy!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        ret, ad_frame = ad_video.read()
        if not ret:
            ad_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, ad_frame = ad_video.read()

        if ret:
            ad_frame_resized = cv2.resize(ad_frame, (400, 200))
            h, w, _ = ad_frame_resized.shape
            frame[:h, :w] = ad_frame_resized

    cv2.imshow("Reklama z detekcja oczu", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc tutaj
        break

""""
Koniec programu (zwolenienie miejsca)
cap.release() – zwalnia zasoby kamery, czyli kończy działanie kamery.
ad_video.release() – zwalnia zasoby pliku wideo, kończy odtwarzanie reklamy.
cv2.destroyAllWindows() – zamyka wszystkie otwarte okna z obrazami, które zostały otwarte przez cv2.imshow().
"""
cap.release()
ad_video.release()
cv2.destroyAllWindows()
