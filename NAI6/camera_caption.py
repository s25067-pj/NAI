import cv2 as cv

# Open first available capture device
cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()

    # Translate original frame to gray scale
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("Original stream", frame)
    cv.imshow("Gray stream ", gray_frame)

    # Closed when q is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

