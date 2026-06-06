import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Cannot read camera")
        break

    cv2.imshow("Test Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
