import cv2

face_class = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

eye_class = cv2.CascadeClassifier("Cascades/haarcascade_eye.xml")

smile = cv2.CascadeClassifier("Cascades/haarcascade_smile.xml")

video = cv2.VideoCapture(0)

def detect(stream):

    g_image = cv2.cvtColor(stream, cv2.COLOR_BGR2GRAY)

    faces = face_class.detectMultiScale(g_image, 1.1, 5, minSize=(40, 40))

    for (x, y, w, h) in faces:

        cv2.rectangle(stream, (x, y), (x + w, y + h), (0, 255, 0), 4)

    eyes = eye_class.detectMultiScale(g_image, 1.1, 5, minSize=(40, 40))

    for (x, y, w, h) in eyes:

        cv2.rectangle(stream, (x, y), (x + w, y + h), (0, 255, 0), 4)

    smiles = eye_class.detectMultiScale(g_image, 1.1, 5, minSize=(40, 40))

    for (x, y, w, h) in smiles:

        cv2.rectangle(stream, (x, y), (x + w, y + h), (0, 255, 0), 4)

    return faces
    return eyes
    return smiles

while True:

    res, frame = video.read()

    if res == False:

        break

    faces = detect(frame)

    cv2.imshow("Visify", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break

video.release()

cv2.destroyAllWindows()

