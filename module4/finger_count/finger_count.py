"""
Пример на OpenCV: определение количества пальцев, показанных в камеру.
Используются OpenCV (захват и отрисовка) и MediaPipe (детекция руки и ключевых точек).
Запуск: python finger_count.py
Выход: Q или Esc.
"""
import cv2
import mediapipe as mp

# Инициализация MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# Индексы кончиков пальцев и соответствующих «оснований» (для проверки «палец поднят»)
# MediaPipe: 4 — кончик большого, 3 — основание большого; 8,12,16,20 — кончики остальных; 6,10,14,18 — основания
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]  # точка ниже кончика по оси Y (при ладони к камере)


def count_fingers(landmarks, width: int, height: int, is_left: bool = True) -> int:
    """
    По 21 ключевой точке руки считаем поднятые пальцы.
    Для большого пальца: у левой руки кончик левее основания, у правой — правее.
    Для остальных: кончик выше основания по Y (ладонь к камере).
    """
    count = 0
    if not landmarks:
        return 0

    # Большой палец: кончик 4, основание 3. По X: левая рука — кончик левее (x4 < x3), правая — кончик правее (x4 > x3)
    x4, x3 = landmarks[4].x, landmarks[3].x
    if is_left and x4 < x3 - 0.02:
        count += 1
    elif not is_left and x4 > x3 + 0.02:
        count += 1

    # Остальные 4 пальца: кончик выше основания по Y (кончик.y < основание.y)
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        if landmarks[tip].y < landmarks[pip].y:
            count += 1

    return count


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть камеру.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # зеркально
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        num_fingers = 0
        if results.multi_hand_landmarks:
            handedness = results.multi_handedness or []
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                is_left = True
                if i < len(handedness) and handedness[i].classification:
                    is_left = handedness[i].classification[0].label == "Left"
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1),
                )
                num_fingers = count_fingers(hand_landmarks.landmark, w, h, is_left=is_left)

        cv2.putText(
            frame,
            f"Пальцев: {num_fingers}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Finger count (Q / Esc — выход)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
