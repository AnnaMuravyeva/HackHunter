import csv
import copy
import argparse
import time
import sys
from collections import deque, Counter

import cv2 as cv
import numpy as np
import mediapipe as mp

from model import KeyPointClassifier

# ——— choose a soft pink for all overlays ———
PINK = (147,  20, 255)   # BGR for RGB(255,20,147) (deep pink)

class LetterSmoother:
    """Mode-filter a stream of discrete letter-IDs (0=A…25=Z)."""
    def __init__(self, window_size=5):
        self.window = deque(maxlen=window_size)

    def add(self, idx):
        """
        idx: int in [0..25] for A..Z, or None for a ‘gap’.
        Returns the most common int once window is full, else None.
        """
        if idx is None:
            self.window.clear()
            return None
        self.window.append(idx)
        if len(self.window) == self.window.maxlen:
            return Counter(self.window).most_common(1)[0][0]
        return None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960, help='capture width')
    parser.add_argument("--height", type=int, default=540, help='capture height')
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    parser.add_argument("--dup_hold_time", type=float, default=3.0,
                        help='Seconds to hold the same letter before duplicating')
    parser.add_argument("--rec_cooldown", type=float, default=1.0,
                        help='Seconds minimum between recognitions')
    parser.add_argument("--post_word_pause", type=float, default=3.0,
                        help='Seconds pause after word end before next word detection')
    return parser.parse_args()


def main():
    args = get_args()

    # set up capture
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # prepare branded window
    cv.namedWindow('Hand Gesture Recognition', cv.WINDOW_NORMAL)
    cv.resizeWindow('Hand Gesture Recognition', args.width, args.height)

    # splash screen: "PREPARE"
    splash = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    font, scale, th = cv.FONT_HERSHEY_DUPLEX, 3.5, 8
    text = "PREPARE"
    (tw, tht), _ = cv.getTextSize(text, font, scale, th)
    x = (args.width  - tw) // 2
    y = (args.height + tht) // 2
    cv.putText(splash, text, (x, y), font, scale, PINK, th, cv.LINE_AA)
    cv.imshow('Hand Gesture Recognition', splash)
    cv.waitKey(2000)  # show for 2s

    # brief pause for positioning
    print("Get ready: position your hand for the first letter…")
    time.sleep(3)

    # initialize Mediapipe & classifier
    hands = mp.solutions.hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    classifier = KeyPointClassifier()
    smoother   = LetterSmoother(window_size=5)

    prev_letter_id = None
    last_rec_time  = 0.0
    last_change_time = 0.0

    # load valid words
    with open('word_list.txt') as wf:
        VALID_WORDS = {w.strip().upper() for w in wf if w.strip()}

    word_buffer = []
    in_word = False

    while True:
        if cv.waitKey(10) == 27:
            break

        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        output = copy.deepcopy(frame)

        # hand detection
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        letter_id = None
        x = y = 0
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            pts = np.array([[int(p.x * frame.shape[1]),
                             int(p.y * frame.shape[0])]
                            for p in lm.landmark], dtype=int)
            x, y, w, h = cv.boundingRect(pts)

            # normalize to first landmark
            base_x, base_y = pts[0]
            flat = []
            for px, py in pts:
                flat.extend([px - base_x, py - base_y])
            maxv = max(map(abs, flat)) or 1.0
            pre = [v / maxv for v in flat]

            raw_id = classifier(pre)
            letter_id = smoother.add(raw_id)

        # end-of-word gap?
        if letter_id is None:
            if in_word:
                word = ''.join(word_buffer)
                print("Recognized word →", word if word in VALID_WORDS else word)
                word_buffer.clear()
                in_word = False

                # pink numeric countdown
                n = int(args.post_word_pause)
                H, W = frame.shape[:2]
                font_c, sc_c, th_c = cv.FONT_HERSHEY_DUPLEX, 4.0, 10
                for i in range(n, 0, -1):
                    blank = np.zeros_like(frame)
                    s = str(i)
                    (tw_, th_), _ = cv.getTextSize(s, font_c, sc_c, th_c)
                    xx = (W - tw_) // 2
                    yy = (H + th_) // 2
                    cv.putText(blank, s, (xx, yy), font_c, sc_c, PINK, th_c, cv.LINE_AA)
                    cv.imshow('Hand Gesture Recognition', blank)
                    if cv.waitKey(1000) == 27:
                        cap.release()
                        cv.destroyAllWindows()
                        sys.exit(0)

            prev_letter_id   = None
            last_change_time = 0.0

        else:
            now = time.time()
            # new letter?
            if letter_id != prev_letter_id:
                if now - last_rec_time >= args.rec_cooldown:
                    ch = chr(ord('A') + letter_id)
                    cv.putText(output, ch, (x, y - 10),
                               cv.FONT_HERSHEY_DUPLEX, 2.0, PINK, 5, cv.LINE_AA)
                    word_buffer.append(ch)
                    in_word = True
                    prev_letter_id = letter_id
                    last_rec_time  = now
                    last_change_time = now

            # same letter held → duplicate
            else:
                if (now - last_change_time >= args.dup_hold_time and
                    now - last_rec_time      >= args.rec_cooldown):
                    ch = chr(ord('A') + letter_id)
                    cv.putText(output, ch, (x, y - 10),
                               cv.FONT_HERSHEY_DUPLEX, 2.0, PINK, 5, cv.LINE_AA)
                    word_buffer.append(ch)
                    last_rec_time    = now
                    last_change_time = now

        cv.imshow('Hand Gesture Recognition', output)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()


