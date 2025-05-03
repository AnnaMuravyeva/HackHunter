import csv
import copy
import argparse
import itertools
import time
import sys
from collections import deque, Counter

import cv2 as cv
import numpy as np
import mediapipe as mp

from model import KeyPointClassifier

# ——— smoothing class to stabilize per-frame letter IDs ———
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
    parser.add_argument("--post_word_pause", type=float, default=2.0,
                        help='Seconds pause after word end before next word detection')
    return parser.parse_args()


def main():
    args = get_args()
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    hands = mp.solutions.hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    classifier = KeyPointClassifier()

    smoother = LetterSmoother(window_size=5)
    prev_letter_id = None

    # timing
    last_rec_time = 0.0       # last time any letter was recognized
    last_change_time = 0.0    # time when prev_letter_id was first set

    # load allowed words
    with open('word_list.txt') as wf:
        VALID_WORDS = {w.strip().upper() for w in wf if w.strip()}

    word_buffer = []
    in_word = False

    # initial pause to allow positioning
    print("Get ready: position your hand for the first letter...")
    time.sleep(5)

    while True:
        if cv.waitKey(10) == 27:
            break

        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        output = copy.deepcopy(frame)

        # process image
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        letter_id = None
        x = y = 0
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            pts = np.array([
                [int(p.x * frame.shape[1]), int(p.y * frame.shape[0])]
                for p in lm.landmark
            ], dtype=int)
            x, y, w, h = cv.boundingRect(pts)

            # preprocess landmarks
            pts_list = pts.tolist()
            base_x, base_y = pts_list[0]
            flat = []
            for px, py in pts_list:
                flat.extend([px - base_x, py - base_y])
            maxv = max(map(abs, flat))
            pre = [v / maxv for v in flat]

            # classify
            raw_id = classifier(pre)
            letter_id = smoother.add(raw_id)

        # handle gap (end-of-word)
        if letter_id is None:
            if in_word:
                word = ''.join(word_buffer)
                if word in VALID_WORDS:
                    print(f"Recognized word → {word}")
                else:
                    print(f"Word → {word}")
                word_buffer.clear()
                in_word = False
                # pause before next word with blank screen
                blank = np.zeros_like(frame)
                start = time.time()
                while time.time() - start < args.post_word_pause:
                    cv.imshow('Hand Gesture Recognition', blank)
                    if cv.waitKey(1) == 27:
                        cap.release()
                        cv.destroyAllWindows()
                        sys.exit(0)
            prev_letter_id = None
            last_change_time = 0.0

        else:
            now = time.time()
            # new letter different from previous
            if letter_id != prev_letter_id:
                if now - last_rec_time >= args.rec_cooldown:
                    ch = chr(ord('A') + letter_id)
                    cv.putText(output, ch, (x, y - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    word_buffer.append(ch)
                    in_word = True
                    prev_letter_id = letter_id
                    last_rec_time = now
                    last_change_time = now

            # same letter held
            else:
                if now - last_change_time >= args.dup_hold_time and now - last_rec_time >= args.rec_cooldown:
                    ch = chr(ord('A') + letter_id)
                    cv.putText(output, ch, (x, y - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    word_buffer.append(ch)
                    last_rec_time = now
                    last_change_time = now

        cv.imshow('Hand Gesture Recognition', output)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
