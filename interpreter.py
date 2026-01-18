import os
import time
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import keyboard


MODEL_PATH = "hand_model.keras"
LABELS_PATH = "labels.npy"

DATASET_DIR = "dataset"

SEQ_LEN = 30          # frames per clip
CONF_THRESH = 0.70

# recording specs
MOTION_START = 0.012
MOTION_END   = 0.008
END_HOLD_FRAMES = 8
MIN_SIGN_FRAMES = 12
MAX_SIGN_FRAMES = 60

SMOOTH_N = 5
DUPLICATE_COOLDOWN_S = 0.5

PRINT_DEBUG = True

current_label = "none"

is_typing_label = False
label_prompt_requested = False
enter_hotkey_id = None

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


# MARK: Init model
model = None
label_names = None
if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    label_names = np.load(LABELS_PATH, allow_pickle=True)
else:
    print("No model found")


# MARK: Hand calculations
def normalize_one_hand(hand_lm) -> np.ndarray:
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], dtype=np.float32)
    pts -= pts[0]
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale
    return pts.flatten().astype(np.float32)


def two_hand_features(results) -> np.ndarray:
    left = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)

    if not results.multi_hand_landmarks or not results.multi_handedness:
        return np.concatenate([left, right])

    for hand_lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
        side = handed.classification[0].label  # "Left" or "Right"
        feat = normalize_one_hand(hand_lm)
        if side == "Left":
            left = feat
        elif side == "Right":
            right = feat

    return np.concatenate([left, right])


def fit_to_length(frames: np.ndarray, seq_len: int) -> np.ndarray:
    n = frames.shape[0]
    if n == seq_len:
        return frames
    if n > seq_len:
        start = (n - seq_len) // 2
        return frames[start:start + seq_len]
    pad = np.repeat(frames[-1:], seq_len - n, axis=0)
    return np.concatenate([frames, pad], axis=0)


# MARK: saving
def save_clip(clip_frames: np.ndarray, label: str):
    os.makedirs(DATASET_DIR, exist_ok=True)
    label_dir = os.path.join(DATASET_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    ts = int(time.time() * 1000)
    path = os.path.join(label_dir, f"clip_{ts}.npz")
    np.savez_compressed(path, X=clip_frames.astype(np.float32), y=str(label))
    print(f"Saved clip: {path}  shape={clip_frames.shape}")


# MARK: predict + smoothing
recent_preds = deque(maxlen=SMOOTH_N)
last_word = ""
last_word_time = 0.0

def predict_clip(clip_T: np.ndarray):
    if model is None or label_names is None:
        return ("(no model)", 0.0)

    x = np.expand_dims(clip_T, axis=0)  # [1, T, 126]
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return (str(label_names[idx]), float(probs[idx]))


def majority_vote(items):
    vals, counts = np.unique(np.array(items), return_counts=True)
    return str(vals[np.argmax(counts)])


def should_output(word: str, conf: float):
    global last_word, last_word_time

    if conf < CONF_THRESH:
        return False

    now = time.time()
    if word == last_word and (now - last_word_time) < DUPLICATE_COOLDOWN_S:
        return False

    last_word = word
    last_word_time = now
    return True


# MARK: whatever else
def request_label_prompt():
    global label_prompt_requested, is_typing_label
    if is_typing_label or label_prompt_requested:
        return
    label_prompt_requested = True


# MARK: Main
def run():
    global enter_hotkey_id, label_prompt_requested, is_typing_label, current_label

    enter_hotkey_id = keyboard.add_hotkey('enter', request_label_prompt, trigger_on_release=False)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not opened")
        return

    prev_feat = None

    # segmenting
    auto_recording = False
    auto_frames = []
    low_motion_frames = 0

    # recording
    manual_recording = False
    manual_frames = []

    # FPS
    t0 = time.time()
    frames = 0
    fps = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame")
                break

            # Label making
            if label_prompt_requested and not is_typing_label:
                label_prompt_requested = False
                is_typing_label = True

                if enter_hotkey_id is not None:
                    keyboard.remove_hotkey(enter_hotkey_id)
                    enter_hotkey_id = None

                print("\nType name for new label:")
                typed = input("> ").strip()
                if typed:
                    current_label = typed
                    print("Set next label to:", current_label)
                else:
                    print("Label unchanged. Current:", current_label)

                enter_hotkey_id = keyboard.add_hotkey('enter', request_label_prompt, trigger_on_release=False)
                is_typing_label = False

            frame = cv2.flip(frame, 1)

            if is_typing_label:
                cv2.putText(frame, "Entering name", (8, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow("ASL Interpreter", frame)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
                continue

            # MARK: landmarks
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

            feat = two_hand_features(results)

            # motion score
            motion = 0.0 if prev_feat is None else float(np.mean(np.abs(feat - prev_feat)))
            prev_feat = feat

            # MARK: segementation attempt
            hand_present = bool(results.multi_hand_landmarks)

            if not auto_recording:
                if hand_present and motion > MOTION_START:
                    auto_recording = True
                    auto_frames = [feat]
                    low_motion_frames = 0
            else:
                auto_frames.append(feat)

                if (motion < MOTION_END) or (not hand_present):
                    low_motion_frames += 1
                else:
                    low_motion_frames = 0

                if len(auto_frames) >= MAX_SIGN_FRAMES:
                    low_motion_frames = END_HOLD_FRAMES

                if low_motion_frames >= END_HOLD_FRAMES:
                    auto_recording = False
                    segment = np.array(auto_frames, dtype=np.float32)
                    auto_frames = []
                    low_motion_frames = 0

                    if segment.shape[0] >= MIN_SIGN_FRAMES:
                        clip_T = fit_to_length(segment, SEQ_LEN)
                        pred, conf = predict_clip(clip_T)

                        recent_preds.append(pred)
                        smoothed = majority_vote(recent_preds)

            key = cv2.waitKey(1) & 0xFF

            # SPACE manual capture toggles
            if key == 32:
                if not manual_recording:
                    manual_recording = True
                    manual_frames = []
                    if PRINT_DEBUG:
                        print("Recording started for label:", current_label)
                else:
                    manual_recording = False
                    if len(manual_frames) >= MIN_SIGN_FRAMES:
                        clip = fit_to_length(np.array(manual_frames, np.float32), SEQ_LEN)
                        save_clip(clip, current_label)
                    else:
                        print("Recording too short")
                    manual_frames = []

            if manual_recording:
                manual_frames.append(feat)

            if key == 27:
                break

            # MARK: ui
            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps = frames / dt
                frames = 0
                t0 = time.time()

            last_pred = recent_preds[-1] if recent_preds else ""
            overlay_lines = [
                f"FPS: {fps:.1f}",
                f"Motion: {motion:.4f}  AutoRec:{auto_recording}  ManualRec:{manual_recording}",
                f"Next label: {current_label}"
            ]

            y0 = 200
            for i, line in enumerate(overlay_lines):
                cv2.putText(frame, line, (8, y0 + 18 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, str(last_pred), (5, 100), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 0, 0), 1)

            cv2.imshow("ASL Interpreter", frame)

    cap.release()
    cv2.destroyAllWindows()

    try:
        if enter_hotkey_id is not None:
            keyboard.remove_hotkey(enter_hotkey_id)
    except Exception:
        pass


if __name__ == "__main__":
    run()
