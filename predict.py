import torch
import cv2
import numpy as np
from torchvision import transforms
import time
from PIL import Image
import os
from model import LightASLModel

MODEL_PATH = "asl_model_best.pth"
IMG_SIZE = 128
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found.")
    model = LightASLModel(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")
    return model

def process_frame(frame, hand_roi=None):
    if hand_roi is not None:
        x, y, w, h = hand_roi
        padding = int(max(w, h) * 0.2)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2*padding)
        h = min(frame.shape[0] - y, h + 2*padding)
        if w > 0 and h > 0:
            frame = frame[y:y+h, x:x+w]
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(image)
    return tensor

def draw_prediction(frame, prediction, confidence, x=50, y=50):
    text = f"{prediction} ({confidence:.2f})"
    text_size = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
    cv2.rectangle(frame, (x, y - text_size[1] - 10), (x + text_size[0] + 10, y + 10), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 5, y), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
    return frame

def draw_hand_roi(frame, roi):
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICKNESS)
    return frame

def detect_hand(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hand_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(hand_contour) > 5000:
            return cv2.boundingRect(hand_contour)
    return None

def main():
    model = load_model()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fixed_size = min(frame_width, frame_height) // 2
    x = (frame_width - fixed_size) // 2
    y = (frame_height - fixed_size) // 2
    fixed_roi = (x, y, fixed_size, fixed_size)
    cv2.namedWindow('ASL Recognition', cv2.WINDOW_NORMAL)
    use_hand_detection = False
    last_prediction_time = time.time()
    prediction_cooldown = 0.2
    last_prediction = "N/A"
    confidence = 0.0
    print("\n----- ASL Recognition with Webcam -----")
    print("Press 'q' to quit")
    print("Press 'm' to switch between fixed ROI and hand detection")
    print("Press 's' to save the current frame")
    print("Place your hand in the green box and show an ASL sign")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break
        frame = cv2.flip(frame, 1)
        if use_hand_detection:
            hand_roi = detect_hand(frame)
            if hand_roi is None:
                hand_roi = fixed_roi
        else:
            hand_roi = fixed_roi
        draw_hand_roi(frame, hand_roi)
        current_time = time.time()
        if current_time - last_prediction_time > prediction_cooldown:
            last_prediction_time = current_time
            input_tensor = process_frame(frame, hand_roi)
            with torch.no_grad():
                input_batch = input_tensor.unsqueeze(0).to(device)
                output = model(input_batch)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                confidence, pred_idx = torch.max(probabilities, 0)
                last_prediction = class_names[pred_idx.item()]
                confidence = confidence.item()
        draw_prediction(frame, last_prediction, confidence)
        cv2.putText(frame, "Press 'm' to toggle hand detection", (10, frame_height - 60), 
                    FONT, 0.5, TEXT_COLOR, 1)
        cv2.putText(frame, "Press 's' to save frame", (10, frame_height - 40), 
                    FONT, 0.5, TEXT_COLOR, 1)
        cv2.putText(frame, "Press 'q' to quit", (10, frame_height - 20), 
                    FONT, 0.5, TEXT_COLOR, 1)
        mode_text = "Mode: Hand Detection" if use_hand_detection else "Mode: Fixed ROI"
        cv2.putText(frame, mode_text, (10, 30), FONT, 0.6, TEXT_COLOR, 1)
        cv2.imshow('ASL Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('m'):
            use_hand_detection = not use_hand_detection
            print(f"Switched to {'hand detection' if use_hand_detection else 'fixed ROI'} mode")
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"asl_capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved frame as {filename}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    except Exception as e:
        print(f"Error: {e}")
