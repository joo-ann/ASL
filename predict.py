import cv2
import torch
from PIL import Image
import numpy as np
from model import ASLModel
import torchvision.transforms as transforms
from collections import deque

# Load the trained model
model = ASLModel(29)  # 29 classes (26 letters + nothing, del, space)
model.load_state_dict(torch.load('asl_model_best.pth', map_location=torch.device('cpu')))
model.eval()

# Define the transformation for the input image - MATCHING THE TRAINING EXACTLY
transform = transforms.Compose([
    transforms.Resize((200, 200)),  # Match the training size
    transforms.ToTensor()
])

# Define the class labels
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["nothing", "del", "space"]

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'c' to capture and predict, 'g' to toggle guide overlay, or 'q' to quit.")

# Prediction smoothing
prediction_history = deque(maxlen=5)
confidence_threshold = 0.7

# Flag to control guide visibility
show_guide = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    display_frame = frame.copy()
    
    # If guide mode is enabled, draw the calibration overlay
    if show_guide:
        height, width = display_frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Draw hand positioning guide
        cv2.rectangle(display_frame, (center_x-100, center_y-100), 
                     (center_x+100, center_y+100), (0, 255, 0), 2)
        cv2.putText(display_frame, "Position hand here", 
                   (center_x-95, center_y-110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add instructions
        cv2.putText(display_frame, "Keep hand in good lighting", 
                   (10, height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, "Use plain background", 
                   (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the webcam feed
    cv2.imshow("ASL Recognition", display_frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):  # Capture and predict
        # Convert the frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Apply transformations
        input_tensor = transform(image).unsqueeze(0)
        
        # Save processed image for debugging
        processed_img = input_tensor.squeeze(0).permute(1, 2, 0).numpy()
        cv2.imwrite('processed_input.jpg', 
                   cv2.cvtColor((processed_img*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            confidence, predicted = torch.max(probabilities, 0)
            predicted_idx = predicted.item()
            predicted_letter = classes[predicted_idx]
            
            # Add to prediction history
            prediction_history.append((predicted_idx, confidence.item()))
            
            # Get smoothed prediction (most common in history)
            if len(prediction_history) >= 3:
                # Count occurrences and get the most frequent
                pred_counts = {}
                for p, c in prediction_history:
                    if c >= confidence_threshold:  # Only count high confidence predictions
                        pred_counts[p] = pred_counts.get(p, 0) + 1
                
                if pred_counts:
                    smoothed_idx = max(pred_counts, key=pred_counts.get)
                    smoothed_letter = classes[smoothed_idx]
                    
                    # Calculate average confidence for this prediction
                    avg_conf = np.mean([c for p, c in prediction_history if p == smoothed_idx])
                    
                    print(f"Predicted: {predicted_letter} (Raw confidence: {confidence:.2f})")
                    print(f"Smoothed: {smoothed_letter} (Avg confidence: {avg_conf:.2f})")
                else:
                    print(f"Low confidence prediction: {predicted_letter} ({confidence:.2f})")
            else:
                if confidence >= confidence_threshold:
                    print(f"Predicted Letter: {predicted_letter} (Confidence: {confidence:.2f})")
                else:
                    print(f"Low confidence prediction: {predicted_letter} (Confidence: {confidence:.2f})")

    elif key == ord('g'):  # Toggle guide overlay
        show_guide = not show_guide
        print(f"Guide overlay {'enabled' if show_guide else 'disabled'}")

    elif key == ord('q'):  # Quit
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()