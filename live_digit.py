"""
Parker Cai
March 28, 2026

CS 5330 - Project 5: Recognition using Deep Networks
Extension: Live video digit recognition using trained CNN
"""

import sys
import cv2
import torch
import numpy as np

from train import Network


# Helper functions
def show_help():
    """Print available key bindings to terminal."""
    print("Live Digit Recognition")
    print("  +/-        - Zoom in/out")
    print("  Arrow keys - Pan (while zoomed in)")
    print("  s          - Save screenshot to results/")
    print("  i          - Toggle invert mode (white-on-black vs black-on-white)")
    print("  h          - Show this help menu")
    print("  q          - Quit")


def preprocess_roi(roi, invert=True):
    """Convert an ROI crop to a normalized 28x28 tensor matching MNIST format.

    MNIST digits are centered, anti-aliased, and fill ~20x20 pixels inside a 28x28 frame.
    We replicate that by: threshold to find the digit bounding box, crop, scale to fit
    20x20, center on a 28x28 black canvas, then use grayscale (not binary) for anti-aliasing.
    Returns (tensor, preview) where preview is the 28x28 uint8 image for display.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Gaussian to find digit region — handles uneven webcam lighting
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    if not invert:
        thresh = 255 - thresh

    # Find the bounding box of all white pixels (the digit)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        # No digit found — return blank
        blank = np.zeros((28, 28), dtype=np.uint8)
        tensor = torch.zeros(1, 1, 28, 28)
        return tensor, blank

    x, y, bw, bh = cv2.boundingRect(coords)

    # Crop digit region, use threshold as mask to zero out paper background
    # while keeping anti-aliased grayscale strokes from the digit itself
    digit_gray = blurred[y : y + bh, x : x + bw]
    digit_mask = thresh[y : y + bh, x : x + bw]
    if invert:
        digit_crop = (255 - digit_gray) * (digit_mask > 0).astype(np.uint8)
    else:
        digit_crop = digit_gray * (digit_mask > 0).astype(np.uint8)

    # Scale to fit inside 20x20 (MNIST convention) preserving aspect ratio
    scale = 20.0 / max(bw, bh)
    new_w, new_h = max(1, int(bw * scale)), max(1, int(bh * scale))
    scaled = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center on a 28x28 black canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = scaled

    # Convert to float tensor and normalize with MNIST stats (mean=0.1307, std=0.3081)
    tensor = torch.from_numpy(canvas).float() / 255.0
    tensor = (tensor - 0.1307) / 0.3081
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28) batch + channel dims
    return tensor, canvas


def main(argv):
    """Live webcam digit recognition with CNN inference on a centered ROI box."""

    # Load trained CNN
    model = Network()
    model.load_state_dict(torch.load("results/mnist_model.pth", weights_only=True))
    model.eval()

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    show_help()

    # State variables
    invert = True  # assume dark digit on light paper by default
    save_count = 0  # screenshot counter
    zoom = 1.0  # digital zoom level (1.0 = no zoom, higher = closer)
    pan_x = 0.0  # pan offset as fraction of cropped region (-1.0 to 1.0)
    pan_y = 0.0
    PAN_STEP = 0.1  # how far each arrow key press pans

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        h, w = frame.shape[:2]

        # Digital zoom + pan: crop a sub-region of the frame then scale back
        if zoom > 1.0:
            zh, zw = int(h / zoom), int(w / zoom)
            # Center offset + pan offset, clamped so we don't go out of bounds
            cx, cy = w // 2, h // 2
            x_off = int(cx + pan_x * (w - zw) / 2) - zw // 2
            y_off = int(cy + pan_y * (h - zh) / 2) - zh // 2
            x_off = max(0, min(x_off, w - zw))
            y_off = max(0, min(y_off, h - zh))
            frame = cv2.resize(frame[y_off : y_off + zh, x_off : x_off + zw], (w, h))

        display = frame.copy()

        # ROI: centered square, 40% of the smaller frame dimension
        roi_size = int(min(h, w) * 0.4)
        x1 = (w - roi_size) // 2
        y1 = (h - roi_size) // 2
        x2, y2 = x1 + roi_size, y1 + roi_size

        # Draw ROI box
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Extract ROI and run inference
        roi = frame[y1:y2, x1:x2]
        tensor, preview = preprocess_roi(roi, invert=invert)

        with torch.no_grad():
            output = model(tensor)
        probs = torch.exp(output).squeeze()  # log_softmax -> probabilities
        confidence, predicted = probs.max(0)
        conf_pct = confidence.item() * 100
        digit = predicted.item()

        # Only show prediction when model is reasonably confident
        if conf_pct > 50:
            digit_text = str(digit)
            color = (0, 255, 0)
        else:
            digit_text = "?"
            color = (0, 0, 255)

        # Overlay: confidence % on top, large digit below, centered above ROI
        roi_cx = (x1 + x2) // 2
        conf_text = f"{conf_pct:.0f}%"
        (cw, _), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(
            display,
            conf_text,
            (roi_cx - cw // 2, y1 - 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
        )
        (dw, _), _ = cv2.getTextSize(digit_text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 5)
        cv2.putText(
            display,
            digit_text,
            (roi_cx - dw // 2, y1 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            color,
            5,
        )

        # Inset: show the 28x28 preprocessed image the model actually sees
        inset_size = 80
        inset = cv2.resize(
            preview, (inset_size, inset_size), interpolation=cv2.INTER_NEAREST
        )
        inset_bgr = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)
        # White border around the inset so it stands out
        cv2.rectangle(
            inset_bgr, (0, 0), (inset_size - 1, inset_size - 1), (255, 255, 255), 1
        )
        display[10 : 10 + inset_size, 10 : 10 + inset_size] = inset_bgr
        cv2.putText(
            display,
            "28x28 Preview",
            (10, 10 + inset_size + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )

        # Invert mode indicator (to match white-on-black from dataset)
        mode_text = "Mode: invert" if invert else "Mode: normal"
        cv2.putText(
            display,
            mode_text,
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        cv2.imshow("Live Digit Recognition", display)

        # Key handling (waitKeyEx for arrow key support on Windows)
        key = cv2.waitKeyEx(1)
        if key == ord("q") or key == 27:
            break
        elif key == ord("s"):
            save_count += 1
            path = f"results/live_screenshot_{save_count}.png"
            cv2.imwrite(path, display)
            print(f"Saved screenshot -> {path}")
        elif key == ord("i"):
            invert = not invert
            print(f"Invert mode: {'ON' if invert else 'OFF'}")
        elif key in (ord("+"), ord("=")):
            zoom = min(zoom + 0.25, 4.0)
            print(f"Zoom: {zoom:.1f}x")
        elif key == ord("-"):
            zoom = max(zoom - 0.25, 1.0)
            if zoom == 1.0:
                pan_x, pan_y = 0.0, 0.0  # reset pan when fully zoomed out
            print(f"Zoom: {zoom:.1f}x")
        elif key in (2424832, 65361):  # Left arrow (Win / Linux)
            pan_x = max(pan_x - PAN_STEP, -1.0)
        elif key in (2555904, 65363):  # Right arrow
            pan_x = min(pan_x + PAN_STEP, 1.0)
        elif key in (2490368, 65362):  # Up arrow
            pan_y = max(pan_y - PAN_STEP, -1.0)
        elif key in (2621440, 65364):  # Down arrow
            pan_y = min(pan_y + PAN_STEP, 1.0)
        elif key == ord("h"):
            show_help()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
