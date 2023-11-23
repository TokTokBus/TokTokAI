import cv2
import torch
import time

class ObjectDetection:
    def __init__(self):
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")
        self.cap = cv2.VideoCapture(0)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            results = self.model(frame)
            current_detection = []

            for detection in results.xyxy[0]:
                class_id = int(detection[5])
                class_label = self.model.names[class_id]
                if class_label == "bus":
                    current_detection.append(class_label)

            frame = results.render()[0]
            cv2.imshow("Frame", frame)

            # 버스 인지되면 캡처            
            if "bus" in current_detection:
                #time.sleep(2)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"bus_capture_{timestamp}.png"
                cv2.imwrite(filename, frame)
                print(f"버스 인지됨, {filename}로 저장")
                return filename
                break 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    obj_detector = ObjectDetection()
    captured_filename = obj_detector.run()

    # Read the captured image
    if captured_filename:
        img = cv2.imread(captured_filename)
        if img is not None:
            cv2.imshow("Captured Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Failed to read the captured image: {captured_filename}")