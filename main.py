from utils import read_video, save_video
import os
from detections import  CarDetection, LicencePlateDetection

def main():
    input_video_path = "input_videos/video3.mp4"

    # Debug: mostrar path absoluto
    abs_path = os.path.abspath(input_video_path)
    print(f"🧭 Buscando video en: {abs_path}")
    print(f"📂 Existe?: {os.path.exists(abs_path)}")

    video_frames = read_video(input_video_path)
    print(f"🎞️ Frames leídos: {len(video_frames)}")

    if len(video_frames) == 0:
        print("⚠️ No se leyeron frames, abortando guardado.")
        return

    #Detect Car
    car_detector = CarDetection(model_path="yolo11n.pt")
    car_detections = car_detector.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/car_detection.pkl")

    #Detect Licence Plate
    licence_plate_detector = LicencePlateDetection(model_path="models/best.pt")
    licence_plate_detections, licence_plate_texts = licence_plate_detector.detect_frames(video_frames)

    #Draw Car Bounding Boxes
    output_video_frames = car_detector.draw_bboxed(video_frames, car_detections)
    #Draw Licence Plate Bounding Boxes
    output_video_frames = licence_plate_detector.draw_bbox(output_video_frames, licence_plate_detections, licence_plate_texts)
    save_video(video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()
