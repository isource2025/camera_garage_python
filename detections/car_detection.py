import cv2
import pickle
from ultralytics import YOLO

class CarDetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Detecta autos en una lista de frames usando un modelo Ultralytics YOLO v8+.

        Args:
            frames (list): Lista de frames (np.array) a procesar.
            read_from_stub (bool): Si True, carga resultados guardados desde stub_path.
            stub_path (str): Ruta al archivo pickle para leer/escribir detecciones.

        Returns:
            list: Lista de detecciones por frame. Cada elemento es una lista de diccionarios:
                  {"box": [x1, y1, x2, y2], "class": int, "conf": float}
        """
        car_detections = []

        # Cargar desde stub si corresponde
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                car_detections = pickle.load(f)
            return car_detections

        # Detectar frame por frame
        for frame in frames:
            results = self.model(frame)  # llamado moderno de YOLOv8
            car_list = []

            for r in results:
                # Asegurarse de que existan cajas detectadas
                if r.boxes is not None:
                    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                        car_list.append({
                            "box": box.cpu().numpy().tolist(),  # convertir a lista simple
                            "class": int(cls.cpu().numpy()),
                            "conf": float(conf.cpu().numpy())
                        })
            car_detections.append(car_list)

        # Guardar stub si corresponde
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(car_detections, f)

        return car_detections


    def detect(self, frame):
        results = self.model.predict(frame, iou = 0.1, conf = 0.30)[0]
        id_name_dict = results.name
        car_list = []
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            cls_id = int(box.cls.tolist()[0])
            cls_name = id_name_dict[cls_id]
            if cls_name == "car":
                car_list.append(result)
        return car_list

    def draw_bboxed(self, video_frames, car_detections):
        output_videos_frames = []
        for frame, car_list in zip(video_frames, car_detections):
            for bbox_dict in car_list:
                x1, y1, x2, y2 = bbox_dict["box"]  # tomar la lista dentro del dict
                cv2.putText(frame, f"Car", (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (255, 255, 0), 2)
            output_videos_frames.append(frame)  # ahora fuera del for interno
        return output_videos_frames
