from ultralytics import YOLO
import os

def yolo_inference(save_dir = "runs/detect/val_yolo"):
    
    # 加载模型
    model = YOLO('yolo/weights/yolov5lu.pt')

    os.makedirs(save_dir, exist_ok=True)
    
    results = model.val(
        data='ultralytics/cfg/datasets/coco-car100.yaml',
        batch=16,
        conf=0.3,
        iou=0.6,
        imgsz=416,
        save_txt=True,  # 保存检测结果为YOLO格式的.txt文件
        save_conf=True,  # 在txt文件中保存置信度（可选）
        project=save_dir,
        name=''
    )

    det_result_txt_dir = os.path.join(results.save_dir, "labels")

    return results, det_result_txt_dir

if __name__ == "__main__":
    results, det_resulttxt_dir = yolo_inference()
    print(results)
    print(det_resulttxt_dir)
    print(results.save_dir)