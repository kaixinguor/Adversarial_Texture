#yolo task=detect mode=predict model=yolo/weights/yolov5lu.pt source='dataset/coco2017_car/sub100/images/val2017/000000315187.jpg'
#yolo task=detect mode=predict model=yolo/weights/yolov5lu.pt source='tesst_results/ori_padded/000000315187.jpg'
#yolo task=detect mode=predict model=yolo/weights/yolov5lu.pt source='tesst_results/adv_padded/000000315187.jpg'
yolo val model=yolo/weights/yolov5lu.pt data=ultralytics/cfg/datasets/coco-car100.yaml batch=16 conf=0.3 iou=0.6 imgsz=416
