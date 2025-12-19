from ultralytics import YOLO

model = YOLO(model='yolov12n.yaml', task="detect")

# Train the model
results = model.train(
  data='coco8.yaml',
  epochs=100,
  imgsz=640,
  # scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
  # mosaic=1.0,
  # mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
  # copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("000000000036.jpg")
results[0].show()