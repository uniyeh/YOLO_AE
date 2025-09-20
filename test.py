from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yoloe-11-ae.yaml")
print("create model done.")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="coco8.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=608,  # Image size for training
    device="device=0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    batch=48,
    project='ae_experiments',
    name='gpu_training_v1'
)

print("train model done.")
# Evaluate the model's performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("https://ultralytics.com/images/bus.jpg")  # Predict on an image
results[0].show()  # Display results

# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model