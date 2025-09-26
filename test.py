from ultralytics import YOLO

# Load a model
model = YOLO('yolo11-ae-single.yaml')  # build a new model from scratch

# Use the model
results = model.train(
    data='coco.yaml',
    epochs=100
)  # train the model
results = model.val()  # evaluate model performance on the validation set
results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
results = model.export(format='onnx')  # export the model to ONNX format    