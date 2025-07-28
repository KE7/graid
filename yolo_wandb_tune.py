# import wandb
# from ultralytics import YOLO
# from wandb.integration.ultralytics import add_wandb_callback

# wandb.login()

# model_gpu_map = {
#     "yolov5m": [3],
#     "yolov5m6": [2],
#     "yolov6x": [1],
#     "yolov6l": [0],
# }

# config = {
#     "epochs": 30,
#     "iterations": 100,
#     "imgsz": 640,
#     "batch": 32,
#     "save": True,
#     "plots": True,
#     "optimizer": "AdamW"
# }

# wandb.init(project="bdd_ultra", name="yolov5m", job_type="tuning", config=config)

# for model_name, device in model_gpu_map.items():
#     model = YOLO(f"{model_name}.pt")

#     add_wandb_callback(model, enable_model_checkpointing=True)
    

# model.tune(
#     data="bdd_ultra.yaml", 
#     epochs=config["epochs"], 
#     iterations=config["iterations"], 
#     imgsz=config["imgsz"], 
#     device=config["device"], 
#     batch=config["batch"], 
#     save=config["save"], 
#     plots=config["plots"], 
#     optimizer=config["optimizer"],
# )

# wandb.finish()


import wandb
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import threading

wandb.login()

model_gpu_map = {
    "yolov5mu.pt": [3],
    "yolov5m6u.pt": [2],
    "yolov6l.yaml": [1],
    "yolov10b.pt": [0],
}

config = {
    "data": "bdd_ultra.yaml",
    "epochs": 30,
    "iterations": 100,
    "imgsz": 736,
    "batch": 16,
    "save": True,
    "plots": True,
    "optimizer": "AdamW",
    "amp": False,
}

threads = []

for model_name, device in model_gpu_map.items():

    def tune_model(model_name, device, config):
        wandb.init(project="bdd_ultra", name=model_name, job_type="tuning", config=config)

        model = YOLO(model_name)

        add_wandb_callback(model, enable_model_checkpointing=True)

        model.tune(
            device=device,
            **config,
        )

    threads.append(
        threading.Thread(
            target=tune_model,
            args=(model_name, device, config),
        )
    )

    threads[-1].start()

# wait for all threads to finish
for thread in threads:
    thread.join()

wandb.finish()