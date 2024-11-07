import numpy as np
import torch
import os
from lightning import Trainer
from model import MultiClassVegetationModel
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from veg_dataset import VegetationDataModule
from datetime import datetime
from inference import validation_inference, training_inference
import json

def all_callbacks(run_name):
    logger = TensorBoardLogger(
                "./logs",
                name=run_name,
                # default_hp_metric=False,
    )
    
    cp = ModelCheckpoint(
            os.path.abspath("./saved_models")
            + f"/{run_name}",
            monitor="val/loss_mask",
            mode="min",
            save_last=True,
            # save_best_only=True,
            save_top_k=1,
        )

    early_stopping = EarlyStopping(
            monitor="val/loss_mask",  
            patience=10,              
            mode="min",               
            verbose=True              
        )
    
    os.makedirs('./results',exist_ok=True)
    
    return logger, [cp, early_stopping, LearningRateMonitor()]


def training_setup():
    mask_patches_np = np.load('./data/mask_patches_np.npy')
    image_patches_np = np.load('./data/image_patches_np.npy')

    vdm = VegetationDataModule(image_patches=image_patches_np, mask_patches=mask_patches_np, batch_size=16)
    vdm.setup()
    
    
    formatted_date = datetime.now().strftime("%d%m_%H%M")
    backbone = "resnet50"
    arch = "unet"
    run_name = f'{arch}_{backbone}_{formatted_date}'

    logger, callbacks = all_callbacks(run_name=run_name)
    unet_model = MultiClassVegetationModel(
        out_channels=4,
        in_channels=5, 
        backbone=backbone,
        # encoder_depth=3,
        # decoder_channels=(256, 128, 64), 
        class_weights=vdm.class_weights)

    trainer = Trainer(
            accelerator="gpu",
            devices=1,
            num_nodes=1,
            logger=logger,
            callbacks=callbacks,
            max_epochs=50,
    )
    
    return trainer, unet_model, vdm, run_name


if __name__ == "__main__":
    trainer, unet_model, vdm, run_name = training_setup()
    torch.set_float32_matmul_precision('medium')
    
    INFERENCE = True
    FROM_SAVED_MODEL = False
    
    if FROM_SAVED_MODEL:
        saved_model_path = './saved_models/unet_resnet50_0711_1922/epoch=49-step=5550.ckpt'
        unet_model.load_state_dict(torch.load(saved_model_path)['state_dict'], strict=False)
        run_name = saved_model_path.split('/')[-2]
    else:
        trainer.fit(unet_model, datamodule=vdm)
    
    if INFERENCE:
        metrics = {
            "validation_metrics": validation_inference(
                validation_dataloader=vdm.val_dataloader(), 
                model=unet_model, 
                run_name=run_name
            ),
            "training_metrics": training_inference(
                training_dataloader=vdm.train_dataloader(), 
                model=unet_model, 
                run_name=run_name
            )
        }
        with open(f'./results/metrics_{run_name}.txt', 'w') as f:
            json.dump(metrics, f, indent=4)