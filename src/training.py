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
from inference import validation_inference, training_inference, test_inference
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


def training_setup(image_patches_path, mask_patches_path, in_channels, out_channels, backbone, architecture, batch_size=16, epochs=50):
    mask_patches_np = np.load(mask_patches_path)
    image_patches_np = np.load(image_patches_path)

    vdm = VegetationDataModule(image_patches=image_patches_np, mask_patches=mask_patches_np, batch_size=batch_size)
    vdm.setup()
    
    formatted_date = datetime.now().strftime("%d%m_%H%M")
    # backbone = "resnet50"
    # arch = "unet"
    run_name = f'{architecture}_{backbone}_outch_{out_channels}_{formatted_date}'

    logger, callbacks = all_callbacks(run_name=run_name)
    unet_model = MultiClassVegetationModel(
        out_channels=out_channels,
        in_channels=in_channels, 
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
            max_epochs=epochs,
    )
    
    return trainer, unet_model, vdm, run_name


if __name__ == "__main__":
    image_patches_path, mask_patches_path = './data/image_patches_np.npy' ,'./data/mask_patches_np.npy'
    trainer, unet_model, vdm, run_name = training_setup(
        image_patches_path=image_patches_path, mask_patches_path=mask_patches_path,  in_channels=5, out_channels=4,
        backbone='resnet50', architecture='unet', batch_size=16)
    
    
    SEEN_DATA_INFERENCE = True
    UNSEEN_DATA_INFERENCE = True

    FROM_SAVED_MODEL = True
    
    if FROM_SAVED_MODEL:
        saved_model_path = './saved_models/patch_unet_resnet50_4patch_unet_resnet50-{epoch}/last.ckpt'
        unet_model.load_state_dict(torch.load(saved_model_path)['state_dict'], strict=False)
        run_name = saved_model_path.split('/')[-2]
    else:
        torch.set_float32_matmul_precision('medium')
        trainer.fit(unet_model, datamodule=vdm)
    
    os.makedirs('./results',exist_ok=True)
    if SEEN_DATA_INFERENCE:
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
            
    if UNSEEN_DATA_INFERENCE:
        test_inference(test_path='./data/test_tif/10_20201012_Multi_ortho.tif', model=unet_model, run_name=run_name)