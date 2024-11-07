import segmentation_models_pytorch as smp
import torch
from torch import nn
from lightning import LightningModule

architectures = [smp.Unet, smp.UnetPlusPlus, smp.Linknet, smp.FPN, smp.FPN, smp.PSPNet, smp.PAN, smp.DeepLabV3,
                 smp.DeepLabV3Plus]
arch_names = [m.__name__.replace("Plus", "+") for m in architectures]
arch_dict = {name: m for name, m in zip(arch_names, architectures)}

class MultiClassVegetationModel(LightningModule):
    def __init__(
        self,
        out_channels,
        in_channels=3,
        architecture="Unet",
        backbone="resnet18",
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16),
        encoder_weights="imagenet",
        lr=1e-4,
        apply_softmax=False,
        apply_sigmoid=False,
        class_weights=None  # Add class_weights as a parameter
    ):
        super().__init__()

        arch = arch_dict[architecture]
        self.model = arch(in_channels=in_channels,
                          classes=out_channels,
                          encoder_name=backbone,
                          encoder_depth=encoder_depth,
                          decoder_channels=decoder_channels,
                          encoder_weights=encoder_weights)
        self.lr = lr
        self.apply_softmax = apply_softmax
        self.apply_sigmoid = apply_sigmoid
        self.num_classes = out_channels
        
        # Initialize class weights for the loss function
        self.class_weights = class_weights
        
        # Update the loss function to use the class weights (if provided)
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, img):
        prediction = self.model(img)
        if self.apply_softmax:
            return torch.softmax(prediction, dim=1)
        elif self.apply_sigmoid:
            return torch.sigmoid(prediction)
        else:
            return prediction

    def shared_step(self, batch):
        x, y = batch
        mask = self(x)
        mask_t = y

        # Calculate the loss with class weights
        cce_loss_mask = self.loss_fn(mask, mask_t)

        return cce_loss_mask

    def training_step(self, batch, step):
        loss = self.shared_step(batch)
        self.log("train/loss_mask", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, step):
        loss = self.shared_step(batch)
        self.log("val/loss_mask", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss_mask",  # Replace with your actual metric
            },
        }
