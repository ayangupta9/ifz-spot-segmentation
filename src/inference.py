import torch
import torchmetrics
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
from PIL import Image
import os

def cf_matrix_cal(y_true, y_pred, save_path):
    conf_matrix = confusion_matrix(y_true=y_true.view(-1).numpy(), y_pred=y_pred.view(-1).numpy())

    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

    # Plot the confusion matrix with percentages
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(conf_matrix_percent, annot=True, fmt=".2f", cmap="viridis", cbar=True,
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_pred))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix for Multi-Class Classification (Percentage)")
    
    fig.canvas.draw()  # Draw the canvas to access it as an image
    img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    img.save(save_path)  # Save using PIL
    plt.close(fig)  # Close the Matplotlib figure to free memory
    # plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # plt.close()


def validation_inference(validation_dataloader, model, run_name):
    model = model.cuda()
    model.eval()
    val_imgs, val_msks, val_preds = [], [], []

    for img, mask in validation_dataloader:    
        mask = mask.unsqueeze(1)
        val_msks.append(mask)
        
        # pred = torch.softmax(unet_model(img.cuda()).detach().cpu(), dim=1)
        # pred = torch.softmax(fpn_model(img.cuda()).detach().cpu(), dim=1)
        pred = torch.softmax(model(img.cuda()).detach().cpu(), dim=1)
        pred = torch.argmax(pred,dim=1)
        val_preds.append(pred)

    # val_imgs = torch.concat(val_imgs,dim=0)
    val_msks = torch.concat(val_msks, dim=0)
    val_preds = torch.concat(val_preds).unsqueeze(1)
    
    metrics = calc_metrics(num_classes=4, y_true=val_msks, y_pred=val_preds)
    cf_matrix_cal(y_true=val_msks, y_pred=val_preds, save_path=f"./results/{run_name}_val_cfmatrix.png")
    return metrics
    
def training_inference(training_dataloader, model, run_name):
    model = model.cuda()
    model.eval()

    train_imgs, train_msks, train_preds = [], [], []

    for img, mask in training_dataloader:
        mask = mask.unsqueeze(1)
        train_msks.append(mask)
        
        # pred = torch.softmax(unet_model(img.cuda()).detach().cpu(), dim=1)
        # pred = torch.softmax(fpn_model(img.cuda()).detach().cpu(), dim=1)
        pred = torch.softmax(model(img.cuda()).detach().cpu(), dim=1)
        pred = torch.argmax(pred,dim=1)
        train_preds.append(pred)

    # train_imgs = torch.concat(train_imgs,dim=0)
    train_msks = torch.concat(train_msks, dim=0)
    train_preds = torch.concat(train_preds).unsqueeze(1)
    
    metrics = calc_metrics(num_classes=4, y_true=train_msks, y_pred=train_preds)
    cf_matrix_cal(y_true=train_msks, y_pred=train_preds, save_path=f"./results/{run_name}_train_cfmatrix.png")
    return metrics
    
    
def calc_metrics(num_classes, y_true: torch.Tensor, y_pred: torch.Tensor):
    iou_fn = torchmetrics.JaccardIndex(task='multiclass', average='none', num_classes=num_classes).cpu()
    f1_fn = torchmetrics.F1Score(task='multiclass', average='none', num_classes=num_classes).cpu()
    precision_fn = torchmetrics.Precision(task="multiclass", average='none', num_classes=num_classes).cpu()
    recall_fn = torchmetrics.Recall(task="multiclass", average='none', num_classes=num_classes).cpu()
    

    ious = iou_fn(y_true.cpu(), y_pred.cpu()).tolist()
    f1s = f1_fn(y_true.cpu(), y_pred.cpu()).tolist()
    precisions = precision_fn(y_true.cpu(), y_pred.cpu()).tolist()
    recalls = recall_fn(y_true.cpu(), y_pred.cpu()).tolist()
    
    return \
        {
            "iou" : ious,
            "f1": f1s,
            "precision" : precisions,
            "recall" :recalls
        }
    
    
    
    
    