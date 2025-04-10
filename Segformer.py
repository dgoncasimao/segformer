from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import os
import numpy as np
from sklearn.metrics import jaccard_score
import transformers
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F

#Configurations
IMAGE_DIR = 'C:/Users/Public/segformer/dataset/train/images'
MASK_DIR = 'C:/Users/Public/segformer/dataset/train/masks'
TEST_IMAGE_DIR = 'C:/Users/Public/segformer/dataset/test/images'
TEST_MASK_DIR = 'C:/Users/Public/segformer/dataset/test/masks'
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Dataset 
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, feature_extractor, resize=(512,512)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.resize = resize
        self.feature_extractor = feature_extractor
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.images[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, self.masks[idx])).convert("L")

         # Resize images and masks
        image = image.resize(self.resize)
        mask = mask.resize(self.resize)
        
        encoding = self.feature_extractor(images=image, return_tensors="pt")

        pixel_values = encoding['pixel_values'].squeeze(0)
        
        mask = np.array(mask)
        mask = (mask > 127).astype(np.uint8)
        mask= np.clip(mask, 0 , NUM_CLASSES -1)
        mask = torch.tensor(mask, dtype=torch.long)

        encoding['labels'] = mask
        encoding['pixel_values'] = pixel_values
        return encoding
    
#Metrics    
def compute_iou(preds, labels, num_classes):
    preds = preds.cpu() if preds.is_cuda else preds
    labels = labels.cpu() if labels.is_cuda else labels
    
    # Flatten preds and labels tensors
    preds = preds.flatten()
    labels = labels.flatten()
    
    # Print the shapes of preds and labels
    print(f"preds shape: {preds.shape}, labels shape: {labels.shape}")

    # Compute the IoU for each class
    iou_list = []
    for i in range(num_classes):
        # Filter out pixels where the ground truth and predictions don't match
        intersection = torch.sum((preds == i) & (labels == i))
        union = torch.sum((preds == i) | (labels == i))
        
        # IoU is intersection over union
        iou = intersection.float() / (union.float() + 1e-10)  # Avoid division by zero
        iou_list.append(iou.item())
        
    return sum(iou_list) / len(iou_list)
   
# For binary segmentation
loss_fn = BCEWithLogitsLoss()

def compute_dice_coefficient(preds, labels, num_classes=2):
    dice_scores = []
    for i in range(num_classes):
        intersection = torch.sum((preds == i) & (labels == i))
        total_pixels = torch.sum((preds == i) + (labels == i))
        dice = (2 * intersection.float()) / (total_pixels.float() + 1e-10)  # Avoid division by zero
        dice_scores.append(dice.item())
    return sum(dice_scores) / len(dice_scores)


def dice_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def compute_pixel_accuracy(preds, labels):
    # Flatten the tensors to treat all pixels as a 1D vector
    preds = preds.flatten()
    labels = labels.flatten()


    # Count correctly predicted pixels
    correct_pixels = (preds == labels).sum().item()

    # Total pixels
    total_pixels = labels.numel()

    # Pixel accuracy
    accuracy = correct_pixels / total_pixels
    return accuracy


def compute_metrics(p):
    logits, labels = p
    
    logits = torch.tensor(logits).to(DEVICE)
    labels = torch.tensor(labels).to(DEVICE)
    
    if labels.shape[1:] != logits.shape[2:]:
        labels = F.interpolate(labels.unsqueeze(1).float(),
                               size=logits.shape[2:],
                               mode='nearest').squeeze(1).long()
        
    preds = torch.argmax(logits, dim=1) 
    
    targets_onehot = torch.nn.functional.one_hot(labels, num_classes=NUM_CLASSES)
    targets_onehot = targets_onehot.permute(0,3,1,2).float()
    
    if targets_onehot.shape[2:] != logits.shape[2:]:
        targets_onehot = F.interpolate(targets_onehot, size=logits.shape[2:], mode='nearest')
    
    bce_loss = loss_fn(logits, targets_onehot) 
    dice = compute_dice_coefficient(preds, labels, num_classes=NUM_CLASSES)
    iou = compute_iou(preds, labels, num_classes=NUM_CLASSES)
    accuracy = compute_pixel_accuracy(preds, labels)
    
    return {
        "accuracy": accuracy,
        "dice": dice,
        "iou": iou,
        "bce": bce_loss.item(),
    }

#class VisualizationCallback(transformers.TrainerCallback):
#    def __init__(self, model, num_images=3, test_dataloader=None):
#        self.model = model
#        self.num_images = num_images
#        self.writer = SummaryWriter(log_dir="./logs")
#        self.test_batch = next(iter(test_dataloader)) if test_dataloader else None
#        
#    def on_log(self, args, state, control, logs=None, **kwargs):
#        # Check if the logs have been updated
#        if 'eval_loss' in logs and self.test_batch:
#            # Get the model predictions on a batch of validation data
#            self.model.eval()
#            with torch.no_grad():
#                # Get a batch from the eval dataset using the provided dataloader
#                inputs = {key: value.to(DEVICE) for key, value in self.test_batch.items()}

                # Perform the forward pass
#                outputs = self.model(**inputs)

                # Get the predicted segmentation map
#                logits = outputs.logits
#                predicted_class = torch.argmax(logits, dim=1)

#                class_colors = np.array([[0,0,0], [255,255,255]])

#                for i in range(self.num_images):
#                    # Get a single predicted mask and corresponding input image
#                    pred_mask = predicted_class[i]
#                    input_image = inputs['pixel_values'][i].cpu().numpy().transpose(1, 2, 0)  # To HWC format

                    # Normalize and convert to [0, 1] for TensorBoard image logging
#                    input_image = np.clip(input_image, 0, 1)

                    # Overlay predicted mask with original image using transparency
#                    pred_mask_colored = class_colors[pred_mask]
#                    overlay_image = (1 - 0.6) * input_image + 0.6 *(pred_mask_colored / 255.0)

                    # Log the input image and the overlayed prediction to TensorBoard
#                    input_image_tensor = T.ToTensor()(input_image)
                    
#                    self.writer.add_image(f"Input Image {i}", input_image_tensor, global_step=state.global_step)

#                    overlay_image_tensor = T.ToTensor()(overlay_image)
#                    self.writer.add_image(f"Overlayed Predicted Mask {i}", overlay_image_tensor, global_step=state.global_step)

#                    gt_mask = inputs['labels'][i].cpu().numpy()
#                    gt_colored = class_colors[gt_mask]
#                    gt_tensor = T.ToTensor()(gt_colored/255.0)
#                    self.writer.add_image(f"Ground Truth Mask {i}", gt_tensor, global_step=state.global_step)

def visualize_predictions(model, dataloader, device, num_images=3):
    model.eval()
    batch = next(iter(dataloader))
    # Move batch data to device
    inputs = {key: value.to(device) for key, value in batch.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits  # Shape: (B, num_classes, H_pred, W_pred)
    preds = torch.argmax(logits, dim=1)  # Shape: (B, H_pred, W_pred)

    # Define class colors for segmentation (e.g., background: black, foreground: white)
    class_colors = np.array([[0, 0, 0], [255, 255, 255]])

    for i in range(min(num_images, preds.shape[0])):
        # Convert input image to numpy (HWC) and clip values for visualization
        image = inputs['pixel_values'][i].cpu().numpy().transpose(1, 2, 0)
        image = np.clip(image, 0, 1)

        # Get predicted mask and colorize it using class_colors
        pred_mask = preds[i].cpu().numpy()  # Shape: (H_pred, W_pred)
        colored_pred_mask = class_colors[pred_mask]  # Shape: (H_pred, W_pred, 3)

        # Upsample the colored predicted mask to match the image dimensions
        # Convert colored_pred_mask to a tensor and add a batch dimension
        colored_pred_mask_tensor = torch.tensor(colored_pred_mask.transpose(2, 0, 1)).unsqueeze(0).float()  # Shape: (1, 3, H_pred, W_pred)
        # Use 'nearest' interpolation to preserve class boundaries
        upsampled_mask_tensor = F.interpolate(colored_pred_mask_tensor,
                                              size=(image.shape[0], image.shape[1]),
                                              mode='nearest')
        # Remove the batch dimension and convert back to (H, W, 3) numpy array
        upsampled_mask = upsampled_mask_tensor.squeeze(0).permute(1, 2, 0).numpy()

        # Create the overlay by blending the image and the upsampled predicted mask
        overlay = (0.4 * image + 0.6 * (upsampled_mask / 255.0))

        # Plot original image, predicted mask and overlay
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(pred_mask, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.axis("off")

        plt.show()

#Loading
feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
train_dataset = SegmentationDataset(IMAGE_DIR, MASK_DIR, feature_extractor)
test_dataset = SegmentationDataset(TEST_IMAGE_DIR, TEST_MASK_DIR, feature_extractor)
test_dataloader = DataLoader(test_dataset, batch_size=4)

#Loading model
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)
model.to(DEVICE)

#Training parameters
training_args = TrainingArguments(
    output_dir="./segformer-checkpoints",
    per_device_train_batch_size=4,
    num_train_epochs=10, 
    logging_dir="./logs",
    save_total_limit=2,
    save_steps=200,
    eval_strategy="steps",
    logging_steps=10,
    eval_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)


# Initialize the callback
#visualization_callback = VisualizationCallback(model=model, num_images=3, test_dataloader=test_dataloader)

# Add the callback to the Trainer
#trainer.add_callback(visualization_callback)

trainer.train()

trainer.evaluate()

def test_simgle_image(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")
    encoding = processor(images=image, return_tensor="pt").to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    plt.imshow(pred, cmap='gray')
    plt.title("Predicted Mask")
    plt.show()
    
    visualize_predictions(model, test_dataloader, DEVICE, num_images=3)
# test_single_image(model, feature_extractor, "C:/Users/Public/segformer/dataset/test/images/image1.jpg")