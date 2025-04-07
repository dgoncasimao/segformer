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
from torchvision.utils import make_grid
import torchvision.transforms as T
from torch.nn import BCEWithLogitsLoss

#Configurations
IMAGE_DIR = 'C:/Users/Public/segformer/dataset/train/images'
MASK_DIR = 'C:/Users/Public/segformer/dataset/train/masks'
TEST_IMAGE_DIR = 'C:/Users/Public/segformer/dataset/test/images'
TEST_MASK_DIR = 'C:/Users/Public/segformer/dataset/test/masks'
NUM_CLASSES = 2


#Dataset 
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, feature_extractor):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.feature_extractor = feature_extractor
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.images[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, self.masks[idx])).convert("L")

        encoding = self.feature_extractor(images=image, return_tensors="pt")

        pixel_values = encoding['pixel_values'].squeeze(0)
        
        mask = np.array(mask)

        mask= np.clip(mask, 0 , NUM_CLASSES -1)
        mask = torch.tensor(mask, dtype=torch.long)

        encoding['labels'] = mask
        encoding['pixel_values'] = pixel_values
        return encoding
    
#Metrics    
def compute_iou(preds, labels, num_classes):
    preds = torch.tensor(preds) if not isinstance(preds, torch.Tensor) else preds
    labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
    
    # Flatten preds and labels tensors
    preds = preds.view(-1)
    labels = labels.view(-1)

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
        iou_list.append(iou)
        
    return torch.mean(torch.tensor(iou_list))
   
# For binary segmentation
loss_fn = BCEWithLogitsLoss()

def compute_dice_coefficient(preds, labels, num_classes=2):
    dice_scores = []
    for i in range(num_classes):
        intersection = torch.sum((preds == i) & (labels == i))
        total_pixels = torch.sum((preds == i) + (labels == i))
        dice = (2 * intersection.float()) / (total_pixels.float() + 1e-10)  # Avoid division by zero
        dice_scores.append(dice)
    return torch.mean(torch.tensor(dice_scores)).item()


def dice_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def compute_pixel_accuracy(preds, labels):
    # Flatten the tensors to treat all pixels as a 1D vector
    preds = preds.view(-1)
    labels = labels.view(-1)

    # Count correctly predicted pixels
    correct_pixels = (preds == labels).sum().item()

    # Total pixels
    total_pixels = labels.numel()

    # Pixel accuracy
    accuracy = correct_pixels / total_pixels
    return accuracy


def compute_metrics(p):
    preds, labels = p
    preds = torch.argmax(preds, dim=1) 
    
    # Calculate IoU
    dice = compute_dice_coefficient(preds, labels, num_classes=2)
    iou = compute_iou(preds, labels, num_classes=NUM_CLASSES)
    accuracy = compute_pixel_accuracy(preds, labels)
    
    return {
        "accuracy": accuracy,
        "dice": dice,
        "iou": iou,
    }

class VisualizationCallback(transformers.TrainerCallback):
    def __init__(self, model, num_images=3, test_dataloader=None):
        self.model = model
        self.num_images = num_images
        self.writer = SummaryWriter(log_dir="./logs")
        self.test_dataloader = test_dataloader

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Check if the logs have been updated
        if 'eval_loss' in logs:
            # Get the model predictions on a batch of validation data
            self.model.eval()
            with torch.no_grad():
                # Get a batch from the eval dataset using the provided dataloader
                inputs = next(iter(self.test_dataloader))
                
                # Move inputs to the model's device
                inputs = {key: value.to(model.device) for key, value in inputs.items()}

                # Perform the forward pass
                outputs = self.model(**inputs)

                # Get the predicted segmentation map
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=1)

                class_colors = np.array([[0,0,0], [255,255,255]])

                for i in range(self.num_images):
                    # Get a single predicted mask and corresponding input image
                    pred_mask = predicted_class[i]
                    input_image = inputs['pixel_values'][i].cpu().numpy().transpose(1, 2, 0)  # To HWC format

                    # Normalize and convert to [0, 1] for TensorBoard image logging
                    input_image = np.clip(input_image, 0, 1)

                    # Overlay predicted mask with original image using transparency
                    pred_mask_colored = class_colors[pred_mask]
                    overlay_image = np.where(pred_mask_colored == [0, 0, 0], input_image, pred_mask_colored)

                    # Log the input image and the overlayed prediction to TensorBoard
                    input_image_tensor = T.ToTensor()(input_image)
                    self.writer.add_image(f"Input Image {i}", input_image_tensor, global_step=state.global_step)

                    overlay_image_tensor = T.ToTensor()(overlay_image)
                    self.writer.add_image(f"Overlayed Predicted Mask {i}", overlay_image_tensor, global_step=state.global_step)


                if state.global_step % 100 == 0: 
                    plt.imshow(pred_mask)
                    plt.title(f"Epoch {state.epoch} - Predicted Mask")
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

#Training parameters
training_args = TrainingArguments(
    output_dir="./segformer-checkpoints",
    per_device_train_batch_size=4,
    num_train_epochs=1, 
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
visualization_callback = VisualizationCallback(model=model, num_images=3, test_dataloader=test_dataloader)

# Add the callback to the Trainer
trainer.add_callback(visualization_callback)

trainer.train()

trainer.evaluate()