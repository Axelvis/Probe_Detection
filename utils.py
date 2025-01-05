import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, random_split
from PIL import Image
import os
import json
import time
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.patches as patches
from tqdm import tqdm
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import time
import psutil

# ============================
# Dataset Preparation
# ============================
class ProbeDataset(torch.utils.data.Dataset):
    """
    Custom dataset for loading probe images and their annotations.
    Groups annotations by image ID and prepares bounding boxes and labels.
    """
    def __init__(self, image_dir, annotation_file, transforms=None):
        self.image_dir = image_dir
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            self.image_id_to_annotations = {}
            
            # Group annotations by image_id
            for ann in data['annotations']:
                if ann['image_id'] not in self.image_id_to_annotations:
                    self.image_id_to_annotations[ann['image_id']] = []
                self.image_id_to_annotations[ann['image_id']].append(ann)
            
            self.image_id_to_file = {img['id']: img['file_name'] for img in data['images']}
        self.transforms = transforms
        self.image_ids = list(self.image_id_to_file.keys())  # Image IDs list

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        file_name = self.image_id_to_file[image_id]
        img_path = os.path.join(self.image_dir, file_name)
        img = Image.open(img_path).convert("RGB")

        # Get all annotations for this image_id
        annotations = self.image_id_to_annotations.get(image_id, [])
        boxes = []
        for ann in annotations:
            x1, y1, w, h = ann['bbox']
            x2 = x1 + w
            y2 = y1 + h
            boxes.append([x1, y1, x2, y2])

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor([1] * len(boxes), dtype=torch.int64)  # 1 for "probe"
        }

        if self.transforms:
            img = self.transforms(img)
        else:
            img = F.to_tensor(img)

        return img, target


# ============================
# Model Preparation
# ============================
def get_model(num_classes):
    """
    Initializes and returns a Faster R-CNN model with a specified number of classes (here only 2 will be used).
    """
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

# ============================
# Training and Validation
# ============================
def train_model(model, train_loader, val_loader, device, num_epochs):
    """
    Fine-Tune the Faster R-CNN model and evaluates it on the validation set after each epoch.
    Tracks training loss, validation loss, and IoU.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Reduce LR every 5 epochs
    model.to(device)

    train_losses, val_losses, ious = [], [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, targets in train_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")

        # Step the scheduler
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        epoch_ious = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                predictions = model(images)
                val_loss += sum(loss for loss in loss_dict.values()).item()

                for i, prediction in enumerate(predictions):
                    pred_boxes = prediction['boxes'].cpu().numpy()
                    pred_scores = prediction['scores'].cpu().numpy()
                    gt_boxes = targets[i]['boxes'].cpu().numpy()
                    if len(pred_scores) > 0:
                        # Keep only the box with the highest confidence
                        max_score_idx = pred_scores.argmax()
                        pred_boxes = pred_boxes[max_score_idx:max_score_idx+1]
                        pred_scores = pred_scores[max_score_idx:max_score_idx+1]

                    if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                        iou_per_image = []
                        for pred_box in pred_boxes:
                            ious_single = [calculate_iou(pred_box, gt_box) for gt_box in gt_boxes]
                            iou_per_image.append(max(ious_single) if ious_single else 0)
                        epoch_ious.extend(iou_per_image)

        val_losses.append(val_loss)
        ious.append(sum(epoch_ious) / len(epoch_ious) if epoch_ious else 0)

        print(f"Validation Loss: {val_loss:.4f}, Mean IoU: {ious[-1]:.4f}")

    return train_losses, val_losses, ious


# ============================
# Additional Functions
# ============================
def compute_metrics(pred_boxes, gt_boxes, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0

    for pred_box in pred_boxes:
        ious = [calculate_iou(pred_box, gt_box) for gt_box in gt_boxes]
        if max(ious) > iou_threshold:
            tp += 1
        else:
            fp += 1

    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def calculate_iou(boxA, boxB):
    # intersection coordinates
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # area of the intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)  

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]) 
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])  

    # Compute the IoU
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou



def plot_training_metrics(train_losses, val_losses, ious):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Loss Trends')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, ious, label='Mean IoU')
    plt.title('Mean IoU Trends')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()

    plt.tight_layout()
    plt.show()
def visualize_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for idx, result in enumerate(results):
        img = F.to_pil_image(result['image'])
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Draw predicted boxes
        for box, score in zip(result['pred_boxes'], result['pred_scores']):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv, f"Pred: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw ground truth boxes
        for box in result['gt_boxes']:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_cv, "GT", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Save the image
        output_path = os.path.join(output_dir, f"result_{idx}.jpg")
        cv2.imwrite(output_path, img_cv)
def plot_metrics(ious, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(ious, bins=20, range=(0, 1), alpha=0.7, color='blue', edgecolor='black')
    plt.title("Distribution of IoUs")
    plt.xlabel("IoU")
    plt.ylabel("Frequency")
    plt.grid(axis='y')
    plt.savefig(os.path.join(output_dir, "iou_distribution.png"))
    plt.show()
def benchmark_inference(model, dataloader, device, warmup=5, iterations=100):
    """
    Benchmark model inference speed.
    Args:
        model: The trained model.
        dataloader: DataLoader for validation/testing.
        device: Device ('cuda' or 'cpu') to run the inference.
        warmup: Number of warmup iterations to exclude from benchmarking.
        iterations: Number of iterations to include in benchmarking.
    Returns:
        avg_time_per_batch: Average time taken per batch.
        avg_time_per_image: Average time taken per image.
        throughput: Number of images processed per second.
    """
    model.eval()
    total_time = 0
    total_images = 0

    # Warmup iterations
    print("Warming up...")
    for i, (images, _) in enumerate(dataloader):
        if i >= warmup:
            break
        images = [img.to(device) for img in images]
        with torch.no_grad():
            _ = model(images)
    print("Warmup complete.")

    # Benchmarking iterations
    print("Benchmarking...")
    for i, (images, _) in enumerate(dataloader):
        if i >= iterations:
            break
        images = [img.to(device) for img in images]
        batch_size = len(images)
        total_images += batch_size

        start_time = time.time()
        with torch.no_grad():
            _ = model(images)
        end_time = time.time()

        total_time += (end_time - start_time)

    avg_time_per_batch = total_time / iterations
    avg_time_per_image = avg_time_per_batch / batch_size
    throughput = total_images / total_time

    print(f"Average Time Per Batch: {avg_time_per_batch:.4f} seconds")
    print(f"Average Time Per Image: {avg_time_per_image:.4f} seconds")
    print(f"Throughput: {throughput:.2f} images/second")
    
    return avg_time_per_batch, avg_time_per_image, throughput
def measure_memory_and_model_size(model, device, val_loader):
    """
    Measures memory usage and model size.
    """
    model_size = torch.save(model.state_dict(), "temp_model.pth")  # Save temporarily
    model_size_mb = os.path.getsize("temp_model.pth") / (1024 * 1024)
    os.remove("temp_model.pth")  # Clean up temporary file

    # Measure GPU memory if on CUDA
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        for images, _ in val_loader:
            images = [img.to(device) for img in images]
            with torch.no_grad():
                _ = model(images)  # Run inference to measure memory
            break  # Single batch is enough to measure memory

        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_memory_mb = "N/A (CPU-only)"

    # Measure RAM usage
    ram_usage_mb = psutil.virtual_memory().used / (1024 * 1024)

    return model_size_mb, peak_memory_mb, ram_usage_mb
# ============================
# Test Image Prediction
# ============================
def evaluate_model(model, val_loader, device):
    model.eval()
    ious = []
    results = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                gt_boxes = targets[i]['boxes'].cpu().numpy()

                if len(pred_scores) > 0:
                    # Keep only the box with the highest confidence
                    max_score_idx = pred_scores.argmax()
                    pred_boxes = pred_boxes[max_score_idx:max_score_idx+1]
                    pred_scores = pred_scores[max_score_idx:max_score_idx+1]

                # Calculate IoU for each prediction against ground truth
                iou_per_image = []
                for pred_box in pred_boxes:
                    ious_single = []
                    for gt_box in gt_boxes:
                        ious_single.append(calculate_iou(pred_box, gt_box))
                    iou_per_image.append(max(ious_single) if ious_single else 0)
                    
                ious.extend(iou_per_image)

                results.append({
                    'image': images[i].cpu(),
                    'pred_boxes': pred_boxes,
                    'gt_boxes': gt_boxes,
                    'pred_scores': pred_scores
                })

    return ious, results
def predict_on_test_image(model, image_path, device):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    img_tensor = transform(img).to(device)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    best_box, best_score = None, None
    if len(prediction['scores']) > 0:
        max_score_idx = prediction['scores'].argmax().item()
        best_box = prediction['boxes'][max_score_idx].cpu().numpy()
        best_score = prediction['scores'][max_score_idx].item()

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    if best_box is not None:
        x1, y1, x2, y2 = best_box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 10, f"Conf: {best_score:.2f}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.axis('off')
    plt.show()
