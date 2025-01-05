from utils import *


# Configurable Variables
DATASET_PATH = '/probe_dataset/probe_images'
ANNOTATION_FILE = '/probe_dataset/probe_labels.json'

MODEL_SAVE_PATH ='/fasterrcnn_probe.pth' #Link to the model you either trained here OR downloaded with my link provided in the Readme

TRAIN=False #True if you want to train the model

TEST=False #True if you want to see the prediction of the model on a custom image 
TEST_IMAGE_PATH = '/path/to/test/image.png'

# ==============================================
# Preprocessing of the dataset
# ==============================================
dataset = ProbeDataset(image_dir=DATASET_PATH, annotation_file=ANNOTATION_FILE)

# Split dataset into training and validation sets (80%/20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))



# ============================
# Main Execution : Training
# ============================
if TRAIN:
    # Model Setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(num_classes=2)  # 1 class (probe) + background

    # Train 
    num_epochs = 15
    train_losses, val_losses, ious = train_model(model, train_loader, val_loader, device, num_epochs)

    # Save the  model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # Plot training metrics
    plot_training_metrics(train_losses, val_losses, ious)

# ============================
# Main Execution : Evaluation
# ============================
# Path
model_path = MODEL_SAVE_PATH

# Load Model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model(num_classes=2)  # 1 class (probe) + background
model.load_state_dict(torch.load(model_path))

model.to(device)

# Evaluate the model on validation set
print("Evaluating model on validation set...")
ious, results = evaluate_model(model, val_loader, device)
print(f"Mean IoU on validation set: {sum(ious) / len(ious):.4f}")

# Visualize results
visualize_results(results, "/kaggle/working/")

# Plot IoU distribution
plot_metrics(ious, "/kaggle/working/")

# Benchmark Inference Speed
print("Benchmarking inference speed...")
avg_batch_time, avg_image_time, throughput = benchmark_inference(model, val_loader, device, warmup=5, iterations=50)
print(f"Average Time Per Batch: {avg_batch_time:.4f} seconds")
print(f"Average Time Per Image: {avg_image_time:.4f} seconds")
print(f"Throughput: {throughput:.2f} images/second")

print("Evaluating model resource usage...")
model_size_mb, peak_memory_mb, ram_usage_mb = measure_memory_and_model_size(model, device, val_loader)

print(f"Model Size: {model_size_mb:.2f} MB")
if device.type == 'cuda':
    print(f"Peak GPU Memory Usage: {peak_memory_mb:.2f} MB")
print(f"RAM Usage: {ram_usage_mb:.2f} MB")

if TEST:
    # ============================
    # Test Image Prediction Execution (Specify the test image in 'test_image_path')
    # ============================
    # Paths
    model_path = MODEL_SAVE_PATH
    test_image_path = TEST_IMAGE_PATH  # Path to a test image

    # Load Model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(num_classes=2)  # 1 class (probe) + background
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Predict on Test Image
    best_box, best_score = predict_on_test_image(model, test_image_path, device)

    if best_box is not None:
        print(f"Predicted box: {best_box}, Confidence: {best_score:.2f}")
    else:
        print("No predictions made for the test image.")