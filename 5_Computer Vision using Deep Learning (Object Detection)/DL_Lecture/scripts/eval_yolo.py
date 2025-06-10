# Simple eval script for YOLOv5
import torch
import yaml
import sys
import os
import subprocess
import shutil
from tqdm import tqdm
from torchvision import transforms

# Import local modules
sys.path.append("..")  # Add parent directory to path
from DL_Lecture.utils.data_prepro import MaskDataset
from DL_Lecture.utils.metrics import get_batch_statistics, ap_per_class

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load configuration
if len(sys.argv) >= 2:
    params_filename = sys.argv[1]
    print("Using config file:", params_filename)
else:
    params_filename = '../config/mask_yolo.yaml'
    print("Using default config file:", params_filename)

with open(params_filename, 'r', encoding="UTF8") as f:
    params = yaml.safe_load(f)

# Print key parameters
print("Dataset task:", params['task'])
print("Test images path:", params['data_files']['image_test_file'])
print("Test annotations path:", params['data_files']['annotation_test_file'])

# Load test dataset
data_transform = transforms.Compose([transforms.ToTensor()])
test_data = MaskDataset(
    data_transform,
    params['data_files']['image_test_file'],
    params['data_files']['annotation_test_file']
)
print(f"Loaded test dataset with {len(test_data)} samples")

# Timestamp for run directory
timestamp = "1746608731"  # Update as needed
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print(f"Run directory: {out_dir}")

# Check for model path
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints/best.pt"))
print(f"Looking for model at: {checkpoint_dir}")

# Try to load YOLOv5
try:
    import yolov5

    print("Successfully imported yolov5 package")
    use_package = True
except ImportError:
    print("yolov5 package not found, trying torch hub...")
    use_package = False

# Load model
if os.path.exists(checkpoint_dir):
    print(f"Found model at {checkpoint_dir}")
    if use_package:
        try:
            model = yolov5.load(checkpoint_dir)
            print("Loaded model using yolov5 package")
        except Exception as e:
            print(f"Error loading with yolov5 package: {e}")
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=checkpoint_dir)
            print("Loaded model using torch hub")
    else:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=checkpoint_dir)
        print("Loaded model using torch hub")
else:
    print(f"No model found at {checkpoint_dir}, using YOLOv5s")
    if use_package:
        try:
            model = yolov5.load('yolov5s.pt')
            print("Loaded YOLOv5s using yolov5 package")
        except Exception as e:
            print(f"Error loading YOLOv5s with package: {e}")
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            print("Loaded YOLOv5s using torch hub")
    else:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        print("Loaded YOLOv5s using torch hub")

model.to(device)


# Helper function to convert YOLOv5 results to Faster R-CNN format
def yolo_to_faster_rcnn_format(results, threshold=0.5):
    pred_dict = {
        'boxes': [],
        'labels': [],
        'scores': []
    }

    for *box, conf, cls_id in results.xyxy[0].cpu().numpy():
        if conf >= threshold:
            pred_dict['boxes'].append(torch.tensor(box))
            pred_dict['labels'].append(torch.tensor(int(cls_id) + 1))  # +1 for 1-indexed
            pred_dict['scores'].append(torch.tensor(conf))

    if pred_dict['boxes']:
        pred_dict['boxes'] = torch.stack(pred_dict['boxes'])
        pred_dict['labels'] = torch.stack(pred_dict['labels'])
        pred_dict['scores'] = torch.stack(pred_dict['scores'])
    else:
        pred_dict['boxes'] = torch.tensor([])
        pred_dict['labels'] = torch.tensor([])
        pred_dict['scores'] = torch.tensor([])

    return pred_dict


# Evaluate model on test dataset
print("\nEvaluating model on test dataset...")
labels = []
preds_adj_all = []
annot_all = []

# Get image files from directory
image_files = sorted(os.listdir(params['data_files']['image_test_file']))
print(f"Found {len(image_files)} image files")

# Run inference on test dataset
for idx in tqdm(range(len(test_data)), desc="Processing test images"):
    try:
        # Get image and annotation
        img, annotation = test_data[idx]

        # Get image filename
        if idx < len(image_files):
            img_filename = image_files[idx]
        else:
            print(f"Warning: Index {idx} out of bounds for image files list")
            continue

        # Full path to image
        img_path = os.path.join(params['data_files']['image_test_file'], img_filename)

        # Add labels to list
        labels.extend(annotation['labels'].tolist())

        # Run inference
        results = model(img_path)

        # Convert to Faster R-CNN format
        pred_dict = yolo_to_faster_rcnn_format(results, threshold=0.5)

        # Add to prediction and annotation lists
        preds_adj_all.append([pred_dict])
        annot_all.append([annotation])
    except Exception as e:
        print(f"Error processing {img_filename if 'img_filename' in locals() else idx}: {e}")
        continue

print(f"Successfully processed {len(preds_adj_all)} images")

# Calculate metrics
print("\nCalculating metrics...")
sample_metrics = []
for batch_i in range(len(preds_adj_all)):
    try:
        batch_metrics = get_batch_statistics(preds_adj_all[batch_i], annot_all[batch_i], iou_threshold=0.5)
        sample_metrics += batch_metrics
    except Exception as e:
        print(f"Error calculating batch {batch_i}: {e}")

# Aggregate metrics
if sample_metrics:
    try:
        true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(
            true_positives, pred_scores, pred_labels, torch.tensor(labels)
        )

        mAP = torch.mean(AP)
        print(f"\nResults:")
        print(f"mAP: {mAP:.4f}")

        class_names = ['Wear', 'Incorrect', 'Not Wear']
        print("\nPer-class results:")
        for i, c in enumerate(ap_class):
            class_name = class_names[c] if c < len(class_names) else f"Class {c}"
            print(f"{class_name}: AP={AP[i]:.4f}, Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")
    except Exception as e:
        print(f"Error calculating final metrics: {e}")
else:
    print("No valid predictions found for evaluation.")

print("\nEvaluation complete.")