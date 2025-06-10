# Simple inference script for test images without annotations with timing measurements
import sys
import os
import torch
import yaml
import cv2
import numpy as np
import time
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

# Print at the very beginning to confirm this script is running
print("Script starting...")
print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())


# Custom transform for images
def transform_image(image_path):
    # Load image using PIL
    image = Image.open(image_path).convert('RGB')

    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Apply transformation
    img_tensor = transform(image)

    return img_tensor, image.size


# Custom plot function with color-coding and smaller text
def plot_image_from_output(img_tensor, output, img_path, return_img=False):
    # Load the original image with OpenCV to maintain original size
    img_np = cv2.imread(img_path)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Get image dimensions
    height, width = img_np.shape[:2]

    # Class names exactly as defined in training
    class_names = ['Not Wear', 'Wear', 'Incorrect']

    # Define colors for each class (B, G, R) in OpenCV format
    color_map = {
        1: (0, 0, 255),  # Red for "Not Wear"
        2: (0, 255, 0),  # Green for "Wear"
        3: (0, 165, 255)  # Orange for "Incorrect"
    }

    # Draw bounding boxes with class labels
    if 'boxes' in output and len(output['boxes']) > 0:
        boxes = output['boxes'].cpu().numpy()
        labels = output['labels'].cpu().numpy() if 'labels' in output else []
        scores = output['scores'].cpu().numpy() if 'scores' in output else []

        for i, box in enumerate(boxes):
            # Get coordinates
            x1, y1, x2, y2 = box.astype(int)

            # Get class label
            if len(labels) > i:
                label_idx = int(labels[i])

                # Ensure we use the correct class name by index
                class_name = class_names[label_idx - 1] if 0 <= label_idx - 1 < len(
                    class_names) else f"Class {label_idx}"
                color = color_map.get(label_idx, (255, 255, 255))

                # Debug print to verify mapping
                if not return_img:  # Only print when displaying, not when saving
                    print(f"Detection {i + 1}: Label index {label_idx} -> Class name '{class_name}'")
            else:
                class_name = "Unknown"
                color = (255, 255, 255)

            # Get score if available
            score_text = f"{scores[i]:.2f}" if len(scores) > i else ""

            # Draw bounding box
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)

            # Prepare text to display
            if score_text:
                display_text = f"{class_name}: {score_text}"
            else:
                display_text = class_name

            # Calculate text position
            # Draw the text with smaller font
            font_scale = 0.5  # Smaller font
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(display_text, font, font_scale, 1)[0]

            # Draw background rectangle for text
            cv2.rectangle(img_np, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)

            # Draw text
            cv2.putText(img_np, display_text, (x1, y1 - 5), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the image or return it
    if return_img:
        return img_np
    else:
        # For display in Jupyter or other environments
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 10))
            plt.imshow(img_np)
            plt.axis('off')
            plt.show()
        except ImportError:
            # If matplotlib is not available
            print("Matplotlib not available. Cannot display image.")

    return None


# Function to convert YOLO output to Faster R-CNN format
def yolo_to_faster_rcnn_format(results, threshold=0.5):
    pred_dict = {
        'boxes': [],
        'labels': [],
        'scores': []
    }

    for *box, conf, cls_id in results.xyxy[0].cpu().numpy():
        if conf >= threshold:
            pred_dict['boxes'].append(torch.tensor(box))
            # Convert class index from 0-indexed to 1-indexed
            pred_dict['labels'].append(torch.tensor(int(cls_id) + 1))
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


# Main function
def main():
    # Get config file path
    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
    else:
        params_filename = '../config/mask_yolo.yaml'

    print(f"Using config file: {params_filename}")

    # Load the configuration
    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get test image directory from config
    image_dir = params['data_files']['image_test_file']
    print(f"Using test image directory: {image_dir}")

    # Check if the directory exists
    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' does not exist.")
        sys.exit(1)

    # Set timestamp and directories
    timestamp = "1746610118"  # Update this to your timestamp
    out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp)))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints/best.pt"))
    print(f"Looking for model at: {checkpoint_dir}")

    # Prepare results directory
    results_dir = os.path.join(out_dir, "custom_inference_results")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to {results_dir}")

    # Try to import YOLOv5
    try:
        import yolov5
        print("Using yolov5 package")
        use_package = True
    except ImportError:
        print("YOLOv5 package not found, using torch hub")
        use_package = False

    # Load model
    model_load_start = time.time()
    if os.path.exists(checkpoint_dir):
        print(f"Found model at {checkpoint_dir}")
        if use_package:
            try:
                model = yolov5.load(checkpoint_dir)
            except Exception as e:
                print(f"Error with yolov5.load: {e}")
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=checkpoint_dir)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=checkpoint_dir)
    else:
        print(f"Model not found at {checkpoint_dir}, using YOLOv5s")
        if use_package:
            try:
                model = yolov5.load('yolov5s.pt')
            except Exception as e:
                print(f"Error loading YOLOv5s with package: {e}")
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    model.to(device)
    model_load_time = time.time() - model_load_start
    print(f"Model loaded and moved to device in {model_load_time:.4f} seconds")

    # Get image filenames
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = [f for f in os.listdir(image_dir) if os.path.splitext(f.lower())[1] in image_extensions]
    print(f"Found {len(image_files)} images in directory")

    # Print class mapping for verification
    class_names = ['Not Wear', 'Wear', 'Incorrect']
    print("\nClass mapping:")
    for idx, name in enumerate(class_names):
        print(f"Class {idx + 1}: {name}")

    # Ask user if they want to display all images
    show_all_images = input("Do you want to display all images? (y/n): ").lower() == 'y'
    save_all_images = input("Do you want to save all result images? (y/n): ").lower() == 'y'

    # Prepare to collect timing statistics
    total_inference_time = 0
    inference_times = []

    # Create a CSV file to store timing results
    timing_file = os.path.join(results_dir, 'inference_timing.csv')
    with open(timing_file, 'w') as f:
        f.write("Image,LoadTime,InferenceTime,ProcessingTime,TotalTime\n")

    # Process all images
    print("\nProcessing all images...")
    for img_idx, img_filename in enumerate(tqdm(image_files)):
        # Full path to image
        img_path = os.path.join(image_dir, img_filename)

        # Run inference directly on the image file
        try:
            # Measure load time
            load_start = time.time()
            img_tensor, _ = transform_image(img_path)
            load_time = time.time() - load_start

            # Measure inference time
            inference_start = time.time()
            results = model(img_path)
            inference_time = time.time() - inference_start

            # Convert results
            process_start = time.time()
            pred_dict = yolo_to_faster_rcnn_format(results)
            process_time = time.time() - process_start

            # Track statistics
            total_time = load_time + inference_time + process_time
            total_inference_time += inference_time
            inference_times.append(inference_time)

            # Write timing to CSV
            with open(timing_file, 'a') as f:
                f.write(f"{img_filename},{load_time:.6f},{inference_time:.6f},{process_time:.6f},{total_time:.6f}\n")

            # Display results if requested
            if show_all_images or img_idx == 0:  # Always show the first image
                print(f"\nImage {img_idx + 1}: {img_filename}")
                print(
                    f"Load time: {load_time:.6f}s, Inference time: {inference_time:.6f}s, Processing time: {process_time:.6f}s")
                print("Predictions:")
                print("Labels:", pred_dict['labels'])
                print("Boxes:", pred_dict['boxes'])
                print("Scores:", pred_dict['scores'])

                # Print prediction classes
                for i, label in enumerate(pred_dict['labels']):
                    label_idx = label.item()
                    class_name = class_names[label_idx - 1] if 0 <= label_idx - 1 < len(
                        class_names) else f"Class {label_idx}"
                    score = pred_dict['scores'][i].item() if len(pred_dict['scores']) > i else 0
                    print(f"  Pred {i + 1}: Class {label_idx} = {class_name} (score: {score:.2f})")

                plot_image_from_output(img_tensor, pred_dict, img_path)

            # Save results
            if save_all_images:
                try:
                    # Save the prediction visualization
                    result_img = plot_image_from_output(img_tensor, pred_dict, img_path, return_img=True)
                    result_path = os.path.join(results_dir, f"pred_{img_filename}")
                    # Convert RGB to BGR for OpenCV
                    result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(result_path, result_img_bgr)
                except Exception as e:
                    print(f"Error saving image {img_filename}: {e}")

        except Exception as e:
            print(f"Error processing image {img_filename}: {e}")
            continue

    # Calculate timing statistics
    avg_inference_time = total_inference_time / len(image_files) if image_files else 0
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

    if inference_times:
        min_time = min(inference_times)
        max_time = max(inference_times)
        median_time = sorted(inference_times)[len(inference_times) // 2]
        fps_min = 1.0 / max_time if max_time > 0 else 0
        fps_max = 1.0 / min_time if min_time > 0 else 0
    else:
        min_time = max_time = median_time = fps_min = fps_max = 0

    # Print timing summary
    print("\n===== Inference Timing Summary =====")
    print(f"Total images processed: {len(image_files)}")
    print(f"Average inference time: {avg_inference_time * 1000:.2f} ms")
    print(f"Median inference time: {median_time * 1000:.2f} ms")
    print(f"Min inference time: {min_time * 1000:.2f} ms")
    print(f"Max inference time: {max_time * 1000:.2f} ms")
    print(f"Average FPS: {fps:.2f}")
    print(f"Min FPS: {fps_min:.2f}")
    print(f"Max FPS: {fps_max:.2f}")

    # Save timing summary
    timing_summary_file = os.path.join(results_dir, 'timing_summary.txt')
    with open(timing_summary_file, 'w') as f:
        f.write("===== Inference Timing Summary =====\n")
        f.write(f"Total images processed: {len(image_files)}\n")
        f.write(f"Average inference time: {avg_inference_time * 1000:.2f} ms\n")
        f.write(f"Median inference time: {median_time * 1000:.2f} ms\n")
        f.write(f"Min inference time: {min_time * 1000:.2f} ms\n")
        f.write(f"Max inference time: {max_time * 1000:.2f} ms\n")
        f.write(f"Average FPS: {fps:.2f}\n")
        f.write(f"Min FPS: {fps_min:.2f}\n")
        f.write(f"Max FPS: {fps_max:.2f}\n")

    print(f"\nTiming summary saved to {timing_summary_file}")
    print(f"Detailed timing data saved to {timing_file}")
    print("\nScript completed successfully")


if __name__ == "__main__":
    main()