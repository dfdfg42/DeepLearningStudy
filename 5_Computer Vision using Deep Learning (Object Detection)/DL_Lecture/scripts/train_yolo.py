import time
import sys
import yaml
import random
import os
import torch
import shutil
import subprocess
from tqdm import tqdm
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from DL_Lecture.utils.data_prepro import MaskDataset


def main():
    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/mask_yolo.yaml'

    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    # 랜덤 시드 세팅
    if 'random_seed' in params:
        seed = params['random_seed']
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # 데이터 경로 및 파일 수 확인
    if params['task'] == "Mask":
        # 데이터 개수
        print('train 데이터 annotations 수 : {}'.format(len(os.listdir(params['data_files']['annotation_file']))))
        print('train 데이터 images 수 : {}'.format(len(os.listdir(params['data_files']['image_file']))))
        print('val 데이터 annotations 수 : {}'.format(len(os.listdir(params['data_files']['annotation_val_file']))))
        print('val 데이터 images 수 : {}'.format(len(os.listdir(params['data_files']['image_val_file']))))

    # 타임스탬프 생성 및 출력 디렉토리 설정
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp)))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    summary_dir = os.path.join(out_dir, "summaries")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 텐서보드 초기화
    writer = SummaryWriter(summary_dir)

    # YOLOv5 데이터셋 형식으로 변환
    def convert_to_yolo_format(image_path, annotation_path, output_path):
        """
        Convert annotation format to YOLOv5 format using MaskDataset class
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            os.makedirs(os.path.join(output_path, 'images'))
            os.makedirs(os.path.join(output_path, 'labels'))

        # Get the dataset to access annotations properly
        data_transform = transforms.Compose([transforms.ToTensor()])
        dataset = MaskDataset(data_transform, image_path, annotation_path)

        print(f"Converting {len(dataset)} images to YOLO format...")

        # Get image files
        image_files = os.listdir(image_path)

        # Process each image and its annotation
        for idx in tqdm(range(len(dataset))):
            # Get image and annotation
            img, annotation = dataset[idx]

            # We don't have direct access to image paths, so we'll use the index
            # Let's assume images are ordered the same as in the dataset
            if idx < len(image_files):
                img_name = image_files[idx]
            else:
                print(f"Warning: Index {idx} out of bounds for image files list")
                continue

            # Copy image file to output directory
            try:
                shutil.copy(
                    os.path.join(image_path, img_name),
                    os.path.join(output_path, 'images', img_name)
                )
            except Exception as e:
                print(f"Error copying image {img_name}: {e}")
                continue

            # Get image dimensions
            img_height, img_width = img.shape[1], img.shape[2]

            # Create YOLO format label file
            try:
                label_filename = os.path.splitext(img_name)[0] + '.txt'
                with open(os.path.join(output_path, 'labels', label_filename), 'w') as f:
                    for box_idx in range(len(annotation['boxes'])):
                        box = annotation['boxes'][box_idx]
                        label = annotation['labels'][box_idx].item() - 1  # Convert to 0-indexed classes

                        # Convert box coords from [x1, y1, x2, y2] to normalized [x_center, y_center, width, height]
                        x1, y1, x2, y2 = box
                        x_center = ((x1 + x2) / 2) / img_width
                        y_center = ((y1 + y2) / 2) / img_height
                        width = abs(x2 - x1) / img_width
                        height = abs(y2 - y1) / img_height

                        f.write(f"{label} {x_center} {y_center} {width} {height}\n")
            except Exception as e:
                print(f"Error creating YOLO label for {img_name}: {e}")
                continue

    # YOLOv5 데이터셋 생성
    data_transform = transforms.Compose([transforms.ToTensor()])

    train_data = MaskDataset(data_transform, params['data_files']['image_file'],
                             params['data_files']['annotation_file'])
    val_data = MaskDataset(data_transform, params['data_files']['image_val_file'],
                           params['data_files']['annotation_val_file'])

    temp_dataset_dir = os.path.join(out_dir, "yolo_dataset")
    train_dir = os.path.join(temp_dataset_dir, "train")
    val_dir = os.path.join(temp_dataset_dir, "val")

    print("Converting dataset to YOLO format...")
    convert_to_yolo_format(params['data_files']['image_file'], params['data_files']['annotation_file'], train_dir)
    convert_to_yolo_format(params['data_files']['image_val_file'], params['data_files']['annotation_val_file'], val_dir)

    # YOLOv5 데이터셋 YAML 파일 생성
    dataset_yaml = {
        'path': temp_dataset_dir,
        'train': os.path.join(train_dir, 'images'),
        'val': os.path.join(val_dir, 'images'),
        'nc': 3,  # 클래스 수 (마스크 데이터셋은 3개의 클래스: )
        'names': ['Not Wear', 'Wear', 'Incorrect']
    }

    dataset_yaml_path = os.path.join(temp_dataset_dir, 'dataset.yaml')
    with open(dataset_yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f)

    # YOLOv5 설치 확인 및 로드
    try:
        # 첫 번째 방법: yolov5 패키지 사용
        import yolov5
        use_package = True
        print("Using yolov5 package")
    except ImportError:
        try:
            # 두 번째 방법: torch hub 사용
            print("YOLOv5 package not found. Using torch hub instead...")
            use_package = False
        except Exception as e:
            print(f"Error: {e}")
            print("Installing YOLOv5...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            use_package = False

    # YOLOv5 모델 로딩
    model = None
    if use_package:
        try:
            model = yolov5.load('yolov5s.pt')  # .pt 확장자 추가
            print("Successfully loaded YOLOv5 model using package")
        except Exception as e:
            print(f"Error loading model with yolov5 package: {e}")
            use_package = False

    # 학습 파라미터 설정
    train_args = {
        'data': dataset_yaml_path,
        'epochs': params['max_epochs'],
        'batch_size': params['batch_size'],
        'imgsz': 640,
        'project': out_dir,
        'name': 'yolo_train',
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'workers': 4,
        'seed': params['random_seed'] if 'random_seed' in params else 0,
        'lr0': params['lr'],
        'weight_decay': params.get('l2_reg_lambda', 0.0005),  # 기본값 설정
        'momentum': params.get('momentum', 0.9),  # 기본값 설정
        'save_period': 1,  # Save model every epoch
    }

    # 학습 실행
    start_time = time.time()

    # 패키지 또는 torch hub에 따라 다른 훈련 방식 사용
    if use_package and model is not None:
        try:
            print("Training with yolov5 package...")
            results = model.train(**train_args)
            # 최종 모델 저장
            final_model_path = os.path.join(checkpoint_dir, 'best.pt')
            shutil.copy(os.path.join(out_dir, 'yolo_train', 'weights', 'best.pt'), final_model_path)
            print(f"Best model saved to {final_model_path}")

            # TensorBoard에 결과 기록
            writer.add_scalar("Train/Final_mAP", results.results_dict['metrics/mAP_0.5'], params['max_epochs'])
            writer.add_scalar("Val/Final_mAP", results.results_dict['metrics/mAP_0.5_0.95'], params['max_epochs'])
        except Exception as e:
            print(f"Error during training with yolov5 package: {e}")
            use_package = False

    if not use_package or model is None:
        # torch hub 방식으로 훈련 - Ultralytics YOLOv5를 직접 호출
        print("Using YOLOv5 from Ultralytics GitHub repository")

        # YOLOv5 리포지토리 클론 (이미 설치되어 있지 않은 경우)
        if not os.path.exists("yolov5"):
            print("Cloning YOLOv5 repository...")
            subprocess.check_call(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
        else:
            print("YOLOv5 repository already exists")

        # 훈련 명령 설정
        train_cmd = [
            sys.executable, "yolov5/train.py",
            "--data", dataset_yaml_path,
            "--epochs", str(params['max_epochs']),
            "--batch-size", str(params['batch_size']),
            "--img", "640",
            "--project", out_dir,
            "--name", "yolo_train",
            "--weights", "yolov5s.pt"
        ]

        if torch.cuda.is_available():
            train_cmd.append("--device")
            train_cmd.append("0")  # GPU 0 사용

        print(f"Running command: {' '.join(train_cmd)}")

        # 훈련 실행
        try:
            subprocess.check_call(train_cmd)

            # 최종 모델 저장
            weights_dir = os.path.join(out_dir, 'yolo_train', 'weights')
            if os.path.exists(os.path.join(weights_dir, 'best.pt')):
                final_model_path = os.path.join(checkpoint_dir, 'best.pt')
                shutil.copy(os.path.join(weights_dir, 'best.pt'), final_model_path)
                print(f"Best model saved to {final_model_path}")
            else:
                print(f"Warning: best.pt not found in {weights_dir}")

            # 결과를 TensorBoard에 기록하기 위해 results.csv 파일 읽기
            try:
                import pandas as pd
                results_file = os.path.join(out_dir, 'yolo_train', 'results.csv')
                if os.path.exists(results_file):
                    results_df = pd.read_csv(results_file)
                    final_map = results_df['metrics/mAP_0.5'].iloc[-1] if 'metrics/mAP_0.5' in results_df.columns else 0
                    final_map_95 = results_df['metrics/mAP_0.5:0.95'].iloc[
                        -1] if 'metrics/mAP_0.5:0.95' in results_df.columns else 0

                    writer.add_scalar("Train/Final_mAP", final_map, params['max_epochs'])
                    writer.add_scalar("Val/Final_mAP", final_map_95, params['max_epochs'])
                else:
                    print(f"Warning: results.csv not found at {results_file}")
            except Exception as e:
                print(f"Error reading results file: {e}")
        except Exception as e:
            print(f"Error during training with subprocess: {e}")

    # 학습 결과 출력
    training_time = (time.time() - start_time) / 60
    print('========================================')
    print(f"Training completed in {training_time:.2f} minutes")

    writer.close()
    print("Done!")


if __name__ == '__main__':
    main()