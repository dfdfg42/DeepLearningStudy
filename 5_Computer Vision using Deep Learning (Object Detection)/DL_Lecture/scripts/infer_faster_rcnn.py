import torch
import yaml
import sys
import os
import time
import numpy as np
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from DL_Lecture.utils.data_prepro import MaskDataset, collate_fn, plot_image_from_output


def main():
    print('Faster R-CNN Mask Detection evaluation with timing')

    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/mask_faster_rcnn.yaml'

    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    # GPU 사용이 가능하면 사용하고, 불가능하면 CPU 활용
    print("GPU Available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    timestamp = "1746483608"  # 필요에 따라 수정
    out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp)))

    # 데이터 로드
    if params['task'] == "Mask":
        print('test 데이터 annotations 수 : {}'.format(len(os.listdir(params['data_files']['annotation_test_file']))))
        print('test 데이터 images 수 : {}'.format(len(os.listdir(params['data_files']['image_test_file']))))

        data_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        test_data = MaskDataset(data_transform, params['data_files']['image_test_file'],
                                params['data_files']['annotation_test_file'])

    # 결과 디렉토리 생성
    results_dir = os.path.join(out_dir, "inference_results")
    os.makedirs(results_dir, exist_ok=True)

    # 배치 크기를 가져옴
    batch_size = params['batch_size']
    print(f"Batch size: {batch_size}")

    # 데이터 로더 생성 - 배치 단위로 데이터 로드
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, collate_fn=collate_fn)

    # 총 이미지 수와 배치 수 계산
    total_images = len(test_data)
    total_batches = len(test_loader)
    print(f"Total images: {total_images}")
    print(f"Total batches: {total_batches}")

    # 학습 모델 생성
    model_load_start = time.time()
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)

    # 저장된 state 불러오기
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints/best.pth"))
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    model_load_time = time.time() - model_load_start
    print(f"Model loading time: {model_load_time:.4f} seconds")

    def make_prediction(model, img, threshold):
        preds = model(img)
        for id in range(len(preds)):
            idx_list = []

            for idx, score in enumerate(preds[id]['scores']):
                if score > threshold:
                    idx_list.append(idx)

            preds[id]['boxes'] = preds[id]['boxes'][idx_list].cpu()
            preds[id]['labels'] = preds[id]['labels'][idx_list].cpu()
            preds[id]['scores'] = preds[id]['scores'][idx_list].cpu()

        return preds

    # 타이밍 통계를 저장할 리스트
    batch_inference_times = []
    per_image_inference_times = []

    # 타이밍 결과를 저장할 CSV 파일
    timing_file = os.path.join(results_dir, 'faster_rcnn_inference_timing.csv')
    with open(timing_file, 'w') as f:
        f.write("Batch,BatchSize,InferenceTime,InferenceTimePerImage\n")

    # 이미지 카운터
    image_count = 0

    # 추론 실행
    with torch.no_grad():
        for batch_idx, (imgs, annotations) in enumerate(tqdm(test_loader, desc="Processing batches")):
            # 이미지를 디바이스로 이동
            imgs = list(img.to(device) for img in imgs)

            # 배치 크기 (마지막 배치는 적을 수 있음)
            current_batch_size = len(imgs)

            # 추론 시간 측정 시작
            inference_start = time.time()

            # 추론 실행
            preds = make_prediction(model, imgs, 0.5)

            # 추론 시간 측정 완료
            inference_time = time.time() - inference_start
            per_image_time = inference_time / current_batch_size

            # 타이밍 정보 저장
            batch_inference_times.append(inference_time)
            per_image_inference_times.extend([per_image_time] * current_batch_size)

            # CSV에 결과 저장
            with open(timing_file, 'a') as f:
                f.write(f"{batch_idx},{current_batch_size},{inference_time:.6f},{per_image_time:.6f}\n")

            # 첫 번째 배치의 결과만 출력
            if batch_idx == 0:
                _idx = 0  # 첫 번째 이미지
                print(f"\nExample from Batch {batch_idx}, Image {_idx}")
                print("Inference time for this batch:", inference_time * 1000, "ms")
                print("Inference time per image:", per_image_time * 1000, "ms")
                print("Target labels:", annotations[_idx]['labels'])
                print("Target boxes:", annotations[_idx]['boxes'])
                plot_image_from_output(imgs[_idx], annotations[_idx])
                print("Prediction labels:", preds[_idx]['labels'])
                print("Prediction boxes:", preds[_idx]['boxes'])
                plot_image_from_output(imgs[_idx], preds[_idx])

            # 이미지 카운터 증가
            image_count += current_batch_size

    # 타이밍 통계 계산
    batch_times = np.array(batch_inference_times)
    image_times = np.array(per_image_inference_times)

    # 배치 단위 통계
    avg_batch_time = np.mean(batch_times)
    median_batch_time = np.median(batch_times)
    min_batch_time = np.min(batch_times)
    max_batch_time = np.max(batch_times)

    # 이미지 단위 통계
    avg_image_time = np.mean(image_times)
    median_image_time = np.median(image_times)
    min_image_time = np.min(image_times)
    max_image_time = np.max(image_times)

    # FPS 계산
    fps = 1.0 / avg_image_time if avg_image_time > 0 else 0

    # 통계 출력
    print("\n===== Inference Timing Summary =====")
    print(f"Total batches processed: {total_batches}")
    print(f"Total images processed: {image_count}")
    print(f"Batch size: {batch_size}")

    print("\nBatch statistics:")
    print(f"Average batch inference time: {avg_batch_time * 1000:.2f} ms")
    print(f"Median batch inference time: {median_batch_time * 1000:.2f} ms")
    print(f"Min batch inference time: {min_batch_time * 1000:.2f} ms")
    print(f"Max batch inference time: {max_batch_time * 1000:.2f} ms")

    print("\nPer-image statistics:")
    print(f"Average inference time: {avg_image_time * 1000:.2f} ms per image")
    print(f"Median inference time: {median_image_time * 1000:.2f} ms per image")
    print(f"Min inference time: {min_image_time * 1000:.2f} ms per image")
    print(f"Max inference time: {max_image_time * 1000:.2f} ms per image")
    print(f"Average throughput: {fps:.2f} FPS")

    # 통계 저장
    summary_file = os.path.join(results_dir, 'faster_rcnn_timing_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("===== Faster R-CNN Inference Timing Summary =====\n")
        f.write(f"Total batches processed: {total_batches}\n")
        f.write(f"Total images processed: {image_count}\n")
        f.write(f"Batch size: {batch_size}\n")

        f.write("\nBatch statistics:\n")
        f.write(f"Average batch inference time: {avg_batch_time * 1000:.2f} ms\n")
        f.write(f"Median batch inference time: {median_batch_time * 1000:.2f} ms\n")
        f.write(f"Min batch inference time: {min_batch_time * 1000:.2f} ms\n")
        f.write(f"Max batch inference time: {max_batch_time * 1000:.2f} ms\n")

        f.write("\nPer-image statistics:\n")
        f.write(f"Average inference time: {avg_image_time * 1000:.2f} ms per image\n")
        f.write(f"Median inference time: {median_image_time * 1000:.2f} ms per image\n")
        f.write(f"Min inference time: {min_image_time * 1000:.2f} ms per image\n")
        f.write(f"Max inference time: {max_image_time * 1000:.2f} ms per image\n")
        f.write(f"Average throughput: {fps:.2f} FPS\n")

    print(f"\nTiming summary saved to {summary_file}")
    print(f"Detailed timing data saved to {timing_file}")
    print("\nEvaluation completed successfully")


if __name__ == "__main__":
    main()