import time
import sys
import yaml
import random
import os
import tqdm
import torchvision
from torchvision import transforms
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.tensorboard import SummaryWriter
from DL_Lecture.utils.data_prepro import MaskDataset, collate_fn
from DL_Lecture.utils.metrics import get_batch_statistics, ap_per_class

def main():

    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/mask_faster_rcnn.yaml'

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
    print(device)
    torch.backends.cudnn.benchmark = True

    # 데이터 로드
    if params['task'] == "Mask":
        # 데이터 개수
        print('train 데이터 annotations 수 : {}'.format(len(os.listdir(params['data_files']['annotation_file']))))
        print('train 데이터 images 수 : {}'.format(len(os.listdir(params['data_files']['image_file']))))
        print('val 데이터 annotations 수 : {}'.format(len(os.listdir(params['data_files']['annotation_val_file']))))
        print('val 데이터 images 수 : {}'.format(len(os.listdir(params['data_files']['image_val_file']))))

        data_transform = transforms.Compose([  # transforms.Compose : list 내의 작업을 연달아 할 수 있게 호출하는 클래스
            transforms.ToTensor()  # ToTensor : numpy 이미지에서 torch 이미지로 변경
        ])

        train_data = MaskDataset(data_transform, params['data_files']['image_file'], params['data_files']['annotation_file'])
        val_data = MaskDataset(data_transform, params['data_files']['image_val_file'], params['data_files']['annotation_val_file'])

    # 배치 단위로 네트워크에 데이터를 넘겨주는 Data loader
    train_loader = torch.utils.data.DataLoader(train_data, params['batch_size'], collate_fn=collate_fn)
    dev_loader = torch.utils.data.DataLoader(val_data, params['batch_size'], collate_fn=collate_fn)

    # 학습 모델 생성

    def get_model_instance_segmentation(num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model

    model = get_model_instance_segmentation(4).to(device) # 모델을 지정한 device로 올려줌

    optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'], weight_decay=params['l2_reg_lambda'])  # model.parameters -> 가중치 w들을 의미

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp)))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    summary_dir = os.path.join(out_dir, "summaries")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    writer = SummaryWriter(summary_dir) # TensorBoard를 위한 초기화
     # training 시작
    start_time = time.time()
    global_steps = 0
    highest_val_mAP = 0.0
    print('========================================')
    print("Start training...")
    for epoch in range(params['max_epochs']):
        train_loss = 0
        train_batch_cnt = 0
        model.train()
        for imgs, annotations in train_loader:

            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            train_loss += losses.item()
            train_batch_cnt += 1

            losses.backward()# 가중치 w에 대해 loss를 미분
            optimizer.step()# 가중치들을 업데이트
            optimizer.zero_grad()

            writer.add_scalar("Batch/Loss", losses.item(), global_steps)

            global_steps += 1
            if (global_steps) % 100 == 0:
                print('Epoch [{}], Step [{}], Loss: {:.4f}'.format(epoch+1, global_steps, losses.item()))

        train_ave_loss = train_loss / train_batch_cnt # 학습 데이터의 평균 loss
        training_time = (time.time() - start_time) / 60
        writer.add_scalar("Train/Loss", train_ave_loss, epoch)
        print('========================================')
        print("epoch:", epoch + 1, "/ global_steps:", global_steps)
        print("training dataset average loss: %.3f" % train_ave_loss)
        print("training_time: %.2f minutes" % training_time)

        model.eval()
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

        # validation (for early stopping)
        labels = []
        preds_adj_all = []
        annot_all = []

        for imgs, annotations in dev_loader:
            imgs = list(img.to(device) for img in imgs)

            for t in annotations:
                labels += t['labels']

            with torch.no_grad():
                preds_adj = make_prediction(model, imgs, 0.5)
                preds_adj = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in preds_adj]
                preds_adj_all.append(preds_adj)
                annot_all.append(annotations)

        sample_metrics = []
        for batch_i in range(len(preds_adj_all)):
            sample_metrics += get_batch_statistics(preds_adj_all[batch_i], annot_all[batch_i], iou_threshold=0.5)

        true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))]  # 배치가 전부 합쳐짐
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels,
                                                           torch.tensor(labels))
        print(AP)
        val_mAP = torch.mean(AP)
        print(f'val_mAP : {val_mAP}')
        writer.add_scalar("Val/mAP", val_mAP, epoch)

        if val_mAP > highest_val_mAP:  # validation accuracy가 경신될 때
            save_path = checkpoint_dir + '/epoch_' + str(epoch + 1) + '.pth'
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict()},
                       save_path)

            save_path = checkpoint_dir + '/best.pth'
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict()},
                       save_path)  # best accuracy에 도달할 때만 모델을 저장함으로써 early stopping
            highest_val_mAP = val_mAP
        epoch += 1


if __name__ == '__main__':
    main()