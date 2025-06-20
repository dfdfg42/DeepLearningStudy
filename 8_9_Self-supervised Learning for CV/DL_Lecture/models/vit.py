from transformers import ViTModel

from torch import nn


class ViT_model(nn.Module):
    #모델 초기화
    def __init__(self, model_name, num_classes = 10):
        super().__init__()
        # 사전 훈련된 모델을 허깅페이스에서 로드
        #매개변수로 전달된 모델을 불러옴
        self.vit = ViTModel.from_pretrained(model_name)
        #분류기 정의 , vit의 출력을 받아 최종 클래스 개수로 매핑 -> 히든 사이즈 값을 가져와 입력 차원으로 사용
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    #순전파
    def forward(self, x):
        #vit 모델 실행시킴 입력 'x' 를 vit 모델에 통과시킴
        outputs = self.vit(**x)
        #vit의 마지막 트랜스포머의 블록 출력을 가져옴
        last_hidden_states = outputs.last_hidden_state


        # vit 출력에서 cls토큰에 해당하는 임베딩 추출
        #추출된 cls토큰을 분류기에 통과시켜 최종 점수 계산
        x = self.classifier(last_hidden_states[:, 0])
        return x
