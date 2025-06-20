import os
import yaml
from dataclasses import dataclass, field
from typing import Optional

import datasets

from transformers import (
    HfArgumentParser,
    pipeline,
    TrainingArguments
)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


def main():
    with open('../config/tc_transformers.yaml', 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)
    parser = HfArgumentParser((TrainingArguments, ModelArguments))
    training_args, model_args = parser.parse_args_into_dataclasses()


    timestamp = "1749570504"

    # 실제 모델이 저장된 runs 폴더 사용
    base_output_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
    output_dir = os.path.join(base_output_dir, timestamp)

    # timestamp 폴더 존재 여부 확인 및 사용 가능한 옵션 표시
    if not os.path.exists(output_dir):
        if os.path.exists(base_output_dir):
            available_timestamps = [f for f in os.listdir(base_output_dir)
                                    if os.path.isdir(os.path.join(base_output_dir, f))]
            print(f"❌ Timestamp directory not found: {output_dir}")
            print(f"📁 Available timestamps in {base_output_dir}:")
            for ts in available_timestamps:
                print(f"  - {ts}")
            raise FileNotFoundError(
                f"Please update 'timestamp' variable to one of: {available_timestamps}"
            )
        else:
            raise FileNotFoundError(f"Runs directory not found: {base_output_dir}")

    print(f"✅ Using model from: {output_dir}")

    # 데이터셋 로드 및 샘플 문장 선택
    if params['task'] == "SST2":
        eval_dataset = datasets.load_dataset("nyu-mll/glue", "sst2", split="validation", cache_dir=model_args.cache_dir)
        sentence = eval_dataset['sentence'][0]
        print(f"📝 Sample sentence (SST2): {sentence}")

    elif params['task'] == "COLA":
        eval_dataset = datasets.load_dataset("nyu-mll/glue", "cola", split="validation", cache_dir=model_args.cache_dir)
        sentence = eval_dataset['sentence'][0]
        print(f"📝 Sample sentence (COLA): {sentence}")

    elif params['task'] == "MR":
        # MR 데이터셋 로드 방식 선택
        mr_source = params.get('mr_source', 'huggingface')

        if mr_source == 'huggingface':
            # Hugging Face datasets에서 로드
            mr_dataset_name = params.get('mr_dataset_name', 'rotten_tomatoes')

            if mr_dataset_name == 'rotten_tomatoes':
                eval_dataset = datasets.load_dataset("rotten_tomatoes", split="test", cache_dir=model_args.cache_dir)
                sentence = eval_dataset['text'][0]
                print(f"📝 Sample sentence (MR - Rotten Tomatoes): {sentence}")

            elif mr_dataset_name == 'mattymchen/mr':
                eval_dataset = datasets.load_dataset("mattymchen/mr", split="test", cache_dir=model_args.cache_dir)
                sentence = eval_dataset['sentence'][0]
                print(f"📝 Sample sentence (MR - SentEval): {sentence}")

            elif mr_dataset_name == 'stanfordnlp/imdb':
                eval_dataset = datasets.load_dataset("stanfordnlp/imdb", split="test", cache_dir=model_args.cache_dir)
                sentence = eval_dataset['text'][0]
                print(f"📝 Sample sentence (MR - IMDB): {sentence}")

            else:
                try:
                    eval_dataset = datasets.load_dataset(mr_dataset_name, split="test", cache_dir=model_args.cache_dir)
                    # 첫 번째 텍스트 컬럼 찾기
                    text_columns = ['text', 'sentence', 'review', 'content']
                    text_column = None
                    for col in text_columns:
                        if col in eval_dataset.column_names:
                            text_column = col
                            break
                    if text_column is None:
                        text_column = eval_dataset.column_names[0]

                    sentence = eval_dataset[text_column][0]
                    print(f"📝 Sample sentence (MR - {mr_dataset_name}): {sentence}")

                except Exception as e:
                    print(f"❌ Error loading {mr_dataset_name}: {e}")
                    # 기본값으로 rotten_tomatoes 사용
                    eval_dataset = datasets.load_dataset("rotten_tomatoes", split="test",
                                                         cache_dir=model_args.cache_dir)
                    sentence = eval_dataset['text'][0]
                    print(f"📝 Sample sentence (MR - Rotten Tomatoes fallback): {sentence}")
        else:
            # CSV에서 로드
            data_params = params.get('data_files', {})
            data_files = {}
            data_files["test"] = data_params['test_file']
            eval_dataset = datasets.load_dataset("csv", data_files=data_files, split="test")

            # 텍스트 컬럼 자동 감지
            possible_text_columns = ['text', 'review', 'sentence', 'content', 'document']
            text_column = None

            for col in possible_text_columns:
                if col in eval_dataset.column_names:
                    text_column = col
                    break

            if text_column is None:
                text_column = eval_dataset.column_names[0]
                print(f"⚠️ Using '{text_column}' as text column")

            sentence = eval_dataset[text_column][0]
            print(f"📝 Sample sentence (MR - CSV): {sentence}")

    # 추론 파이프라인 생성
    print(f"🚀 Creating inference pipeline...")
    try:
        inference = pipeline(
            'text-classification',
            model=output_dir,
            tokenizer=output_dir,
            device=0 if os.environ.get('CUDA_VISIBLE_DEVICES') else -1  # GPU 사용 가능하면 GPU 사용
        )

        print('=' * 20, 'Inference Results', '=' * 20)
        result = inference(sentence)
        print(f"Input: {sentence}")
        print(f"Prediction: {result}")
        print('=' * 55)

        # 추가 테스트 문장들 (선택사항)
        if params['task'] == "SST2":
            test_sentences = [
                "This movie is amazing!",
                "I hate this film.",
                "The plot was confusing but the acting was great."
            ]
        elif params['task'] == "COLA":
            test_sentences = [
                "The cat sat on the mat.",
                "Cat the on sat mat the.",  # 비문법적
                "John gave Mary a book."
            ]
        elif params['task'] == "MR":
            test_sentences = [
                "This movie is fantastic and entertaining!",
                "Boring and poorly written.",
                "The cinematography was beautiful but the story was weak."
            ]

        print("\n🧪 Additional test sentences:")
        for i, test_sentence in enumerate(test_sentences, 1):
            result = inference(test_sentence)
            print(f"{i}. Input: {test_sentence}")
            print(f"   Prediction: {result}")

    except Exception as e:
        print(f"❌ Error creating pipeline: {e}")
        print(
            f"Make sure the model directory contains all necessary files (config.json, pytorch_model.bin, tokenizer files)")


if __name__ == "__main__":
    main()