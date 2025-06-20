import logging
import os
import yaml
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
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
    parser = HfArgumentParser((DataTrainingArguments, TrainingArguments, ModelArguments))
    data_args, training_args, model_args = parser.parse_args_into_dataclasses()

    with open('../config/tc_transformers.yaml', 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    timestamp = "1749638927"

    # 실제 모델이 저장된 runs 폴더 사용
    base_output_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))  # trainer_output 대신 runs 사용
    output_dir = os.path.join(base_output_dir, timestamp)

    # timestamp 폴더 존재 여부 확인 및 사용 가능한 옵션 표시
    if not os.path.exists(output_dir):
        if os.path.exists(base_output_dir):
            available_timestamps = [f for f in os.listdir(base_output_dir)
                                    if os.path.isdir(os.path.join(base_output_dir, f))]
            logger.error(f"Timestamp directory not found: {output_dir}")
            logger.info(f"Available timestamps in {base_output_dir}:")
            for ts in available_timestamps:
                logger.info(f"  - {ts}")
            raise FileNotFoundError(
                f"Please update 'timestamp' variable to one of: {available_timestamps}"
            )
        else:
            raise FileNotFoundError(f"Runs directory not found: {base_output_dir}")

    logger.info(f"Using model from: {output_dir}")

    # 데이터 파라미터
    data_params = params.get('data_files', {})

    # 데이터셋 로드
    if params['task'] == "SST2":
        eval_dataset = datasets.load_dataset("nyu-mll/glue", "sst2", split="validation", cache_dir=model_args.cache_dir)
        text_column = "sentence"

    elif params['task'] == "COLA":
        eval_dataset = datasets.load_dataset("nyu-mll/glue", "cola", split="validation", cache_dir=model_args.cache_dir)
        text_column = "sentence"

    elif params['task'] == "MR":
        # MR 데이터셋 로드 방식 선택
        mr_source = params.get('mr_source', 'huggingface')

        if mr_source == 'huggingface':
            # Hugging Face datasets에서 로드
            mr_dataset_name = params.get('mr_dataset_name', 'rotten_tomatoes')

            if mr_dataset_name == 'rotten_tomatoes':
                eval_dataset = datasets.load_dataset("rotten_tomatoes", split="test", cache_dir=model_args.cache_dir)
                text_column = "text"
            elif mr_dataset_name == 'mattymchen/mr':
                eval_dataset = datasets.load_dataset("mattymchen/mr", split="test", cache_dir=model_args.cache_dir)
                text_column = "sentence"
            elif mr_dataset_name == 'stanfordnlp/imdb':
                eval_dataset = datasets.load_dataset("stanfordnlp/imdb", split="test", cache_dir=model_args.cache_dir)
                text_column = "text"
            else:
                try:
                    eval_dataset = datasets.load_dataset(mr_dataset_name, split="test", cache_dir=model_args.cache_dir)
                except:
                    eval_dataset = datasets.load_dataset(mr_dataset_name, split="validation",
                                                         cache_dir=model_args.cache_dir)
                text_column = "text"
        else:
            # CSV에서 로드
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
                logger.warning(f"Using '{text_column}' as text column")

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(output_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(output_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(output_dir, config=config, local_files_only=True)
    model.to(device)

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    def preprocess_function(examples):
        # Tokenize the texts
        return tokenizer(
            examples[text_column],  # 동적으로 결정된 텍스트 컬럼 사용
            padding=padding,
            max_length=params["max_seq_length"],
            truncation=True,
        )

    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    # Get the metric function (scikit-learn 에러 대비)
    try:
        metric = evaluate.load("accuracy")

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            return metric.compute(predictions=preds, references=p.label_ids)

    except ImportError as e:
        logger.warning(f"Could not load evaluate accuracy metric: {e}")
        logger.warning("Using manual accuracy calculation instead")

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            accuracy = (preds == p.label_ids).mean()
            return {"accuracy": accuracy}

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("*** Predict ***")

    predictions, labels, metrics = trainer.predict(eval_dataset, metric_key_prefix="predict")

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)


if __name__ == "__main__":
    main()