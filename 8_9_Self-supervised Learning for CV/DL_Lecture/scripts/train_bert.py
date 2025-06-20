import logging
import os
import time
import random
import sys
import yaml
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np

import transformers
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
    set_seed,
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

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    with open('../config/tc_transformers.yaml', 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    training_args.num_train_epochs = params['max_epochs']
    training_args.evaluation_strategy = "steps"
    training_args.save_steps = params['save_steps']
    training_args.eval_steps = params['eval_steps']
    training_args.warmup_ratio = params['warmup_ratio']

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Downloading and loading xnli dataset from the hub.

    # 데이터 파라미터 (CSV 로드시에만 필요)
    data_params = params.get('data_files', {})

    # 데이터 로드
    if params['task'] == "SST2":  # huggingface datasets로 부터 sst2 dataset load
        train_dataset = datasets.load_dataset("nyu-mll/glue", "sst2", split="train", cache_dir=model_args.cache_dir, )
        eval_dataset = datasets.load_dataset("nyu-mll/glue", "sst2", split="validation",
                                             cache_dir=model_args.cache_dir, )
        label_list = train_dataset.features["label"].names
        text_column = "sentence"

    elif params['task'] == "COLA":  # huggingface datasets로 부터 cola dataset load
        train_dataset = datasets.load_dataset("nyu-mll/glue", "cola", split="train", cache_dir=model_args.cache_dir, )
        eval_dataset = datasets.load_dataset("nyu-mll/glue", "cola", split="validation",
                                             cache_dir=model_args.cache_dir, )
        label_list = train_dataset.features["label"].names
        text_column = "sentence"

    elif params['task'] == "MR":  # MR dataset load
        # MR 데이터셋 로드 방식 선택 (yaml 설정에서 결정)
        mr_source = params.get('mr_source', 'huggingface')  # 'huggingface' 또는 'csv'

        if mr_source == 'huggingface':
            # Hugging Face datasets에서 로드
            mr_dataset_name = params.get('mr_dataset_name', 'rotten_tomatoes')

            if mr_dataset_name == 'rotten_tomatoes':
                train_dataset = datasets.load_dataset("rotten_tomatoes", split="train", cache_dir=model_args.cache_dir)
                eval_dataset = datasets.load_dataset("rotten_tomatoes", split="validation",
                                                     cache_dir=model_args.cache_dir)
                text_column = "text"
            elif mr_dataset_name == 'mattymchen/mr':
                train_dataset = datasets.load_dataset("mattymchen/mr", split="train", cache_dir=model_args.cache_dir)
                eval_dataset = datasets.load_dataset("mattymchen/mr", split="test", cache_dir=model_args.cache_dir)
                text_column = "sentence"
            elif mr_dataset_name == 'stanfordnlp/imdb':
                train_dataset = datasets.load_dataset("stanfordnlp/imdb", split="train", cache_dir=model_args.cache_dir)
                eval_dataset = datasets.load_dataset("stanfordnlp/imdb", split="test", cache_dir=model_args.cache_dir)
                text_column = "text"
            else:
                # 사용자 정의 데이터셋
                train_dataset = datasets.load_dataset(mr_dataset_name, split="train", cache_dir=model_args.cache_dir)
                eval_dataset = datasets.load_dataset(mr_dataset_name, split="validation",
                                                     cache_dir=model_args.cache_dir)
                text_column = "text"  # 기본값

            label_list = train_dataset.features["label"].names

        else:  # csv 방식
            # CSV에서 로드 (기존 방식)
            data_files = {}
            data_files["train"] = data_params['train_file']
            data_files["validation"] = data_params['val_file']
            train_dataset = datasets.load_dataset("csv", data_files=data_files, split="train")
            eval_dataset = datasets.load_dataset("csv", data_files=data_files, split="validation")

            # MR 데이터셋의 컬럼명 확인 및 설정
            logger.info(f"Train dataset columns: {train_dataset.column_names}")
            logger.info(f"Train dataset sample: {train_dataset[0]}")

            # 일반적인 텍스트 컬럼명들을 확인하여 자동으로 매핑
            possible_text_columns = ['text', 'review', 'sentence', 'content', 'document']
            text_column = None

            for col in possible_text_columns:
                if col in train_dataset.column_names:
                    text_column = col
                    logger.info(f"Using '{col}' as text column for MR dataset")
                    break

            if text_column is None:
                # 첫 번째 컬럼을 텍스트로 가정
                text_column = train_dataset.column_names[0]
                logger.warning(f"Could not find standard text column. Using '{text_column}' as text column")

            # 라벨 리스트 생성
            label_list = train_dataset.unique("label")
            logger.info(f"Label list for MR: {label_list}")

            # MR 데이터셋 전처리 함수 (컬럼명 표준화)
            def standardize_mr_dataset(examples):
                # 텍스트 컬럼을 'sentence'로 표준화
                examples['sentence'] = examples[text_column]
                return examples

            # MR 데이터셋 전처리 적용
            train_dataset = train_dataset.map(
                standardize_mr_dataset,
                batched=True,
                desc="Standardizing MR train dataset columns"
            )

            eval_dataset = eval_dataset.map(
                standardize_mr_dataset,
                batched=True,
                desc="Standardizing MR eval dataset columns"
            )

            text_column = "sentence"  # 표준화 후 컬럼명

    # Labels
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label={str(i): label for i, label in enumerate(label_list)},
        label2id={label: i for i, label in enumerate(label_list)},
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

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
            examples[text_column],
            padding=padding,
            max_length=params["max_seq_length"],
            truncation=True,
        )

    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
    # Log a few random samples from the training set:

    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    # Get the metric function
    metric = evaluate.load("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    timestamp = str(int(time.time()))
    training_args.output_dir = os.path.abspath((os.path.join(training_args.output_dir, timestamp)))

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training

    train_result = trainer.train()
    metrics = train_result.metrics

    trainer.save_model()  # Saves the tokenizer too for easy upload

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation

    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate(eval_dataset=eval_dataset)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()