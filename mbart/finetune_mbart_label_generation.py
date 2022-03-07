import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import MBartTokenizer, MBartForConditionalGeneration, IntervalStrategy
from transformers import Trainer, TrainingArguments


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--model_name', default='mbart-', type=str, help="model name")
argparser.add_argument('--data_path', default='', type=str, help="path to training data")
argparser.add_argument('--save_path', default='trained_models', type=str, help="save directory")
argparser.add_argument('--top_terms_strategy', default='tfidf', type=str, help="tfidf or sent")
args = argparser.parse_args()

print("\n"+"-"*10, "Finetuning mBART for label generation", "-"*10)
print("model_name:", args.model_name)
print("save_path:", args.save_path)
print("data_path:", args.data_path)
print("top_terms_strategy:", args.top_terms_strategy)
print("-"*50 + "\n")


DATA_DIR = args.data_path

model_suffix = args.model_suffix

MODEL_OUT_DIR = os.path.join(args.save_path,
                             args.model_name + "_" +
                             args.top_terms_strategy + "_" +
                             args.label_strategy +
                             model_suffix + ".pt")
DEBUG = True
DISABLE_TQDM = False

EPOCHS = 5
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_EVAL = 32
WARMUP = 50
WEIGHT_DECAY = 0.01
STEPS_SAVE = 6000
STEPS_EVAL = 2000
STEPS_LOGGING = 100

########################################################################################################################
# check cuda
########################################################################################################################
cuda_available = torch.cuda.is_available()

########################################################################################################################
# load model and tokenizer
########################################################################################################################
print("Loading model and tokenizer")

tokenizer = MBartTokenizer.from_pretrained(
    "facebook/mbart-large-cc25", src_lang="fi_FI", tgt_lang="fi_FI")

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
model.resize_token_embeddings(len(tokenizer))
model.config.decoder_start_token_id = tokenizer.lang_code_to_id["fi_FI"]


########################################################################################################################
# Load data from csv file and prepare them
########################################################################################################################
print("Loading data")

file_suffix = args.model_suffix

train_filename = "train_" + args.top_terms_strategy + "_" + args.label_strategy + "_" + file_suffix + ".csv"
valid_filename = "valid_" + args.top_terms_strategy + "_" + args.label_strategy + "_" + file_suffix + ".csv"

print("Train data:", train_filename)
print("Valid data:", valid_filename)

dataset = load_dataset(
    "csv",
    data_files={
        "train": os.path.join(DATA_DIR, train_filename),
        "validation": os.path.join(DATA_DIR, valid_filename),
    },
    delimiter=",",
)


def convert_to_features(example_batch):
    inp_enc = tokenizer.batch_encode_plus(
        example_batch["source"],
        padding="max_length",
        max_length=32,
        truncation=True
    )
    trg_enc = tokenizer.batch_encode_plus(
        example_batch["target"],
        padding="max_length",
        max_length=32,
        truncation=True,
    )
    # dec_inp_ids = shift_tokens_right(lbs, model.config.pad_token_id)
    lbs = np.array(trg_enc["input_ids"])
    lbs[lbs[:, :] == model.config.pad_token_id] = -100
    encodings = {
        "input_ids": inp_enc["input_ids"],
        "attention_mask": inp_enc["attention_mask"],
        # 'decoder_input_ids': decoder_input_ids,
        "labels": lbs,
    }
    return encodings

print("Converting text to features")
dataset["train"] = dataset["train"].map(convert_to_features, batched=True)
dataset["validation"] = dataset["validation"].map(convert_to_features, batched=True)
columns = [
    "input_ids",
    "labels",
    #  'decoder_input_ids',
    "attention_mask",
]
dataset.set_format(type="torch", columns=columns)

########################################################################################################################
# Training
########################################################################################################################
print("Start training")
training_args = TrainingArguments(
    output_dir=MODEL_OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE_TRAIN,
    per_device_eval_batch_size=BATCH_SIZE_EVAL,
    warmup_steps=WARMUP,
    weight_decay=WEIGHT_DECAY,
    logging_dir=os.path.join(MODEL_OUT_DIR, "logs"),
    save_steps=STEPS_SAVE,
    evaluation_strategy=IntervalStrategy.STEPS,
    eval_steps=STEPS_EVAL,
    logging_strategy=IntervalStrategy.STEPS,
    logging_steps=STEPS_LOGGING,
    disable_tqdm=DISABLE_TQDM,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

# train model
trainer.train()

# save model
print("Saving model")
trainer.save_model()

print("Done finetuning mBART! Finetuned model saved at", MODEL_OUT_DIR)
