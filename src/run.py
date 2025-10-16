from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from utils import load_and_preprocess, compute_metrics

def run(use_lora=False):
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized, data_collator = load_and_preprocess("data/sentimentdataset.csv", tokenizer)

    if use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        lora_config = LoraConfig(
            r=8, lora_alpha=32, target_modules=["q_lin","v_lin","k_lin"], lora_dropout=0.1, bias="none", task_type=TaskType.SEQ_CLS
        )
        model = get_peft_model(model, lora_config)
        # Freeze non-LoRA layers
        for name, param in model.named_parameters():
            if "lora" not in name and "classifier" not in name and "pre_classifier" not in name:
                param.requires_grad = False
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    training_args = TrainingArguments(
        output_dir="./out/lora" if use_lora else "./out/full",
        do_train=True,
        do_eval=True,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print(f"\n===== Training {'LoRA' if use_lora else 'Full Fine-Tuning'} =====")
    trainer.train()
    results = trainer.evaluate()
    print(f"\n===== Evaluation Results (LoRA={use_lora}): =====")
    print(results)
    return results

import pandas as pd

if __name__ == "__main__":
    # Run both training methods
    full_results = run(use_lora=False)
    lora_results = run(use_lora=True)

    # Create summary table
    results_table = pd.DataFrame([full_results, lora_results])

    print("\n===== Summary Table =====")
    print(results_table.to_string(index=False))

