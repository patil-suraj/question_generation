from pipelines import pipeline
from run_qg import run_qg

args_dict = {
    "model_name_or_path": "valhalla/t5-small-e2e-qg",
    "model_type": "t5",
    "tokenizer_name_or_path": "t5_qg_tokenizer",
    "output_dir": "t5-doctor",
    "train_file_path": "data\doctor\doctor_train_data_e2e_qg_t5.pt",
    "valid_file_path": "data\doctor\doctor_valid_data_e2e_qg_t5.pt",
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4,
    # "learning_rate": 0.0,
    "num_train_epochs": 900,
    "seed": 42,
    "do_train": True,
    "do_eval": True,
    "evaluate_during_training": True,
    "logging_steps": 100
}
# start training
run_qg(args_dict)

nlp = pipeline("e2e-qg", model="t5-doctor")
text = "I have diarrhea."
print(nlp(text))
print(nlp("diarrhea."))
print(nlp("diarrhea"))
print(nlp("diarrhea "))
print(nlp("diarrhea </s>"))
