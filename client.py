
import torch
import flwr as fl
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from model import load_model
from utils import load_data
from data_collator import DataCollatorSpeechSeq2SeqWithPadding

class ASRClient(fl.client.NumPyClient):
    def __init__(self):
        print("[CLIENT INIT] Starting client on CPU")
        self.device = torch.device("cpu")

        print("[CLIENT INIT] Loading model...")
        self.model = load_model().to(self.device)
        print("[CLIENT INIT] Model loaded")

        print("[CLIENT INIT] Loading data...")
        self.train_dataset, self.processor = load_data("train")
        self.test_dataset, _ = load_data("test")
        print("[CLIENT INIT] Data loaded")

    def get_parameters(self, config):
        print("[CLIENT] Sending model parameters to server")
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        print("[CLIENT] Receiving parameters from server")
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict({k: torch.tensor(v) for k, v in state_dict.items()})

    def fit(self, parameters, config):
        # Update model parameters
        self.set_parameters(parameters)
        print("[CLIENT] Starting local training with Trainer...")

        # Define compute_metrics within fit so self.processor is available
        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            decoded_preds = self.processor.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.processor.batch_decode(labels, skip_special_tokens=True)
            correct = sum(
                p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels)
            )
            return {"accuracy": correct / len(decoded_preds) if decoded_preds else 0.0}

        # Prepare training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir="./fl_output",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=2,
            logging_steps=50,
            evaluation_strategy="epoch",
            predict_with_generate=True,
        )

        # Instantiate the custom data collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )

        # Initialize the Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            data_collator=data_collator,
            tokenizer=self.processor.tokenizer,
            compute_metrics=compute_metrics,
        )

        # Train and evaluate
        trainer.train()
        metrics = trainer.evaluate()
        print(f"[CLIENT] Evaluation Metrics: {metrics}")

        # Return updated parameters, dataset size, and accuracy metric
        return (
            [val.cpu().numpy() for val in self.model.state_dict().values()],
            len(self.train_dataset),
            {"accuracy": metrics.get("accuracy", 0.0)},
        )

    def evaluate(self, parameters, config):
        # Not used: evaluation handled inside fit()
        self.set_parameters(parameters)
        print("[CLIENT] evaluate() called; using Trainer for evaluation in fit().")
        return None

if __name__ == "__main__":
    print("[MAIN] Launching client…")
    fl.client.start_numpy_client(server_address="localhost:8080", client=ASRClient())

