import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import pandas as pd
import json
from typing import Dict, Any, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.supported_models = {
            "llama2-7b": "NousResearch/Llama-2-7b-chat-hf",
            "llama2-13b": "NousResearch/Llama-2-13b-chat-hf",
        }
        self.default_training_params = {
            "num_train_epochs": 1,
            "per_device_train_batch_size": 4,
            "learning_rate": 2e-4,
            "weight_decay": 0.001,
            "max_steps": -1,
            "warmup_ratio": 0.03,
            "report_to": "none"  # Disable wandb tracking
        }

    def load_dataset(self, file_path: str) -> list:
        """Load dataset from various file formats"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext in ['.json', '.jsonl']:
                with open(file_path, 'r') as f:
                    data = [json.loads(line) for line in f]
            elif file_ext == '.csv':
                data = pd.read_csv(file_path).to_dict('records')
            elif file_ext == '.xlsx':
                data = pd.read_excel(file_path).to_dict('records')
            elif file_ext == '.txt':
                with open(file_path, 'r') as f:
                    data = [{"text": line.strip()} for line in f if line.strip()]
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            return data
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def prepare_model(self, model_name: str, use_4bit: bool = True) -> tuple:
        """Prepare the model and tokenizer for fine-tuning"""
        try:
            base_model = self.supported_models.get(model_name)
            if not base_model:
                raise ValueError(f"Unsupported model: {model_name}")

            compute_dtype = getattr(torch, "float16")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=use_4bit,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=False,
            ) if use_4bit else None

            # Load model with device placement
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=quant_config,
                device_map={"": 0} if device == "cuda" else None,
                trust_remote_code=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            return model, tokenizer
        except Exception as e:
            logger.error(f"Error preparing model: {str(e)}")
            raise

    def start_training(
        self,
        model_name: str,
        dataset_path: str,
        output_dir: str,
        training_params: Dict[str, Any] = None,
        lora_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Start the fine-tuning process"""
        try:
            # Load and prepare model
            model, tokenizer = self.prepare_model(model_name)
            
            # Load dataset
            dataset = self.load_dataset(dataset_path)
            
            # Prepare PEFT config
            peft_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=64,
                bias="none",
                task_type="CAUSAL_LM",
                **(lora_config or {})
            )

            # Prepare training arguments
            train_params = {**self.default_training_params, **(training_params or {})}

            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                peft_config=peft_config,
                dataset_text_field="text",
                max_seq_length=None,
                tokenizer=tokenizer,
                args=train_params,
                packing=False,
            )

            # Start training
            trainer.train()
            
            # Save the model
            trainer.model.save_pretrained(output_dir)
            trainer.tokenizer.save_pretrained(output_dir)

            return {
                "status": "success",
                "message": "Training completed successfully",
                "output_dir": output_dir
            }

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }