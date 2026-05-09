
from datasets.utils import logging
from src.models.GPTModel import GPTModel
from src.entity.config_entity import GPTConfig,ModelTrainingConfig
from src.constants import TOKENIZER_NAME,DEVICE,MAX_NEW_TOKEN,TOP_K,TEMPERATURE,ALLOWED_SPECIAL_TOKEN
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import tiktoken
import os

from src.data_access.download_gpt_weights import GPT_Weight_Downloader
from src.constants import EOT
from tqdm import tqdm

import logging



class GPTDatasetV1(Dataset):
            def __init__(self, txt, tokenizer, max_length, stride):
                self.input_ids = []
                self.target_ids = []

                # Tokenize the entire text
                token_ids = tokenizer.encode(txt, allowed_special={ALLOWED_SPECIAL_TOKEN})

                # Use a sliding window to chunk the book into overlapping sequences of max_length
                for i in range(0, len(token_ids) - max_length, stride):
                    input_chunk = token_ids[i:i + max_length]
                    target_chunk = token_ids[i + 1: i + max_length + 1]
                    self.input_ids.append(torch.tensor(input_chunk))
                    self.target_ids.append(torch.tensor(target_chunk))

            def __len__(self):
                return len(self.input_ids)

            def __getitem__(self, idx):
                return self.input_ids[idx], self.target_ids[idx]

class MyModel():
    def __init__(self,load_pretrained_weights=False,model_size:str="124M",models_dir="gpt"):
        import dataclasses
        self.config=dataclasses.asdict(GPTConfig())
        self.model=GPTModel(self.config)
        self.tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
        if load_pretrained_weights:
            logging.info(f"loading pretreined weights of model_size: {model_size} and models_dir: {models_dir}")
            gpt_weight_downloader=GPT_Weight_Downloader()

            settings,params=gpt_weight_downloader.download_and_load_gpt2(model_size=model_size,models_dir=models_dir)
            self.config.update(settings)
            # Map 'n_ctx' for context length to our config key 'context_length'
            if "n_ctx" in settings:
                self.config["context_length"] = settings["n_ctx"]
            # GPT-2 pretrained weights include Q/K/V biases; enable them before rebuilding
            # self.config["qkv_bias"] = True
            self.model=GPTModel(self.config)
            logging.info("Updated settings gpt config to ",settings)
            gpt_weight_downloader.load_weights_into_gpt(gpt=self.model,params=params)
            
    
    @staticmethod
    def calc_loss_batch(input_batch, target_batch, model, device):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss

    @staticmethod
    def calc_loss_loader(data_loader, model, device, num_batches=None):
        total_loss = 0.
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            # Reduce the number of batches to match the total number of batches in the data loader
            # if num_batches exceeds the number of batches in the data loader
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = MyModel.calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches
    
    @staticmethod
    def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

        # Initialize the tokenizer
        tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)

        # Create dataset
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

        return dataloader

    def load_model(self,model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    

    @staticmethod
    def evaluate_model(model, train_loader, val_loader, device, eval_iter):
        model.eval()
        with torch.no_grad():
            train_loss = MyModel.calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
            val_loss = MyModel.calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()
        return train_loss, val_loss
    
    @staticmethod
    def text_to_token_ids(tokenizer, text):
        return torch.tensor(tokenizer.encode(text, allowed_special={ALLOWED_SPECIAL_TOKEN}), dtype=torch.long).unsqueeze(0)
    @staticmethod
    def token_ids_to_text(token_ids, tokenizer):
        return tokenizer.decode(token_ids.squeeze(0).cpu().numpy().tolist())
    @staticmethod
    def generate_text_simple(model, idx, max_new_tokens, context_size):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token_id), dim=1)
        return idx
    @staticmethod
    def generate_and_print_sample(model, tokenizer, device, start_context):
        model.eval()
        context_size = model.pos_emb.weight.shape[0]
        encoded = MyModel.text_to_token_ids(tokenizer, start_context).to(device)
        with torch.no_grad():
            token_ids = MyModel.generate_text_simple(
                model=model, idx=encoded,
                max_new_tokens=50, context_size=context_size
            )
        decoded_text = MyModel.token_ids_to_text(token_ids, tokenizer)
        logging.info(decoded_text.replace("\n", " "))  # Compact print format
        model.train()
    @staticmethod
    def train_model(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer,model_training_config):
        
        train_losses, val_losses,track_tokens_seen= [], [],[]

        token_seen,global_step = 0,-1

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            model.train()
            pbar=tqdm(train_loader,desc=f"Training Epoch {epoch+1}",leave=True,dynamic_ncols=True,total=len(train_loader))
            for step,(input_batch, target_batch) in enumerate(pbar):
                optimizer.zero_grad()
                loss = MyModel.calc_loss_batch(input_batch, target_batch, model, device)
                
                loss.backward()
                optimizer.step()
                token_seen+=input_batch.numel()
                global_step += 1
                pbar.set_postfix({
                    "step": step+1,
                    "loss": f"{loss.item():.3f}"
                })

                if global_step % eval_freq == 0:
                    train_loss,val_loss=MyModel.evaluate_model(
                        model, train_loader, val_loader, device, eval_iter
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(token_seen)
                    # logging.info(f"Ep {epoch+1} (Step {global_step:06d}): "
                    #   f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                    dir_path:str=os.path.join(os.path.dirname(model_training_config.trained_model_file_path),str(epoch))
                    os.makedirs(dir_path, exist_ok=True)
                    model_file_path:str=os.path.join(dir_path,"model.pth")
                    plot_folder_path:str=os.path.join(dir_path,"plots")

                    epochs_tensor = torch.linspace(0, model_training_config.num_epochs, len(train_losses))
                    MyModel.plot_losses(epochs_tensor, track_tokens_seen, train_losses, val_losses, dir_to_save_plots=plot_folder_path)
                    torch.save(model.state_dict(), model_file_path)

                    
                    logging.info(f"Model saved to the dir {dir_path}")
                    logging.info(
                f"Ep {epoch+1} Step {global_step}: Train {train_loss:.3f}, Val {val_loss:.3f}"
            )
            # Print a sample text after each epoch
            MyModel.generate_and_print_sample(
                model, tokenizer, device, start_context
            )
        return train_losses, val_losses, track_tokens_seen
    


    def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, dir_to_save_plots):
        fig, ax1 = plt.subplots(figsize=(5, 3))

        # Plot training and validation loss against epochs
        ax1.plot(epochs_seen, train_losses, label="Training loss")
        ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

        # Create a second x-axis for tokens seen
        ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
        ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
        ax2.set_xlabel("Tokens seen")

        fig.tight_layout()  # Adjust layout to make room

        os.makedirs(dir_to_save_plots, exist_ok=True)
        plt.savefig(os.path.join(dir_to_save_plots, "loss.pdf"))
        # plt.show()



    def fit(self,X,model_training_config:ModelTrainingConfig):  
        self.model.to(DEVICE)

        train_losses, val_losses, tokens_seen = MyModel.train_model(
            model=self.model,
            train_loader=X["train"],
            val_loader=X["val"],
            optimizer=torch.optim.AdamW(self.model.parameters(), lr=model_training_config.learning_rate, weight_decay=model_training_config.weight_decay),
            device=DEVICE,
            num_epochs=model_training_config.num_epochs,
            eval_freq=model_training_config.eval_freq,
            eval_iter=model_training_config.eval_iter,
            start_context=model_training_config.start_context,
            tokenizer=self.tokenizer,
            model_training_config=model_training_config
        )

                # Note:  
        epochs_tensor = torch.linspace(0, model_training_config.num_epochs, len(train_losses))
        MyModel.plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, dir_to_save_plots=model_training_config.dir_to_save_plots)
        os.makedirs(os.path.dirname(model_training_config.trained_model_file_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_training_config.trained_model_file_path)



    @staticmethod
    def generate_text_advanced(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eot_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)

            if temperature>0:
                
                next_token_logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    top_k_values, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < top_k_values[:, -1, None]] = -float('Inf')

                next_token_id = torch.multinomial(torch.nn.functional.softmax(next_token_logits, dim=-1), num_samples=1)
                if eot_id is not None and next_token_id.item() == eot_id:
                    logging.debug(f"Model predicted {EOT}")
                    return idx
                idx = torch.cat((idx, next_token_id), dim=1)
            else:
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                if eot_id is not None and next_token_id.item() == eot_id:
                    logging.debug(f"Model predicted {EOT}")
                    return idx
                idx = torch.cat((idx, next_token_id), dim=1)    

        return idx
    

    def predict(self,text:str,max_new_tokens:int=MAX_NEW_TOKEN,temperature:float=TEMPERATURE,top_k:int=TOP_K):
        self.model.to(DEVICE)
        self.model.eval()
        eot_id = self.tokenizer.encode(EOT, allowed_special={EOT})[0]
        token_ids = MyModel.generate_text_advanced(
        model=self.model,
        idx=MyModel.text_to_token_ids(self.tokenizer, text=text).to(DEVICE),
        max_new_tokens=max_new_tokens,
        context_size=self.config["context_length"],
        temperature=temperature,
        top_k=top_k,
        eot_id=eot_id
        )
        return MyModel.token_ids_to_text(token_ids, self.tokenizer)

    @staticmethod
    def generate_text_advanced_yield(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eot_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)

            if temperature>0:
                
                next_token_logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    top_k_values, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < top_k_values[:, -1, None]] = -float('Inf')

                next_token_id = torch.multinomial(torch.nn.functional.softmax(next_token_logits, dim=-1), num_samples=1)
                if eot_id is not None and next_token_id.item() == eot_id:
                    logging.debug(f"Model predicted {EOT}")
                    break
                idx = torch.cat((idx, next_token_id), dim=1)
                yield next_token_id
            else:
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                if eot_id is not None and next_token_id.item() == eot_id:
                    logging.debug(f"Model predicted {EOT}")
                    break
                idx = torch.cat((idx, next_token_id), dim=1)    
                yield next_token_id

    def predict_yeild(self,text:str,max_new_tokens:int=MAX_NEW_TOKEN,temperature:float=TEMPERATURE,top_k:int=TOP_K):
        self.model.to(DEVICE)
        self.model.eval()
        eot_id = self.tokenizer.encode(EOT, allowed_special={EOT})[0]
        idx = MyModel.text_to_token_ids(self.tokenizer, text=text).to(DEVICE)
        
        buffer = ""
        stop_words = [
            "User:", "user:", "User :", "user :", "\nUser", "\nuser",
            "Assistant:", "assistant:", "assistent:", "\nAssistant", "\nassistant", "\nassistent"
        ]
        
        for token_id in MyModel.generate_text_advanced_yield(
            model=self.model,
            idx=idx,
            max_new_tokens=max_new_tokens,
            context_size=self.config["context_length"],
            temperature=temperature,
            top_k=top_k,
            eot_id=eot_id
        ):
            decoded_token = MyModel.token_ids_to_text(token_id, self.tokenizer)
            buffer += decoded_token
            
            # Check if any complete stop word is found
            should_return = False
            for stop_word in stop_words:
                if stop_word in buffer:
                    yield buffer.split(stop_word)[0]
                    return
            
            # Check if the end of the buffer partially matches a stop word
            is_partial = False
            for stop_word in stop_words:
                for i in range(1, len(stop_word)):
                    if buffer.endswith(stop_word[:i]):
                        is_partial = True
                        break
                if is_partial:
                    break
            
            if not is_partial:
                yield buffer
                buffer = ""
                
        if buffer:
            yield buffer