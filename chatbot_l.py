from transformers import AutoModelForCausalLM, Qwen2Tokenizer
import torch
from peft import PeftModel
from typing import Optional
class Chatbot:
    def __init__(self, model_path, lora_path: Optional[str] = None):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        if lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        self.chat_history = []
        self.system_prompt = {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}
        
    def get_response(self, user_input):
        self.chat_history.append({"role": "user", "content": user_input})
        messages = [self.system_prompt] + self.chat_history

        while True:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            tokens = self.tokenizer(text, return_tensors="pt")
            if len(tokens.input_ids[0]) > self.tokenizer.model_max_length and len(messages) > 2:
                messages.pop(1)
            else:
                break
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            repetition_penalty=1.2,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.pad_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self.chat_history.append({"role": "assistant", "content": response})
        return response
    
    def new_session(self):
        self.chat_history = []
        return "Starting new conversation..."

def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    lora_path = None
    model_path = "/home/xxf/HW/NLP_project/Qwen2.5-3B"
    # lora_path = "/home/xxf/HW/NLP_project/outputs/Qwen3B/checkpoint-25880"
    chatbot = Chatbot(model_path, lora_path)
    
    print("Chatbot is ready!")
    print("Enter \\quit to end the conversation")
    print("Enter \\newsession to start a new conversation")
    
    while True:
        user_input = input("\nUser: ").strip()
        
        if user_input.lower() == "\\quit":
            print("Goodbye!")
            break
        elif user_input.lower() == "\\newsession":
            print(chatbot.new_session())
            continue
        
        response = chatbot.get_response(user_input)
        print(f"\nAssistant: {response}")

if __name__ == "__main__":
    main()