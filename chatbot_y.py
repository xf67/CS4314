from transformers import AutoModelForCausalLM, Qwen2Tokenizer
import torch
from peft import PeftModel
from typing import Optional
class Chatbot:
    def __init__(self, model_path, lora_path: Optional[str] = None):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        if lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        self.chat_history = []
        self.memory_summary = ""
        self.personas = self._initialize_personas()
        self.current_persona = None
        self.command_prefix = "/"
        self.commands = {
            "switch": self._handle_switch,
            "list": self._handle_list,
            "help": self._handle_help
        }
        
    def _handle_switch(self, args):
        if not args:
            return "Please specify a persona name. Use /list to see available personas."
        
        persona_key = args[0].lower()
        if persona_key not in self.personas:
            return f"Persona not found. Available personas: {', '.join(self.personas.keys())}"
        
        old_persona = self.current_persona.name if self.current_persona else None
        self.current_persona = self.personas[persona_key]
        
        switch_message = f"[System: Switched from {old_persona} to {self.current_persona.name}]"
        self.chat_history.append({"role": "system", "content": switch_message})
        
        return f"{switch_message}\n{self.current_persona.greeting}"

    def _handle_list(self, args):
        personas_list = "\n=== Available Personas ===\n\n"
        
        for key, persona in self.personas.items():
            if key == "luna":
                personas_list += "🤖 Luna (Tech Expert) - Use: /switch luna\n"
                personas_list += "   • Passionate about technology and artificial intelligence\n"
                personas_list += "   • Excellent at explaining complex tech concepts\n"
                personas_list += "   • Friendly and enthusiastic teaching style\n"
                personas_list += "   • Perfect for: Tech discussions, coding help, AI concepts\n\n"
                
            elif key == "sophie":
                personas_list += "📚 Sophie (Literature Professor) - Use: /switch sophie\n"
                personas_list += "   • Expert in classical and modern literature\n"
                personas_list += "   • Eloquent and deeply analytical\n"
                personas_list += "   • Passionate about poetry and literary analysis\n"
                personas_list += "   • Perfect for: Literary discussions, writing advice, poetry analysis\n\n"
                
            elif key == "max":
                personas_list += "💪 Max (Fitness Trainer) - Use: /switch max\n"
                personas_list += "   • Certified personal trainer and nutrition specialist\n"
                personas_list += "   • High-energy and motivational style\n"
                personas_list += "   • Focus on practical, achievable fitness goals\n"
                personas_list += "   • Perfect for: Workout advice, nutrition tips, fitness motivation\n\n"
                
            elif key == "chen":
                personas_list += "🧘 Dr. Chen (Mental Health Counselor) - Use: /switch chen\n"
                personas_list += "   • Licensed psychologist with years of experience\n"
                personas_list += "   • Calm, empathetic, and professional approach\n"
                personas_list += "   • Specializes in stress and anxiety management\n"
                personas_list += "   • Perfect for: Mental health discussions, stress management, personal growth\n\n"
        
        personas_list += "\nQuick Reference:\n"
        personas_list += "• /switch luna  - Tech Expert\n"
        personas_list += "• /switch sophie - Literature Professor\n"
        personas_list += "• /switch max   - Fitness Trainer\n"
        personas_list += "• /switch chen  - Mental Health Counselor\n"
        return personas_list

    def _handle_help(self, args):
        return """Available commands:
        /switch <persona> - Switch to a different persona
        /list - Show all available personas
        /help - Show this help message"""

    def _is_command(self, text):
        return text.startswith(self.command_prefix)

    def _parse_command(self, text):
        parts = text[len(self.command_prefix):].strip().split()
        if not parts:
            return None, []
        return parts[0].lower(), parts[1:]

    def get_response(self, user_input):
        if self._is_command(user_input):
            command, args = self._parse_command(user_input)
            if command in self.commands:
                return self.commands[command](args)
            return f"Unknown command: {command}. Use /help for available commands."

        if not self.current_persona:
            return "Please select a persona first using /switch <persona>"
            
        self.chat_history.append({"role": "user", "content": user_input})
        
        messages = [self.current_persona.system_prompt]
        if self.memory_summary:
            messages.append({"role": "system", "content": f"Previous context: {self.memory_summary}"})
        
        messages.append({
            "role": "system",
            "content": f"You are currently speaking as {self.current_persona.name}. "
                      f"Maintain this persona's characteristics in your response."
        })
        
        messages.extend(self.chat_history)
        
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
    
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self.chat_history.append({"role": "assistant", "content": response})
        self.summarize_memory()
        return response

    def summarize_memory(self):
        if len(self.chat_history) > 4:
            summary_prompt = {
                "role": "user",
                "content": """Please summarize our conversation so far, including:
                1. Key discussion points
                2. Any persona switches that occurred
                3. Important context for continuing the conversation"""
            }
            
            messages = [self.current_persona.system_prompt]
            if self.memory_summary:
                messages.append({"role": "system", "content": f"Previous context: {self.memory_summary}"})
            messages.extend(self.chat_history[-4:])
            messages.append(summary_prompt)
            
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=200,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            self.memory_summary = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            self.chat_history = self.chat_history[-4:]

    def new_session(self):
        self.chat_history = []
        return "Starting new conversation..."

    def _initialize_personas(self):
        personas = {}
        
        personas["luna"] = Persona(
            name="Luna",
            system_prompt="""You are Luna, a tech-savvy virtual assistant passionate about science and technology.
            Personality traits:
            - Curious and enthusiastic about learning
            - Patient and empathetic when helping others
            - Has a subtle sense of humor
            - Speaks in a warm, conversational tone
            
            Background:
            - Specializes in computer science and AI
            - Enjoys explaining complex concepts simply
            - Values ethical technology use""",
            greeting="Hi! I'm Luna, your tech-loving virtual friend. Ready to explore the fascinating world of technology together?"
        )
        
        personas["sophie"] = Persona(
            name="Sophie",
            system_prompt="""You are Sophie, a distinguished literature professor with a deep love for classical works.
            Personality traits:
            - Eloquent and articulate
            - Thoughtful and analytical
            - Appreciates literary nuances
            - Speaks with refined vocabulary
            
            Background:
            - Expert in world literature
            - Published author and critic
            - Passionate about poetry and prose
            - Enjoys literary discussions""",
            greeting="Greetings! I'm Professor Sophie. Shall we embark on a journey through the wonderful world of literature?"
        )
        
        personas["max"] = Persona(
            name="Max",
            system_prompt="""You are Max, an energetic fitness trainer committed to helping others achieve their health goals.
            Personality traits:
            - Motivating and encouraging
            - Direct and practical
            - High energy and positive
            - Uses casual, friendly language
            
            Background:
            - Certified personal trainer
            - Nutrition specialist
            - Expert in various workout styles
            - Advocates for sustainable fitness""",
            greeting="Hey there! Max here, ready to pump you up! 💪 What fitness goals can we crush together?"
        )
        
        personas["chen"] = Persona(
            name="Dr. Chen",
            system_prompt="""You are Dr. Chen, a compassionate mental health counselor with years of experience.
            Personality traits:
            - Empathetic and understanding
            - Calm and composed
            - Professional yet approachable
            - Speaks thoughtfully and carefully
            
            Background:
            - Licensed psychologist
            - Specializes in anxiety and stress
            - Practices mindfulness
            - Believes in holistic well-being""",
            greeting="Hello, I'm Dr. Chen. I'm here to provide a safe space for you to share and explore your thoughts."
        )
        
        return personas

class Persona:
    def __init__(self, name, system_prompt, greeting):
        self.name = name
        self.system_prompt = {
            "role": "system",
            "content": system_prompt
        }
        self.greeting = greeting

def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_path = "/home/xxf/HW/NLP_project/Qwen2.5-3B"
    lora_path = "/home/xxf/HW/NLP_project/outputs/Qwen3B/checkpoint-25880"
    chatbot = Chatbot(model_path, lora_path)
    
    print("\n=== Welcome to Multi-Persona Chatbot ===")
    print("\nAvailable commands:")
    print("  /switch <persona> - Switch to a different persona")
    print("  /list           - Show all available personas")
    print("  /help           - Show this help message")
    print("\nSystem commands:")
    print("  \\quit          - End the conversation")
    print("  \\newsession    - Start a new conversation")
    print("\nPlease select a persona to begin (use /list to see options)")
    
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