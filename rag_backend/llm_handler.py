from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

class LLMHandler:
    """
    Wrapper for language models (Salamandra or similar chat-instruct models).
    Handles initialization, prompt formatting, and response generation.
    """
    def __init__(self, model_name, cache_dir, quantization=False, system_prompt=None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.quantization = quantization
        # Load prompts from config
        self.default_system_prompt = system_prompt
        self.default_system_prompt_pre = self._default_prompt_pre()
        self.default_system_prompt_post = self._default_prompt_post()
        self.model, self.tokenizer = self._load_model_and_tokenizer()
    # ----------------------------
    # Initialization
    # ----------------------------
    def _load_model_and_tokenizer(self):
        """Load the language model and tokenizer."""
        model_name = self.model_name
        cache_dir = self.cache_dir
        use_quantization = self.quantization

        print(f"[LLMHandler] Loading model: {model_name}")
        print(f"[LLMHandler] Quantization: {use_quantization}")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.bfloat16,
            cache_dir=cache_dir,
            quantization_config=quantization_config if use_quantization else None
        )
        
        print(f"[LLMHandler] Model loaded successfully on device: {model.device}")
        return model, tokenizer

    # ----------------------------
    # System prompts
    # ----------------------------
    def _format_prompt(self, prompt_template):
        """Format prompt template with current date."""
        date_string = datetime.today().strftime('%Y-%m-%d')
        return prompt_template.format(date=date_string)

    def _default_prompt_pre(self):
        """Prefix for context in RAG mode."""
        return (
            "These are the context to respond the question:\n"
            "-------------------------\n"
        )

    def _default_prompt_post(self):
        """Suffix for context in RAG mode."""
        return (
            "\n-------------------------\n\n"
            "The user will now ask a question related to the context. \n"
            "Answer based only on the provided context. \n"
        )

    # ----------------------------
    # Prompt preparation
    # ----------------------------
    def _prepare_prompt(self, messages):
        """Apply chat template to role-based messages."""
        date_string = datetime.today().strftime('%Y-%m-%d')
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            date_string=date_string
        )

    def _tokenize_prompt(self, prompt):
        """Convert text prompt into token IDs on the correct device."""
        return self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

    # ----------------------------
    # Response generation
    # ----------------------------
    def generate(self, messages, max_new_tokens=200, temperature=0.3, top_p=0.95):
        """
        Generate a response for a chat conversation.

        Args:
            messages (list): [{"role": "user"/"assistant"/"system", "content": "..."}]
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling parameter
            
        Returns:
            str: Generated response
        """
        # Prepare inputs
        prompt = self._prepare_prompt(messages)
        #print("[LLMHandler] Prepared prompt:")
        #print(prompt)
        #print("-" * 60)
        
        inputs = self._tokenize_prompt(prompt)

        # Generate
        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3
        )

        # Decode and clean
        generated_tokens = outputs[0][len(inputs[0]):]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        #print("[LLMHandler] Generated response:")
        #print(response)
        #print("-" * 60)

        # Stop sequence cleanup
        stop_sequence = "Non o sei."
        if stop_sequence in response:
            response = response.split(stop_sequence)[0] + stop_sequence
            
        return response.strip()
    
    def get_model_info(self):
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "quantization": self.generator_config.quantization,
            "device": str(self.model.device),
            "status": "loaded"
        }