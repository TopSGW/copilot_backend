import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

class DeepSeekVLV2Pipeline:
    def __init__(self, model_path="deepseek-ai/deepseek-vl2-small", device="cuda"):
        """
        Initialize the DeepSeek-VL2 pipeline by loading the processor, tokenizer, and model.
        """
        self.device = device
        # Load processor and tokenizer
        self.processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        
        # Load the model and set to evaluation mode
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.model = self.model.to(torch.bfloat16).to(device).eval()
    
    def load_images(self, conversation):
        """
        Load PIL images from the conversation image paths.
        """
        return load_pil_images(conversation)
    
    def prepare_inputs(self, conversation, pil_images, system_prompt=""):
        """
        Prepare the inputs for the model given the conversation and images.
        """
        prepared = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=system_prompt
        ).to(self.device)
        return prepared
    
    def generate_response(self, prepare_inputs):
        """
        Generate a response using incremental prefilling and text generation.
        """
        with torch.no_grad():
            # Obtain inputs embeddings from the model's image encoder.
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
            
            # Use incremental prefilling for efficient memory usage.
            inputs_embeds, past_key_values = self.model.incremental_prefilling(
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
                chunk_size=512  # adjust chunk size if needed
            )
            
            # Generate the response tokens.
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
                past_key_values=past_key_values,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
            )
            
            # Decode the generated tokens, removing the prompt tokens.
            answer = self.tokenizer.decode(
                outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(),
                skip_special_tokens=False
            )
        return answer

DeepSeekpipeline = DeepSeekVLV2Pipeline(model_path="deepseek-ai/deepseek-vl2-small", device="cuda")

# def main():
#     # Define a conversation with interleaved image and text content.
#     conversation = [
#         {
#             "role": "<|User|>",
#             "content": (
#                 "This is image_1: <image>\n"
#                 "This is image_2: <image>\n"
#                 "This is image_3: <image>\n Can you tell me what are in the images?"
#             ),
#             "images": [
#                 "images/multi_image_1.jpeg",
#                 "images/multi_image_2.jpeg",
#                 "images/multi_image_3.jpeg",
#             ],
#         },
#         {"role": "<|Assistant|>", "content": ""}
#     ]
    
#     # Initialize the pipeline
#     pipeline = DeepSeekVLV2Pipeline(model_path="deepseek-ai/deepseek-vl2-small", device="cuda")
    
#     # Load images from conversation
#     pil_images = pipeline.load_images(conversation)
    
#     # Prepare the inputs
#     prepared_inputs = pipeline.prepare_inputs(conversation, pil_images, system_prompt="")
    
#     # Generate the response
#     answer = pipeline.generate_response(prepared_inputs)
    
#     # Display the formatted result using the processor's formatting.
#     print(f"{prepared_inputs['sft_format'][0]}\n{answer}")

# if __name__ == "__main__":
#     main()
