import requests
import torch
import transformers
from PIL import Image

from transformers.generation.streamers import TextStreamer
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.mm_utils import tokenizer_image_token, process_images



if __name__ == "__main__":
    model_path = 'toshi456/chat-vector-llava-v1.5-7b-ja'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device=="cuda" else torch.float32

    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path, 
        device_map=device,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=1024,
        padding_side="right",
        use_fast=False,
    )
    model.get_model().vision_tower.load_model()
    model = model.to(device)

    eos_token_id_list = [
        tokenizer.eos_token_id,
        tokenizer.bos_token_id,
    ]

    # image pre-process
    image_url = "https://huggingface.co/rinna/bilingual-gpt-neox-4b-minigpt4/resolve/main/sample.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')

    if not isinstance(image, list):
        image = [image]
    
    image_tensor = process_images(image, model.get_model().vision_tower.image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    # create prompt
    # ユーザー: <image>\n{prompt}
        conv_mode = "llava_llama_2"
    conv = conv_templates[conv_mode].copy()
    prompt = "猫の隣には何がありますか？"
    inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors='pt'
    ).unsqueeze(0)
    if device == "cuda":
        input_ids = input_ids.to(device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    streamer = TextStreamer(tokenizer, skip_prompt=True, timeout=20.0)

    # parameter
    temperature = 0.0
    top_p = 1.0
    max_new_tokens=256

    # predict
    with torch.inference_mode():
        model.generate(
            inputs=input_ids,
            images=image_tensor,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            eos_token_id=eos_token_id_list,
        )

    """猫の隣には、コンピューター（パソコン）があります。<s>"""
