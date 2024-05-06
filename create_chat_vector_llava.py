import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from llava.model.builder import load_pretrained_model


if __name__ == "__main__":
    vlm_model_name = "liuhaotian/llava-v1.5-7b"
    vlm_tokenizer, vlm_model, image_processor, context_len = load_pretrained_model(
        model_path=vlm_model_name,
        model_base=None,
        model_name="llava-v1.5-7b",
        load_bf16=True,
        device_map="cpu",
        device="cpu"
    )

    ja_model_name = "elyza/ELYZA-japanese-Llama-2-7b"
    ja_tokenizer = AutoTokenizer.from_pretrained(ja_model_name)
    ja_model = AutoModelForCausalLM.from_pretrained(ja_model_name, torch_dtype=torch.bfloat16, device_map="cpu")

    eng_model_name = "meta-llama/Llama-2-7b-hf"
    eng_tokenizer = AutoTokenizer.from_pretrained(eng_model_name)
    eng_model = AutoModelForCausalLM.from_pretrained(eng_model_name, torch_dtype=torch.bfloat16, device_map="cpu")

    # ボキャブラリ数を確認
    print(f"llava-v1.5-7b: {vlm_tokenizer.vocab_size}")  # -> 32000
    print(f"Llama-2-7b-hf: {eng_tokenizer.vocab_size}")  # -> 32000
    print(f"ELYZA-japanese-Llama-2-7b: {ja_tokenizer.vocab_size}")  # -> 32000

    # tokenizerの一致確認
    text = "こんにちは世界！🤓でござるよドゥフフw"
    print(vlm_tokenizer(text))
    print(eng_tokenizer(text))
    print(ja_tokenizer(text))

    # 除外対象
    if ja_tokenizer.vocab_size == eng_tokenizer.vocab_size:
        # 語彙拡張なし
        skip_layers = []
    else:
        # 語彙拡張あり
        skip_layers = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    
    for k, v in ja_model.state_dict().items():
        # layernormも除外
        if (k in skip_layers) or ("layernorm" in k):
            continue
        chat_vector = vlm_model.state_dict()[k] - eng_model.state_dict()[k]
        new_v = v + chat_vector.to(v.device)
        vlm_model.state_dict()[k].copy_(new_v)

    new_model_name = "chat-vector-llava-v1.5-7b-ja"
    vlm_model.save_pretrained(new_model_name)
    vlm_tokenizer.save_pretrained(new_model_name)