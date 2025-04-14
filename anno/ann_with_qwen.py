import os
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    pass

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import tqdm
import argparse

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in
               ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif', 'TIF', 'tiff'])

def generate_captain_for_one_image(model, processor, system_prompt, prompt, image_url):
    messages = [
        {"role": "system", "content": system_prompt}, # System message
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_url,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def generate_captain_for_images(model, processor, system_prompt, prompt, image_urls):
    captions = []
    for image_url in image_urls:
        caption = generate_captain_for_one_image(model, processor, system_prompt, prompt, image_url)
        captions.append(caption)
    return captions

def generate_captain_for_one_folder(model, processor, system_prompt, prompt, folder_path, captain_output_path):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if is_image_file(f)]
    os.makedirs(captain_output_path, exist_ok=True)
    for image_file in image_files:
        image_name = image_file.split('/')[-1]
        if os.path.exists(os.path.join(captain_output_path, image_name + '.txt')):
            print('Caption for image', image_name, 'already exists.')
            continue
        caption = generate_captain_for_one_image(model, processor, system_prompt, prompt, image_file)
        print("image_name: ", image_name)
        print("caption: ", caption)
        print("")
        with open(os.path.join(captain_output_path, image_name + '.txt'), 'w') as f:
            f.write(caption)
        print('Caption for image', image_name, 'is saved.')

def load_txt(txt_path):
    with open(txt_path, 'r') as f:
        prompt = f.read()
    return prompt

def load_prompt(prompt_type):
    if prompt_type == "ivf":
        prompt = load_txt('prompt_ivf.txt')
    elif prompt_type == "med":
        prompt = load_txt('prompt_med.txt')
    else:
        raise ValueError("Invalid task type. Please choose 'ivf' or 'med'.")
    return prompt

def generate_captain_for_one_pair(model, processor, system_prompt, prompt, A_image, B_image):
    messages = [
        {"role": "system", "content": system_prompt}, # System message
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": A_image,
                },
                {
                    "type": "image",
                    "image": B_image,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def generate_captain_for_datasets(model, processor, system_prompt, prompt, A_dir, B_dir, captain_output_path):
    A_image_files = [os.path.join(A_dir, f) for f in os.listdir(A_dir) if is_image_file(f)]
    B_image_files = [os.path.join(B_dir, f) for f in os.listdir(B_dir) if is_image_file(f)]
    # using name to match image pair, because the order of images may be different and the name of images are the same
    A_image_files.sort()
    B_image_files.sort()
    os.makedirs(captain_output_path, exist_ok=True)
    for A_image_file, B_image_file in zip(A_image_files, B_image_files):
        A_image_name = A_image_file.split('/')[-1]
        B_image_name = B_image_file.split('/')[-1]
        if os.path.exists(os.path.join(captain_output_path, A_image_name + '_' + B_image_name + '.txt')):
            print('Caption for images', A_image_name, B_image_name, 'already exists.')
            continue
        caption = generate_captain_for_one_pair(model, processor, system_prompt, prompt, A_image_file, B_image_file)
        print("A_image_name: ", A_image_name)
        print("B_image_name: ", B_image_name)
        print("caption: \n", caption)
        print("")
        with open(os.path.join(captain_output_path, A_image_name + '_' + B_image_name + '.txt'), 'w') as f:
            f.write(caption)
        print('Caption for images', A_image_name, B_image_name, 'is saved.')

def parse_args():
    parser = argparse.ArgumentParser(description='Generate captions for images in a folder.')
    parser.add_argument('--model_path', type=str, default="./Qwen2-VL/Qwen2-VL-7B-Instruct", help='Path to the model.')
    parser.add_argument('--folder_A_path', type=str, default="")
    parser.add_argument('--folder_B_path', type=str, default="")
    parser.add_argument('--captain_output_path', type=str, default="")
    parser.add_argument('--for_split', type=str, default="")
    parser.add_argument('--task_type', type=str, choices=["ivf", "med"], default="med", help='Task type, ivf or med.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path,)
    # for system prompt
    system_prompt = load_prompt(args.task_type)
    # prompt is just the user's question
    prompt_ivf = "Describe for the image pair, the first is visible, the second is infrared."
    prompt_med = "Describe for the image pair, the first is CT or PET or SPECT, the second is MRI."
    prompt_type = args.task_type
    if prompt_type == "ivf":
        prompt = prompt_ivf
    elif prompt_type == "med":
        prompt = prompt_med
    else:
        raise ValueError("Invalid task type. Please choose 'ivf' or 'med'.")
    if len(args.for_split) == 0:
        generate_captain_for_datasets(model, processor, system_prompt, prompt, args.folder_A_path, args.folder_B_path, args.captain_output_path)
        print('All captions are generated and saved.')
    else:
        splits_big = ['CT-MRI', 'PET-MRI', 'SPECT-MRI']
        splits_small = ['train', 'test']
        for s1 in splits_big:
            for s2 in splits_small:
                path_a = os.path.join(args.for_split, s1, s2, s1.split('-')[0])
                path_b = os.path.join(args.for_split, s1, s2, s1.split('-')[1])
                path_text = os.path.join(args.for_split, s1, s2, 'text')
                generate_captain_for_datasets(model, processor, system_prompt, prompt, path_a, path_b, path_text)
                print("captions for split", s1, s2, "are generated and saved.")
        print('All captions are generated and saved.')
