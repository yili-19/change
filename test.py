from openai import OpenAI
import base64
# openai_api_key = "EMPTY"
# openai_api_base = "http://localhost:8000/v1"

model_selected = "gpt-4o-mini"
# model_selected = "Qwen/Qwen2.5-7B-Instruct"

client = OpenAI(
    api_key="sk-E9HHdz2kqB3naVskSAG9Xwqj02JW3tET8hNx9tUK2ghDsi0H",
    base_url="https://xiaoai.plus/v1",
)
system_prompt = '''
    You are an expert in the field of image change caption. 
    You will be provided with two images in sequence: the first image is the original image, and the second image is the changed image. 
    Your goal is to observe whether the objects in the image have changed and describe the changes that have occurred.
    Each pair of images can have one change at most.
    Note that the object within the image may not have changed and to exclude the interference of perspective changes.
    Here are a few examples.
    If no change has occurred, your answer should be like this: "there is no change", "nothing has changed", "the two scenes seem identical", "the scene is the same as before", "the scene remains the same", "nothing was modified", "the scene remains the same", "the scene is the same as before", "the two scenes seem identical", "no change was made".
    If the image has changed, your answer should be like this:: "the large brown matte block that is on the right side of the large brown cylinder is gone", "the large brown rubber cube that is in front of the large blue matte cylinder is missing", "the tiny yellow cylinder turned brown", "the small matte object became brown", "the tiny yellow rubber cylinder that is behind the rubber block became brown", "the matte cylinder became brown", "the small yellow matte cylinder on the left side of the brown cube changed to brown", "the small yellow cylinder changed to brown", "the small yellow rubber cylinder left of the brown cube turned brown", "the tiny yellow rubber cylinder behind the small metallic sphere turned brown", "the block changed its location", "the red metal object changed its location", "the red block is in a different location", "the red metallic cube changed its location", "the small red metallic block that is to the right of the small matte ball changed its location", "the red cube is in a different location", "the small red shiny cube right of the tiny brown object changed its location", "the small red metallic block that is to the right of the tiny brown metal sphere is in a different location", "the small red metallic block to the right of the small green object is in a different location","the tiny matte ball has been added", "the tiny gray thing has been newly placed", "the tiny gray object has been newly placed", "the small gray thing has been added", "the tiny gray rubber ball that is right of the large matte ball has appeared". 
'''
from tqdm import tqdm


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(bef_image_path, aft_image_path):
    bef_base64_image = encode_image(bef_image_path)
    aft_base64_image = encode_image(aft_image_path)
    response = client.chat.completions.create(
    model=model_selected,
    messages=[
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": system_prompt,
            },
            {
            "type": "image_url",
            "image_url": {
                "url":  f"data:image/jpeg;base64,{bef_base64_image}"
            },
            },
                        {
            "type": "image_url",
            "image_url": {
                "url":  f"data:image/jpeg;base64,{aft_base64_image}"
            },
            },
        ],
        }
    ],
    )
    return response.choices[0].message.content

base_dir = ""
import os
import numpy as np
import json
def main():
    d_imgs = sorted(os.listdir(base_dir + 'images'))
    s_imgs = sorted(os.listdir(base_dir + 'sc_images'))
    n_imgs = sorted(os.listdir(base_dir + 'nsc_images'))

    result_sents_pos = json.load(open('sc_results.json', 'r'))
    result_sents_neg = json.load(open('nsc_results.json', 'r'))
    result_sents_pos = []
    result_sents_neg = []
    done = [sents['image_id'] for sents in result_sents_pos]
    for idx in tqdm(range(100)):
        image_id = d_imgs[idx].split('_')[-1]
        if(image_id in done): continue
        d_img = os.path.join(base_dir + 'images', d_imgs[idx])
        s_img = os.path.join(base_dir + 'sc_images', s_imgs[idx])
        n_img = os.path.join(base_dir + 'nsc_images', n_imgs[idx])

        sent_pos = analyze_image(d_img, s_img)
        sent_neg = analyze_image(d_img, n_img)

        result_sents_pos.append({
           "caption": sent_pos, 
           "image_id": image_id
           })
        
        result_sents_neg.append({
           "caption": sent_neg, 
           "image_id": image_id + '_n'
           })
        if(idx % 5 == 0):
            json.dump(result_sents_pos, open('sc_results.json', 'w'))
            json.dump(result_sents_neg, open('nsc_results.json', 'w'))
    
    json.dump(result_sents_pos, open('sc_results.json', 'w'))
    json.dump(result_sents_neg, open('nsc_results.json', 'w'))




if __name__ == "__main__":
    main() 