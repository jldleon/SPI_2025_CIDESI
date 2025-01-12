import json
import PIL.Image as Image
import torch
import gradio as gr
import matplotlib.pyplot as plt
from argparse import Namespace
from time import sleep

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

from src import init_model, prompt_refinement, save_prompts
from src import load_model_SDXL, load_model_stable_diffusion
from src import CMM_MobileNet

TMP_DIR = "./gradio_tmp"
IMG_PATH = TMP_DIR + "/gen_img.png"
TEMP_JSON_KW = TMP_DIR + "/keywords.json"
TEMP_JSON_PRMT = TMP_DIR + "/prompts.json"

PROMPT_LMS = [ "gpt2", "gemma" ] #"gpt2" -  "google/gemma-7b"
IMGS_GENAI = [ "SDXL", "SD2" ] # "stabilityai/stable-diffusion-2-1" - "stabilityai/stable-diffusion-xl-base-1.0"
CKPT_NUM_SDXL = 4
CKPT_NUM_SD2 = 4

DEVICE_SELECTOR = True
SIMULATE_GEN = True
FAKE_OPTS = [ f"{k}_{i+2}" for k in ["castle", "cityscape", "desert", "forest",
                                     "garden", "jungle", "mermaid", "ocean", "robot",
                                     "spaceship"] 
             for i in range(3) ]


vars = Namespace()
vars.cuda_id = None
vars.kw = None
vars.prompts = None
vars.pipe = None
model_llm, tokenizer_llm = None, None
cmm_metric:CMM_MobileNet = None


def cuda_select( device_dict:dict[str:int], device_name:str ):
    # device_dict -> { device_name:int|None }
    vars.cuda_id = device_dict[ device_name ]
    new_drop = gr.Dropdown(info="")
    if vars.cuda_id!=None:
        free, total = torch.cuda.mem_get_info( vars.cuda_id )
        new_drop = gr.Dropdown( info=f"Free memory {round( free/total, 2)} % of {round(total/1024**3, 3)} GB" )
    return new_drop

def process_kw(kw_str:str): # gradio_tmp
    vars.kw = { k:int(w) for k,w in map( lambda x: x.strip().split(":") , kw_str.split(",") )}
    kw_json = {'concept' : { 'keywords' : [ {k:w} for k,w in vars.kw.items() ] } }
    with open(TEMP_JSON_KW, "w") as outfile: json.dump(kw_json, outfile)
    # Make prompt
    vars.prompts = prompt_refinement(TEMP_JSON_KW, model_llm, tokenizer_llm)
    new_prompt_outs = [ gr.TextArea(prompt, placeholder=None) for prompt in vars.prompts['concept'] ]
    #new_prompt_column = gr.Column( visible=True )
    new_prompt_slct_col = gr.Row(visible=True)
    new_device_row = gr.Row( visible=DEVICE_SELECTOR )
    new_img_row = gr.Row(visible=True)
    return new_prompt_outs + [new_prompt_slct_col, new_device_row, new_img_row]

def save_img_tensor( tensor:torch.Tensor, save_dir:str ) -> None:
    fig, ax = plt.subplots()
    img = ax.imshow(tensor.numpy())
    ax.axis("off")
    plt.savefig(save_dir, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def generate_img( promt_idx:int, model_genai_name:str, model_llm:str ):
    #select_prompt = { 'concept': [ prompts['concept'][promt_idx] ] }
    #save_prompts( TEMP_JSON_PRMT, select_prompt) # DEP
    prompt = vars.prompts['concept'][promt_idx-1]
    if model_genai_name == "SDXL":
        if vars.pipe is None or not isinstance(vars.pipe, StableDiffusionXLPipeline):
            vars.pipe = load_model_SDXL(CKPT_NUM_SDXL, False)
        #generate_image_SDXL(prompt, IMG_PATH, CKPT_NUM_SDXL)
    elif model_genai_name == "SD2":
        if vars.pipe is None or not isinstance(vars.pipe, StableDiffusionPipeline):
            vars.pipe = load_model_stable_diffusion(CKPT_NUM_SD2, False)
        #generate_image_stable_diffusion(prompt, IMG_PATH, CKPT_NUM_SD2)
    vars.pipe.to( f"cuda:{vars.cuda_id}" if vars.cuda_id is not None else "cpu")
    img_gen = vars.pipe(prompt, num_inference_steps=CKPT_NUM_SDXL, guidance_scale=0).images[0]
    save_img_tensor( img_gen, IMG_PATH )
    gen_img = Image.open( IMG_PATH )

    mcmm, scmm, acmm = cmm_metric.calculate( vars.kw, gen_img, "all", "all", 3, False )
    metrics_str = f"Weight adhesion:\t{ cmm_metric.calc_vals['w_a'] }\nClass adhesion:\t\t{cmm_metric.calc_vals['c_a'] }\n"
    metrics_str += f"\nM-CMM:\t{mcmm}\nS-CMM:\t{scmm}\nA-CMM:\t{acmm}"

    new_img = gr.Image( gen_img )
    new_metrics = gr.TextArea( metrics_str )
    return [ new_img, new_metrics ]

def simulate_generation( image_name:int, model_genai_name:str, model_llm_name:str ):
    if model_genai_name == "SDXL":
        sleep(5)
    elif model_genai_name == "SD2":    
        sleep(20)
    fake_img = f"./experiments/experiment_{model_llm_name}_{model_genai_name.lower()}/images/{image_name}.png"
    gen_img = Image.open( fake_img ).convert("RGB")

    mcmm, scmm, acmm = cmm_metric.calculate( vars.kw, gen_img, "all", "all", 3, False )
    metrics_str = f"Weight adhesion:\t{ cmm_metric.calc_vals['w_a'] }\nClass adhesion:\t\t{cmm_metric.calc_vals['c_a'] }\n"
    metrics_str += f"\nM-CMM:\t{mcmm}\nS-CMM:\t{scmm}\nA-CMM:\t{acmm}"

    new_img = gr.Image( gen_img )
    new_metrics = gr.TextArea( metrics_str )
    return [ new_img, new_metrics ]

if __name__=="__main__":

    # Loads for prompt
    model_llm, tokenizer_llm = init_model( "gpt2" )
    cmm_metric = CMM_MobileNet()

    devs = {"Cpu":None}
    for i in range(torch.cuda.device_count()): devs[ f"Cuda {i} - {torch.cuda.get_device_name(i)}"  ] = i

    with gr.Blocks() as demo:
        # Prompting
        with gr.Blocks() as prompt_section:
            with gr.Row():
                with gr.Column():
                    kw_in = gr.TextArea( label="Input keywords and weights", lines=1,
                                        info="Keywords that will be converted into a prompt. Each one with its weight, separed by ':', \
                                        and each pair keyword:weight, separed by comma.",
                                        placeholder="keyword_1:weight_1, keyword_2:weight_2, .." )
                    llm_selector = gr.Dropdown( PROMPT_LMS, value=PROMPT_LMS[0], label="LLM for prompt", visible=False )
                    get_prompt_btn = gr.Button( "Generate prompts" )
            with gr.Row():
                with gr.Column(visible=True) as prompt_column:
                    with gr.Row(): # Column gets 3 textAreas horizotaly, but is too much space
                        prompt_out_1 = gr.TextArea( label="Prompt 1", lines=2, placeholder="No prompt", interactive=False)
                        prompt_out_2 = gr.TextArea( label="Prompt 2", lines=2, placeholder="No prompt", interactive=False)
                        prompt_out_3 = gr.TextArea( label="Prompt 3", lines=2, placeholder="No prompt", interactive=False)
                    prompt_outs = [ prompt_out_1, prompt_out_2, prompt_out_3 ]
                    with gr.Column(visible=False) as prompt_slct_col:
                        prompt_selector = gr.Dropdown( [1,2,3], value=1, label="Generation prompt", interactive=True )
                        genai_selector = gr.Dropdown( IMGS_GENAI, value=IMGS_GENAI[0], label="Model for image generation" )
                        if SIMULATE_GEN:
                            img_selector = gr.Dropdown( FAKE_OPTS, FAKE_OPTS[0], label="Test image" )
                        gen_img_btn = gr.Button( "Generate" )

        # Image gen show area
        with gr.Blocks() as image_section:
            with gr.Row( visible=False ) as image_row:
                #with gr.Column(): # Shows img and below the metric
                img_1 = gr.Image( interactive=False )
                metrics_1 = gr.TextArea( label="Metrics", lines=2, interactive=False )

        # Device selector
        with gr.Blocks() as device_selector:
            with gr.Row( visible=False ) as device_row:
                dropdown_devices = gr.Dropdown( opts:=(list(devs.keys()) ), value=opts[0], label="Device", interactive=True )

        # ------ Bindings ------
        dropdown_devices.select( lambda x: cuda_select(devs, x), inputs=dropdown_devices, outputs=dropdown_devices )
        get_prompt_btn.click( process_kw, inputs=kw_in, 
                             outputs = prompt_outs + [prompt_slct_col, image_row, device_row] )
        if not SIMULATE_GEN:
            gen_img_btn.click( generate_img, inputs=[prompt_selector, genai_selector, llm_selector], outputs=[ img_1, metrics_1 ] )
        else:
            gen_img_btn.click( simulate_generation, inputs=[img_selector, genai_selector, llm_selector], outputs=[ img_1, metrics_1 ] )

    demo.launch( share=True )