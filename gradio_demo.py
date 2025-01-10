import json
import PIL.Image
import torch, PIL
import gradio as gr
from argparse import Namespace

from src import init_model, prompt_refinement, save_prompts
from src import load_model_SDXL, generate_image_SDXL
from src import load_model_stable_diffusion, generate_image_stable_diffusion, pipe
from src import CMM_MobileNet

TMP_DIR = "./gradio_tmp"
IMG_PATH = TMP_DIR + "/gen_img.png"
TEMP_JSON_KW = TMP_DIR + "/keywords.json"
TEMP_JSON_PRMT = TMP_DIR + "/prompts.json"

PROMPT_LMS = [ "gpt2", "gemma" ] #"gpt2" -  "google/gemma-7b"
IMGS_GENAI = [ "SDXL", "SD2" ] # "stabilityai/stable-diffusion-2-1" - "stabilityai/stable-diffusion-xl-base-1.0"
CKPT_NUM_SDXL = 4
CKPT_NUM_SD2 = 4

DEVICE_SELECTOR = False


use_cuda = None
vars = Namespace()
vars.kw = None
vars.prompts = None
model_llm, tokenizer_llm = None, None
cmm_metric:CMM_MobileNet = None


def cuda_select( device_dict:dict[str:int], device_name:str ):
    # device_dict -> { device_name:int|None }
    use_cuda = device_dict[ device_name ]
    new_drop = gr.Dropdown(info="")
    if use_cuda!=None:
        free, total = torch.cuda.mem_get_info( use_cuda )
        new_drop = gr.Dropdown( info=f"Free memory {round( free/total, 2)} %" )
    return new_drop

def process_kw(kw_str:str): # gradio_tmp
    vars.kw = { k:int(w) for k,w in map( lambda x: x.strip().split(":") , kw_str.split(",") )}
    kw_json = {'concept' : { 'keywords' : [ {k:w} for k,w in vars.kw.items() ] } }
    with open(TEMP_JSON_KW, "w") as outfile: json.dump(kw_json, outfile)
    # Make prompt
    vars.prompts = prompt_refinement(TEMP_JSON_KW, model_llm, tokenizer_llm)
    new_prompt_outs = [ gr.TextArea(prompt) for prompt in vars.prompts['concept'] ]
    new_prompt_column = [ gr.Column( visible=True ) ]
    return new_prompt_outs + new_prompt_column

def generate_img( promt_idx:int, model_genai_name:str, model_llm_name:str ):
    #select_prompt = { 'concept': [ prompts['concept'][promt_idx] ] }
    #save_prompts( TEMP_JSON_PRMT, select_prompt) # DEP
    prompt = vars.prompts['concept'][promt_idx]
    if model_genai_name == "SDXL":
        if pipe is None:
            load_model_SDXL(CKPT_NUM_SDXL)
        generate_image_SDXL(prompt, IMG_PATH, CKPT_NUM_SDXL)
        total_images_generated_counter += 1
    elif model_genai_name == "SD2":
        if pipe is None:
            load_model_stable_diffusion(CKPT_NUM_SD2)
        generate_image_stable_diffusion(prompt, IMG_PATH, CKPT_NUM_SD2)
    gen_img = PIL.Image.open( IMG_PATH )

    mcmm, scmm, acmm = cmm_metric.calculate( vars.kw, gen_img, "all", "all", 3, False )
    metrics_str = f"Weight adhesion: { cmm_metric.calc_vals['w_a'] }\nClass adhesion: {cmm_metric.calc_vals['w_a'] }\n"
    metrics_str += f"MCMM: {mcmm}\nSCMM: {scmm}\nACMM: {acmm}"

    new_img = gr.Image( gen_img )
    new_metrics = gr.TextArea( metrics_str )
    new_row = gr.Row(visible=True)
    return [ new_img, new_metrics, new_row ]

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
                with gr.Column(visible=False) as prompt_column:
                    with gr.Row(): # Column gets 3 textAreas horizotaly, but is too much space
                        prompt_out_1 = gr.TextArea( label="Prompt 1", interactive=False)
                        prompt_out_2 = gr.TextArea( label="Prompt 2", interactive=False)
                        prompt_out_3 = gr.TextArea( label="Prompt 3", interactive=False)
                    prompt_outs = [ prompt_out_1, prompt_out_2, prompt_out_3 ]
                    with gr.Row():
                        prompt_selector = gr.Dropdown( [1,2,3], value=1, label="Generation prompt" )
                        genai_selector = gr.Dropdown( IMGS_GENAI, value=IMGS_GENAI[0], label="Model for image generation" )
                        gen_img_btn = gr.Button( "Generate" )

        # Image gen show area
        with gr.Blocks() as image_section:
            with gr.Row( visible=False ) as image_row:
                #with gr.Column(): # Shows img and below the metric
                img_1 = gr.Image( interactive=False )
                metrics_1 = gr.TextArea( label="Metrics", lines=2, interactive=False )

        # Device selector
        with gr.Blocks() as device_selector:
            with gr.Row( visible=DEVICE_SELECTOR ):
                dropdown_devices = gr.Dropdown( opts:=(list(devs.keys()) ), value=opts[0], label="Device", interactive=True )

        # ------ Bindings ------
        dropdown_devices.select( lambda x: cuda_select(devs, x), inputs=dropdown_devices, outputs=dropdown_devices )
        get_prompt_btn.click( process_kw, inputs=kw_in, outputs= prompt_outs+[prompt_column] )
        gen_img_btn.click( generate_img, inputs=[prompt_selector, genai_selector, llm_selector],
                          outputs=[ img_1, metrics_1, image_row ] )

    demo.launch( share=False )