from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
import gradio as gr

# Set credentials to use the model
watsonx_API = ""
project_id = ""

# Model and project settings
model_id = "meta-llama/llama-2-13b-chat"

# Set necessary parameters
generate_params = {
    GenParams.MAX_NEW_TOKENS: 250

}

model = Model(
    model_id = 'meta-llama/llama-2-13b-chat', # you can also specify like: ModelTypes.LLAMA_2_70B_CHAT
    params = generate_params,
    credentials={
        "apikey": watsonx_API,
        "url": "https://jp-tok.dataplatform.cloud.ibm.com"
    },
    project_id= project_id
    )

# # Initialize the model
# model = Model(model_id, my_credentials, gen_parms, project_id, space_id, verify)

# Function to generate a response from the model
def generated_response(prompt_txt):
    generated_response = model.generate(prompt_txt)

    # Extract and return the generated text
    generated_text = generated_response["results"][0]["generated_text"]
    return generated_text

# Create Gradio interface
chat_application = gr.Interface(
    fn=generated_response,
    allow_flagging="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Watson.ai Chatbot",
    description="Ask any question and the chatbot will try to answer."
)

# Launch the app
chat_application.launch()