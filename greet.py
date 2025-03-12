import gradio as gr

def process_text(input_text):
    # Convert text to uppercase
    uppercase_text = input_text.upper()
    # Count characters
    char_count = len(input_text)
    return uppercase_text, f"Character count: {char_count}"

# Create the Gradio interface
with gr.Interface(
    fn=process_text,
    inputs=[
        gr.Textbox(label="Enter your text here", placeholder="Type something...")
    ],
    outputs=[
        gr.Textbox(label="Uppercase Result"),
        gr.Textbox(label="Statistics")
    ],
    title="Text Transformer",
    description="A simple app that converts text to uppercase and counts characters",
    theme=gr.themes.Soft()
) as demo:
    demo.launch()
