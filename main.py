import gradio as gr
import requests
import os
import chardet
import fitz  # PyMuPDF for reading PDF documents
import json

def verify_api_connection():
    url = "http://localhost:11434/api/generate"
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({
        "model": "mistral",
        "prompt": "Say Hello back to me, and only say Hello"
    })
    try:
        response = requests.post(url, data=data, headers=headers)
        print("Raw Response:", response.text)  # Log raw response data for inspection

        # Attempt to parse JSON response
        response_data = response.json()

        if response.status_code == 200 and "Hello" in response_data.get("choices", [{}])[0].get("text", ""):
            print("Successfully connected to the Ollama Mistral LLM API.")
            return True
        else:
            print(f"Failed to connect to the Ollama Mistral LLM API. Status Code: {response.status_code}, Response: {response.text}")
            return False
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}. Response content: '{response.text}'")
        return False
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while trying to connect to the Ollama Mistral LLM API: {e}")
        return False

def load_document(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        text = ''
        with fitz.open(file_path) as doc:
            text = ''.join(page.get_text() for page in doc)
    elif file_extension.lower() in ['.csv', '.txt', '.md']:
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding'] or 'utf-8'
            text = raw_data.decode(encoding, errors='ignore')
    else:
        text = "Unsupported file format."
    return text

def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            document_text = load_document(file_path)
            if document_text and document_text != "Unsupported file format.":
                documents.append(document_text)
    return documents

def ask_ollama(question, documents):
    payload = {"documents": documents, "query": question}
    headers = {"Content-Type": "application/json"}
    response = requests.post("http://localhost:11434/api/generate", json=payload, headers=headers)
    if response.status_code == 200:
        try:
            response_data = response.json()
            return response_data.get("choices", [{"text": "No response or unexpected response structure."}])[0]["text"]
        except json.JSONDecodeError:
            return "Error in parsing response."
    else:
        return f"An error occurred: {response.status_code} -> {response.text}"

def gradio_interface(folder_path, question):
    if not verify_api_connection():
        return "API connection verification failed. Please check the API status."
    documents = load_documents(folder_path)
    return ask_ollama(question, documents)

# Setting up the Gradio interface
def setup_interface():
    with gr.Blocks() as demo:
        gr.Markdown("### Document Inquiry - Ollama Mistral LLM")
        with gr.Row():
            folder_input = gr.Textbox(label="Enter the path to your document repository", placeholder="Path to folder containing your text or PDF documents...")
            question_input = gr.Textbox(label="Enter your question", placeholder="What would you like to ask?")
            submit_button = gr.Button("Ask Ollama")

        output = gr.Textbox(label="Ollama's Response")

        submit_button.click(fn=gradio_interface, inputs=[folder_input, question_input], outputs=output)

    return demo

if __name__ == "__main__":
    demo = setup_interface()
    demo.launch()
