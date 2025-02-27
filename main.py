from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
import os, asyncio, uuid  
from dotenv import load_dotenv
from genai_detectLan_andTranslate.src.splitter import loadDataAndCreateChunks
from genai_detectLan_andTranslate.src.embeddingAndvectorDB import process_pdf_and_store, retrieve_answer
from langchain.callbacks import get_openai_callback

load_dotenv()
UPLOAD_FOLDER = 'pdfs'
ALLOWED_EXTENSIONS = {'pdf'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

openai_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX")
huggingface_api_token = os.getenv("HUGGINGFACE_API_READ_TOKEN")

app = Flask(__name__)
app.secret_key = "fdew4fw34fr3T44gIIV&Ihg78@d" 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class genAI_class:
    def __init__(self, user_file):
        self.user_filename = user_file
        self.ai_instance = None
    
    async def process_ai(self):
        if self.ai_instance is None:
            data_chunks = await loadDataAndCreateChunks(self.user_filename)
            await process_pdf_and_store(pinecone_key, pinecone_index, data_chunks)
            self.ai_instance = await retrieve_answer(openai_key,pinecone_key,pinecone_index)
        return self.ai_instance

# async def main():
#     user_file = 'genai.pdf'
#     ai_obj = genAI_class(user_file)
#     await ai_obj.process_ai()

@app.route('/')
async def home():
    return render_template('index.html')

user_sessions = {}
@app.route('/upload', methods=['POST'])
async def upload_pdf():
    """Handle file uploads and create session"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())

        user_id = session['user_id']
        session['filename'] = filename
        print(session.get('filename'))
        user_sessions[user_id] = genAI_class(file_path)
        return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 200
    else:
        return jsonify({'error': 'Invalid file format. Only PDFs are allowed up to 10MB.'}), 400

@app.route('/query', methods=['POST'])
async def query():
    """Process user query using stored PDF and AI instance"""
    data = request.get_json()
    user_input = data.get('query')

    user_id = session.get('user_id')
    filename = session.get('filename')
    print(f"-----\n{session.get('filename')}")
    if not user_id or not filename:
        return jsonify({'error': 'No file uploaded. Please upload a file first.'}), 400
    print(app.config['UPLOAD_FOLDER'])

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found. Please upload again.'}), 404

    # ai_obj = user_sessions.get(user_id)
    # if not ai_obj:
    with get_openai_callback() as cb:
        ai_obj = genAI_class(file_path)
        user_sessions[user_id] = ai_obj

        ask_me = await ai_obj.process_ai()
        res = ask_me({'query': user_input})
        response = res['result']
        print(f"total_tokens : {cb.total_tokens}")
        print(f"prompt_tokens : {cb.prompt_tokens}")
        print(f"completion_tokens : {cb.completion_tokens}")
        print(f"total_cost : {cb.total_cost}") 
        return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)