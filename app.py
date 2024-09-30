from flask import Flask, render_template, request, jsonify
from src import WhitespaceTokenizer, RegexTokenizer, BPETokenizer, CustomHFTokenizer, CustomSPTokenizer
from collections import Counter
import os
from werkzeug.utils import secure_filename
import tempfile
import re
import logging

app = Flask(__name__)

# Initialize tokenizers
tokenizers = {
    'whitespace': WhitespaceTokenizer(),
    'regex': RegexTokenizer(pattern='gpt2'),
    'bpe': BPETokenizer(),
    'hf': CustomHFTokenizer(),
    'sp': CustomSPTokenizer()
}

# Load pre-trained vocabularies if they exist
for name, tokenizer in tokenizers.items():
    if name == 'hf':
        vocab_file = 'trained_vocabs/hf_vocab.json'
        if os.path.exists(vocab_file):
            tokenizer.load(vocab_file)
            logging.info(f"Loaded pre-trained vocabulary for {name} tokenizer")
        else:
            logging.warning(f"Vocabulary file {vocab_file} not found for {name} tokenizer")
    elif name == 'sp':
        vocab_file = 'trained_vocabs/sp_vocab.model'
    elif name == 'regex':
        vocab_file = 'trained_vocabs/regex_vocab_gpt2.txt'
    elif name == 'bpe':
        vocab_file = 'trained_vocabs/bpe_vocab.txt'
    else:
        vocab_file = f'trained_vocabs/{name}_vocab.txt'
    
    if os.path.exists(vocab_file):
        tokenizer.load(vocab_file)
        print(f"Loaded pre-trained vocabulary for {name} tokenizer")

import re

def add_space_except_first(string_list):
    if not string_list:
        return []
    
    result = []
    for i, s in enumerate(string_list):
        if '<unk>' in s:
            # Split the token around '<unk>'
            parts = re.split(r'(<unk>)', s)
            for j, part in enumerate(parts):
                if part == '<unk>':
                    if j > 0 and parts[j-1]:  # If there's a part before '<unk>'
                        if i > 0 or j > 0:  # Add space if it's not the first token
                            result.append(' ' + parts[j-1])
                        else:
                            result.append(parts[j-1])
                    if i > 0 or j > 0:  # Add space before '<unk>' if it's not the very first token
                        result.append(' <unk>')
                    else:
                        result.append('<unk>')
                elif j == len(parts) - 1 and part:  # Last part after '<unk>'
                    result.append(part)
        else:
            if i > 0 and s not in ['</w>']:
                result.append(' ' + s)
            else:
                result.append(s)
    
    return result


@app.route('/')
def index():
    return render_template('index.html', tokenizers=tokenizers.keys())

@app.route('/tokenize', methods=['POST'])
def tokenize():
    data = request.get_json()
    text = data.get('text', '')
    tokenizer_names = data.get('tokenizers', [])

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    results = {}

    for name in tokenizer_names:
        tokenizer = tokenizers[name]
        try:
            tokens = tokenizer.tokenize(text)
            logging.debug(f"{name} tokenizer - tokens: {tokens}")
            
            encoded = tokenizer.encode(text)
            logging.debug(f"{name} tokenizer - encoded: {encoded}")
            
            decoded = tokenizer.decode(encoded)
            logging.debug(f"{name} tokenizer - decoded: {decoded}")

            if isinstance(decoded, str):
                decoded_tokens = decoded.split()
            else:
                decoded_tokens = decoded
            
            decoded_tokens = add_space_except_first(decoded_tokens)
            logging.debug(f"{name} tokenizer - decoded_tokens: {decoded_tokens}")

            results[name] = {
                'tokens': tokens,
                'encoded': encoded,
                'decoded': decoded_tokens,
                'frequencies': dict(Counter(tokens))
            }
        except Exception as e:
            logging.error(f"Error with {name} tokenizer: {str(e)}")
            results[name] = {'error': str(e)}

    return jsonify(results)

@app.route('/train', methods=['POST'])
def train():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected file'}), 400

    tokenizer_name = request.form['tokenizer']
    tokenizer = tokenizers[tokenizer_name]

    # Create a temporary directory to store uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in files:
            if file and file.filename.endswith('.txt'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(temp_dir, filename)
                file.save(filepath)

        # Train the tokenizer
        if isinstance(tokenizer, (BPETokenizer, CustomHFTokenizer, CustomSPTokenizer)):
            tokenizer.train(temp_dir)
        else:
            corpus = ""
            for filename in os.listdir(temp_dir):
                with open(os.path.join(temp_dir, filename), 'r', encoding='utf-8') as f:
                    corpus += f.read() + "\n"
            tokenizer.fit(corpus)

    # Save the new vocabulary
    if tokenizer_name == 'hf':
        vocab_file = 'trained_vocabs/hf_vocab.json'
    elif tokenizer_name == 'sp':
        vocab_file = 'trained_vocabs/sp_vocab.model'
    else:
        vocab_file = f'trained_vocabs/{tokenizer_name}_vocab.txt'
    
    tokenizer.save(vocab_file)

    return jsonify({'message': f'{tokenizer_name} tokenizer trained successfully on {len(files)} files'})


if __name__ == '__main__':
    app.run(debug=True)