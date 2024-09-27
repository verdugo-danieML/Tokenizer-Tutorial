from flask import Flask, render_template, request, jsonify
from src import WhitespaceTokenizer, RegexTokenizer, BPETokenizer, CustomHFTokenizer, CustomSPTokenizer
from collections import Counter
import os
from werkzeug.utils import secure_filename
import tempfile
import re

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
        tokens = tokenizer.tokenize(text)
        
        encoded = tokenizer.encode(text)
        
        decoded_tokens = tokenizer.decode(encoded)
        print(decoded_tokens)
        if isinstance(decoded_tokens, str):
            decoded_tokens = decoded_tokens.split()  # Split the string into a list of tokens
        decoded_tokens = add_space_except_first(decoded_tokens)
        print(decoded_tokens)
        

        results[name] = {
            'tokens': tokens,
            'encoded': encoded,
            'decoded': decoded_tokens,
            'frequencies': dict(Counter(tokens))
        }

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