from flask import Flask, request, jsonify
from transformers import pipeline
from google.cloud import translate_v2 as translate
from google.cloud import texttospeech



app = Flask(__name__)

# Load Hugging Face grammar correction model
grammar_model = pipeline('text2text-generation', model='vennify/t5-base-grammar-correction')

# Initialize Google Cloud Translate and Text-to-Speech clients
translate_client = translate.Client()
tts_client = texttospeech.TextToSpeechClient()

@app.route('/process-text', methods=['POST'])
def process_text():
    data = request.json
    user_text = data['user_text']
    target_language = data['target_language']
    
    # Step 1: Correct grammar using Hugging Face model
    corrected_grammar = check_grammar_huggingface(user_text)
    
    # Step 2: Translate feedback into the selected language
    translated_feedback = translate_text(corrected_grammar, target_language)
    
    # Step 3: Optionally convert feedback to speech using Google Text-to-Speech
    speech_url = convert_to_speech(translated_feedback, target_language)
    
    return jsonify({"feedback": translated_feedback, "speech_url": speech_url})

def check_grammar_huggingface(user_text):
    corrected_text = grammar_model(user_text, max_length=512, clean_up_tokenization_spaces=True)
    return corrected_text[0]['generated_text']

def translate_text(text, target_language):
    result = translate_client.translate(text, target_language=target_language)
    return result['translatedText']

def convert_to_speech(text, target_language):
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code=target_language, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    
    # Save the speech file locally and return the path (for testing purposes)
    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)

    return "/output.mp3"

if __name__ == '__main__':
    app.run(debug=True)