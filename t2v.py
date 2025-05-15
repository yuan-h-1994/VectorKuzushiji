import sys
from datetime import datetime
from pytz import timezone
from google.cloud import texttospeech
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file('api.json')
client = texttospeech.TextToSpeechClient(credentials=credentials)

with open(sys.argv[1]) as f:
    s = f.read()

synthesis_input = texttospeech.SynthesisInput(
  text=s
#  text='Classes on human-object interaction, such as "holding phone", calculate a score by the percentage of objects detected. Here we only focus on objects that are close to the human hand. We also used EfficientDet, pre-trained in MS-COCOCO for object detection.'
)

voice = texttospeech.VoiceSelectionParams(
#  language_code='ja-JP',
#  name='ja-JP-Wavenet-D',
  language_code='en-US',
  name='en-US-Wavenet-J',
  ssml_gender=texttospeech.SsmlVoiceGender.MALE)
#  ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)

audio_config = texttospeech.AudioConfig(
  audio_encoding=texttospeech.AudioEncoding.MP3, 
  speaking_rate=1.05 )
#  pitch = 1.1 )

response = client.synthesize_speech(
request={
            "input": synthesis_input,
            "voice":voice,
            "audio_config":audio_config
                })
now = datetime.now(timezone('Asia/Tokyo'))
filename = now.strftime('%Y-%m-%d_%H%M%S.mp3')
with open(filename, 'wb') as out:
  out.write(response.audio_content)
  print(f'Audio content written to file {filename}')
