from tempfile import TemporaryDirectory
import wave
from pydub import AudioSegment
from pydub.playback import play
import os
import librosa
import glob

list_wav = glob.glob("/home/nas-wks01/users/uapv2202503/Test_anas/data"+"/**/*.wav",recursive=True)


def audio_split(wavefile,tempo):
    path =os.path.split(wavefile)
    file_name = path[1]
    foler_path = path[0]
    newAudio = AudioSegment.from_wav(wavefile)
    for idx,i in enumerate(range(0,len(newAudio)-tempo,tempo)):
        split_audio = newAudio[0:i]+newAudio[i+tempo:]
        split_audio.export(foler_path +'/'+str(idx+1)+file_name,format="wav")
for waves in list_wav:

    audio_split(waves,1000)
