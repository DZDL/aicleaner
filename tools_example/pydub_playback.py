from pydub import AudioSegment
from pydub.playback import play

sound = AudioSegment.from_file("temporal/output.wav", format="wav")
play(sound)