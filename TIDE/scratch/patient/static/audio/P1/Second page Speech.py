import gtts
import playsound

text1 = " PATIENT one"
text2 = "(HDT Based personalized management  )  (Time In Range 87.1 % ) (Time spent in hypo 5.4 %) (time spent in hyper 7.5%)"
text3 = "(conventional Insulin Therapy ) (Time In Range 63 %) ( time spent in hypo 8.6 %) (time spent in hyper 27.6%)"
text = text1+text2+text3
#text = text + str(NumberHyperEvents)
print(text)

sound = gtts.gTTS(text, lang="en")
sound.save("save.mp3")
playsound.playsound("save.mp3")