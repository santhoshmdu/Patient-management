import gtts
import playsound

#NumberHyperEvents = 80
text1 = " PATIENT One "
text2 = "(Co-Morbidity conditions Hypothyroidism  Dyslipidemia )  (Target BGL Range 80 to 180 mg per dl ) ( oral drugs Glycomet 250 miligrams)"
text = text1+text2
#text = text + str(NumberHyperEvents)
print(text)

sound = gtts.gTTS(text, lang="en")
sound.save("save.mp3")
playsound.playsound("save.mp3")