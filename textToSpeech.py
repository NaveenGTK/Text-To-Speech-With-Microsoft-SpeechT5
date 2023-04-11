# Following pip packages need to be installed:
# !pip install git+https://github.com/huggingface/transformers sentencepiece datasets

import tkinter
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import os
import soundfile as sf
import pygame
from tkinter import *
import customtkinter

customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

root = customtkinter.CTk()
root.title('Text To Audio Software - Made With Microsoft SpeechT5')
 
root.geometry("500x400")
 
pygame.mixer.init()# initialise the pygame
 
def play():
    path = os.getcwd()
    if os.path.isfile(path+'\speech.wav'):
        pygame.mixer.music.load("speech.wav")
        pygame.mixer.music.play(loops=0)
    else:
        tkinter.messagebox.showerror(title="ERROR!", message="No audio file detected. Enter text first!")

def convertToAudio():
    pygame.mixer.music.unload()
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    inputT = my_text.get(1.0, END)

    inputs = processor(text=inputT, return_tensors="pt")

    # load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    sf.write("C:/Users/User/Documents/Tensor Projects/speech.wav", speech.numpy(), samplerate=16000)

 
title=customtkinter.CTkLabel(root,text="Speech To Text",
            font=("montserrat",25,"bold"), fg_color=("white", "gray75"), corner_radius=8)
title.pack(side=TOP,fill=X)

my_text = Text(root, width=50, height=10, font=("Helvetica", 14))
my_text.pack(pady=20)
submit_button = customtkinter.CTkButton(master=root, text="Submit Text", font=("Helvetica", 18), command=convertToAudio)
submit_button.place(relx=0.5, rely=0.7, anchor=tkinter.CENTER)

play_button = customtkinter.CTkButton(master=root, text="Listen To Text", font=("Helvetica", 18), command=play)
play_button.place(relx=0.5, rely=0.8, anchor=tkinter.CENTER)
root.mainloop()