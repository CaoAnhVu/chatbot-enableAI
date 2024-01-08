import tkinter as tk
from tkinter import ttk, scrolledtext, IntVar
import tkinter.messagebox as messagebox
from PIL import Image, ImageTk
from model import NeuralNet
import time
import torch
import json
import random
from nltk_utils import bag_of_words, tokenize, stem

class ChatBotGUI:
    def __init__(self, master):
        self.master = master
        master.title("ChatBot")
        self.is_dark_theme = IntVar()
        self.chat_display = scrolledtext.ScrolledText(self.master, wrap=tk.WORD, width=80, height=20, font=('Arial', 16))
        self.chat_display.pack(padx=20, pady=20)
        self.user_img = Image.open("D:\\TRÍ TUỆ NHÂN TẠO\\chatbot-enableAI\\profile.png")
        self.user_img = self.user_img.resize((50, 50), resample=Image.LANCZOS)
        self.user_img = ImageTk.PhotoImage(self.user_img)
        airplane_icon = Image.open("D:\\TRÍ TUỆ NHÂN TẠO\\chatbot-enableAI\\send.png")
        airplane_icon = airplane_icon.resize((20, 20), resample=Image.LANCZOS)
        self.airplane_img = ImageTk.PhotoImage(airplane_icon)
        self.user_input = tk.Entry(master, width=50, font=('Arial', 14))
        self.user_input.pack(padx=20, pady=20, side=tk.LEFT)
        self.send_button = tk.Button(master, text="Send", command=self.send_message, compound=tk.LEFT, image=self.airplane_img, width=150, height=20, font=('Arial', 12))
        self.send_button.place(relx=1, rely=1, anchor=tk.SE, x=-250, y=-20)
        style = ttk.Style()
        style.configure('ThemeToggle.TCheckbutton', indicatoron=False, font=('Arial', 12))
        self.theme_toggle = ttk.Checkbutton(master, text="Dark Theme", variable=self.is_dark_theme, command=self.apply_theme, style='ThemeToggle.TCheckbutton')
        self.theme_toggle.pack(side=tk.RIGHT, padx=10)
        self.master.bind('<Return>', lambda event=None: self.send_message())
        self.new_chat_button = tk.Button(master, text="New Chat", command=self.create_new_chat, font=('Arial', 12))
        self.new_chat_button.pack(side=tk.RIGHT, padx=10, pady=10)
        self.load_model()
        self.display_message_with_typing_animation("Sam: Hi! I'm your ChatBot. How can I assist you today?", 'sam', align='left', color='#09795a')

    def load_model(self):
        with open('D:\\TRÍ TUỆ NHÂN TẠO\\chatbot-enableAI\\chatbot-enableAI\\intents.json', 'r', encoding='utf-8') as f:
            self.intents = json.load(f)
        data = torch.load("D:\\TRÍ TUỆ NHÂN TẠO\\chatbot-enableAI\\data.pth")
        self.model_state = data["model_state"]
        self.all_words = data["all_words"]
        self.tags = data["tags"]
        self.model = NeuralNet(len(self.all_words), 8, len(self.tags))
        self.model.load_state_dict(self.model_state)
        self.model.eval()

    def apply_theme(self):
        if self.is_dark_theme.get():
            self.master.config(bg='#333333')
            self.chat_display.config(bg='#2d2d2d', fg='white')
            self.user_input.config(bg='#2d2d2d', fg='white')
        else:
            self.master.config(bg='white')
            self.chat_display.config(bg='white', fg='black')
            self.user_input.config(bg='white', fg='black')

    def create_new_chat(self):
        result = messagebox.askquestion("New Chat", "Are you sure you want to start a new chat?")
        if result == 'yes':
            self.chat_display.delete(1.0, tk.END)
            self.display_message_with_typing_animation("Sam: Hi! I'm your ChatBot. How can I assist you today?", 'sam', align='left', color='#09795a')

    def send_message(self):
        user_message = self.user_input.get()
        self.user_input.delete(0, tk.END)

        # Decode user input to handle Vietnamese characters
        user_message = user_message.encode('utf-8').decode('utf-8')
        self.display_message("You: " + user_message, 'user', color='#ff12a5')
        response = self.get_model_response(user_message)
        self.display_message_with_typing_animation("Sam: " + response, 'sam', align='left', color='#09795a')

    def get_model_response(self, user_message):
        user_words = tokenize(user_message)
        user_bag = bag_of_words(user_words, self.all_words)
        user_input_tensor = torch.tensor(user_bag, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.model(user_input_tensor)
        predicted_tag = self.tags[torch.argmax(output).item()]
        for intent in self.intents['intents']:
            if intent['tag'] == predicted_tag:
                responses = intent['responses']
                return random.choice(responses)
        return "I'm sorry, I don't understand."

    def display_message(self, message, sender, color='black'):
        # Check if the tag already exists
        if not self.chat_display.tag_names().__contains__(sender):
            # Create the tag if it doesn't exist
            self.chat_display.tag_configure(sender, foreground=color)

        self.chat_display.insert(tk.END, "\n" + message + "\n", sender)
        self.chat_display.yview(tk.END)

    def display_message_with_typing_animation(self, message, sender, align='left', color='black'):
        last_sender = None
        last_line = ""
        for char in message:
            if sender != last_sender:
                align = 'left' if sender == 'sam' else 'left'
                last_sender = sender
            # Check if the tag already exists
            if not self.chat_display.tag_names().__contains__(sender):
                # Create the tag if it doesn't exist
                self.chat_display.tag_configure(sender, foreground=color)

            self.chat_display.insert(tk.END, char, sender)
            last_line += char
            self.chat_display.yview(tk.END)
            self.master.update()
            time.sleep(0.02)

if __name__ == "__main__":
    root = tk.Tk()
    chatbot_gui = ChatBotGUI(root)
    root.mainloop()
