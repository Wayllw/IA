import tkinter as tk
from tkinter import font, filedialog, messagebox
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline


# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Assign a padding token to the GPT-2 tokenizer
tokenizer.pad_token = tokenizer.eos_token  # Use the end-of-sequence token as the padding token

# Load the fine-tuned model
generator = pipeline("text-generation", model="./fine_tuned_model", tokenizer=tokenizer)

# Generate text
def genText():
    prompt=user_entry.get()
    # prompt = "Life is beautiful when"
    output = generator(prompt, max_length=50, num_return_sequences=1)
    text_generated=output[0]["generated_text"]
    mensagem(text_generated)
    login.destroy()


def mensagem(mensagem:str):
    sucesso = tk.Tk()
    sucesso.title("Alert")
    sucesso.eval('tk::PlaceWindow . center')
    bold = font.Font(weight="bold")
    message = tk.Message(sucesso, text=mensagem, font=bold, width=300)
    message.grid(row=1, column=1)
    sucesso.geometry("300x100")
    sucesso.config(bg="#CCCCFF")
    message.configure(bg="#CCCCFF", fg="black")
    sucesso.mainloop()



login = tk.Tk()
bold = font.Font(weight="bold")
login.title("Quote Generator")
login.eval('tk::PlaceWindow . center')

user_label = tk.Label(login, text="Prompt:", height=5, font=bold)
user_label.place(relx=0.5, rely=0.1, anchor="center")

user_entry = tk.Entry(login)
user_entry.place(relx=0.5, rely=0.55 ,width=300,height=200, anchor='center')

confirm_button=tk.Button(login, text="Confirm", command=genText)
confirm_button.place(x=110, y=115, width=200, height=25)
# Tamanho da janela
login.geometry("400x300")
# Edição de cores
login.config(bg="#d7bde2")
user_label.configure(bg="#d7bde2", fg="black")
user_entry.configure(bg="#ebdef0", fg="black")
confirm_button.configure(bg="#ebdef0", fg="black")
login.mainloop()