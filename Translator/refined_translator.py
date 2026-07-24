import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

warnings.filterwarnings("ignore")
model_name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Model trained!")

def translate(text):
    
    inputs = tokenizer(text, return_tensors="pt")
    
    translated_tokens = model.generate(
        **inputs, 
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("hin_Deva"),
        max_length=100
    )
    
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
print("Type an English sentence: ")
print("Type 'quit' to exit.\n")
while True:
    try:
        sentence = input("🇬🇧 English: ").strip()
        
        if sentence.lower() in ("quit", "exit", "q"):
            print("Goodbye! अलविदा!")
            break
            
        if sentence:
            result = translate(sentence)
            print(f"🇮🇳 Hindi:   {result}\n")
            
    except (KeyboardInterrupt, EOFError):
        print("Goodbye! अलविदा!")
        break