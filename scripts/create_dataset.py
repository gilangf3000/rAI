import pandas as pd
import os

# Define the dataset
data = {
    "text": [
        # ENGLISH IMAGE REQUESTS
        "create an image of a futuristic city",
        "generate a picture of a cat",
        "draw me a landscape",
        "make an image of a dog",
        "illustrate a concept of peace",
        "photo of a sunset",
        "render a 3d model of a car",
        "visualize this data as a chart",
        "paint a portrait of a woman",
        "sketch a building",
        "give me a picture of a flower",
        "generate art",
        "create a logo for my brand",
        "show me an image of a bird",
        "I want a photo of a beach",
        
        # INDONESIAN IMAGE REQUESTS
        "buatkan gambar kota masa depan",
        "generate foto kucing",
        "lukis pemandangan alam",
        "buat gambar anjing",
        "ilustrasikan konsep perdamaian",
        "foto matahari terbenam",
        "render model 3d mobil",
        "visualisasikan data ini sebagai grafik",
        "lukis potret wanita",
        "sketsa bangunan",
        "berikan saya gambar bunga",
        "buat seni digital",
        "buat logo untuk brand saya",
        "tampilkan gambar burung",
        "saya ingin foto pantai",
        
        # SPANISH IMAGE REQUESTS
        "crea una imagen de una ciudad futurista",
        "genera una foto de un gato",
        "dibújame un paisaje",
        "haz una imagen de un perro",
        "ilustra un concepto de paz",
        
        # CHINESE IMAGE REQUESTS
        "生成一张未来城市的图片",
        "画一只猫",
        "给我画一个风景",
        "生成一张狗的图片",
        "画一幅和平的插图",
        
        # ENGLISH TEXT REQUESTS
        "what is the weather today?",
        "tell me a joke",
        "how to code in python",
        "explain quantum physics",
        "write a poem about love",
        "who is the president of USA?",
        "translate this to spanish",
        "summary of the news",
        "best restaurants near me",
        "how do I cook pasta?",
        "define artificial intelligence",
        "solve this math problem",
        "chat with me",
        "what is your name?",
        "help me write an email",
        
        # INDONESIAN TEXT REQUESTS
        "bagaimana cuaca hari ini?",
        "ceritakan lelucon",
        "bagaimana cara coding python",
        "jelaskan fisika kuantum",
        "tulis puisi tentang cinta",
        "siapa presiden amerika?",
        "terjemahkan ini ke spanyol",
        "ringkasan berita",
        "restoran terbaik di dekat saya",
        "bagaimana cara memasak pasta?",
        "definisikan kecerdasan buatan",
        "selesaikan masalah matematika ini",
        "mengobrol dengan saya",
        "siapa namamu?",
        "bantu saya menulis email",
        
         # SPANISH TEXT REQUESTS
        "¿qué tiempo hace hoy?",
        "cuéntame un chiste",
        "cómo programar en python",
        "explica la física cuántica",
        "escribe un poema sobre el amor",
        
        # CHINESE TEXT REQUESTS
        "今天天气怎么样？",
        "讲个笑话",
        "如何用python编程",
        "解释量子物理",
        "写一首关于爱的诗"
    ],
    "label": (
        ["IMAGE"] * 15 + ["IMAGE"] * 15 + ["IMAGE"] * 5 + ["IMAGE"] * 5 +
        ["TEXT"] * 15 + ["TEXT"] * 15 + ["TEXT"] * 5 + ["TEXT"] * 5
    )
}

# Create DataFrame
df = pd.DataFrame(data)

# Ensure directory exists
os.makedirs("datasets", exist_ok=True)

# Save to Excel
output_path = "datasets/rAI-beta.xlsx"
df.to_excel(output_path, index=False)

print(f"Dataset created at {output_path} with {len(df)} samples.")
