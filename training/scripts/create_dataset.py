import pandas as pd
import os
import random

# --- CONFIG ---
TARGET_PER_CATEGORY = 300 # 4 categories * 300 = 1200 samples

# --- MULTILINGUAL TEMPLATES ---

# 1. IMAGE TEMPLATES
img_templates = {
    "en": ["create an image of {sub}", "generate a picture of {sub}", "draw {sub}", "illustration of {sub}", "make a photo of {sub} {mod}", "render {sub} {mod}"],
    "id": ["buatkan gambar {sub}", "generate foto {sub}", "lukis {sub}", "ilustrasi {sub}", "buat foto {sub} {mod}", "render {sub} {mod}"],
    "es": ["crea una imagen de {sub}", "genera una foto de {sub}", "dibuja {sub}", "ilustración de {sub} {mod}"],
    "zh": ["生成一张{sub}的图片", "画一个{sub}", "创建{sub}的图像 {mod}", "绘制{sub}"],
    "fr": ["crée une image de {sub}", "génère une photo de {sub}", "dessine {sub}", "illustration de {sub} {mod}"],
    "de": ["erstelle ein bild von {sub}", "generiere ein foto von {sub}", "zeichne {sub}", "illustration von {sub} {mod}"]
}

img_subjects = {
    "en": ["a cat", "a futuristic city", "a dragon", "a mystical forest", "a cyberpunk character", "a spaceship", "a delicious burger", "a mountain landscape", "a superhero", "an anime girl"],
    "id": ["kucing", "kota masa depan", "naga", "hutan mistis", "karakter cyberpunk", "pesawat luar angkasa", "burger lezat", "pemandangan gunung", "superhero", "gadis anime"],
    "es": ["un gato", "una ciudad futurista", "un dragón", "un bosque místico", "un personaje cyberpunk", "una nave espacial", "una hamburguesa", "un paisaje de montaña"],
    "zh": ["猫", "未来城市", "龙", "神秘森林", "赛博朋克角色", "宇宙飞船", "汉堡", "山水画", "超级英雄"],
    "fr": ["un chat", "une ville futuriste", "un dragon", "une forêt mystique", "un personnage cyberpunk", "un vaisseau spatial"],
    "de": ["eine katze", "eine futuristische stadt", "ein drache", "ein mystischer wald", "ein cyberpunk-charakter", "ein raumschiff"]
}

img_modifiers = {
    "en": ["in 4k", "realistic", "oil painting style", "pixel art", "cinematic lighting", "3d render", "studio quality"],
    "id": ["resolusi 4k", "realistis", "gaya lukisan minyak", "pixel art", "pencahayaan sinematik", "render 3d", "kualitas studio"],
    "es": ["en 4k", "realista", "estilo pintura al óleo", "arte de píxeles", "iluminación cinematográfica"],
    "zh": ["4k分辨率", "逼真", "油画风格", "像素艺术", "电影级灯光"],
    "fr": ["en 4k", "réaliste", "style peinture à l'huile", "pixel art", "éclairage cinématique"],
    "de": ["in 4k", "realistisch", "ölmälde-stil", "pixel-art", "kinoreife beleuchtung"]
}

# 2. VIDEO TEMPLATES
vid_templates = {
    "en": ["create a video of {sub}", "make a video about {sub}", "animate {sub}", "generate a clip of {sub} {mod}", "make a gif of {sub}"],
    "id": ["buatkan video {sub}", "bikin video tentang {sub}", "animasikan {sub}", "buat klip {sub} {mod}", "bikin gif {sub}"],
    "es": ["crea un video de {sub}", "haz un video sobre {sub}", "anima {sub}", "genera un clip de {sub}"],
    "zh": ["制作关于{sub}的视频", "生成{sub}的视频", "动画化{sub}", "创建{sub}的剪辑"],
    "fr": ["crée une vidéo de {sub}", "fais une vidéo sur {sub}", "anime {sub}", "génère un clip de {sub}"],
    "de": ["erstelle ein video von {sub}", "mache ein video über {sub}", "animiere {sub}", "generiere einen clip von {sub}"]
}

vid_modifiers = {
    "en": ["slow motion", "looping", "timelapse", "dancing", "running", "exploding"],
    "id": ["gerak lambat", "looping", "timelapse", "menari", "berlari", "meledak"],
    "es": ["cámara lenta", "en bucle", "lapso de tiempo", "bailando", "corriendo"],
    "zh": ["慢动作", "循环", "延时摄影", "跳舞", "奔跑"],
    "fr": ["ralenti", "en boucle", "timelapse", "dansant", "courant"],
    "de": ["zeitlupe", "looping", "zeitraffer", "tanzen", "laufen"]
}

# 3. SEARCH TEMPLATES
search_templates = {
    "en": ["search for {topic}", "google {topic}", "find info about {topic}", "look up {topic}", "who is {topic}", "price of {topic}"],
    "id": ["cari {topic}", "googling {topic}", "cari info {topic}", "cari tahu {topic}", "siapa {topic}", "harga {topic}"],
    "es": ["buscar {topic}", "googlear {topic}", "encontrar información sobre {topic}", "buscar precio de {topic}"],
    "zh": ["搜索{topic}", "谷歌{topic}", "查找关于{topic}的信息", "查询{topic}"],
    "fr": ["rechercher {topic}", "google {topic}", "trouver des infos sur {topic}", "chercher {topic}"],
    "de": ["suche nach {topic}", "google {topic}", "finde infos über {topic}", "nachschlagen {topic}"]
}

search_topics = {
    "en": ["pasta recipes", "bitcoin", "weather today", "tesla stock", "best movies 2024", "python tutorial", "history of rome"],
    "id": ["resep pasta", "bitcoin", "cuaca hari ini", "saham tesla", "film terbaik 2024", "tutorial python", "sejarah romawi"],
    "es": ["recetas de pasta", "bitcoin", "clima hoy", "acciones de tesla", "mejores películas", "tutorial de python"],
    "zh": ["意面食谱", "比特币", "今天天气", "特斯拉股票", "最好的电影", "python教程"],
    "fr": ["recettes de pâtes", "bitcoin", "météo aujourd'hui", "action tesla", "meilleurs films", "tutoriel python"],
    "de": ["pasta rezepte", "bitcoin", "wetter heute", "tesla aktie", "beste filme", "python tutorial"]
}

# 4. TEXT TEMPLATES
text_templates = {
    "en": ["write a {topic}", "tell me a {topic}", "explain {topic}", "how to {topic}", "summary of {topic}", "translate {topic}"],
    "id": ["tulis {topic}", "ceritakan {topic}", "jelaskan {topic}", "cara {topic}", "ringkasan {topic}", "terjemahkan {topic}"],
    "es": ["escribe {topic}", "cuéntame {topic}", "explica {topic}", "cómo {topic}", "resumen de {topic}"],
    "zh": ["写一个{topic}", "讲个{topic}", "解释{topic}", "如何{topic}", "翻译{topic}"],
    "fr": ["écris {topic}", "raconte {topic}", "explique {topic}", "comment {topic}", "résumé de {topic}"],
    "de": ["schreibe {topic}", "erzähle {topic}", "erkläre {topic}", "wie man {topic}", "zusammenfassung von {topic}"]
}

text_topics = {
    "en": ["poem", "joke", "story", "email", "code", "essay", "article", "letter"],
    "id": ["puisi", "lelucon", "cerita", "email", "kode", "esai", "artikel", "surat"],
    "es": ["poema", "chiste", "historia", "correo", "código", "ensayo"],
    "zh": ["诗", "笑话", "故事", "邮件", "代码", "文章"],
    "fr": ["poème", "blague", "histoire", "email", "code", "essai"],
    "de": ["gedicht", "witz", "geschichte", "email", "code", "aufaufsatz"]
}

def generate_sentence(lang, type_dict, subject_dict, mod_dict=None):
    if lang not in type_dict: lang = "en"
    template = random.choice(type_dict[lang])
    
    sub = random.choice(subject_dict.get(lang, subject_dict["en"]))
    
    res = template.replace("{sub}", sub).replace("{topic}", sub)
    
    if "{mod}" in res and mod_dict:
        mod = random.choice(mod_dict.get(lang, mod_dict["en"]))
        res = res.replace("{mod}", mod)
    
    return res.replace("{mod}", "") # Clean up unused modifiers

def create_dataset():
    data = []
    langs = ["en", "id", "es", "zh", "fr", "de"]
    
    print(f"Generating multilingual dataset (Target: {TARGET_PER_CATEGORY} per category)...")

    # Generate samples
    for _ in range(TARGET_PER_CATEGORY):
        lang = random.choice(langs)
        
        # IMAGE
        data.append({"text": generate_sentence(lang, img_templates, img_subjects, img_modifiers), "label": "IMAGE"})
        
        # VIDEO
        data.append({"text": generate_sentence(lang, vid_templates, img_subjects, vid_modifiers), "label": "VIDEO"})
        
        # SEARCH
        data.append({"text": generate_sentence(lang, search_templates, search_topics), "label": "SEARCH"})
        
        # TEXT
        data.append({"text": generate_sentence(lang, text_templates, text_topics), "label": "TEXT"})

    df = pd.DataFrame(data)
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, "rAI-beta.xlsx")
    
    df.to_excel(output_path, index=False)
    print(f"Generated {len(df)} samples at {output_path}")

if __name__ == "__main__":
    create_dataset()
