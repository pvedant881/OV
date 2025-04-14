from flask import Flask, request, render_template_string, session
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from urllib.parse import urljoin
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from markupsafe import Markup

# --- Setup Gemini API ---
GEMINI_API_KEY = 'AIzaSyBF77bxroQkBHJ2Q1PhUlHtJLb8yhruVi8'  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

app = Flask(__name__)

# --- Define local file paths ---
file_paths = [
    'data/BB_BOS_vinyl_banner_category_data_nitinji (3).csv',
    'data/BB_US_stands_customflags_product_data (1).csv',
    'data/Production Support Sheet -2023.xlsx',
    'data/All Questions.xlsx'
]

# --- Websites to crawl ---
websites = [
    "https://www.bannerbuzz.com/",
    "https://www.coversandall.com/"
]

# --- Read local files ---
def read_file(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            return f"Unsupported file format: {file_path}"
        return f"\nData from {os.path.basename(file_path)}:\n{df.head(10).to_string(index=False)}\n"
    except Exception as e:
        return f"Error reading {file_path}: {e}"

# --- Crawl website content deeply (up to 100 pages) ---
def crawl_website(base_url, max_pages=1000):
    visited = set()
    to_visit = [base_url]
    contents = []

    headers = {"User-Agent": "Mozilla/5.0"}
    count = 0

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = " ".join(soup.stripped_strings)
            contents.append(f"--- Content from {url} ---\n{text[:3000]}")
            visited.add(url)
            count += 1

            # Progress logging
            if count % 100 == 0:
                print(f"[{count}] pages crawled...")

            # Add new internal links
            for link in soup.find_all('a', href=True):
                full_url = urljoin(base_url, link['href'])
                if base_url in full_url and full_url not in visited and full_url not in to_visit:
                    to_visit.append(full_url)
        except Exception as e:
            contents.append(f"Error scraping {url}: {e}")
    return "\n\n".join(contents)
    
def markdown_to_html(text):
    # Convert markdown-style links to clickable HTML
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', text)
    # Convert raw URLs to links (optional)
    text = re.sub(r'(https?://\S+)', r'<a href="\1" target="_blank">\1</a>', text)
    # Replace newlines with <br> for HTML line breaks
    text = text.replace('\n', '<br>')
    return Markup(text)

# --- Function to split large data into chunks based on relevance using TF-IDF ---
def chunk_data(data, user_query, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    all_texts = [user_query] + data
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    top_indices = cosine_sim[0].argsort()[-top_n:][::-1]
    return [data[i] for i in top_indices]

# --- Prepare full dataset ---
def prepare_data():
    all_data = []
    print("Reading local files...")
    for file_path in file_paths:
        all_data.append(read_file(file_path))

    print("Crawling websites (this may take time)...")
    for website in websites:
        all_data.append(crawl_website(website))

    print("✅ All data loaded.")
    return all_data

# --- Cache combined data before Flask starts ---
print("🔁 Preparing data, please wait...")
start_time = time.time()
combined_data = prepare_data()
end_time = time.time()
print(f"\n✅ Data ready in {int(end_time - start_time)} seconds.")

# --- Set up session for history ---
app.secret_key = 'your_secret_key'  # Replace with a secure secret key

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'history' not in session:
        session['history'] = []
        # Send opening message when user first visits
        welcome_message = """👋 Hi there! Welcome to OneVoice! Thank you so much for stopping by. 😊
I’m your friendly assistant here to help you find the perfect product, share accurate info, and make your journey smooth and delightful—every step of the way! 🎯

Whether you’re looking for product details, images, pricing, or just need a quick recommendation—we’ve got you covered. 💬✨

Just ask your question, and I’ll fetch the best info straight from our trusted knowledge base. 🔍 Let’s get started! 🚀

**PS:** If you are looking for SKUs, only refer to the numbers."""
        session['history'].append({'role': 'bot', 'text': welcome_message})

    answer = ""

    if request.method == 'POST':
        question = request.form.get('question')

        # Chunk data intelligently based on user's query
        relevant_chunks = chunk_data(combined_data, question)

        # Add to history
        session['history'].append({'role': 'user', 'text': question})

        # Keep last 3 turns of conversation (user-bot)
        recent_history = session['history'][-6:]  # 3 user-bot pairs

        # Build context
        conversation_context = ""
        for h in recent_history:
            role = "User" if h['role'] == 'user' else "Bot"
            conversation_context += f"{role}: {h['text']}\n"

        prompt = f"""
        You are a smart, customer-obsessed support assistant bot trained to help human agents across our 11 personalized print signage, display, and custom cover e-commerce businesses.
 
Your job is to support agents during live interactions with customers — across **calls, chats, and emails** — by offering well-structured, empathetic, and actionable responses that:
 
1. Reflect the tone required for the medium (concise for chat, guided for call scripts, minimalistic but complete for emails).
2. Pull information from our connected **website(s)**, **product catalogs**, and **knowledge base**.
3. Include the following when recommending or explaining products:
   - Product title & description  
   - Starting price or price range  
   - URL to the product page  
   - Product image (if available)  
   - Key features and benefits (in bullet or table format if comparing multiple products)
 
---
 
When the human agent asks you a question (on behalf of a customer), follow this decision tree:
 
**1. Identify the business line** (from the website or selected input) and route the response accordingly.  
**2. Detect if the customer query is:**
   - **Pre-sales** (e.g., product inquiry, comparison, feature question) → Be informative, recommend upsell/cross-sell products.
   - **Post-sales** (e.g., delivery issue, product defect, size concern) → Be empathetic, offer apology where needed, and guide to resolution.
   - **Troubleshooting** → Use the internal knowledge base or product guides.
   - **Unknown or unsupported query** → Acknowledge, and advise the agent to create a ticket or escalate to the right department without guessing.
 
---
 
### 🎯 Tone Guidelines by Channel:
- **Chat**: Keep it brief, helpful, and friendly. Include clickable product link, image, and price if relevant.
- **Email**: Use a professional and appreciative tone. Slightly detailed, use bulleted info if needed.
- **Phone Call Script**: Begin with gratitude or apology, give a direct and empathetic explanation, suggest a solution, and optionally include cross-sell if pre-sales.
 
---
 
### 🔁 Sample Agent Questions:
- "The customer wants to know the difference between vinyl banner and mesh banner."
- "Customer is asking if the sofa cover is waterproof and if it fits L-shaped sofas."
- "Which table covers are heat resistant?"
- "Can you compare options for patio furniture covers under ₹5000?"
- "Customer is asking why their design isn’t uploading properly — what can I tell them?"
 
---
 
### 🔄 Sample Expected Bot Response (for an email):
> Thank you for choosing us! Based on your query, here are two recommended patio furniture covers under ₹5000:  
>
> **1. Waterproof Outdoor Cover**  
> - Price starts at ₹3,299  
> - UV resistant, custom size options  
> - [View Product](https://www.example.com/waterproof-outdoor-cover)  
> - ![Product Image](https://www.example.com/image.jpg)  
>
> **2. Heavy-Duty Fabric Cover**  
> - Price starts at ₹4,799  
> - Windproof straps, 5-year warranty  
> - [View Product](https://www.example.com/heavy-duty-cover)  
> - ![Product Image](https://www.example.com/image2.jpg)
 
If you need installation guidance or bulk quotes, let us know how we can assist further.
 
---
 
### ❌ If you don’t have an answer:
Reply with:
> “I don’t have the exact answer right now. I recommend creating a ticket for the [relevant team/business unit]. Please choose if the customer would prefer a reply over call, chat, or email.”
 
---
 
**Output must never contain assumptions. Only provide what’s verified from product links, catalog, or knowledge source.**
 
Your purpose is to make the agent’s job easier and the customer experience more delightful.

Previous conversation:
{conversation_context}

Current question: "{question}"

Relevant data:
{''.join(relevant_chunks)}

Answer helpfully and clearly:
"""

        try:
            response = model.generate_content(prompt)
            answer = response.text

            # --- Auto convert image URLs to <img> tags ---
            def convert_image_links(text):
                # Match Markdown-style ![desc](url)
                text = re.sub(r'!\[.*?\]\((https?:\/\/[^\s)]+)\)', r'<img src="\1" class="product-image">', text)
                # Match raw image URLs
                text = re.sub(r'(https?:\/\/[^\s]+\.(?:jpg|jpeg|png|gif))', r'<img src="\1" class="product-image">', text)
                return text

            answer = convert_image_links(answer)
            session['history'].append({'role': 'bot', 'text': answer})
        except Exception as e:
            answer = f"Error generating response: {e}"

    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChatBot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            padding: 20px;
        }
        .chat-container {
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .chat-box {
            max-height: 400px;
            overflow-y: auto;
            margin-bottom: 20px;
        }
        .message {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            white-space: pre-wrap; /* Preserve line breaks */
        }
        .user {
            background-color: #d1f7d7;
            text-align: right;
        }
        .bot {
            background-color: #e3e3e3;
            text-align: left;
        }
        .input-container {
            display: flex;
            margin-top: 20px;
        }
        .input-container input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .input-container button {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #4CAF50;
            color: white;
            cursor: pointer;
        }
        .input-container button:hover {
            background-color: #45a049;
        }
        .product-image {
            max-width: 100px;
            max-height: 100px;
            margin: 10px 0;
        }
        a {
            color: #1a0dab;
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <h2>Chat with Our Support Agent</h2>
        <div class="chat-box">
            {% for message in session['history'] %}
                <div class="message {{ message['role'] }}">
                    <strong>{{ message['role'] | capitalize }}:</strong>
                    <div>{% autoescape false %}{{ message['text'] }}{% endautoescape %}</div>
                </div>
            {% endfor %}
        </div>
        <form method="post" class="input-container">
            <input type="text" name="question" placeholder="Ask me anything..." required>
            <button type="submit">Send</button>
        </form>
    </div>
</body>
</html>
    """, answer=answer)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Use PORT from environment variable
    app.run(host='0.0.0.0', port=port, debug=True)
