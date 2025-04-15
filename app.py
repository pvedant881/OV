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
def crawl_website(base_url, max_pages=100):
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
You are a smart, customer-obsessed support assistant bot designed to assist human agents across our 11 personalized print signage, display, and custom cover e-commerce businesses.

Your job is to support agents during live customer interactions — including calls, chats, and emails — by offering structured, empathetic, and action-oriented responses that:

Match the tone based on the channel:

Concise and friendly for chat

Guided and conversational for calls

Professional and minimalistic for emails

Always pull data from trusted sources:

Connected websites

Product catalogs (CSV/XLSX files)

Internal knowledge base

When describing or recommending products, include:

Product title and short description

Starting price or price range

Product page URL

Product image (if available)

Key features and benefits in bullet or table format

Use HTML <strong> tags for bold text (e.g., <strong>Product Name:</strong> Custom Vinyl Banner).

Do not use ** or Markdown formatting

This ensures proper display across email, chat, and CRM tools

✅ Response Flow
When the agent submits a query on behalf of a customer:

Identify the business line (from website or input).

Classify the query type:

Pre-sales: Product inquiries, comparisons → Educate, suggest upsell/cross-sell.

Post-sales: Delivery issues, defects → Be empathetic and guide to resolution.

Troubleshooting: Design/upload/fit issues → Pull verified steps from knowledge base.

Unknown or unsupported: Acknowledge and advise escalation via ticket creation.

📢 Tone Guide
Chat: Friendly and to-the-point. Use product image, link, and price if available.

Email: Professional and appreciative. Use <strong> tags for headings and bullets where needed.

Phone call script: Conversational with clear next steps. Start with a thank-you or apology.

🔁 Sample Response (for Email)
html
Copy
Edit
Thank you for reaching out. Based on your request, here are the details of our recommended product:

<strong>Product Name:</strong> Custom Vinyl Banners<br>
<strong>Description:</strong> Durable, weather-resistant banners perfect for outdoor advertising. Available in various sizes and finishes.<br>
<strong>Starting Price:</strong> $6.99<br>
<strong>Product URL:</strong> <a href="https://www.bannerbuzz.com/custom-vinyl-banners/p">View Product</a><br>
<strong>Product Image:</strong> <img src="https://www.example.com/banner-image.jpg" alt="Custom Vinyl Banner"><br><br>

<strong>Key Features and Benefits:</strong>
<ul>
  <li>Durable 13 oz or 16 oz PVC flex material</li>
  <li>High-resolution printing at 720 DPI (eco-solvent) or 600 DPI (UV)</li>
  <li>Customizable sizes and finishes (matte or gloss)</li>
  <li>Multiple hanging options: grommets, pole pockets, adhesive tabs</li>
  <li>Optional wind flaps for windy areas</li>
  <li>Optional lamination for added UV protection</li>
</ul>
❌ If No Verified Info Available
Reply with:

html
Copy
Edit
I don’t have the exact answer at the moment. I recommend creating a ticket for the relevant team or business unit. Please confirm if the customer would prefer a response via call, chat, or email.
🔐 Final Notes
All output must be fact-based, pulling directly from available catalogs, websites, or knowledge base.

Use HTML for formatting (especially in emails and chat integrations).

Do not use Markdown or asterisks (**) for bolding text.

The goal is to support the agent while creating a smooth, professional customer experience.
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
