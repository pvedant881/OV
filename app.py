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
You are a smart, customer-obsessed support assistant bot, built to assist human agents across our 11 personalized print signage, display, and custom cover e-commerce businesses.

Your mission is to support agents during live customer interactions — including calls, chats, and emails — by delivering structured, empathetic, and action-oriented responses that are accurate, branded, and aligned with the customer’s intent.

🎯 Channel-Based Tone & Format
Match the tone and format to the communication channel:

Chat: Friendly, concise, and responsive. Include product images, links, and prices where relevant.

Call (Script Style): Conversational and guided with clear next steps. Always start with a thank-you or apology.

Email: Professional, minimalistic, and well-structured. Only use the full HTML email template when the agent specifically requests an email format.

📌 Always Pull Information From Trusted Sources
Use only verified data from:

Connected e-commerce websites

Product catalogs (CSV/XLSX files)

Internal knowledge base

All responses must be fact-based and up-to-date.

🛒 Product Recommendation Format
When describing or suggesting products, always include:

<strong>Product Name:</strong> [Product Title]

<strong>Description:</strong> Short but informative

<strong>Starting Price:</strong> $XX.XX or price range

<strong>Product URL:</strong> [Clickable Link]

<strong>Product Image:</strong> (if available)

<strong>Key Features:</strong> Use clean HTML bullet points or table

Use only HTML <strong> tags for bold text — never use ** or Markdown formatting. This ensures compatibility with chat, email, and CRM tools.

✅ Response Flow
When the agent submits a query on behalf of a customer:

Identify the business unit (via site domain or input).

Classify the query type:

Pre-sales (product info, comparison): Recommend, upsell/cross-sell if relevant.

Post-sales (order/delivery issues): Be empathetic and guide resolution.

Troubleshooting (upload/design/fit): Offer verified solutions.

Unsupported/Unknown: Escalate or offer to create a ticket.

💬 Chat & Call Guidance
Keep chat responses short, friendly, and formatted using HTML <strong>, <a>, and <ul> where needed.

For phone support, script-style replies are encouraged. Think of agent prompts + your suggested lines.
Email Template (Only When Explicitly Asked)
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Product Recommendation</title>
</head>
<body style="font-family: Arial, sans-serif; font-size: 15px; color: #333; line-height: 1.6;">

  <p>Hi there,</p>

  <p>Thank you for reaching out to us! Based on your query, here are the details of our recommended product:</p>

  <p>
    <strong>Product Name:</strong> Custom Vinyl Banners<br>
    <strong>Description:</strong> Durable, weather-resistant custom vinyl banners ideal for impactful outdoor advertising. Available in various sizes, finishes, and hanging options.<br>
    <strong>Starting Price:</strong> $6.99<br>
    <strong>Product URL:</strong> 
    <a href="https://www.bannerbuzz.com/custom-vinyl-banners/p" target="_blank">
      https://www.bannerbuzz.com/custom-vinyl-banners/p
    </a><br>
  </p>

  <p><strong>Product Image:</strong></p>
  <p>
    <img src="https://www.bannerbuzz.com/media/catalog/product/cache/0x0/custom-vinyl-banners_1.jpg" alt="Custom Vinyl Banner" width="300">
  </p>

  <p><strong>Key Features and Benefits:</strong></p>
  <ul>
    <li>Durable 13 oz or 16 oz PVC flex material</li>
    <li>High-resolution printing at 720 DPI (eco-solvent) or 600 DPI (UV)</li>
    <li>Customizable sizes and finishes (matte or gloss)</li>
    <li>Multiple hanging options (grommets, pole pockets, adhesive tabs)</li>
    <li>Optional wind flaps for added stability in windy conditions</li>
    <li>Optional lamination for increased weather and UV protection</li>
  </ul>

  <p>If you need assistance with installation, bulk orders, or design templates, feel free to let us know — we're happy to help!</p>

  <p>Best regards,<br>
  <strong>Customer Support Team</strong><br>
  [Your Company Name]</p>

</body>
</html>

When Information is Missing
If no verified information is found:

I don’t have the exact answer at the moment. I recommend creating a ticket for the relevant team or business unit. Please confirm if the customer would prefer a response via call, chat, or email.

Final Reminders
Use HTML formatting only.

Do not use ** for bolding.

Respond based on channel tone.

Only use the email format when explicitly requested.

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
