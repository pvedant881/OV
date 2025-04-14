# Token free code
from flask import Flask, request, render_template_string
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from urllib.parse import urljoin
import socket
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    "https://www.coversandall.com/",
    "https://www.bestofsigns.com/"
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

# --- Crawl website content deeply (up to 10,000 pages) ---
def crawl_website(base_url, max_pages=10):
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

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = ""
    if request.method == 'POST':
        question = request.form.get('question')
        
        # Chunk data intelligently based on user's query
        relevant_chunks = chunk_data(combined_data, question)

        prompt = f"""
        You are a smart assistant with access to local business files and website data.

        A user asked: "{question}"

        Based on the data below, provide a helpful and detailed answer:

        Relevant Data:
        {''.join(relevant_chunks)}
        """
        try:
            response = model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            answer = f"Error: {e}"

    return render_template_string("""
    <html>
    <head><title>Smart Business Chatbot</title></head>
    <body style="font-family: Arial; padding: 30px;">
        <h2>Ask a Question about Your Business</h2>
        <form method="post">
            <textarea name="question" rows="4" cols="80" placeholder="e.g., What are common banner sizes?">{{ request.form.question or "" }}</textarea><br><br>
            <input type="submit" value="Ask Gemini" style="padding: 10px 20px;">
        </form>
        {% if answer %}
            <hr><h3>Gemini Answer:</h3>
            <div style="white-space: pre-wrap;">{{ answer }}</div>
        {% endif %}
    </body>
    </html>
    """, answer=answer)

# --- Run server after data is ready ---
if __name__ == '__main__':
    local_ip = socket.gethostbyname(socket.gethostname())
    print(f"\n🌐 Chatbot is running at: http://{local_ip}:5000")
    app.run(host='0.0.0.0', port=5000)
