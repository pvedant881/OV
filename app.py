from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Do You Like Me?</title>
        <style>
            body {
                font-family: 'Comic Sans MS', cursive, sans-serif;
                margin: 0;
                padding: 0;
                height: 100vh;
                background: linear-gradient(135deg, #f8bbd0, #fce4ec);
                overflow: hidden;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            #questionContainer {
                position: relative;
                width: 90%;
                max-width: 400px;
                height: 330px;
                background-color: #ffffff;
                border: 2px solid #ec407a;
                border-radius: 20px;
                padding: 20px;
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
                text-align: center;
                z-index: 2;
            }
            h2 {
                color: #d81b60;
                font-size: 24px;
                margin-bottom: 40px;
            }
            button {
                position: absolute;
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                color: white;
                font-weight: bold;
                font-size: 16px;
                cursor: pointer;
                transition: transform 0.3s ease;
            }
            #yesBtn {
                background-color: #66bb6a;
                left: 60%;
                top: 180px;
                transform: translateX(-50%);
            }
            #noBtn {
                background-color: #ef5350;
                left: 40%;
                top: 180px;
                transform: translateX(-50%);
            }
            #heartLoader {
                margin-top: 20px;
                font-size: 24px;
                display: none;
            }
            #resultContainer {
                display: none;
                font-size: 24px;
                color: #d81b60;
                margin-top: 20px;
            }
            .heart {
                position: absolute;
                width: 20px;
                height: 20px;
                background: red;
                transform: rotate(-45deg);
                animation: float 6s infinite ease-in-out;
                opacity: 0.7;
                z-index: 1;
            }
            .heart::before,
            .heart::after {
                content: "";
                position: absolute;
                width: 20px;
                height: 20px;
                background: red;
                border-radius: 50%;
            }
            .heart::before {
                top: -10px;
                left: 0;
            }
            .heart::after {
                left: 10px;
                top: 0;
            }
            @keyframes float {
                0% { transform: translateY(0) rotate(-45deg); opacity: 0.7; }
                100% { transform: translateY(-120vh) rotate(-45deg); opacity: 0; }
            }
            @media (max-width: 480px) {
                h2 {
                    font-size: 20px;
                }
                button {
                    padding: 10px 20px;
                    font-size: 14px;
                }
                #questionContainer {
                    height: 310px;
                }
            }
        </style>
    </head>
    <body>
        <div id="questionContainer">
            <h2>Do you like me? 💖</h2>
            <button id="noBtn">No</button>
            <button id="yesBtn">Yes</button>
            <div id="heartLoader">❤️ Loading...</div>
            <div id="resultContainer">💘 Yay! I like you too! 🥰💖</div>
        </div>

        <script>
            for (let i = 0; i < 30; i++) {
                const heart = document.createElement("div");
                heart.className = "heart";
                heart.style.left = Math.random() * window.innerWidth + "px";
                heart.style.animationDelay = Math.random() * 5 + "s";
                heart.style.width = heart.style.height = 10 + Math.random() * 20 + "px";
                document.body.appendChild(heart);
            }

            const noBtn = document.getElementById("noBtn");
            const yesBtn = document.getElementById("yesBtn");
            const heartLoader = document.getElementById("heartLoader");
            const resultContainer = document.getElementById("resultContainer");
            const questionContainer = document.getElementById("questionContainer");

            noBtn.addEventListener("mouseover", () => {
                const newX = Math.floor(Math.random() * (questionContainer.offsetWidth - 100));
                const newY = Math.floor(Math.random() * (questionContainer.offsetHeight - 100));
                noBtn.style.left = newX + "px";
                noBtn.style.top = newY + "px";
            });

            yesBtn.addEventListener("click", () => {
                heartLoader.style.display = "block";
                resultContainer.style.display = "none";
                setTimeout(() => {
                    heartLoader.style.display = "none";
                    resultContainer.style.display = "block";
                }, 2000);
            });
        </script>
    </body>
    </html>
    '''

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
