from flask import Flask, request, jsonify
from chatbot import chatbot_response

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if user_message:
        bot_response = chatbot_response(user_message)
        return jsonify({"response": bot_response})
    else:
        return jsonify({"response": "Sorry, I didn't understand that."}), 400

if __name__ == "__main__":
    app.run(debug=True)
