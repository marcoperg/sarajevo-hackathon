from flask import Flask, jsonify
import random

app = Flask(__name__)

@app.route('/', methods=['GET'])
def get_random():
    random_value = random.random()
    return jsonify({'random_value': int(random_value * 100)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)