from flask import Flask, jsonify,render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    # Serve the index page
    return render_template('index.html')


@app.route('/run-ml', methods=['GET'])
def run_ml():
    try:
        # Execute your Python code as a subprocess
        subprocess.run(['python', 'runningNew.py'], check=True)
        return jsonify({"message": "ML Code executed successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
