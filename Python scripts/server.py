import main

from flask import Flask, request
app = Flask(__name__)


@app.route("/init")
def init():
    filename = request.args.get('filename')
    main.init(file_name=filename)
    return "OK"


@app.route('/predictions', methods=['POST'])
def return_prediction():
    input_json = request.get_json(force=True)
    output = main.make_prediction(input=input_json)
    # return [str(1-output[0]), str(1-output[1])]
    # output = main.make_prediction(input=input_json)
    return str(output)


if __name__ == "__main__":
    app.run()
