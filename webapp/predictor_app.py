import flask
from flask import request
from predictor_api import make_prediction, cols

app = flask.Flask(__name__)


@app.route("/predict", methods=["GET", "POST"])
def predict():

    # x_input, prediction = make_prediction(request.args.to_dict())

    # return flask.render_template('predictor.html',
    #                              x_input = x_input,
    #                              cols = cols,
    #                              prediction = prediction)        

    df, prediction = make_prediction(request.args.to_dict())
    # df = df[cols]
    return flask.render_template('predictor.html',
                                 columns = cols,
                                 df = df,
                                 prediction = prediction)        


if __name__=='__main__':
    app.run(debug=True)

