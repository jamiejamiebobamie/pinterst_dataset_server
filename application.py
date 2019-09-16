import boto3

from flask_restplus import Api, Resource, fields
from flask import Flask, jsonify, request, make_response, abort, render_template, redirect, url_for

application = app = Flask(__name__)
api = Api(app, version='1.0', title='Flower Identification API', description='Flower Species Identification')
ns = api.namespace('myNamed_API', description='Methods')
single_parser = api.parser()
single_parser.add_argument('pl', type=float, help= 'petal length')
single_parser.add_argument('pw', type=float, help= 'petal width')
single_parser.add_argument('sl', type=float, help= 'sepal length')
single_parser.add_argument('sw', type=float, help= 'sepal width')

def whatKindOfFlower(PL=None, PW=None, SL=None, SW=None):
    """Takes in four numbers and outputs a dictionary of probability of flower species based around the given input."""
    Flower = {'setosa_percent': 0,
              'virginica_percent':0,
              'versicolor_percent':0
             }

    #Sepal length
    if SL:
        if SL >= 1 and SL <= 1.9:
            Flower['setosa_percent'] += .25
        if SL >= 3 and SL <= 5.1:
            Flower['versicolor_percent'] += .25
        if SL >= 4.5 and SL <= 6.9:
            Flower['virginica_percent'] += .25

    #Sepal width
    if SW:
        if SW >= .1 and SW <= .6:
            Flower['setosa_percent'] += .25
        if SW >= 1 and SW <= 1.8:
            Flower['versicolor_percent'] += .25
        if SW >= 1.4 and SW <= 2.5:
            Flower['virginica_percent'] += .25

    #Petal length
    if PL:
        if PL >= 4.3 and PL <= 5.8:
            Flower['setosa_percent'] += .25
        if PL >= 4.9 and PL <= 7:
            Flower['versicolor_percent'] += .25
        if PL >= 4.9 and PL <= 7.9:
            Flower['virginica_percent'] += .25

    #Petal width
    if PW:
        if PW >= 2.3 and PW <= 4.4:
            Flower['setosa_percent'] += .25
        if PW >= 2 and PW <= 3.4:
            Flower['versicolor_percent'] += .25
        if PW >= 2.2 and PW <= 3.8:
            Flower['virginica_percent'] += .25

    return Flower

@ns.route('/')
class IdentifyFlower(Resource):
    """Identifies flower."""
    @api.doc(parser=single_parser, description='Enter Petal length, Petal width, Sepal length, Sepal width')
    def get(self):
        """hi."""
        args = single_parser.parse_args()
        PL = args.pl
        PW = args.pw
        SL = args.sl
        SW = args.sw
        r = whatKindOfFlower(PL, PW, SL, SW)
        print(r)
        return {'Flower': r}


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=5000)
