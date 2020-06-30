from itertools import islice

from chartjs.colors import next_color
from chartjs.views.lines import BaseLineChartView
from django.http import HttpResponse
from django.views.generic import TemplateView


def index(request):
    return HttpResponse("Hello, world. You're at the polls index." + \
                        '<li><a href="colors">colors</a></li>' + \
                        '<li><a href="chart">line_chart</a></li>' + \
                        '<li><a href="column_highchart">column_highchart</a></li>')


class LineChartJSONView(BaseLineChartView):
    def get_labels(self):
        """Return 7 labels for the x-axis."""
        return ["January", "February", "March", "April", "May", "June", "July"]

    def get_providers(self):
        """Return names of datasets."""
        return ["Central", "Eastside", "Westside"]

    def get_data(self):
        """Return 3 datasets to plot."""

        return [[75, 44, 92, 11, 44, 95, 35],
                [41, 92, 18, 3, 73, 87, 92],
                [87, 21, 94, 3, 90, 13, 65]]


import os

# import numpy as np
workpath = os.path.dirname(os.path.abspath(__file__))

# with open(os.path.join(workpath, "task8_data.csv"), "r") as f:

import pickle
import numpy as np

with open("../tfidfvector.pkl", "rb") as f:
    tfidfvector = pickle.load(f)

with open("../sentiment_values.pkl", "rb") as f:
    sentiment_values = pickle.load(f)

def cos_sim(x, y):
    return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

tfidf_cossim = [cos_sim(tfidfvector[0], tfidfvector[i+1]) for i in range(5)]

class LineChartJSONView(BaseLineChartView):
    def get_labels(self):
        """ get line numbers top-5"""
        return [1.0 + 1.0*i for i in range(5)]

    def get_providers(self):
        """get feature names"""
        return ["sentiment", "cos_sim(tf-idf)"]

    def get_data(self):
        """ get values """
        return [sentiment_values[1:6], tfidf_cossim]


class ColorsView(TemplateView):
    template_name = "colors.html"

    def get_context_data(self, **kwargs):
        data = super(ColorsView, self).get_context_data(**kwargs)
        data["colors"] = islice(next_color(), 0, 50)
        return data


colors = ColorsView.as_view()
column_highchart = TemplateView.as_view(template_name='column_highchart.html')
line_chart = TemplateView.as_view(template_name='line_chart.html')
line_chart_json = LineChartJSONView.as_view()

if __name__ == "__main__":
    print(tfidfvector[1:5])
    print(sentiment_values[1:5])
    print(tfidf_cossim)