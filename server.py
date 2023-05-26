
import os
from flask import (Flask, render_template, request,
                   send_from_directory, jsonify)
import pandas as pd
# import tagui as t
import model

# preferences = ""

app = Flask(__name__)

def dataframe(table):
    tup = tuple(table.itertuples(index=False, name=None))
    return tup

@app.route('/')
def index():
    return render_template('index.html')

########## chat Message hangling ####################
@app.route('/message', methods=['GET' , 'POST'])
def message():
    print("Inside webhook")
    # if request.method == "GET":
    #     return "Sample"
    # if request.method == "POST":
        # payload = request.json
    data = request.get_json()
    response = (data['queryResult']['fulfillmentText'])
    # message = data.get('message')
    # print(data)
    print(response)
    
    # print(preferences)
    # if response != "":
    if response.startswith("Okay, here's a summary of your preferences"):
        global preferences
        preferences = response
        # if preferences.startswith("Okay, here's a summary of your preferences, "):
        preferences = preferences[len("Okay, here's a summary of your preferences, "):]
        # if preferences.endswith(" Please confirm below."):
        preferences = preferences[:-len(" Please confirm below.")]
        print ("Long Description")
        print(preferences)
        # start = preferences.find("MRT station") + len("MRT station ")
        # end = preferences.find(".", start)
        # station = preferences[start:end]
        # print(station)

    elif response.startswith("Okay Please wait till I process your request"):
    #     samples = ("https://www.propertyguru.com.sg/listing/hdb-for-rent-111b-alkaff-crescent-24347640",
    #                 "https://www.propertyguru.com.sg/listing/hdb-for-rent-108-potong-pasir-avenue-1-24453302",
    #                 "https://www.propertyguru.com.sg/listing/24399936/for-rent-st-michael-s-place")
        # print("Inside results handling")
        # create the response message
        # fulfillment_message = {
        # "text": samples[0]}
        # # create the webhook response
        # webhook_response = {
        #     "fulfillmentMessages": [fulfillment_message]
        # }
        samples = model.run_process(preferences)
        print("Inside results handling")
        print(preferences)
        print(samples)
        payload = {"Sug1" : samples[0], "Sug2" : samples[1], "Sug3" : samples[2]}

        return jsonify(payload)


    
        # user_response = (payload['queryResult']['queryText'])
        # bot_response = (payload['queryResult']['fulfillmentText'])
        # if bot_response.startswith("Okay, here's a summary"):
        # if user_response or bot_response != "":
            # print (bot_response)
            # print (bot_response)

            # render_template('InfoPage.html')
    
    # else:
        # return render_template('ErrorPage.html')
    return "4000"









if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)