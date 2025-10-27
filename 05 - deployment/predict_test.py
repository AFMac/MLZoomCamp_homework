import requests

url = "http://127.0.0.1:9696/predict"

#customer = {
#    "gender": "female",
#    "seniorcitizen": 0,
#    "partner": "yes",
#    "dependents": "no",
#    "phoneservice": "no",
#    "multiplelines": "no_phone_service",
#    "internetservice": "dsl",
#    "onlinesecurity": "no",
#    "onlinebackup": "yes",
#    "deviceprotection": "no",
#    "techsupport": "no",
#    "streamingtv": "no",
#    "streamingmovies": "no",
#    "contract": "month-to-month",
#    "paperlessbilling": "yes",
#    "paymentmethod": "electronic_check",
#    "tenure": 1,
#    "monthlycharges": 29.85,
#    "totalcharges": 29.85
#}

client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}
print(requests.post(url, json = client).json())
