import pickle

def main():
    model_file = 'pipeline_v1.bin'

    with open(model_file, 'rb') as f_in:
        pipeline = pickle.load(f_in)

    test = {
        "lead_source": "paid_ads",
        "number_of_courses_viewed": 2,
        "annual_income": 79276.0
    }

    y_pred = pipeline.predict_proba(test)[0,1]

    print(f'Probability of conversion = {y_pred}')

if __name__ == "__main__":
    main()
