Business needs
Predict the obesity level of people using Classification

Requirements

    python 3.7

    numpy==1.26.4
    pandas==2.2.1
    sklearn==1.4.1.post1

Running:

    To run the demo, in pipeline folder execute:
        python predict.py

    After running the script in models folder will be generated <prediction_results.csv>
    The file has 'ObesityLevel_pred' column with the result value.

    The input is expected  csv file in models folder with a name <new_input.csv>. The file should have all features columns.

Training a Model:

    Before you run the training script for the first time, you will need to create dataset. The file <train_data.csv> should contain all features columns and target for prediction Price. Run split_data.py in src folder to split your dataset into <train_data.csv> and <new_input.csv>.
    After running the script the "param_dict.pickle"  and "finalized_model.saw" will be created in models folder.
    Run the training script in pipeline folder:
        python train.py

    The model accuracy is 72%
    There is no fraud check.
