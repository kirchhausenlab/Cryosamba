# Cryosamba

## UI

From cryosamba/automate

```bash
pip install streamlit
cd automate
streamlit run main.py
```

<img src="https://github.com/kirchhausenlab/Cryosamba/blob/main/how_to_open_cryosamba.gif" width="800"/>

You can setup the environment, train models, make configs and run inferences on them from here.

## Terminal

### Setup CryoSamba

Please navigate into the cryosamba directory and open it in an IDE (VScode, Pycharm) of your choice and run the following code to setup cryosamba on your machine.

```bash
cd automate/scripts
chmod -R 755 .
```

The above lines install all the necessary packages and dependencies, generate an environment.yml for conda environments, and activate the environment to run the models.

### Training the Model

Stay in the same directory and run the following commands:
`setup_experiment.sh`
The above code will simply ask you for a specific configuration for your model including the locations of your training data and data path, max frame gap, number of iterations etc. You can choose the defaults if you like however beware that you MUST provide the train_dir, data_path and max_frame_gap.
After this you can run:
`train_data.sh`
to train your data. You can pick which GPU to use, or a collection of GPUs.

### Inference

Inference will look very similar to training the model. If you're in the training folder, run the following:

```bash
cd ../ && cd inference
./setup_inference.sh
```

Similar to the training data, the above code will prompt you to generate a configuration for your model. Once done, please run `inference.sh` to get inference data.
