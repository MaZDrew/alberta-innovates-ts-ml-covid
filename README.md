# Albert Innovates Covid-19

This project was created for https://albertainnovates.ca/impact/newsroom/covid-19-hackathon/.
The aim of this project was to provide predictions of future COVID-19 statistics via machine learning. We decided to make a LSTM (Long Short Term Memory) Model that runs 'n' simulations for each statistic of interest, and then averages them out for the "final" prediction.

The Covid-19 dataset for this project is provided from https://covidapi.info.
A visualization of the data can be found on https://ml-covid.web.app/. 

## Installation

Clone this repository

```bash
$ git clone https://github.com/MaZDrew/alberta-innovates-ts-ml-covid.git
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies

```python
$ pip install -r requirements.txt
```

## Usage

If you wish to write the values to a firebase database you must add your `serviceAccountKey.json` to the root of the project directory, which can be found in your firebase project settings. You must also replace the `databaseURL` inside of `modules/database.py` with the database URL of your firebase project. It will also write the history of the previous day to keep a catalog of all previously ran predictions for comparison and accuracy checking.

To generate predictions for other countries or globally add its country code e.g `CAN` or `global` as a string to the `scopes` array.

Predictions can be generated for: `Deaths`, `Confirmed`, `Recovered`, and `Concurrent`, as well as their respective rates.
`Concurrent` is understood as being equal to `Confirmed` - `Recovered` - `Deaths`

An example of how to use the model can be found in `app.py`
To begin training the model.

```python
$ python app.py
```

## Contributing

All Pull requests are welcome!
This project was created in less then a month, and was very much a learning project, so there are many improvements that can be made.

## License
[MIT](https://choosealicense.com/licenses/mit/)
