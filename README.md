# Albert Innovates Covid-19

This project was created for https://albertainnovates.ca/impact/newsroom/covid-19-hackathon/.
The aim of this project was to provide a better understanding of the impacts of Covid-19. We decided to make a LSTM (Long Short Term Memory) Model that can predict the next week of data given some data from the past.

The Covid-19 dataset for this project is provided from https://covidapi.info.

## Installation

Create a new folder and navigate to its directory root

```bash
$ git clone https://github.com/MaZDrew/alberta-innovates-ts-ml-covid.git
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies

```python
pip install -r requirements.txt
```

To begin training the model run the example file

```python
$ python app.py
```

## Usage

If you want to write the values to a firebase database you must add your `serviceAccountKey.json` to the root of the project directory and replace the `databaseURL` inside of `modules/database.py` with the database URL of your firebase project. It will also write the history of the previous day to keep a catalouge of all previous predictions for comparison and accuracy checking.

An example of how to use the model can be found in `app.py`

This example will run data for all available statistics Deaths, Confirmed, Recovered and Active cases (and there rates) for Canada and the United States.

To generate predictions for other countries or globally add its country code e.g `CAN` or `global` as a string to the `scopes` array.

## Contributing

All Pull requests are welcome!
This project was created in less then a month, so there are many improvements that can be made.

## License
[MIT](https://choosealicense.com/licenses/mit/)
