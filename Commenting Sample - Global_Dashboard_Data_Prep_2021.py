import pandas as pd
import numpy as np
import os
from datetime import datetime
import re
from ScrapeData import ZipScrape
import ScrapeData
import logging


"""
-------------------------------------------------------------------------------
                                    CONFIG
-------------------------------------------------------------------------------
"""

full_date = datetime.today().strftime("%d-%b-%Y")
y_date = datetime.today().year
# Set the path to where the data is located, by default the Scraper will put data in the location below
# Need to be connected to the vpn to access the folder where the agg_long data is
path = os.environ['ONEDRIVE'] + f"\\Documents\\Data\\{y_date}\\BIS\\bis_raw\\"
agg_long_path = "PATH_TO_FILE/aggregated_long.csv"
save_path = path + "transformed\\"

# Checks to see if the directory exists for the save_path, if not creates it
ScrapeData.check_dir(save_path)

logging.basicConfig(filename="BIS_Data.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


"""
-------------------------------------------------------------------------------
                                    BIS DATA
-------------------------------------------------------------------------------
"""

def compare_file_date(file):
    """
    Compares the date of the BIS .csv to see if we need to get a new file
    :type file: str
    :return: boolean
    """
    file_date = datetime.fromtimestamp(os.stat(path + file).st_ctime).strftime('%Y-%m')  # file created date
    compare_date = datetime.today().strftime("%Y-%m")
    # Compares the date by both year and month, ensures file is updated bi-annually
    if file_date < compare_date and ((file_date[-2:] >= '06' and compare_date[-2:] < '06') or
                                     (file_date[-2:] < '06' and compare_date[-2:] >= '06')):
        return False
    else:
        return True


def get_latest_file(files):
    """
    Gets the most recent BIS .csv (by created date) from a list of all .csv files in the "path" defined above
    :type files: list
    :return: string
    """
    ret_file = ""
    for i in range(len(files) - 1):
        if ret_file:
            if os.stat(path + files[i]).st_ctime > os.stat(path + ret_file).st_ctime:
                ret_file = files[i]
            else:
                ret_file = ret_file
        elif os.stat(path + files[i]).st_ctime > os.stat(path + files[i + 1]).st_ctime:
            ret_file = files[i]
        else:
            ret_file = files[i + 1]
    return ret_file


class CleanAndPrepBISData(ZipScrape):

    def __init__(self):
        super().__init__()
        self.data_list = [x for x in os.listdir(path) if x.find(".csv")]

    def get_csv(self):
        for tries in range(2):
            # Creates a list of all csv files if the file has the extension .csv
            csv = [x for x in self.data_list if x.split(".")[-1] == "csv"]
            if not csv:
                # Calls the ZipScrape class in ScrapeData.py to get the BIS data from the web
                super().get_data()
            else:
                file = get_latest_file(csv)
                if compare_file_date(file):
                    file_data = pd.read_csv(path + file)
                    return file_data
                else:
                    super().get_data()

    def transform_data(self):
        df = self.get_csv()
        df = pd.melt(df, id_vars=["Derivatives measure", "Derivatives instrument", "Derivatives risk category",
                                  "Derivatives reporting country", "Derivatives counterparty sector",
                                  "Derivatives counterparty country", "DER_CURR_LEG1", "DER_CURR_LEG2",
                                  "Derivatives maturity"],
                     value_vars=["1998-S1", "1998-S2", "1999-S1", "1999-S2", "2000-S1", "2000-S2", "2001-S1",
                                 "2001-S2", "2002-S1", "2002-S2", "2003-S1", "2003-S2", "2004-S1", "2004-S2",
                                 "2005-S1", "2005-S2", "2006-S1", "2006-S2", "2007-S1", "2007-S2", "2008-S1",
                                 "2008-S2", "2009-S1", "2009-S2", "2010-S1", "2010-S2", "2011-S1", "2011-S2",
                                 "2012-S1", "2012-S2", "2013-S1", "2013-S2", "2014-S1", "2014-S2", "2015-S1",
                                 "2015-S2", "2016-S1", "2016-S2", "2017-S1", "2017-S2", "2018-S1", "2018-S2",
                                 "2019-S1", "2019-S2", "2020-S1", "2020-S2"],
                     var_name="Date", value_name="Notional Value")

        # Delete S1 and S2 from the dates and apply Jan and Jul to the end
        new_dates = [
            (lambda x, y: x + "-01-01" if y == "S1" else x + "-07-01")(date.split("-")[0], date.split("-")[1])
            for date in df['Date']
            ]
        # Replace columns in value_data with new columns
        df['Date'] = new_dates
        return df

    def clean_dataset(self):
        # Drop columns and rows based on the data that is needed to be retained
        clean_data = self.transform_data()
        clean_data.dropna(subset=['Notional Value'], inplace=True)
        clean_data = clean_data[clean_data['Derivatives measure'] == 'Notional amounts outstanding']
        clean_data = clean_data[clean_data['Derivatives risk category'] != 'Total (all risk categories)']
        clean_data = clean_data[clean_data['Derivatives counterparty sector'] == 'Total (all counterparties)']
        clean_data = clean_data[clean_data['DER_CURR_LEG1'] == 'TO1']
        clean_data = clean_data[clean_data['DER_CURR_LEG2'] == 'TO1']
        clean_data = clean_data[clean_data['Derivatives maturity'] != 'Total (all maturities)']
        return clean_data

"""
-------------------------------------------------------------------------------
                                    OSC DATA
-------------------------------------------------------------------------------
"""

class CleanAndPrepOSCData:

    def __init__(self):
        self.data = pd.DataFrame()

    def _load_data(self):
        try:
            self.data = pd.read_csv(agg_long_path, parse_dates=['Dates'], index_col=['Dates'])
        except Exception as e:
            print("Error loading aggregated_long file, confirm file location.")
            logger.error(e)

    def _clean_data(self):
        """
            Cleans the dataset:
            1) Drop the data for turnover and then 'Outstanding_/_Turnover' column
            2) Fix the TypeErrors for Notional (some str, some NaN)
            3) Drop NaN and 0.0 from Notional
        """
        if not self.data.empty:
            # 1) Drop turnover
            self.data = self.data.loc[self.data['Outstanding_/_Turnover'] == 'outstanding'].copy()
            self.data.drop(labels='Outstanding_/_Turnover', inplace=True, axis=1)

            # 2) Fix TypeErrors, use RegEx to replace "," in the numbers read as Str, and converts to Float
            self.data['Notional'] = [(lambda x: float(x))(re.sub(r'(,)', '', x) if re.search(r',', str(x)) else x)
                                     for x in self.data['Notional']]

            # 3) Drop NaN and 0.0 from Notional
            self.data.dropna(subset=['Notional'], inplace=True)
            self.data = self.data.loc[self.data['Notional'] != 0.0]

    def _transform_data(self):
        """
            Transformations:
            1) Convert Tenor to match BIS data (ie, Up to and including a 1 year)
            2) Convert Asset_Class to match BIS data (ie, credit --> Credit Derivatives)
            3) Adjust the date to the first of the month (ie, 7-Jul-20 --> 2020-07-01)
            4) Aggregate Outstanding Notional by TR, Asset Class, Adjusted Date, and Maturity
        """
        if not self.data.empty:
            # 1) Convert tenor
            tenors = {'0-3 months': "Up to and including 1 year", '3-6 months': "Up to and including 1 year",
                      '6-12 months': "Up to and including 1 year", '12-24 months': "Over 1 year and up to 5 years",
                      '24-60 months': "Over 1 year and up to 5 years", '60+ months': "Over 5 years", np.NaN: "No Tenor",
                      'unknown months': "No Tenor"}
            self.data['Maturity'] = [(lambda x: tenors[x])(x) for x in self.data['Tenor']]

            # 2) Convert Asset_Class
            asset_class = {'credit': "Credit Derivatives", 'comm': "Commodities", 'equity': "Equity",
                           'fx': "Foreign exchange", 'rates': "Interest rate"}
            self.data['Asset_Class'] = [(lambda x: asset_class[x])(x) for x in self.data['Asset_Class']]

            # 3) Adjust dates
            self.data['Adjusted_Date'] = [(lambda x: datetime(x.year, x.month, 1))(x) for x in self.data.index]

            # 4) Aggregate notional outstanding
            cols_to_keep = ['TR', 'Asset_Class', 'Adjusted_Date', 'Maturity', 'Notional']
            self.data = self.data[cols_to_keep]
            self.data = self.data.groupby(['Adjusted_Date', 'TR',
                                           'Asset_Class', 'Maturity'])['Notional'].agg(['count', 'sum']).reset_index()
            self.data.set_index('Adjusted_Date', inplace=True)

    def get_osc_data(self):
        self._load_data()
        self._clean_data()
        self._transform_data()
        return self.data

"""
-------------------------------------------------------------------------------
                                    EXECUTE
-------------------------------------------------------------------------------
"""

if __name__ == "__main__":
    bis_data = CleanAndPrepBISData().clean_dataset()
    if bis_data:
        print("BIS data successfully cleaned and transformed")
    else:
        print("Error getting BIS dataset")
    osc_data = CleanAndPrepOSCData().get_osc_data()
    if osc_data:
        print("OSC data successfully cleaned and transformed")
    else:
        print("Error getting OSC dataset")
    # Output data to csv in the defined save_path with the date of execution
    bis_data.to_csv(f"{save_path}\\BIS_DATA_{full_date}.csv", index=True)
    osc_data.to_csv(f"{save_path}\\OSC_DATA_{full_date}.csv", index=True)
