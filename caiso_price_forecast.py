# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:32:00 2023

@author: lzhao
"""

# import datetime
import holidays
import pandas as pd
import numpy as np
from functools import reduce
from dateutil import relativedelta
import lightgbm as lgb
import pytz

# YesEnergy datatypes.
DATATYPE_DALMP = "DALMP"
DATATYPE_RTLMP = "RTLMP"
DATATYPE_REG_UP = "DAM RU_CLR_PRC"
DATATYPE_REG_DOWN = "DAM RD_CLR_PRC"
DATATYPE_SPINNING_RESERVES = "DAM SP_CLR_PRC"

DATATYPE_GENERATION_SOLAR_DA_SCHEDULE = "GENERATION_SOLAR_DA_SCHEDULE"
DATATYPE_LOAD_FORECAST_2DAY = "LOAD_FORECAST_2DAY"

DATATYPE_RTLOAD = "RTLOAD"
DATATYPE_SOLAR_FORECAST = "SOLAR_FORECAST"
DATATYPE_WIND_DA = "WIND_DA"
DATATYPE_WIND_FORECAST = "WIND_FORECAST"

TZ = pytz.timezone('US/Pacific')


class caiso_price_forecast_base(object):
    def __init__(self, name='caiso price forecast model'):
        self.name = name
        # self.tz = pytz.timezone('US/Pacific')

        self.WEATHER_ROLLING_PERIOD = 24
        self._WEATHER_STATIONS = ["KCQT",
                                  "KEDW",
                                  "KHWD",
                                  "KLGB",
                                  "KMHV",
                                  "KSAN",
                                  "KSBD",
                                  ]

    @property
    def YES_ENERGY_ITEMS(self):
        return {}

    @property
    def YES_ENERGY_TIME_TRANSFORMATIONS(self):
        YES_ENERGY_TIME_TRANSFORMATIONS = {
            "DALMP": [24, 48],
            "RTLMP": [48],
            "CAISO_LOAD_FORECAST_2DAY": [24, 168],
            "PGE_LOAD_FORECAST_2DAY": [24, 168],
            "SCE_LOAD_FORECAST_2DAY": [24, 168],
            "SDGE_LOAD_FORECAST_2DAY": [24, 168],
            "VEA_LOAD_FORECAST_2DAY": [24, 168],
            "CAISO_LOAD_FORECAST_RESIDUAL": [168],
            "PGE_LOAD_FORECAST_RESIDUAL": [168],
            "SCE_LOAD_FORECAST_RESIDUAL": [168],
            "SDGE_LOAD_FORECAST_RESIDUAL": [168],
            "VEA_LOAD_FORECAST_RESIDUAL": [168],
        }
        return YES_ENERGY_TIME_TRANSFORMATIONS

    @property
    def WEATHER_STATIONS(self):
        return self._WEATHER_STATIONS

    @WEATHER_STATIONS.setter
    def WEATHER_STATIONS(self, weather_stations_list):
        self._WEATHER_STATIONS = weather_stations_list

    @property
    def WEATHER_ITEMS(self):
        """


        Returns
        -------
        weather_items : LIST
            use temp/cloud coverage/wind speed as base case model.

        """
        weather_items = []
        for station in self.WEATHER_STATIONS:
            weather_items.extend([f"Temperature ({station})",
                                  f"Cloud Coverage ({station})",
                                  f"Wind Speed ({station})"])
        return weather_items

    @property
    def TIME_ITEMS(self):
        TIME_ITEMS = [
            "Day Of Year",
            "Day Of Week",
            "Epoch",
            "Hour Of Day",
            "Holiday",
        ]
        return TIME_ITEMS

    def holidays_by_year(self):
        holidays_by_year = {}
        for year in range(2015, 2100):
            holidays_by_year.update({year: [hd for hd, hn in holidays.US(years=year).items()]})
        return holidays_by_year

    @property
    def FEATURES(self):
        # base_features = self

        yes_energy_time_transformed_items = []
        for yes_energy_item, parameters in self.YES_ENERGY_TIME_TRANSFORMATIONS.items():
            for param in parameters:
                transformed_features_name = f'{yes_energy_item}_SHIFT_PT{param}H'
                yes_energy_time_transformed_items.append(transformed_features_name)

        weather_transformed_itmes = [f'{weather_item}_ROLLING_MEAN_PT{self.WEATHER_ROLLING_PERIOD}H' for weather_item in
                                     self.WEATHER_ITEMS]

        features = self.BASE_FEATURES + self.WEATHER_ITEMS + yes_energy_time_transformed_items + weather_transformed_itmes + self.TIME_ITEMS

        return features

    def define_calendar_data(self, calendar_ts):
        # Extend 72 hours
        calendar_ts_end = TZ.localize(calendar_ts.DATETIME.iloc[-1])
        extend_ts = pd.DataFrame(
            {'DATETIME': [calendar_ts_end + relativedelta.relativedelta(hours=i) for i in range(1, 73)]})
        extend_ts.DATETIME = extend_ts.DATETIME.dt.tz_convert(TZ)
        extend_ts.DATETIME = extend_ts.DATETIME.apply(lambda c: c.replace(tzinfo=None))

        calendar_ts = pd.concat([calendar_ts, extend_ts], ignore_index=True)

        calendar_ts['Day Of Year'] = calendar_ts.DATETIME.dt.day_of_year
        calendar_ts['Day Of Week'] = calendar_ts.DATETIME.dt.day_of_week
        calendar_ts['Hour Of Day'] = calendar_ts.DATETIME.dt.hour
        calendar_ts['Epoch'] = (calendar_ts.DATETIME - pd.Timestamp('1970-01-01')).dt.total_seconds()
        # Get the unique years in the data
        unique_years = calendar_ts.DATETIME.dt.year.unique()

        # Create a series that maps each date to its corresponding holiday flag
        holiday_flags = pd.Series(dtype=int, index=calendar_ts.index)
        for year in unique_years:
            holiday_dates = self.holidays_by_year()[year]
            mask = (calendar_ts.DATETIME.dt.year == year) & (calendar_ts.DATETIME.dt.date.isin(holiday_dates))
            holiday_flags[mask] = 1

        # Fill any remaining NaN values with 0 (not a holiday)
        holiday_flags = holiday_flags.fillna(0)
        calendar_ts['Holiday'] = holiday_flags
        calendar_data = {time_item: calendar_ts[['DATETIME', time_item]].copy() for time_item in self.TIME_ITEMS}

        return calendar_data

    def apply_time_transformations_on_weather_data(self, weather_data, calendar_ts):
        for weather_item in self.WEATHER_ITEMS:
            weather_ts = weather_data[weather_item].copy()
            weather_ts = calendar_ts[['DATETIME']].merge(weather_ts, on='DATETIME', how='left')
            weather_ts.fillna(method='ffill', inplace=True)

            transformed_features_name = f'{weather_item}_ROLLING_MEAN_PT{self.WEATHER_ROLLING_PERIOD}H'
            transformed_features = weather_ts.copy()
            transformed_features.AVGVALUE = transformed_features.AVGVALUE.rolling(self.WEATHER_ROLLING_PERIOD,
                                                                                  min_periods=1).mean()

            weather_data.update({weather_item: weather_ts})
            weather_data.update({transformed_features_name: transformed_features})
        return weather_data

    def apply_time_transformations_on_yes_energy_data(self, yes_energy_data):
        # Apply yes_energy time transformations.
        for yes_energy_item, parameters in self.YES_ENERGY_TIME_TRANSFORMATIONS.items():
            for param in parameters:
                transformed_features_name = f'{yes_energy_item}_SHIFT_PT{param}H'
                transformed_features = yes_energy_data[yes_energy_item].copy()

                features_end = TZ.localize(transformed_features.DATETIME.iloc[-1])
                extended_set = pd.DataFrame([{'DATETIME': features_end + relativedelta.relativedelta(hours=i),
                                              'AVGVALUE': np.nan} for i in range(1, param + 1)])
                extended_set.DATETIME = extended_set.DATETIME.dt.tz_convert(TZ)
                extended_set.DATETIME = extended_set.DATETIME.apply(lambda c: c.replace(tzinfo=None))

                transformed_features = pd.concat([transformed_features, extended_set],
                                                 ignore_index=True).sort_values(by='DATETIME')

                transformed_features.AVGVALUE = transformed_features.AVGVALUE.shift(param)

                yes_energy_data.update({transformed_features_name: transformed_features})
        return yes_energy_data

    def apply_timestamp_adjust_on_input_data(self, data, calibrate_ts_name='Hour Of Day'):
        data_refine = {}
        calibrate_ts = data[calibrate_ts_name].copy()
        for key, sub_data in data.items():
            if key in self.FEATURES + self.TARGETS:
                if key in sub_data.columns:
                    temp = sub_data[['DATETIME', key]].copy()
                else:
                    temp = sub_data[['DATETIME', 'AVGVALUE']].copy()
                    temp.rename(columns={'AVGVALUE': key}, inplace=True)
                if key != calibrate_ts_name:
                    temp = calibrate_ts[['DATETIME']].merge(temp, on='DATETIME', how='left')
                    temp.ffill(inplace=True)
                temp.DATETIME = temp.DATETIME.apply(lambda c: c - relativedelta.relativedelta(hours=1))
                temp.drop_duplicates(subset=['DATETIME'], keep='first', inplace=True)
                temp.set_index("DATETIME", inplace=True)
                data_refine.update({key: temp})
        return data_refine

    def define_load_residual_data(self, load_actual_name, load_forecast_name, yes_energy_data, calendar_ts):
        load_actual_ts = yes_energy_data[load_actual_name].copy()
        load_forecast_ts = yes_energy_data[load_forecast_name].copy()

        load_actual_ts.rename(columns={'AVGVALUE': 'AVGVALUE_ACTUAL'}, inplace=True)
        load_forecast_ts.rename(columns={'AVGVALUE': 'AVGVALUE_FORECASTED'}, inplace=True)

        # load_residual_ts = load_actual_ts.copy()
        load_residual_ts = load_actual_ts.merge(load_forecast_ts, on=['DATETIME', 'TIMEZONE'], how='outer')
        load_residual_ts.ffill(inplace=True)

        load_residual_ts.loc[:, 'AVGVALUE'] = load_residual_ts.apply(
            lambda r: r.AVGVALUE_ACTUAL - r.AVGVALUE_FORECASTED,
            axis=1)

        load_residual_ts = calendar_ts[['DATETIME']].merge(load_residual_ts, on='DATETIME', how='left').ffill()

        load_residual_ts.drop_duplicates(subset=["DATETIME", "TIMEZONE"], inplace=True)
        return load_residual_ts


class caiso_as_price_forecast(caiso_price_forecast_base):
    def __init__(self, name='as forecast'):
        super().__init__(name)

    @property
    def YES_ENERGY_ITEMS(self):

        # YesEnergy price objects.
        PRICE_NODE = 20000004682  # TH_SP15_GEN-APND
        HUB_AS_CAISO = 10002484314
        HUB_AS_CAISO_EXP = 10002484315
        HUB_AS_SP26 = 10002484316
        HUB_AS_SP26_EXP = 10002484317

        # YesEnergy load objects.
        ZONE_CAISO = 10000328798
        ZONE_PGE = 10000328796
        ZONE_SCE = 10000328795
        ZONE_SDGE = 10000328797
        ZONE_VEA = 10002423972

        # YesEnergy generation objects.
        ZONE_NP15 = 10002494909
        ZONE_SP15 = 10002494908

        YES_ENERGY_ITEMS = {
            # Energy prices.
            "DALMP": (DATATYPE_DALMP, PRICE_NODE),
            "RTLMP": (DATATYPE_RTLMP, PRICE_NODE),
            # Ancillary services prices.
            "REG_UP_CAISO": (DATATYPE_REG_UP, HUB_AS_CAISO),
            "REG_DOWN_CAISO": (DATATYPE_REG_DOWN, HUB_AS_CAISO),
            "SPINNING_RESERVES_CAISO": (DATATYPE_SPINNING_RESERVES, HUB_AS_CAISO),
            "REG_UP_CAISO_EXP": (DATATYPE_REG_UP, HUB_AS_CAISO_EXP),
            "REG_DOWN_CAISO_EXP": (DATATYPE_REG_DOWN, HUB_AS_CAISO_EXP),
            "SPINNING_RESERVES_CAISO_EXP": (DATATYPE_SPINNING_RESERVES, HUB_AS_CAISO_EXP),
            "REG_UP_SP26": (DATATYPE_REG_UP, HUB_AS_SP26),
            "REG_DOWN_SP26": (DATATYPE_REG_DOWN, HUB_AS_SP26),
            "SPINNING_RESERVES_SP26": (DATATYPE_SPINNING_RESERVES, HUB_AS_SP26),
            "REG_UP_SP26_EXP": (DATATYPE_REG_UP, HUB_AS_SP26_EXP),
            "REG_DOWN_SP26_EXP": (DATATYPE_REG_DOWN, HUB_AS_SP26_EXP),
            "SPINNING_RESERVES_SP26_EXP": (DATATYPE_SPINNING_RESERVES, HUB_AS_SP26_EXP),
            
            # Load forecasts.
            "CAISO_LOAD_FORECAST_2DAY": (DATATYPE_LOAD_FORECAST_2DAY, ZONE_CAISO),
            "PGE_LOAD_FORECAST_2DAY": (DATATYPE_LOAD_FORECAST_2DAY, ZONE_PGE),
            "SCE_LOAD_FORECAST_2DAY": (DATATYPE_LOAD_FORECAST_2DAY, ZONE_SCE),
            "SDGE_LOAD_FORECAST_2DAY": (DATATYPE_LOAD_FORECAST_2DAY, ZONE_SDGE),
            "VEA_LOAD_FORECAST_2DAY": (DATATYPE_LOAD_FORECAST_2DAY, ZONE_VEA),
            # Actual load.
            "CAISO_RTLOAD": (DATATYPE_RTLOAD, ZONE_CAISO),
            "PGE_RTLOAD": (DATATYPE_RTLOAD, ZONE_PGE),
            "SCE_RTLOAD": (DATATYPE_RTLOAD, ZONE_SCE),
            "SDGE_RTLOAD": (DATATYPE_RTLOAD, ZONE_SDGE),
            "VEA_RTLOAD": (DATATYPE_RTLOAD, ZONE_VEA),
            # Scheduled renewable generation.
            "NP15_GENERATION_SOLAR_DA_SCHEDULE": (DATATYPE_GENERATION_SOLAR_DA_SCHEDULE, ZONE_NP15),
            "NP15_WIND_DA": (DATATYPE_WIND_DA, ZONE_NP15),
            "SP15_GENERATION_SOLAR_DA_SCHEDULE": (DATATYPE_GENERATION_SOLAR_DA_SCHEDULE, ZONE_SP15),
            "SP15_WIND_DA": (DATATYPE_WIND_DA, ZONE_SP15),
            # Forecasted renewable generation.
            "NP15_SOLAR_FORECAST": (DATATYPE_SOLAR_FORECAST, ZONE_NP15),
            "NP15_WIND_FORECAST": (DATATYPE_WIND_FORECAST, ZONE_NP15),
            "SP15_SOLAR_FORECAST": (DATATYPE_SOLAR_FORECAST, ZONE_SP15),
            "SP15_WIND_FORECAST": (DATATYPE_WIND_FORECAST, ZONE_SP15),
        }

        return YES_ENERGY_ITEMS

    @property
    def YES_ENERGY_TIME_TRANSFORMATIONS(self):
        YES_ENERGY_TIME_TRANSFORMATIONS = {
            "DALMP": [24, 48],
            "RTLMP": [48],
            "REG_UP": [24, 48, 72],
            "REG_DOWN": [24, 48, 72],
            "SPINNING_RESERVES": [24, 48, 72],
            "CAISO_LOAD_FORECAST_2DAY": [24, 168],
            "PGE_LOAD_FORECAST_2DAY": [24, 168],
            "SCE_LOAD_FORECAST_2DAY": [24, 168],
            "SDGE_LOAD_FORECAST_2DAY": [24, 168],
            "VEA_LOAD_FORECAST_2DAY": [24, 168],
            "CAISO_LOAD_FORECAST_RESIDUAL": [168],
            "PGE_LOAD_FORECAST_RESIDUAL": [168],
            "SCE_LOAD_FORECAST_RESIDUAL": [168],
            "SDGE_LOAD_FORECAST_RESIDUAL": [168],
            "VEA_LOAD_FORECAST_RESIDUAL": [168],

        }


        return YES_ENERGY_TIME_TRANSFORMATIONS

    @property
    def BASE_FEATURES(self):
        base_features = [
            "CAISO_LOAD_FORECAST_2DAY",
            "PGE_LOAD_FORECAST_2DAY",
            "SCE_LOAD_FORECAST_2DAY",
            "SDGE_LOAD_FORECAST_2DAY",
            "VEA_LOAD_FORECAST_2DAY",
            "NP15_GENERATION_SOLAR_DA_SCHEDULE_OR_FORECAST",
            "NP15_WIND_DA_OR_FORECAST",
            "SP15_GENERATION_SOLAR_DA_SCHEDULE_OR_FORECAST",
            "SP15_WIND_DA_OR_FORECAST",
        ]
        return base_features

    @property
    def TARGETS(self):
        return ['REG_UP', 'REG_DOWN', 'SPINNING_RESERVES']

    def define_data(self, yes_energy_data):
        """


        Parameters
        ----------
        yes_energy_data : DICT
            dict of yes energy data {yes_energy_item:data} .
        weather_data : DICT
            dict of weather data .

        Returns
        -------
        formated input and target data.

        """

        calendar_ts = yes_energy_data['DALMP'][['DATETIME']].copy()  ##use dalmp as example
        calendar_data = self.define_calendar_data(calendar_ts)

        # Since the scheduled datatypes may not be available in time for the model to use them,
        #   we use layering to set up the corresponding forecasts as a backup.
        for zone in ("NP15", "SP15"):
            for scheduled, forecasted in (
                    (DATATYPE_GENERATION_SOLAR_DA_SCHEDULE, DATATYPE_SOLAR_FORECAST),
                    (DATATYPE_WIND_DA, DATATYPE_WIND_FORECAST),
            ):
                # Pop the un-layered time series objects.
                scheduled_name = f"{zone}_{scheduled}"
                forecasted_name = f"{zone}_{forecasted}"
                scheduled_ts = yes_energy_data.pop(scheduled_name)
                forecasted_ts = yes_energy_data.pop(forecasted_name).copy()

                scheduled_ts.rename(columns={'AVGVALUE': 'AVGVALUE_SCHEDULED'}, inplace=True)
                forecasted_ts.rename(columns={'AVGVALUE': 'AVGVALUE_FORECASTED'}, inplace=True)

                # Create and insert the layered time series.
                layered_name = f"{scheduled_name}_OR_FORECAST"

                layered_ts = forecasted_ts.merge(scheduled_ts, on='DATETIME', how='outer').sort_values("DATETIME")

                layered_ts.loc[:, 'AVGVALUE'] = layered_ts.apply(
                    lambda r: r.AVGVALUE_FORECASTED if np.isnan(r.AVGVALUE_SCHEDULED) else r.AVGVALUE_SCHEDULED,
                    axis=1)
                layered_ts.drop_duplicates(subset=["DATETIME"], inplace=True)

                layered_name = f"{scheduled_name}_OR_FORECAST"

                yes_energy_data[layered_name] = layered_ts[['DATETIME', 'AVGVALUE']]

        # Compute load forecast residuals as features.
        for zone in ("CAISO", "PGE", "SCE", "SDGE", "VEA"):
            load_actual_name = f"{zone}_{DATATYPE_RTLOAD}"
            load_forecast_name = f"{zone}_{DATATYPE_LOAD_FORECAST_2DAY}"
            load_residual_name = f"{zone}_LOAD_FORECAST_RESIDUAL"

            load_residual_ts = self.define_load_residual_data(load_actual_name, load_forecast_name,
                                                              yes_energy_data,
                                                              calendar_ts=calendar_data[self.TIME_ITEMS[0]])
            yes_energy_data[load_residual_name] = load_residual_ts[['DATETIME', 'AVGVALUE']]

        # Sum reg_up and reg_down prices across hubs.
        reg_up_ts_sequence = ["REG_UP_CAISO", "REG_UP_CAISO_EXP", "REG_UP_SP26", "REG_UP_SP26_EXP"]
        reg_down_ts_sequence = ["REG_DOWN_CAISO", "REG_DOWN_CAISO_EXP", "REG_DOWN_SP26", "REG_DOWN_SP26_EXP"]
        spinning_reserves_ts_sequence = ["SPINNING_RESERVES_CAISO", "SPINNING_RESERVES_CAISO_EXP",
                                         "SPINNING_RESERVES_SP26", "SPINNING_RESERVES_SP26_EXP"]

        # reg_up_ts_set = [yes_energy_data[c]
        temp_set = [yes_energy_data[c].rename(columns={'AVGVALUE': c}) for c in reg_up_ts_sequence]
        reg_up_sum = reduce(lambda l, r: l.merge(r, on=['DATETIME', 'TIMEZONE'], how='left'), temp_set)
        reg_up_sum.ffill(inplace=True)
        reg_up_sum.loc[:, 'AVGVALUE'] = reg_up_sum.apply(lambda r: sum([r[col] for col in reg_up_ts_sequence]),
                                                         axis=1)
        reg_up_sum = calendar_ts[['DATETIME']].merge(reg_up_sum, on='DATETIME', how='left').ffill()
        yes_energy_data.update({"REG_UP": reg_up_sum})

        temp_set = [yes_energy_data[c].rename(columns={'AVGVALUE': c}) for c in reg_down_ts_sequence]
        reg_down_sum = reduce(lambda l, r: l.merge(r, on=['DATETIME', 'TIMEZONE'], how='left'), temp_set)
        reg_down_sum.ffill(inplace=True)
        reg_down_sum.loc[:, 'AVGVALUE'] = reg_down_sum.apply(lambda r: sum([r[col] for col in reg_down_ts_sequence]),
                                                             axis=1)
        reg_down_sum = calendar_ts[['DATETIME']].merge(reg_down_sum, on='DATETIME', how='left').ffill()
        yes_energy_data.update({"REG_DOWN": reg_down_sum})

        temp_set = [yes_energy_data[c].rename(columns={'AVGVALUE': c}) for c in spinning_reserves_ts_sequence]
        spin_reserve_sum = reduce(lambda l, r: l.merge(r, on=['DATETIME', 'TIMEZONE'], how='left'), temp_set)
        spin_reserve_sum.ffill(inplace=True)
        spin_reserve_sum.loc[:, 'AVGVALUE'] = spin_reserve_sum.apply(
            lambda r: sum([r[col] for col in spinning_reserves_ts_sequence]),
            axis=1)
        spin_reserve_sum = calendar_ts[['DATETIME']].merge(spin_reserve_sum, on='DATETIME', how='left').ffill()
        yes_energy_data.update({"SPINNING_RESERVES": spin_reserve_sum})

        yes_energy_data = self.apply_time_transformations_on_yes_energy_data(yes_energy_data)

        data = {**yes_energy_data, **calendar_data}
        data_refine = self.apply_timestamp_adjust_on_input_data(data)

        return data_refine

    @property
    def model_parameters(self):
        ru_params = {"objective": "mae",
                     "num_boost_round": 900,
                     "max_depth": 8,
                     "min_child_weight": 0,
                     "learning_rate": 0.04,
                     "num_leaves": 80
                     }
        rd_params = {"objective": "mae",
                     "num_boost_round": 600,
                     "max_depth": 8,
                     "min_child_weight": 55,
                     "learning_rate": 0.04,
                     "num_leaves": 80}
        sr_params = {
            "objective": "mae",
            "num_boost_round": 900,
            "max_depth": 8,
            "min_child_weight": 0,
            "learning_rate": 0.04,
            "num_leaves": 80
        }
        return {'RU': ru_params, 'RD': rd_params, 'SR': sr_params}

    @property
    def MODELS(self):
        params = self.model_parameters
        """
        ru_model = lgb.LGBMRegressor(
            objective='mae',
            num_boost_round=900,
            max_depth=8,
            min_child_weight=0,
            learning_rate=0.04,
            num_leaves=80)

        rd_model = lgb.LGBMRegressor(
            objective='mae',
            num_boost_round=600,
            max_depth=8,
            min_child_weight=55,
            learning_rate=0.04,
            num_leaves=80)

        sr_model = lgb.LGBMRegressor(
            objective='mae',
            num_boost_round=900,
            max_depth=8,
            min_child_weight=0,
            learning_rate=0.04,
            num_leaves=80)
        """
        model_dict = {}
        for model_name in ['RU', 'RD', 'SR']:
            model = lgb.LGBMRegressor(**params[model_name])
            model_dict.update({model_name: model})
        return model_dict
