import datetime
import warnings
from decimal import Decimal, ROUND_HALF_UP

import pymannkendall as mk

from data_processing.grading_functions import *
from data_processing.instaBeats_functions import *

warnings.filterwarnings("ignore")


def extract_df_date_range(df, strt_d, end_d, min_d, max_d):
    if strt_d != "" and end_d != "":
        if (strt_d < min_d and end_d > max_d) or (strt_d < min_d and end_d < min_d) or (
                strt_d > max_d and end_d > max_d) or (strt_d > max_d and end_d < min_d):
            strt_d = min_d
            end_d = max_d
            df_test = df
            df = df[df['Date'] >= strt_d]
            df = df[df['Date'] <= end_d]
            if df.shape[0] < 10:
                return df_test
            return df
        else:
            df = df[df['Date'] >= strt_d]
            df = df[df['Date'] <= end_d]
            return df
    elif strt_d != "":
        if strt_d < min_d:
            strt_d = min_d
        end_d = max_d
        df_test = df
        df = df[df['Date'] >= strt_d]
        df = df[df['Date'] <= end_d]
        if df.shape[0] < 10:
            return df_test
        return df
    elif end_d != "":
        df_test = df
        if end_d > max_d:
            end_d = max_d
        strt_d = min_d
        df = df[df['Date'] >= strt_d]
        df = df[df['Date'] <= end_d]
        if df.shape[0] < 10:
            return df_test
        return df
    else:
        return df


def extract_ehi_columns(df, patient_dataframe, act_code, start_date, end_date, usr, wndw_sz):
    df = df.dropna(subset=['code'])
    df = df[['subject', 'effectiveDateTime', 'code', 'valueQuantity', 'resourceType']]
    df = pd.concat([df.drop(['subject'], axis=1), df['subject'].apply(pd.Series)], axis=1)
    # inserting column at appropriate place
    ref = df['reference']
    df.drop(labels=['reference'], axis=1, inplace=True)
    df.insert(0, 'subject_reference', ref)

    df['effectiveDateTime'] = df['effectiveDateTime'].apply(lambda x: x[0:x.find('.')])
    df['effectiveDateTime'] = pd.to_datetime(df['effectiveDateTime'])
    df['effectiveDateTime'] = df['effectiveDateTime'].apply(lambda x: x.replace(microsecond=0, second=0))

    df_test = df['code'].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    # drop the not required columns
    df.drop(labels=['code', 'text'], axis=1, inplace=True)
    df_test = df['coding'].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    # drop the not required columns
    df.drop(labels=['coding'], axis=1, inplace=True)
    df_test = df[0].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    # drop the not required columns
    df.drop(labels=['system', 0], axis=1, inplace=True)
    df.rename(columns={'code': 'activity_code'}, inplace=True)
    df = pd.concat([df, df['valueQuantity'].apply(pd.Series)], axis=1)
    # inserting column at appropriate place
    ref = df['value']
    df.insert(3, 'obs', ref)
    df.drop(labels=['value', 'valueQuantity', 'unit', 'system', 'code'], axis=1, inplace=True)

    df['subject_reference'] = df['subject_reference'].astype("str")
    df['subject_reference'] = df['subject_reference'].apply(
        lambda x: x.replace("Patient/", "", 1) if x.startswith("Patient/") else x)
    df = df[df.activity_code == act_code]

    ref_date = df['effectiveDateTime'].dt.date
    ref_time = df['effectiveDateTime'].dt.time

    df.insert(3, "Date", ref_date)
    df.insert(4, "Time", ref_time)
    date_min = min(df['Date'])
    date_max = max(df['Date'])
    # print("EHI_COL")
    # print("Size of df before date filtering:", df.shape)
    df = extract_df_date_range(df, start_date, end_date, date_min, date_max)
    # print("Size of df after date filtering:", df.shape)
    round_lst = lambda lst: [round(x, 2) for x in lst]

    if usr == 'avg':
        # average
        avg_val = df.groupby(['subject_reference', 'Date'])['obs'].mean().reset_index()
        df = avg_val.groupby('subject_reference')['obs'].agg(list).reset_index()
        df['Rolling AVG'] = df['obs'].apply(lambda x: sliding_average(x, wndw_sz)).dropna()
        df = df[df['Rolling AVG'].apply(lambda x: len(x) > 0)]

        # If you want to reset the index of the resulting DataFrame
        df = df.reset_index(drop=True)

        # calculating average of rolling averages and assigning it to
        # Average column in the dataframe
        df['Latest'] = df['Rolling AVG'].apply(lambda x: x[-1] if x else None)
        # rounding off the values to 2 decimal places
        df['obs'] = df['obs'].apply(round_lst)
        df['Rolling AVG'] = df['Rolling AVG'].apply(round_lst)
    elif usr == 'latest':
        # latest
        df['effectiveDateTime'] = pd.to_datetime(df['effectiveDateTime'])
        latest_val = df.loc[df.groupby(['Date', 'subject_reference'])['effectiveDateTime'].idxmax()].reset_index()
        df = latest_val.groupby('subject_reference')['obs'].agg(list).reset_index()
        df = df[df['obs'].apply(lambda x: len(x) >= wndw_sz)]

        # Apply sliding average to the 'obs' column and create a new 'Rolling AVG' column
        df['Rolling AVG'] = df['obs'].apply(lambda x: sliding_average(x, wndw_sz))

        # Filter rows with non-empty 'Rolling AVG' lists
        df = df[df['Rolling AVG'].apply(lambda x: len(x) > 0)]

        # If you want to reset the index of the resulting DataFrame
        df = df.reset_index(drop=True)

        # calculating average of rolling averages and assigning it to
        # Average column in the dataframe
        df['Latest'] = df['Rolling AVG'].apply(lambda x: x[-1] if x else None)
        # rounding off the values to 2 decimal places
        df['obs'] = df['obs'].apply(round_lst)
        df['Rolling AVG'] = df['Rolling AVG'].apply(round_lst)
    df = pd.merge(df, patient_dataframe)
    # print("Size of df after merging:", df.shape)
    min_age, max_age = min(df['age']), max(df['age'])
    min_bmi, max_bmi = min(df['BMI']), max(df['BMI'])
    return df, date_min, date_max, min_age, max_age, min_bmi, max_bmi


def extract_sbp_dbp_columns(df, ehi_value, patient_df, start_date, end_date, usr, wndw_sz):
    df = df[['subject', 'effectiveDateTime', 'code', 'encounter', 'component']]

    df = pd.concat([df.drop(['subject'], axis=1), df['subject'].apply(pd.Series)], axis=1)
    ref = df['reference']
    df.drop(labels=['reference'], axis=1, inplace=True)
    df.insert(0, 'subject_reference', ref)
    df['effectiveDateTime'] = df['effectiveDateTime'].apply(lambda x: x[0:x.find('.')])
    df['effectiveDateTime'] = pd.to_datetime(df['effectiveDateTime'])
    df_test = df['code'].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    # drop the not required columns
    df.drop(labels=['code', 'text'], axis=1, inplace=True)
    df_test = df['coding'].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    # drop the not required columns
    df.drop(labels=['coding'], axis=1, inplace=True)
    df_test = df[0].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    # drop the not required columns
    df.drop(labels=[0, 'system', 'display'], axis=1, inplace=True)
    df.rename(columns={'code': 'activity_code'}, inplace=True)
    df_test = df['encounter'].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    df.drop(labels=['encounter', 0], axis=1, inplace=True)
    df.rename(columns={'reference': 'encounter_reference'}, inplace=True)
    df['subject_reference'] = df['subject_reference'].astype("str")
    df['subject_reference'] = df['subject_reference'].apply(
        lambda x: x.replace("Patient/", "", 1) if x.startswith("Patient/") else x)
    df['encounter_reference'] = df['encounter_reference'].astype("str")
    df['encounter_reference'] = df['encounter_reference'].apply(
        lambda x: x.replace("Encounter/", "", 1) if x.startswith("Encounter/") else x)
    df_test = df['component'].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    # inserting column at appropriate place
    imp_col1 = df[0]
    imp_col2 = df[1]
    df.drop(labels=[0, 1], axis=1, inplace=True)
    df.insert(0, 'component_1', imp_col2)
    df.insert(0, 'component_0', imp_col1)
    if ehi_value == "cadphr-sbp":
        df = extract_sbp(df)
    else:
        df = extract_dbp(df)
    ref_date = df['effectiveDateTime'].dt.date
    ref_time = df['effectiveDateTime'].dt.time
    df.insert(3, "Date", ref_date)
    df.insert(4, "Time", ref_time)
    min_date, max_date = min(df['Date']), max(df['Date'])
    # print("SBP_DBP")
    # print("Size of df before date filtering:", df.shape)
    df = extract_df_date_range(df, start_date, end_date, min_date, max_date)
    # print("Size of df after date filtering:", df.shape)
    round_lst = lambda lst: [round(x, 2) for x in lst]

    if usr == 'avg':
        # average
        avg_val = df.groupby(['subject_reference', 'Date'])['obs'].mean().reset_index()
        df = avg_val.groupby('subject_reference')['obs'].agg(list).reset_index()
        df['Rolling AVG'] = df['obs'].apply(lambda x: sliding_average(x, wndw_sz)).dropna()
        df = df[df['Rolling AVG'].apply(lambda x: len(x) > 0)]
        df = df.reset_index(drop=True)
        df['Latest'] = df['Rolling AVG'].apply(lambda x: x[-1] if x else None)
        # rounding off the values to 2 decimal places
        df['obs'] = df['obs'].apply(round_lst)
        df['Rolling AVG'] = df['Rolling AVG'].apply(round_lst)
    elif usr == 'latest':
        # latest
        df['effectiveDateTime'] = pd.to_datetime(df['effectiveDateTime'])
        latest_val = df.loc[df.groupby(['Date', 'subject_reference'])['effectiveDateTime'].idxmax()].reset_index()
        df = latest_val.groupby('subject_reference')['obs'].agg(list).reset_index()
        df = df[df['obs'].apply(lambda x: len(x) >= wndw_sz)]
        df['Rolling AVG'] = df['obs'].apply(lambda x: sliding_average(x, wndw_sz))
        df = df[df['Rolling AVG'].apply(lambda x: len(x) > 0)]
        df = df.reset_index(drop=True)
        df['Latest'] = df['Rolling AVG'].apply(lambda x: x[-1] if x else None)
        # rounding off the values to 2 decimal places
        df['obs'] = df['obs'].apply(round_lst)
        df['Rolling AVG'] = df['Rolling AVG'].apply(round_lst)
    df = pd.merge(df, patient_df)
    # print("Size of df after merging:", df.shape)
    min_age, max_age = min(df['age']), max(df['age'])
    min_bmi, max_bmi = min(df['BMI']), max(df['BMI'])
    return df, min_date, max_date, min_age, max_age, min_bmi, max_bmi


def extract_sbp(sbp_dataframe):
    # Extracting SBP values
    df_test = sbp_dataframe['component_0'].apply(pd.Series)
    sbp_dataframe = pd.concat([sbp_dataframe, df_test], axis=1)
    # inserting column at appropriate place
    imp_col1 = sbp_dataframe['code']
    imp_col2 = sbp_dataframe['valueQuantity']
    sbp_dataframe.drop(labels=['code', 'valueQuantity'], axis=1, inplace=True)
    sbp_dataframe.insert(0, 'component_0_valueQuantity', imp_col2)
    sbp_dataframe.insert(0, 'component_0_code', imp_col1)
    df_test = sbp_dataframe['component_0_code'].apply(pd.Series)
    sbp_dataframe = pd.concat([sbp_dataframe, df_test], axis=1)
    # inserting column at appropriate place
    imp_col1 = sbp_dataframe['coding']
    sbp_dataframe.drop(labels=['coding'], axis=1, inplace=True)
    sbp_dataframe.insert(0, 'component_0_code_coding', imp_col1)

    df_test = sbp_dataframe['component_0_code_coding'].apply(pd.Series)
    sbp_dataframe = pd.concat([sbp_dataframe, df_test], axis=1)
    # inserting column at appropriate place
    imp_col1 = sbp_dataframe[0]
    sbp_dataframe.drop(labels=[0], axis=1, inplace=True)
    sbp_dataframe.insert(0, 'component_0_code_coding_0', imp_col1)

    df_test = sbp_dataframe['component_0_code_coding_0'].apply(pd.Series)
    sbp_dataframe = pd.concat([sbp_dataframe, df_test], axis=1)
    # inserting column at appropriate place
    imp_col1 = sbp_dataframe['code']
    imp_col2 = sbp_dataframe['display']
    sbp_dataframe.drop(labels=['code'], axis=1, inplace=True)
    sbp_dataframe.insert(0, 'component_0_code_coding_0_code', imp_col1)
    sbp_dataframe.insert(0, 'component_0_code_coding_0_display', imp_col2)

    df_test = sbp_dataframe['component_0_valueQuantity'].apply(pd.Series)
    sbp_dataframe = pd.concat([sbp_dataframe, df_test], axis=1)
    # inserting column at appropriate place
    imp_col1 = sbp_dataframe['unit']
    imp_col2 = sbp_dataframe['value']
    sbp_dataframe.drop(labels=['unit', 'value'], axis=1, inplace=True)
    sbp_dataframe.insert(0, 'component_0_valueQuantity_value', imp_col2)
    sbp_dataframe.insert(0, 'component_0_code_unit', imp_col1)

    sbp_dataframe = sbp_dataframe[['subject_reference', 'effectiveDateTime', 'activity_code', 'encounter_reference',
                                   'component_0_valueQuantity_value']]
    sbp_dataframe.rename(columns={'component_0_valueQuantity_value': 'obs'}, inplace=True)
    return sbp_dataframe


def extract_dbp(dbp_dataframe):
    # Extracting Diastolic Pressure
    df_test = dbp_dataframe['component_1'].apply(pd.Series)
    dbp_dataframe = pd.concat([dbp_dataframe, df_test], axis=1)
    # inserting column at appropriate place
    imp_col1 = dbp_dataframe['code']
    imp_col2 = dbp_dataframe['valueQuantity']
    dbp_dataframe.drop(labels=['code', 'valueQuantity'], axis=1, inplace=True)
    dbp_dataframe.insert(0, 'component_1_valueQuantity', imp_col2)
    dbp_dataframe.insert(0, 'component_1_code', imp_col1)

    df_test = dbp_dataframe['component_1_code'].apply(pd.Series)
    dbp_dataframe = pd.concat([dbp_dataframe, df_test], axis=1)
    # inserting column at appropriate place
    imp_col1 = dbp_dataframe['coding']
    dbp_dataframe.drop(labels=['coding'], axis=1, inplace=True)
    dbp_dataframe.insert(0, 'component_1_code_coding', imp_col1)

    df_test = dbp_dataframe['component_1_code_coding'].apply(pd.Series)
    dbp_dataframe = pd.concat([dbp_dataframe, df_test], axis=1)
    # inserting column at appropriate place
    imp_col1 = dbp_dataframe[0]
    dbp_dataframe.drop(labels=[0], axis=1, inplace=True)
    dbp_dataframe.insert(0, 'component_1_code_coding_0', imp_col1)

    df_test = dbp_dataframe['component_1_code_coding_0'].apply(pd.Series)
    dbp_dataframe = pd.concat([dbp_dataframe, df_test], axis=1)
    # inserting column at appropriate place
    imp_col1 = dbp_dataframe['code']
    imp_col2 = dbp_dataframe['display']
    dbp_dataframe.drop(labels=['code'], axis=1, inplace=True)
    dbp_dataframe.insert(0, 'component_1_code_coding_0_code', imp_col1)
    dbp_dataframe.insert(0, 'component_1_code_coding_0_display', imp_col2)

    df_test = dbp_dataframe['component_1_valueQuantity'].apply(pd.Series)
    dbp_dataframe = pd.concat([dbp_dataframe, df_test], axis=1)
    # inserting column at appropriate place
    imp_col1 = dbp_dataframe['unit']
    imp_col2 = dbp_dataframe['value']
    dbp_dataframe.drop(labels=['unit', 'value'], axis=1, inplace=True)
    dbp_dataframe.insert(0, 'component_1_valueQuantity_value', imp_col2)
    dbp_dataframe.insert(0, 'component_1_code_unit', imp_col1)

    dbp_dataframe = dbp_dataframe[['subject_reference', 'effectiveDateTime', 'activity_code', 'encounter_reference',
                                   'component_1_valueQuantity_value']]

    dbp_dataframe.rename(columns={'component_1_valueQuantity_value': 'obs'}, inplace=True)
    return dbp_dataframe


def extract_observation_features(df, observation_extraction_features):
    df = df[observation_extraction_features]
    return df


def extract_patient_info(df, data_s):
    # extract observation features
    required_features = ['identifier', 'birthDate', 'gender']
    df = extract_observation_features(df, required_features)
    df = data_pop(df, data_s)
    df_test = df['identifier'].apply(pd.Series)
    df_test = df_test[0].apply(pd.Series)
    df = pd.concat([df.drop(['identifier'], axis=1), df_test], axis=1)
    # inserting column at appropriate place
    ind_val = df['value']
    df.drop(labels=['value'], axis=1, inplace=True)
    df.insert(1, 'identifier_value', ind_val)
    # ======================================================================
    df_info = df[['identifier_value', 'birthDate', 'gender']]
    df_info.rename(columns={"identifier_value": "subject_reference"}, inplace=True)
    # calculate age...................
    df_info['birthDate'] = df_info['birthDate'].fillna(-1)
    df_info['birthDate'] = df_info['birthDate'].astype(int)
    current_year = datetime.datetime.now().year
    df_info['age'] = df_info['birthDate'].apply(lambda x: current_year - x if x != -1 else -1)
    df_info['birthDate'] = df_info['birthDate'].astype(str)
    df_info['birthDate'] = df_info['birthDate'].replace('-1', np.nan)
    df_info['age'] = df_info['age'].astype(str)
    df_info['age'] = df_info['age'].replace('-1', np.nan)
    df_info = df_info.drop_duplicates()
    df_info.drop(labels=['birthDate'], axis=1, inplace=True)
    df_info['age'] = df_info['age'].astype(int)
    return df_info


def convert(input_value, input_range_start, input_range_end, output_range_start, output_range_end):
    if input_range_start == input_range_end:
        raise ValueError("input_range_start and input_range_end can't be equal")

    range_factor = (output_range_end - output_range_start) / (input_range_end - input_range_start)
    return ((input_value - input_range_start) * range_factor) + output_range_start


def compute_mssd_sliding_window(valences, window_size):
    if len(valences) < window_size:
        return None  # Return None if not enough data

    results = []
    for start in range(len(valences) - window_size + 1):
        window = valences[start:start + window_size]
        sum_ssd = sum((window[i + 1] - window[i]) ** 2 for i in range(len(window) - 1))
        mssd = sum_ssd / (2 * (len(window) - 1))
        input_range_end = (2 * len(window)) / (len(window) - 1)
        converted_mssd = convert(mssd, 0, input_range_end, 0, 1)

        # check the code previously to this
        mapped_cohen_value = (converted_mssd / input_range_end) * 2 - 1
        mapped_cohen_value = float(Decimal(mapped_cohen_value).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        results.append((mssd, mapped_cohen_value))

    return results


def sliding_std(lst, window_size):
    if len(lst) < window_size:
        return np.nan  # Return NaN for lists with fewer than three elements
    return [np.std(lst[i:i + window_size]) for i in range(len(lst) - window_size + 1)]


def compute_autocorrelation_sliding_window(values, window_size):
    valences_count = len(values)
    if valences_count < window_size:
        return None  # Not enough data for even one window

    results = []
    for start in range(valences_count - window_size + 1):
        window = values[start:start + window_size]
        valence_mean = np.mean(window)
        valence_variance = np.var(window, ddof=0)

        sum_product = 0
        for i in range(len(window) - 1):
            sum_product += (window[i + 1] - valence_mean) * (window[i] - valence_mean)

        computed_value = 1.0  # Default to 1 if variance is zero
        if valence_variance != 0:
            computed_value = sum_product / valence_variance / (len(window) - 1)

        # Map and round computed autocorrelation value
        mapped_cohen_value = convert(computed_value, -1, 1, 0, 1)
        mapped_cohen_value = float(Decimal(mapped_cohen_value).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        results.append((computed_value, mapped_cohen_value))

    return results


def extract_emotion_values(df):
    angle_values = []
    intensity_values = []
    valence_values = []

    for row in df['component']:
        if row:
            for ext in row[0]['extension']:
                if 'calculated-angle' in ext['url']:
                    angle_values.append(ext['valueDecimal'])
                elif 'calculated-intensity' in ext['url']:
                    intensity_values.append(ext['valueDecimal'])
                elif 'calculated-valence' in ext['url']:
                    valence_values.append(ext['valueDecimal'])

    if len(angle_values) > len(df):
        angle_values = angle_values[:len(df)]
    elif len(angle_values) < len(df):
        angle_values += [np.nan] * (len(df) - len(angle_values))

    if len(intensity_values) > len(df):
        intensity_values = intensity_values[:len(df)]
    elif len(intensity_values) < len(df):
        intensity_values += [np.nan] * (len(df) - len(intensity_values))

    if len(valence_values) > len(df):
        valence_values = valence_values[:len(df)]
    elif len(valence_values) < len(df):
        valence_values += [np.nan] * (len(df) - len(valence_values))

    df['angle'] = angle_values
    df['intensity'] = intensity_values
    df['valence'] = valence_values
    df = df.drop('component', axis=1)

    return df


def get_latest_grade(grades_list):
    if not grades_list:
        return None  # Return None if the list is empty
    else:
        return grades_list[-1]


def emotion_variability(df, patient_df, min_sufficiency, sliding_window_size, start_date, end_date):
    df = df[['subject', 'effectiveDateTime', 'code', 'component']]

    df = pd.concat([df.drop(['subject'], axis=1), df['subject'].apply(pd.Series)], axis=1)
    ref = df['reference']
    df.drop(labels=['reference'], axis=1, inplace=True)
    df.insert(0, 'subject_reference', ref)
    df['effectiveDateTime'] = df['effectiveDateTime'].apply(lambda x: x[0:x.find('.')])
    df['effectiveDateTime'] = pd.to_datetime(df['effectiveDateTime'])
    df['effectiveDateTime'] = df['effectiveDateTime'].apply(lambda x: x.replace(microsecond=0, second=0))
    ref_date = df['effectiveDateTime'].dt.date
    ref_time = df['effectiveDateTime'].dt.time
    df.insert(3, "Date", ref_date)
    df.insert(4, "Time", ref_time)
    df_test = df['code'].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    df.drop(labels=['code', 'text'], axis=1, inplace=True)
    df_test = df['coding'].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    # drop the not required columns
    df.drop(labels=['coding'], axis=1, inplace=True)
    df_test = df[0].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    # drop the not required columns
    df.drop(labels=['system', 0], axis=1, inplace=True)
    df.rename(columns={'code': 'activity_code'}, inplace=True)
    date_min = min(df['Date'])
    date_max = max(df['Date'])
    df = extract_df_date_range(df, start_date, end_date, date_min, date_max)

    df = extract_emotion_values(df)
    df = df.dropna(subset=['angle', 'intensity', 'valence'])

    df['converted_valence'] = df['valence'].apply(lambda x: convert(x, -5, 5, -1, 1))
    avg_val = df.groupby(['subject_reference', 'effectiveDateTime'])[
        ['angle', 'intensity', 'valence', 'converted_valence']].mean().reset_index()
    df_result = avg_val.groupby('subject_reference')[['angle', 'intensity', 'valence', 'converted_valence']].agg(
        list).reset_index()
    df_result = df_result[
        (df_result['valence'].apply(len) >= min_sufficiency) & (df_result['angle'].apply(len) >= min_sufficiency) & (
                df_result['intensity'].apply(len) >= min_sufficiency)].reset_index(drop=True)
    df_result['angle'] = df_result['angle'].apply(lambda x: np.round(x, 2))
    df_result['intensity'] = df_result['intensity'].apply(lambda x: np.round(x, 2))
    df_result['valence'] = df_result['valence'].apply(lambda x: np.round(x, 2))
    df_result['valence'] = df_result['valence'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    df_result['converted_valence'] = df_result['converted_valence'].apply(lambda x: np.round(x, 2))
    df_result['converted_valence'] = df_result['converted_valence'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    df_result['emotionvariability'] = df_result['converted_valence'].apply(lambda x: sliding_std(x, sliding_window_size))

    df_result = pd.merge(df_result, patient_df, on='subject_reference', how='inner')
    min_age = min(df_result['age'])
    max_age = max(df_result['age'])
    min_bmi = min(df_result['BMI'])
    max_bmi = max(df_result['BMI'])
    return df_result, date_min, date_max, min_age, max_age, min_bmi, max_bmi


def emotion_instability(df, patient_df, min_sufficiency, sliding_window_size, start_date, end_date):
    df = df[['subject', 'effectiveDateTime', 'code', 'component']]

    df = pd.concat([df.drop(['subject'], axis=1), df['subject'].apply(pd.Series)], axis=1)
    ref = df['reference']
    df.drop(labels=['reference'], axis=1, inplace=True)
    df.insert(0, 'subject_reference', ref)
    df['effectiveDateTime'] = df['effectiveDateTime'].apply(lambda x: x[0:x.find('.')])
    df['effectiveDateTime'] = pd.to_datetime(df['effectiveDateTime'])
    df['effectiveDateTime'] = df['effectiveDateTime'].apply(lambda x: x.replace(microsecond=0, second=0))
    ref_date = df['effectiveDateTime'].dt.date
    ref_time = df['effectiveDateTime'].dt.time
    df.insert(3, "Date", ref_date)
    df.insert(4, "Time", ref_time)
    df_test = df['code'].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    df.drop(labels=['code', 'text'], axis=1, inplace=True)
    df_test = df['coding'].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    # drop the not required columns
    df.drop(labels=['coding'], axis=1, inplace=True)
    df_test = df[0].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    # drop the not required columns
    df.drop(labels=['system', 0], axis=1, inplace=True)
    df.rename(columns={'code': 'activity_code'}, inplace=True)
    date_min = min(df['Date'])
    date_max = max(df['Date'])
    df = extract_df_date_range(df, start_date, end_date, date_min, date_max)
    df = extract_emotion_values(df)
    df = df.dropna(subset=['angle', 'intensity', 'valence'])

    df['converted_valence'] = df['valence'].apply(lambda x: convert(x, -5, 5, -1, 1))
    avg_val = df.groupby(['subject_reference', 'effectiveDateTime'])[
        ['angle', 'intensity', 'valence', 'converted_valence']].mean().reset_index()
    df_result = avg_val.groupby('subject_reference')[['angle', 'intensity', 'valence', 'converted_valence']].agg(
        list).reset_index()
    df_result = df_result[
        (df_result['valence'].apply(len) >= min_sufficiency) & (df_result['angle'].apply(len) >= min_sufficiency) & (
                df_result['intensity'].apply(len) >= min_sufficiency)].reset_index(drop=True)
    df_result['angle'] = df_result['angle'].apply(lambda x: np.round(x, 2))
    df_result['intensity'] = df_result['intensity'].apply(lambda x: np.round(x, 2))
    df_result['valence'] = df_result['valence'].apply(lambda x: np.round(x, 2))
    df_result['valence'] = df_result['valence'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    df_result['converted_valence'] = df_result['converted_valence'].apply(lambda x: np.round(x, 2))
    df_result['converted_valence'] = df_result['converted_valence'].apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    results_df = df_result['converted_valence'].apply(lambda x: compute_mssd_sliding_window(x, sliding_window_size))

    # Split the lists of tuples into two lists (one for each new column)
    df_result['emotioninstability_raw'], df_result['emotioninstability'] = zip(
        *results_df.apply(lambda x: zip(*x) if x else ([], [])))

    # Convert the tuples to lists
    df_result['emotioninstability_raw'] = df_result['emotioninstability_raw'].apply(list)
    df_result['emotioninstability'] = df_result['emotioninstability'].apply(list)

    df_result = pd.merge(df_result, patient_df, on='subject_reference', how='inner')
    min_age = min(df_result['age'])
    max_age = max(df_result['age'])
    min_bmi = min(df_result['BMI'])
    max_bmi = max(df_result['BMI'])
    return df_result, date_min, date_max, min_age, max_age, min_bmi, max_bmi


def emotion_inertia(df, patient_df, min_sufficiency, sliding_window_size, start_date, end_date):
    df = df[['subject', 'effectiveDateTime', 'code', 'component']]

    df = pd.concat([df.drop(['subject'], axis=1), df['subject'].apply(pd.Series)], axis=1)
    ref = df['reference']
    df.drop(labels=['reference'], axis=1, inplace=True)
    df.insert(0, 'subject_reference', ref)
    df['effectiveDateTime'] = df['effectiveDateTime'].apply(lambda x: x[0:x.find('.')])
    df['effectiveDateTime'] = pd.to_datetime(df['effectiveDateTime'])
    df['effectiveDateTime'] = df['effectiveDateTime'].apply(lambda x: x.replace(microsecond=0, second=0))
    ref_date = df['effectiveDateTime'].dt.date
    ref_time = df['effectiveDateTime'].dt.time
    df.insert(3, "Date", ref_date)
    df.insert(4, "Time", ref_time)
    df_test = df['code'].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    df.drop(labels=['code', 'text'], axis=1, inplace=True)
    df_test = df['coding'].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    # drop the not required columns
    df.drop(labels=['coding'], axis=1, inplace=True)
    df_test = df[0].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    # drop the not required columns
    df.drop(labels=['system', 0], axis=1, inplace=True)
    df.rename(columns={'code': 'activity_code'}, inplace=True)
    date_min = min(df['Date'])
    date_max = max(df['Date'])
    df = extract_df_date_range(df, start_date, end_date, date_min, date_max)
    df = extract_emotion_values(df)
    df = df.dropna(subset=['angle', 'intensity', 'valence'])

    df['converted_valence'] = df['valence'].apply(lambda x: convert(x, -5, 5, -1, 1))
    avg_val = df.groupby(['subject_reference', 'effectiveDateTime'])[
        ['angle', 'intensity', 'valence', 'converted_valence']].mean().reset_index()
    df_result = avg_val.groupby('subject_reference')[['angle', 'intensity', 'valence', 'converted_valence']].agg(
        list).reset_index()
    df_result = df_result[
        (df_result['valence'].apply(len) >= min_sufficiency) & (df_result['angle'].apply(len) >= min_sufficiency) & (
                df_result['intensity'].apply(len) >= min_sufficiency)].reset_index(drop=True)
    df_result['angle'] = df_result['angle'].apply(lambda x: np.round(x, 2))
    df_result['intensity'] = df_result['intensity'].apply(lambda x: np.round(x, 2))
    df_result['valence'] = df_result['valence'].apply(lambda x: np.round(x, 2))
    df_result['valence'] = df_result['valence'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    df_result['converted_valence'] = df_result['converted_valence'].apply(lambda x: np.round(x, 2))
    df_result['converted_valence'] = df_result['converted_valence'].apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    results_df = df_result['converted_valence'].apply(
        lambda x: compute_autocorrelation_sliding_window(x, sliding_window_size))

    # Split the lists of tuples into two lists (one for each new column)
    df_result['emotioninertia_raw'], df_result['emotioninertia'] = zip(
        *results_df.apply(lambda x: zip(*x) if x else ([], [])))

    # Convert the tuples to lists
    df_result['emotioninertia_raw'] = df_result['emotioninertia_raw'].apply(list)
    df_result['emotioninertia'] = df_result['emotioninertia'].apply(list)

    df_result = pd.merge(df_result, patient_df, on='subject_reference', how='inner')
    min_age = min(df_result['age'])
    max_age = max(df_result['age'])
    min_bmi = min(df_result['BMI'])
    max_bmi = max(df_result['BMI'])
    return df_result, date_min, date_max, min_age, max_age, min_bmi, max_bmi


def calculate_HRmax(age):
    return 206.9 - (0.67 * age)


def calculate_bmi(df_patient_height_weight):
    df_patient_height_weight['BMI'] = df_patient_height_weight['weight'] / np.power(
        df_patient_height_weight['height'] / 100, 2)
    return df_patient_height_weight


def BMI_range(df, start_BMI, end_BMI):
    if start_BMI == "" and end_BMI == "":
        return df
    if start_BMI == "":
        df = df[df['BMI'] <= end_BMI]
    if end_BMI == "":
        df = df[df['BMI'] >= start_BMI]
    if start_BMI != "" and end_BMI != "":
        df = df[(df['BMI'] > start_BMI) & (df['BMI'] <= end_BMI)]
    return df


def data_pop(df, data_select):
    if data_select == "all":
        return df

    elif data_select == "male" or data_select == "female":
        df = df[df['gender'] == data_select]
    return df


# returning df with desired age range
def age_range(df, start_age, end_age):
    df_test = df
    if start_age == "" and end_age == "":
        return df

    if start_age == "":
        if end_age < min(df['age']):
            return df
        df = df[df['age'] <= end_age]

    elif end_age == "":
        if start_age > max(df['age']):
            return df
        df = df[df['age'] >= start_age]
    else:
        if start_age > max(df['age']) or end_age < min(df['age']):
            return df
        df = df[(start_age <= df['age']) & (df['age'] <= end_age)]
    if df.shape[0] < 5:
        return df_test
    return df


def extract_height(df_all):
    df = df_all[['subject', 'valueQuantity', 'effectiveDateTime']]
    df = pd.concat([df.drop(['subject'], axis=1), df['subject'].apply(pd.Series)], axis=1)
    # inserting column at appropriate place
    ref = df['reference']
    df.drop(labels=['reference'], axis=1, inplace=True)
    df.insert(0, 'subject_reference', ref)
    df = pd.concat([df, df['valueQuantity'].apply(pd.Series)], axis=1)
    # inserting column at appropriate place
    ref = df['value']
    df.insert(3, 'height', ref)
    df.drop(labels=['value', 'valueQuantity', 'unit', 'system', 'code'], axis=1, inplace=True)
    df['subject_reference'] = df['subject_reference'].astype("str")
    df['subject_reference'] = df['subject_reference'].apply(
        lambda x: x.replace("Patient/", "", 1) if x.startswith("Patient/") else x)

    # Convert effectiveDateTime to datetime
    df['effectiveDateTime'] = pd.to_datetime(df['effectiveDateTime'])

    # Keep only unique and latest rows of each subject_reference
    df = df.sort_values(by=['subject_reference', 'effectiveDateTime'], ascending=[True, False])
    df = df.drop_duplicates(subset='subject_reference', keep='first')
    df.drop(labels=['effectiveDateTime'], axis=1, inplace=True)
    return df


def extract_weight(df_all):
    # Select relevant columns
    df = df_all[['subject', 'valueQuantity', 'effectiveDateTime']]

    # Split 'subject' column into subject_reference
    df = pd.concat([df.drop(['subject'], axis=1), df['subject'].apply(pd.Series)], axis=1)

    # Insert subject_reference column
    ref = df['reference']
    df.drop(labels=['reference'], axis=1, inplace=True)
    df.insert(0, 'subject_reference', ref)

    # Expand valueQuantity column
    df = pd.concat([df, df['valueQuantity'].apply(pd.Series)], axis=1)

    # Insert weight column
    ref = df['value']
    df.insert(3, 'weight', ref)

    # Drop unnecessary columns
    df.drop(labels=['value', 'valueQuantity', 'unit', 'system', 'code'], axis=1, inplace=True)

    # Convert subject_reference to string and remove prefix if present
    df['subject_reference'] = df['subject_reference'].astype(str)
    df['subject_reference'] = df['subject_reference'].apply(
        lambda x: x.replace("Patient/", "", 1) if x.startswith("Patient/") else x)

    # Convert effectiveDateTime to datetime
    df['effectiveDateTime'] = pd.to_datetime(df['effectiveDateTime'])

    # Keep only unique and latest rows of each subject_reference
    df = df.sort_values(by=['subject_reference', 'effectiveDateTime'], ascending=[True, False])
    df = df.drop_duplicates(subset='subject_reference', keep='first')
    df.drop(labels=['effectiveDateTime'], axis=1, inplace=True)

    return df


def get_demographic_data(user_data, weight_df, height_df, gender, start_age, end_age, start_BMI, end_BMI):
    patient_df = pd.read_json(user_data)
    df_patient = extract_patient_info(patient_df, gender)
    # print("DEMOGRAPHICS")
    # print("Size of df before age filtering:", df_patient.shape)
    df_patient = age_range(df_patient, start_age, end_age)
    # print("Size of df after age filtering:", df_patient.shape)
    df_height = extract_height(height_df)
    df_weight = extract_weight(weight_df)
    df_height_weight = pd.merge(df_height, df_weight, on='subject_reference', how='inner')
    df_bmi = calculate_bmi(df_height_weight)
    # print("Size of df before bmi filtering:", df_bmi.shape)
    df_bmi = BMI_range(df_bmi, start_BMI, end_BMI)
    # print("Size of df after bmi filtering:", df_bmi.shape)
    df_patient = pd.merge(df_patient, df_bmi, on='subject_reference', how='inner')
    # print("Size of df after merging:", df_patient.shape)
    return df_patient


def extract_pss(df_all, patient_df):
    df = df_all[['subject', 'valueInteger', 'effectiveDateTime']]

    df = pd.concat([df.drop(['subject'], axis=1), df['subject'].apply(pd.Series)], axis=1)
    # inserting column at appropriate place
    ref = df['reference']
    df.drop(labels=['reference'], axis=1, inplace=True)
    df.insert(0, 'subject_reference', ref)
    df = pd.concat([df, df['valueInteger'].apply(pd.Series)], axis=1)
    # inserting column at appropriate place
    ref = df['valueInteger']
    df.insert(3, 'pss_value', ref)
    df.drop(labels=['valueInteger'], axis=1, inplace=True)
    df['subject_reference'] = df['subject_reference'].astype("str")
    df['subject_reference'] = df['subject_reference'].apply(
        lambda x: x.replace("Patient/", "", 1) if x.startswith("Patient/") else x)
    df = pd.merge(df, patient_df)
    # Convert effectiveDateTime to datetime
    df['effectiveDateTime'] = pd.to_datetime(df['effectiveDateTime'])

    # Keep only unique and latest rows of each subject_reference
    df = df.sort_values(by=['subject_reference', 'effectiveDateTime'], ascending=[True, False])
    df = df.drop_duplicates(subset='subject_reference', keep='first')
    ref_date = df['effectiveDateTime'].dt.date
    ref_time = df['effectiveDateTime'].dt.time
    df.insert(3, "Date", ref_date)
    df.insert(4, "Time", ref_time)
    date_min = min(df['Date'])
    date_max = max(df['Date'])
    df.drop(labels=['effectiveDateTime', 0], axis=1, inplace=True)
    df['pss_value'] = df['pss_value'].astype(int)
    min_age, max_age = min(df['age']), max(df['age'])
    min_bmi, max_bmi = min(df['BMI']), max(df['BMI'])
    return df, date_min, date_max, min_age, max_age, min_bmi, max_bmi


def extract_stress_index(df_all, patient_df):
    df = df_all[['subject', 'computedValue', 'ehiGrading', 'effectiveDateTime', 'ehiInterpretation']]

    df = pd.concat([df.drop(['subject'], axis=1), df['subject'].apply(pd.Series)], axis=1)
    # inserting column at appropriate place
    ref = df['reference']
    df.drop(labels=['reference'], axis=1, inplace=True)
    df.insert(0, 'subject_reference', ref)
    df = pd.concat([df, df['computedValue'].apply(pd.Series)], axis=1)
    # inserting column at appropriate place
    ref = df['computedValue']
    df.insert(3, 'stress_value', ref)
    df.drop(labels=['computedValue'], axis=1, inplace=True)
    df['subject_reference'] = df['subject_reference'].astype("str")
    df['subject_reference'] = df['subject_reference'].apply(
        lambda x: x.replace("Patient/", "", 1) if x.startswith("Patient/") else x)
    df.rename(columns={'ehiInterpretation': 'stress_index_interpretation', 'ehiGrading': 'cadphr-sira_grade'},
              inplace=True)
    # Convert effectiveDateTime to datetime
    df['effectiveDateTime'] = pd.to_datetime(df['effectiveDateTime'], format='mixed')

    # Keep only unique and latest rows of each subject_reference
    df = df.sort_values(by=['subject_reference', 'effectiveDateTime'], ascending=[True, False])
    df = df.drop_duplicates(subset='subject_reference', keep='first')
    ref_date = df['effectiveDateTime'].dt.date
    ref_time = df['effectiveDateTime'].dt.time
    df.insert(3, "Date", ref_date)
    df.insert(4, "Time", ref_time)
    date_min = min(df['Date'])
    date_max = max(df['Date'])
    df['stress_value'] = df['stress_value'].astype(int)
    df.drop(labels=[0, 'effectiveDateTime'], axis=1, inplace=True)
    df = pd.merge(df, patient_df)
    min_age, max_age = min(df['age']), max(df['age'])
    min_bmi, max_bmi = min(df['BMI']), max(df['BMI'])
    return df, date_min, date_max, min_age, max_age, min_bmi, max_bmi


def calculate_std_with_sliding_window(l1, ws):
    result = []
    for i in range(len(l1)):
        start = i
        if (start + ws) <= len(l1):
            end = start + ws
        else:
            break
        curr_l = l1[start:end]
        curr_l = [int(x) for x in curr_l]
        var = np.std(curr_l)
        var = np.round(var, 2)
        result.append(var)
    return result


def calculate_mann_kendall(df, total_period, window_size):
    tau_list = []
    trend_list = []
    pvalue_list = []
    for i in range(0, total_period + 1 - window_size, 1):
        start_index = i
        end_index = start_index + window_size
        data_window = df.iloc[start_index:end_index, 1].tolist()
        # perform Mann-Kendall Trend Test
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(data_window)
        tau_list.append(Tau)
        trend_list.append(trend)
        pvalue_list.append(p)
    return tau_list, trend_list, pvalue_list


def extract_format_find_trend(df, window_size):
    ll = df['obs']
    ll10 = ll[-10:]
    df_test = pd.DataFrame({'val': ll10})
    df_test.reset_index(inplace=True)
    df_test.rename(columns={'index': 'date'}, inplace=True)
    total_period = df_test.shape[0]
    tau_list, trend_list, pvalue_list = calculate_mann_kendall(df_test, total_period, window_size)
    df['latest_tau'] = tau_list[0]
    df['latest_p_value'] = pvalue_list[0]
    return df


def process(df, activity_code, patient_dataframe, start_date, end_date, sliding_window, day_condition,
            min_sufficiency,
            max_sufficiency, ehi_value):
    if ehi_value == "cadphr-hrrra" or ehi_value == "cadphr-diabetesriskscore" or ehi_value == "cadphr-cadrisk10" or ehi_value == "cadphr-osariskscore" or ehi_value == 'cadphr-rcvage' or ehi_value == "cadphr-vo2maxra" or ehi_value == "cadphr-ecrfra":
        df_grade, min_date, max_date, min_age, max_age, min_bmi, max_bmi = extract_ehi_grade(df, patient_dataframe,
                                                                                             ehi_value, start_date,
                                                                                             end_date)
        latest_grade = ehi_value + "_grade"
        df_grade = df_grade.rename(columns={'ehi_grade': latest_grade})
        return df_grade, min_date, max_date, min_age, max_age, min_bmi, max_bmi

    elif ehi_value == "cadphr-sbp" or ehi_value == "cadphr-dbp":
        df_roll, min_date, max_date, min_age, max_age, min_bmi, max_bmi = extract_sbp_dbp_columns(df, ehi_value,
                                                                                                  patient_dataframe,
                                                                                                  start_date, end_date,
                                                                                                  day_condition,
                                                                                                  sliding_window)
        if ehi_value == "cadphr-sbp":
            df_patient = df_roll.rename(columns={'Latest': 'SBP'})
            add_grade_column(df_patient, ehi_value)
        else:
            df_patient = df_roll.rename(columns={'Latest': 'DBP'})
            add_grade_column(df_patient, ehi_value)
        return df_patient, min_date, max_date, min_age, max_age, min_bmi, max_bmi

    elif ehi_value == "cadphr-pulsepressure":
        latest_grade = ehi_value + "_grade"
        df[latest_grade] = df.apply(pulsePressure_assign_grade, axis=1)
        return df

    elif ehi_value == "cadphr-emotioninstability" or ehi_value == "cadphr-emotioninertia" or ehi_value == "cadphr-emotionvariability":
        if ehi_value == "cadphr-emotioninertia":
            df, min_date, max_date, min_age, max_age, min_bmi, max_bmi = emotion_inertia(df, patient_dataframe,
                                                                                         min_sufficiency,
                                                                                         sliding_window, start_date,
                                                                                         end_date)
        elif ehi_value == "cadphr-emotionvariability":
            df, min_date, max_date, min_age, max_age, min_bmi, max_bmi = emotion_variability(df, patient_dataframe,
                                                                                             min_sufficiency,
                                                                                             sliding_window, start_date,
                                                                                             end_date)
        elif ehi_value == "cadphr-emotioninstability":
            df, min_date, max_date, min_age, max_age, min_bmi, max_bmi = emotion_instability(df, patient_dataframe,
                                                                                             min_sufficiency,
                                                                                             sliding_window, start_date,
                                                                                             end_date)

        num_rows = len(df)
        num_first_segment = int(num_rows * 0.3333)
        num_second_segment = int(num_rows * 0.3333)
        num_third_segment = num_rows - num_first_segment - num_second_segment
        first_segment_values = np.random.randint(0, 10, num_first_segment)
        second_segment_values = np.random.randint(10, 15, num_second_segment)
        third_segment_values = np.random.randint(15, 22, num_third_segment)
        all_values = np.concatenate((first_segment_values, second_segment_values, third_segment_values))
        np.random.shuffle(all_values)
        df['GAD7'] = all_values
        df['GAD7'] = df['GAD7'].astype(int)

        if ehi_value == "cadphr-emotioninertia":
            df['emotioninertia_grades'] = df.apply(emotioninertia_grading, axis=1)
            df['inertia_latest_grade'] = df['emotioninertia_grades'].apply(get_latest_grade)
        elif ehi_value == "cadphr-emotionvariability":
            df['emotionvariability_grades'] = df['emotionvariability'].apply(emotionvariability_grading)
            df['variability_latest_grade'] = df['emotionvariability_grades'].apply(get_latest_grade)
        elif ehi_value == "cadphr-emotioninstability":
            df['emotioninstability_grades'] = df['emotioninstability'].apply(emotioninstability_grading)
            df['instability_latest_grade'] = df['emotioninstability_grades'].apply(get_latest_grade)
        return df, min_date, max_date, min_age, max_age, min_bmi, max_bmi

    elif ehi_value == 'cadphr-pss4':
        df_ps, min_date, max_date, min_age, max_age, min_bmi, max_bmi = extract_pss(df, patient_dataframe)
        df_ps['cadphr-pss4_grade'] = df_ps.apply(perceived_stress_grading, axis=1)
        return df_ps, min_date, max_date, min_age, max_age, min_bmi, max_bmi

    elif ehi_value == 'cadphr-sira':
        df_si, min_date, max_date, min_age, max_age, min_bmi, max_bmi = extract_stress_index(df, patient_dataframe)
        return df_si, min_date, max_date, min_age, max_age, min_bmi, max_bmi

    else:
        targetHR_flag = 0
        if ehi_value == 'cadphr-targetHR':
            targetHR_flag = 1
            ehi_value = 'cadphr-heartrate'
        df_roll, min_date, max_date, min_age, max_age, min_bmi, max_bmi = extract_ehi_columns(df, patient_dataframe,
                                                                                              activity_code, start_date,
                                                                                              end_date,
                                                                                              day_condition,
                                                                                              sliding_window)
        if ehi_value == 'cadphr-na':
            df_roll = df_roll.rename(columns={'Latest': 'NA'})
            df_roll = na_grading(df_roll)
            return df_roll, min_date, max_date, min_age, max_age, min_bmi, max_bmi

        df_grade = add_grade_column(df_roll, ehi_value)
        if targetHR_flag == 1:
            df_grade['HRmax'] = df_grade['age'].apply(lambda x: calculate_HRmax(x))
            df_grade['HRR'] = df_grade['HRmax'] - df_grade['Latest']
            df_grade['target_hr_lower'] = df_grade['HRR'] * 0.4 + df_grade['Latest']
            df_grade['target_hr_upper'] = df_grade['HRR'] * 0.6 + df_grade['Latest']
            df_grade['Highest HR of User'] = df_grade['Rolling AVG'].apply(lambda x: max(x) if x else None)
        return df_grade, min_date, max_date, min_age, max_age, min_bmi, max_bmi
