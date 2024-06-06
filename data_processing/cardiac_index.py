import numpy as np
import pandas as pd

from data_processing.dataframe_processing_functions import extract_patient_info, age_range, \
    extract_height, extract_weight, extract_sbp, extract_dbp, extract_df_date_range


def extract_ehi_columns(df_all, ehi_value, act_code):
    df = df_all[['subject', 'effectiveDateTime', 'code', 'valueQuantity']]

    df = pd.concat([df.drop(['subject'], axis=1), df['subject'].apply(pd.Series)], axis=1)
    ref = df['reference']
    df.drop(labels=['reference'], axis=1, inplace=True)
    df.insert(0, 'subject_reference', ref)
    df['effectiveDateTime'] = df['effectiveDateTime'].apply(lambda x: x[0:x.find('.')])
    df['effectiveDateTime'] = pd.to_datetime(df['effectiveDateTime'])
    df['effectiveDateTime'] = df['effectiveDateTime'].apply(lambda x: x.replace(microsecond=0, second=0))
    df = df.dropna(subset=['code'])
    df_test = df['code'].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    df.drop(labels=['code', 'text'], axis=1, inplace=True)
    df_test = df['coding'].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    df.drop(labels=['coding'], axis=1, inplace=True)
    df_test = df[0].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    df.drop(labels=['system', 0], axis=1, inplace=True)
    df.rename(columns={'code': 'activity_code'}, inplace=True)
    # df = df[df.activity_code == act_code]
    df = pd.concat([df, df['valueQuantity'].apply(pd.Series)], axis=1)
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
    # df = date_range(df, start_date, end_date)
    # df = age_range(df, start_age, end_age)
    return df


def extract_sbp_dbp_columns(df, ehi_value, code, start_date, end_date):
    df = df[['subject', 'effectiveDateTime', 'code', 'encounter', 'component']]
    df = pd.concat([df.drop(['subject'], axis=1), df['subject'].apply(pd.Series)], axis=1)
    ref = df['reference']
    df.drop(labels=['reference'], axis=1, inplace=True)
    df.insert(0, 'subject_reference', ref)
    df['effectiveDateTime'] = df['effectiveDateTime'].apply(lambda x: x[0:x.find('.')])
    df['effectiveDateTime'] = pd.to_datetime(df['effectiveDateTime'])
    df['effectiveDateTime'] = df['effectiveDateTime'].dt.floor('T')
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

    df = extract_df_date_range(df, start_date, end_date, min_date, max_date)
    return df


def calculate_average(lst):
    return sum(lst) / len(lst)


def sliding_average(lst, window_size):
    return [sum(lst[i:i + window_size]) / window_size for i in range(len(lst) - window_size + 1)]


def roll_avg(df, usr, wndw_sz):
    round_lst = lambda lst: [round(x, 2) for x in lst]

    if usr == 'avg':
        # average
        avg_val = df.groupby(['subject_reference', 'Date'])['obs'].mean().reset_index()
        df_result = avg_val.groupby('subject_reference')['obs'].agg(list).reset_index()
        df_result['Rolling AVG'] = df_result['obs'].apply(lambda x: sliding_average(x, wndw_sz)).dropna()
        df_result = df_result[df_result['Rolling AVG'].apply(lambda x: len(x) > 0)]

        # If you want to reset the index of the resulting DataFrame
        df_result = df_result.reset_index(drop=True)

        # calculating average of rolling averages and assigning it to
        # Average column in the dataframe
        df_result['Average'] = df_result['Rolling AVG'].apply(lambda x: calculate_average(x))
        # rounding off the values to 2 decimal places
        df_result['obs'] = df_result['obs'].apply(round_lst)
        df_result['Rolling AVG'] = df_result['Rolling AVG'].apply(round_lst)
        df_result['Latest_value'] = df_result['Rolling AVG'].apply(lambda x: x[-1] if x else None)

        return df_result
    elif usr == 'latest':
        # latest
        df['effectiveDateTime'] = pd.to_datetime(df['effectiveDateTime'])
        latest_val = df.loc[df.groupby(['Date', 'subject_reference'])['effectiveDateTime'].idxmax()].reset_index()
        df_result = latest_val.groupby('subject_reference')[['obs', 'effectiveDateTime']].agg(list).reset_index()
        df_result = df_result[df_result['obs'].apply(lambda x: len(x) >= wndw_sz)]
        # print(df_result)

        # Apply sliding average to the 'obs' column and create a new 'Rolling AVG' column
        df_result['Rolling AVG'] = df_result['obs'].apply(lambda x: sliding_average(x, wndw_sz))

        # Filter rows with non-empty 'Rolling AVG' lists
        df_result = df_result[df_result['Rolling AVG'].apply(lambda x: len(x) > 0)]

        # If you want to reset the index of the resulting DataFrame
        df_result = df_result.reset_index(drop=True)

        # calculating average of rolling averages and assigning it to
        # Average column in the dataframe
        df_result['Average'] = df_result['Rolling AVG'].apply(lambda x: calculate_average(x))
        # rounding off the values to 2 decimal places
        df_result['obs'] = df_result['obs'].apply(round_lst)
        df_result['Rolling AVG'] = df_result['Rolling AVG'].apply(round_lst)
        df_result['Latest_value'] = df_result['Rolling AVG'].apply(lambda x: x[-1] if x else None)

        # print(df_result)
        return df_result


def subtract_lists(row):
    return [a - b for a, b in zip(row['SBP_OBS'], row['DBP_OBS'])]


def sbp_dbp_reading_cardiac(df, ehi_value, activity_code, start_date, end_date):
    df = extract_sbp_dbp_columns(df, ehi_value, activity_code, start_date, end_date)
    min_date, max_date = min(df['Date']), max(df['Date'])
    if ehi_value == "cadphr-sbp":
        df_patient = df.rename(columns={'obs': 'SBP_OBS'})
    else:
        df_patient = df.rename(columns={'obs': 'DBP_OBS'})
    return df_patient, min_date, max_date


def calculate_bsa(df):
    df['BSA'] = 0.007184 * np.power(df['weight'], 0.425) * np.power(df['height'], 0.725)
    return df


def cardiac_index(sbp_dataframe, dbp_dataframe, heartrate_dataframe, patient_df, height_df, weight_df, k_const, gender, start_age, end_age, start_date, end_date):
    sbp_dbp_code = "85354-9"
    heartrate_code = "40443-4"
    sbp_dataframe, min_date, max_date = sbp_dbp_reading_cardiac(sbp_dataframe, ehi_value='cadphr-sbp',
                                                                activity_code=sbp_dbp_code,
                                                                start_date=start_date,
                                                                end_date=end_date)

    dbp_dataframe, min_date, max_date = sbp_dbp_reading_cardiac(dbp_dataframe, ehi_value='cadphr-dbp',
                                                                activity_code=sbp_dbp_code,
                                                                start_date=start_date,
                                                                end_date=end_date)
    sbp_dataframe = sbp_dataframe[['subject_reference', 'effectiveDateTime', 'SBP_OBS']]
    dbp_dataframe = dbp_dataframe[['subject_reference', 'effectiveDateTime', 'DBP_OBS']]
    df_final = pd.merge(sbp_dataframe, dbp_dataframe, on=['subject_reference', 'effectiveDateTime'], how="inner")
    df_final = df_final.drop_duplicates()

    df_final['Pulse_Pressure'] = df_final["SBP_OBS"] - df_final["DBP_OBS"]
    heartrate_dataframe = extract_ehi_columns(heartrate_dataframe, ehi_value="cadphr-heartrate",
                                              act_code=heartrate_code)

    # heartrate_dataframe = roll_avg(heartrate_dataframe, day_condition, sliding_window)
    heartrate_dataframe = heartrate_dataframe.rename(columns={'obs': 'HR'})
    heartrate_dataframe = heartrate_dataframe[['subject_reference', 'effectiveDateTime', 'HR']]
    # print("Heartrate Dataframe", heartrate_dataframe)
    df_final = pd.merge(df_final, heartrate_dataframe, on=['subject_reference', 'effectiveDateTime'], how="inner")
    # print(df_final)
    df_final = df_final.rename(columns={"SBP_OBS": "SBP", "DBP_OBS": "DBP", "Pulse_Pressure": "PP"})
    df_final['MAP'] = df_final['DBP'] + (1 / 3) * df_final['PP']
    df_final['CO_EST'] = (df_final['PP'] * df_final['HR']) / (df_final['SBP'] + df_final['DBP'])
    df_final['CO_EST_ADJ'] = df_final['CO_EST'] * k_const
    df_patient = extract_patient_info(patient_df, gender)
    df_patient = age_range(df_patient, start_age, end_age)
    df_final = pd.merge(df_patient, df_final, on='subject_reference')
    df_height = extract_height(height_df)
    df_height = df_height.rename(columns={'obs': 'height_cm'})
    df_weight = extract_weight(weight_df)
    df_weight = df_weight.rename(columns={'obs': 'weight_kg'})
    df_height_weight = pd.merge(df_height, df_weight, on='subject_reference')
    df_height_weight = calculate_bsa(df_height_weight)
    df_final = pd.merge(df_height_weight, df_final, on='subject_reference')
    min_age, max_age = min(df_final['age']), max(df_final['age'])
    min_bsa, max_bsa = min(df_final['BSA']), max(df_final['BSA'])
    df_final['CI_DuBoisFormula'] = df_final['CO_EST_ADJ'] / df_final['BSA']
    df_final['CPI'] = df_final['MAP'] * (df_final['CI_DuBoisFormula'] / 451)
    return df_final, min_date, max_date, min_age, max_age, min_bsa, max_bsa
