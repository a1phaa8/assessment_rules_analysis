import numpy as np
import pandas as pd


def filter_df_date_range(df, strt_d, end_d, min_d, max_d):
    if strt_d != "" and end_d != "":
        if (strt_d < min_d and end_d > max_d) or (strt_d < min_d and end_d < min_d) or (
                strt_d > max_d and end_d > max_d) or (strt_d > max_d and end_d < min_d):
            strt_d = min_d
            end_d = max_d
            df = df[df['Date'] >= strt_d]
            df = df[df['Date'] <= end_d]
            return df
        else:
            df = df[df['Date'] >= strt_d]
            df = df[df['Date'] <= end_d]
            return df
    elif strt_d != "":
        if strt_d < min_d:
            strt_d = min_d
        end_d = max_d
        df = df[df['Date'] >= strt_d]
        df = df[df['Date'] <= end_d]
        return df
    elif end_d != "":
        if end_d > max_d:
            end_d = max_d
        strt_d = min_d
        df = df[df['Date'] >= strt_d]
        df = df[df['Date'] <= end_d]
        return df
    else:
        return df


def sliding_average(lst, window_size):
    return [sum(lst[i:i + window_size]) / window_size for i in range(len(lst) - window_size + 1)]


def extract_ehi_computedValue(df, ehi_val, filtered_patient_data, all_patient_data, start_d, end_d, usr, wndw_sz):
    no_rows = 0
    df_result = df[['subject', 'effectiveDateTime', 'computedValue']]
    df_result.rename(columns={'computedValue': 'obs'}, inplace=True)
    # process the subject column as only the value of reference
    df_result = pd.concat([df_result.drop(['subject'], axis=1), df_result['subject'].apply(pd.Series)], axis=1)
    # inserting column at appropriate place
    ref = df_result['reference']
    df_result.drop(labels=['reference'], axis=1, inplace=True)
    df_result.insert(0, 'subject_reference', ref)
    df_result['effectiveDateTime'] = df_result['effectiveDateTime'].apply(lambda x: x[0:x.find('.')])
    df_result['effectiveDateTime'] = pd.to_datetime(df_result['effectiveDateTime'])
    df_result['effectiveDateTime'] = df_result['effectiveDateTime'].apply(lambda x: x.replace(microsecond=0, second=0))

    df_result['Date'] = df_result['effectiveDateTime'].dt.date
    df_result['Time'] = df_result['effectiveDateTime'].dt.time
    date_min = min(df_result['Date'])
    date_max = max(df_result['Date'])
    print("EHI_GRADE")
    print("Size of df before date filtering:", df_result.shape)
    df_result = filter_df_date_range(df_result, start_d, end_d, date_min, date_max)
    print("Size of df after date filtering:", df_result.shape)
    round_lst = lambda lst: [round(x, 2) for x in lst]

    if usr == 'avg':
        # average
        avg_val = df_result.groupby(['subject_reference', 'Date'])['obs'].mean().reset_index()
        df_result = avg_val.groupby('subject_reference')['obs'].agg(list).reset_index()
        df_result['Rolling AVG'] = df_result['obs'].apply(lambda x: sliding_average(x, wndw_sz)).dropna()
        df_result = df_result[df_result['Rolling AVG'].apply(lambda x: len(x) > 0)]

        # If you want to reset the index of the resulting DataFrame
        df_result = df_result.reset_index(drop=True)

        # calculating average of rolling averages and assigning it to
        # Average column in the dataframe
        df_result['Latest'] = df_result['Rolling AVG'].apply(lambda x: x[-1] if x else None)
        # rounding off the values to 2 decimal places
        df_result['obs'] = df_result['obs'].apply(round_lst)
        df_result['Rolling AVG'] = df_result['Rolling AVG'].apply(round_lst)
    elif usr == 'latest':
        # latest
        df_result['effectiveDateTime'] = pd.to_datetime(df_result['effectiveDateTime'])
        latest_val = df_result.loc[
            df_result.groupby(['Date', 'subject_reference'])['effectiveDateTime'].idxmax()].reset_index()
        df_result = latest_val.groupby('subject_reference')['obs'].agg(list).reset_index()
        if ehi_val == 'cadphr-vo2maxra':
            wndw_sz = 1
        df_result = df_result[df_result['obs'].apply(lambda x: len(x) >= wndw_sz)]
        # Apply sliding average to the 'obs' column and create a new 'Rolling AVG' column
        df_result['Rolling AVG'] = df_result['obs'].apply(lambda x: sliding_average(x, wndw_sz))
        # Filter rows with non-empty 'Rolling AVG' lists
        df_result = df_result[df_result['Rolling AVG'].apply(lambda x: len(x) > 0)]

        # If you want to reset the index of the resulting DataFrame
        df_result = df_result.reset_index(drop=True)

        # calculating average of rolling averages and assigning it to
        # Average column in the dataframe
        df_result['Latest'] = df_result['Rolling AVG'].apply(lambda x: x[-1] if x else None)
        # rounding off the values to 2 decimal places
        df_result['obs'] = df_result['obs'].apply(round_lst)
        df_result['Rolling AVG'] = df_result['Rolling AVG'].apply(round_lst)
    df_result_copy = df_result
    df_result = pd.merge(filtered_patient_data, df_result)
    print("Size of df after merging:", df_result.shape)
    if df_result.shape[0] == 0:
        no_rows = 1
        df_result = pd.merge(df_result_copy, all_patient_data)
        print("Size of df after new merging:", df_result.shape)
    min_age = min(df_result['age'])
    max_age = max(df_result['age'])
    min_bmi = min(df_result['BMI'])
    max_bmi = max(df_result['BMI'])
    return df_result, date_min, date_max, min_age, max_age, min_bmi, max_bmi, no_rows


def extract_ehi_grade(df, filtered_patient_data, all_patient_data, ehi_val, start_d, end_d):
    no_rows = 0
    df_result = df[['subject', 'ehiGrading', 'effectiveDateTime', 'computedValue']]
    # process the subject column as only the value of reference
    df_result = pd.concat([df_result.drop(['subject'], axis=1), df_result['subject'].apply(pd.Series)], axis=1)
    # inserting column at appropriate place
    ref = df_result['reference']
    df_result.drop(labels=['reference'], axis=1, inplace=True)
    df_result.insert(0, 'subject_reference', ref)
    df_result['effectiveDateTime'] = df_result['effectiveDateTime'].apply(lambda x: x[0:x.find('.')])
    df_result['effectiveDateTime'] = pd.to_datetime(df_result['effectiveDateTime'])
    df_result['effectiveDateTime'] = df_result['effectiveDateTime'].apply(lambda x: x.replace(microsecond=0, second=0))

    df_result['Date'] = df_result['effectiveDateTime'].dt.date
    df_result['Time'] = df_result['effectiveDateTime'].dt.time
    date_min = min(df_result['Date'])
    date_max = max(df_result['Date'])
    print("EHI_GRADE")
    print("Size of df before date filtering:", df_result.shape)
    df_result = filter_df_date_range(df_result, start_d, end_d, date_min, date_max)
    print("Size of df after date filtering:", df_result.shape)
    df_sorted = df_result.sort_values(by=['subject_reference', 'effectiveDateTime'], ascending=[True, False])
    # Group by 'id' and aggregate 'grade' and 'date' into lists
    df_result = df_sorted.groupby('subject_reference').agg(
        {'ehiGrading': list, 'effectiveDateTime': list, 'computedValue': list}).reset_index()
    latest_grade = ehi_val + "_grade"
    latest_value = ehi_val + "_value"
    # Create a new column 'latest_grade' to store the latest ehiGrading for each 'subject_reference'
    df_result[latest_grade] = df_sorted.groupby('subject_reference')['ehiGrading'].apply(
        lambda x: x.iloc[-1] if not x.empty else None).reset_index(drop=True)
    df_result[latest_value] = df_result['computedValue'].apply(lambda x: x[-1])
    df_result_copy = df_result
    df_result = pd.merge(filtered_patient_data, df_result)
    print("Size of df after merging:", df_result.shape)
    if df_result.shape[0] == 0:
        no_rows = 1
        df_result = pd.merge(df_result_copy, all_patient_data)
        print("Size of df after new merging:", df_result.shape)
    min_age = min(df_result['age'])
    max_age = max(df_result['age'])
    min_bmi = min(df_result['BMI'])
    max_bmi = max(df_result['BMI'])
    return df_result, date_min, date_max, min_age, max_age, min_bmi, max_bmi, no_rows


def add_grade_column(df, ehi_value):
    # Mapping between ehi values and grading functions
    grading_functions = {
        'heartrate': heartrate_grading,
        'hrv': hrv_grading,
        'prq': prq_grading,
        'dprp': dprp_grading,
        'sbp': sbp_grading,
        'dbp': dbp_grading,
        'emotioninstability': emotioninstability_grading,
        'emotioninertia': emotioninertia_grading,
        'emotionvariability': emotionvariability_grading,
        'pulsepressure': pulsePressure_assign_grade,
        'vo2maxra': vo2max_grading,
        'cadrisk10': cadrisk_grading,
        'affect': affect_grades,
        'pa': pa_grading,
        'na': na_grading,
        'resprate': rr_grade
    }

    # Split ehi_value to get the specific grading function
    split_result = ehi_value.split("-")
    grading_function_key = split_result[1]

    # Check if the grading function exists in the mapping
    if grading_function_key in grading_functions:
        # Call the corresponding grading function
        grading_function = grading_functions[grading_function_key]
        df['grades'] = df.apply(grading_function, axis=1)
        latest_grade = ehi_value + "_grade"
        df[latest_grade] = df['grades'].apply(get_latest_grade)
    else:
        # Handle the case where there's no matching grading function
        print(f"No grading function found for {grading_function_key}")

    return df


def affect_grades(df):
    df['cadphr-affect_grade'] = df.apply(lambda row: 1 if row['affect'] <= -3
    else (2 if -3 < row['affect'] <= 0
          else (3 if 0 < row['affect'] <= 3
                else (4 if row['affect'] >= 3
                      else None))), axis=1)
    return df


def emotionvariability_grading(ev_list):
    grade = []

    for ev in ev_list:
        if ev > 0.5:
            grade.append(1)
        elif 0.3 <= ev <= 0.5:
            grade.append(2)
        elif ev < 0.3:
            grade.append(3)

    return grade


def emotioninstability_grading(ei_list):
    grades = []
    for ei in ei_list:
        if ei > 0.5:
            grades.append(1)
        elif 0.3 <= ei <= 0.5:
            grades.append(2)
        elif ei < 0.3:
            grades.append(3)
    return grades


def emotioninertia_grading(row):
    grades = []

    # Check if ei_list is a list
    if isinstance(row['emotioninertia'], list):
        ei_list = row['emotioninertia']

        for ei in ei_list:
            if ei > 0.5:
                grades.append(1)
            elif 0.3 <= ei <= 0.5:
                grades.append(2)
            elif ei < 0.3:
                grades.append(3)
    else:
        # Handle the case where 'emotioninertia' is a single float value
        ei = row['emotioninertia']
        if ei > 0.5:
            grades.append(1)
        elif 0.3 <= ei <= 0.5:
            grades.append(2)
        elif ei < 0.3:
            grades.append(3)

    return grades


def dprp_grading(row):
    grade = []
    rpp_list = row['Rolling AVG']
    for rpp in rpp_list:
        if rpp >= 30000:
            grade.append(1)
        elif 25000 <= rpp < 30000:
            grade.append(2)
        elif 20000 <= rpp < 25000:
            grade.append(3)
        elif 15000 <= rpp < 20000:
            grade.append(4)
        elif 10000 <= rpp < 15000:
            grade.append(5)
        elif rpp < 10000:
            grade.append(6)
        return grade


def get_latest_grade(grades_list):
    if not grades_list:
        return None  # Return None if the list is empty
    else:
        return grades_list[-1]


def prq_grading(row):
    conditions = [
        lambda x: x >= 10 or x < 2,
        lambda x: (8 <= x < 10) or (2 <= x < 2.5),
        lambda x: (5 <= x < 8) or (2.5 <= x < 3),
        lambda x: (4.5 <= x < 5) or (3 <= x < 3.5),
        lambda x: 3.5 <= x < 4.5
    ]
    grades = [1, 2, 3, 4, 5]
    # Apply grading conditions to each value in the 'Rolling AVG' list
    grades_list = [grades[np.argmax([condition(value) for condition in conditions])] for value in row['Rolling AVG']]
    return grades_list


def hrv_grading(row):
    grade = []
    if 18 <= row['age'] <= 34:

        if row['gender'] == 'male':
            for score in row['Rolling AVG']:
                if score < 19.8:
                    grade.append(1)
                elif 19.8 <= score < 59.6:
                    grade.append(2)
                elif score >= 59.6:
                    grade.append(3)
            return grade

        elif row['gender'] == 'female':

            for score in row['Rolling AVG']:
                if score < 20.1:
                    grade.append(1)
                elif 20.1 <= score < 65.7:
                    grade.append(2)
                elif score >= 65.7:
                    grade.append(3)

            return grade

    elif 35 <= row['age'] <= 44:

        if row['gender'] == 'male':
            for score in row['Rolling AVG']:
                if score < 15.5:
                    grade.append(1)
                elif 15.5 <= score < 48.5:
                    grade.append(2)
                elif score >= 48.5:
                    grade.append(3)

            return grade

        elif row['gender'] == 'female':
            for score in row['Rolling AVG']:
                if score < 16.9:
                    grade.append(1)
                elif 16.9 <= score < 53.9:
                    grade.append(2)
                elif score >= 53.9:
                    grade.append(3)

            return grade
    elif 45 <= row['age'] <= 54:
        if row['gender'] == 'male':
            for score in row['Rolling AVG']:
                if score < 12.1:
                    grade.append(1)
                elif 12.1 <= score < 33.9:
                    grade.append(2)
                elif score >= 33.9:
                    grade.append(3)
            return grade
        elif row['gender'] == 'female':

            for score in row['Rolling AVG']:
                if score < 12.7:
                    grade.append(1)
                elif 12.7 <= score < 39.9:
                    grade.append(2)
                elif score >= 39.9:
                    grade.append(3)

            return grade
    elif 55 <= row['age'] <= 64:
        if row['gender'] == 'male':
            for score in row['Rolling AVG']:
                if score < 8.8:
                    grade.append(1)
                elif 8.8 <= score < 31.0:
                    grade.append(2)
                elif score >= 31.0:
                    grade.append(3)

            return grade
        elif row['gender'] == 'female':

            for score in row['Rolling AVG']:
                if score < 9.5:
                    grade.append(1)
                elif 9.5 <= score < 33.3:
                    grade.append(2)
                elif score >= 33.3:
                    grade.append(3)

            return grade
    elif 65 <= row['age'] <= 84:
        if row['gender'] == 'male':
            for score in row['Rolling AVG']:
                if score < 8.4:
                    grade.append(1)
                elif 8.4 <= score < 29.8:
                    grade.append(2)
                elif score >= 29.8:
                    grade.append(3)

            return grade
        elif row['gender'] == 'female':

            for score in row['Rolling AVG']:
                if score < 7.3:
                    grade.append(1)
                elif 7.3 <= score < 30.9:
                    grade.append(2)
                elif score >= 30.9:
                    grade.append(3)

            return grade


def heartrate_grading(row):
    grade = []
    if 18 <= row['age'] <= 25:
        if row['gender'] == 'male':
            for score in row['Rolling AVG']:
                if 49 <= score <= 55:
                    grade.append('5')
                elif 56 <= score <= 60:
                    grade.append('4')
                elif 61 <= score <= 69:
                    grade.append('3')
                elif 70 <= score <= 81:
                    grade.append('2')
                else:
                    grade.append('1')

            return grade
        elif row['gender'] == 'female':

            for score in row['Rolling AVG']:
                if 54 <= score <= 60:
                    grade.append('5')
                elif 61 <= score <= 65:
                    grade.append('4')
                elif 66 <= score <= 73:
                    grade.append('3')
                elif 74 <= score <= 84:
                    grade.append('2')
                else:
                    grade.append('1')

            return grade
    elif 26 <= row['age'] <= 35:
        if row['gender'] == 'male':
            for score in row['Rolling AVG']:
                if 49 <= score <= 54:
                    grade.append('5')
                elif 55 <= score <= 61:
                    grade.append('4')
                elif 62 <= score <= 70:
                    grade.append('3')
                elif 71 <= score <= 82:
                    grade.append('2')
                else:
                    grade.append('1')

            return grade
        elif row['gender'] == 'female':

            for score in row['Rolling AVG']:
                if 54 <= score <= 59:
                    grade.append('5')
                elif 60 <= score <= 64:
                    grade.append('4')
                elif 65 <= score <= 72:
                    grade.append('3')
                elif 73 <= score <= 82:
                    grade.append('2')
                else:
                    grade.append('1')

            return grade
    elif 36 <= row['age'] <= 45:
        if row['gender'] == 'male':
            for score in row['Rolling AVG']:
                if 49 <= score <= 54:
                    grade.append('5')
                elif 55 <= score <= 61:
                    grade.append('4')
                elif 62 <= score <= 70:
                    grade.append('3')
                elif 71 <= score <= 82:
                    grade.append('2')
                else:
                    grade.append('1')

            return grade
        elif row['gender'] == 'female':

            for score in row['Rolling AVG']:
                if 54 <= score <= 59:
                    grade.append('5')
                elif 60 <= score <= 64:
                    grade.append('4')
                elif 65 <= score <= 73:
                    grade.append('3')
                elif 74 <= score <= 84:
                    grade.append('2')
                else:
                    grade.append('1')

            return grade
    elif 46 <= row['age'] <= 55:
        if row['gender'] == 'male':
            for score in row['Rolling AVG']:
                if 50 <= score <= 57:
                    grade.append('5')
                elif 58 <= score <= 63:
                    grade.append('4')
                elif 64 <= score <= 71:
                    grade.append('3')
                elif 72 <= score <= 83:
                    grade.append('2')
                else:
                    grade.append('1')

            return grade
        elif row['gender'] == 'female':
            for score in row['Rolling AVG']:
                if 54 <= score <= 60:
                    grade.append('5')
                elif 61 <= score <= 65:
                    grade.append('4')
                elif 66 <= score <= 73:
                    grade.append('3')
                elif 74 <= score <= 83:
                    grade.append('2')
                else:
                    grade.append('1')

            return grade
    elif 56 <= row['age'] <= 65:
        if row['gender'] == 'male':
            for score in row['Rolling AVG']:
                if 51 <= score <= 56:
                    grade.append('5')
                elif 57 <= score <= 61:
                    grade.append('4')
                elif 62 <= score <= 71:
                    grade.append('3')
                elif 72 <= score <= 81:
                    grade.append('2')
                else:
                    grade.append('1')

            return grade
        elif row['gender'] == 'female':
            for score in row['Rolling AVG']:
                if 54 <= score <= 59:
                    grade.append('5')
                elif 60 <= score <= 64:
                    grade.append('4')
                elif 65 <= score <= 73:
                    grade.append('3')
                elif 74 <= score <= 83:
                    grade.append('2')
                else:
                    grade.append('1')

    elif row['age'] > 65:
        if row['gender'] == 'male':
            for score in row['Rolling AVG']:
                if 50 <= score <= 55:
                    grade.append('5')
                elif 56 <= score <= 61:
                    grade.append('4')
                elif 62 <= score <= 69:
                    grade.append('3')
                elif 70 <= score <= 79:
                    grade.append('2')
                else:
                    grade.append('1')

            return grade
        elif row['gender'] == 'female':
            for score in row['Rolling AVG']:
                if 54 <= score <= 59:
                    grade.append('5')
                elif 60 <= score <= 64:
                    grade.append('4')
                elif 65 <= score <= 72:
                    grade.append('3')
                elif 73 <= score <= 83:
                    grade.append('2')
                else:
                    grade.append('1')

            return grade


# grading for pulse pressure
def pulsePressure_assign_grade(row):
    gender = row['gender']
    pulsepressure = row['PP_obs']
    grade = []

    if gender == 'male':
        for pp in pulsepressure:
            if pp >= 56:
                grade.append('1')
            elif 45 <= pp < 56:
                grade.append('2')
            elif (43 <= pp < 45) or (pp < 37):
                grade.append('3')
            elif 42 <= pp < 43:
                grade.append('4')
            elif 37 <= pp < 42:
                grade.append('5')
    elif gender == 'female':
        for pp in pulsepressure:
            if pp >= 50:
                grade.append('1')
            elif 37 <= pp < 50:
                grade.append('2')
            elif (35 <= pp < 37) or (pp < 26):
                grade.append('3')
            elif 33 <= pp < 35:
                grade.append('4')
            elif 26 <= pp < 33:
                grade.append('5')
    return grade


def sbp_grading(row):
    sbp_list = row['Rolling AVG']

    grades = []

    for sbp in sbp_list:
        if sbp >= 180 or sbp < 70:
            grades.append(1)
        elif (140 <= sbp < 180) or (70 <= sbp < 80):
            grades.append(2)
        elif (130 <= sbp < 140) or (80 <= sbp < 90):
            grades.append(3)
        elif 120 <= sbp < 130 or 90 <= sbp < 110:
            grades.append(4)
        elif 110 <= sbp < 120:
            grades.append(5)

    return grades


def dbp_grading(row):
    dbp_list = row['Rolling AVG']

    grades = []

    for dbp in dbp_list:
        if dbp >= 120 or dbp < 50:
            grades.append(1)
        elif (95 <= dbp < 120) or (50 <= dbp < 60):
            grades.append(2)
        elif 85 <= dbp < 95:
            grades.append(3)
        elif 80 <= dbp < 85 or 60 <= dbp < 70:
            grades.append(4)
        elif 70 <= dbp < 80:
            grades.append(5)
    return grades


def pa_grading(df):
    df['cadphr-pa_grade'] = df['PA'].apply(lambda x: 1 if x < 2 else (2 if 2 <= x <= 4 else 3))
    return df


def na_grading(df):
    df['cadphr-na_grade'] = df['NA'].apply(lambda x: 3 if x < 2 else (2 if 2 <= x <= 4 else 1))
    return df


def vo2max_grading(row):
    vo2max_list = row['Rolling AVG']
    grades = []

    if row['gender'] == 'male':
        if 18 <= row['age'] < 28:
            for vo2max in vo2max_list:
                if vo2max < 31:
                    grades.append(1)
                elif 31 <= vo2max < 41:
                    grades.append(2)
                elif 41 <= vo2max < 48:
                    grades.append(3)
                elif 48 <= vo2max < 54:
                    grades.append(4)
                elif vo2max >= 54:
                    grades.append(5)
            return grades

        elif 28 <= row['age'] < 35:
            for vo2max in vo2max_list:
                if vo2max < 27:
                    grades.append(1)
                elif 27 <= vo2max < 38:
                    grades.append(2)
                elif 38 <= vo2max < 44:
                    grades.append(3)
                elif 44 <= vo2max < 50:
                    grades.append(4)
                elif vo2max >= 50:
                    grades.append(5)
            return grades

        elif 35 <= row['age'] < 43:
            for vo2max in vo2max_list:
                if vo2max < 25:
                    grades.append(1)
                elif 25 <= vo2max < 35:
                    grades.append(2)
                elif 35 <= vo2max < 43:
                    grades.append(3)
                elif 43 <= vo2max < 50:
                    grades.append(4)
                elif vo2max >= 50:
                    grades.append(5)
            return grades

        elif 43 <= row['age'] < 52:
            for vo2max in vo2max_list:
                if vo2max < 19:
                    grades.append(1)
                elif 19 <= vo2max < 32:
                    grades.append(2)
                elif 32 <= vo2max < 36:
                    grades.append(3)
                elif 36 <= vo2max < 43:
                    grades.append(4)
                elif vo2max >= 43:
                    grades.append(5)
            return grades

        elif 52 <= row['age'] < 60:
            for vo2max in vo2max_list:
                if vo2max < 18:
                    grades.append(1)
                elif 18 <= vo2max < 26:
                    grades.append(2)
                elif 26 <= vo2max < 33:
                    grades.append(3)
                elif 33 <= vo2max < 40:
                    grades.append(4)
                elif vo2max >= 40:
                    grades.append(5)
            return grades

        elif 60 <= row['age'] <= 80:
            for vo2max in vo2max_list:
                if vo2max < 17:
                    grades.append(1)
                elif 17 <= vo2max < 22:
                    grades.append(2)
                elif 22 <= vo2max < 26:
                    grades.append(3)
                elif 26 <= vo2max < 32:
                    grades.append(4)
                elif vo2max >= 32:
                    grades.append(5)
            return grades

    elif row['gender'] == 'female':
        if 18 <= row['age'] < 28:
            for vo2max in vo2max_list:
                if vo2max < 21:
                    grades.append(1)
                elif 21 <= vo2max < 29:
                    grades.append(2)
                elif 29 <= vo2max < 34:
                    grades.append(3)
                elif 34 <= vo2max < 39:
                    grades.append(4)
                elif vo2max >= 39:
                    grades.append(5)
            return grades

        elif 28 <= row['age'] < 35:
            for vo2max in vo2max_list:
                if vo2max < 19:
                    grades.append(1)
                elif 19 <= vo2max < 27:
                    grades.append(2)
                elif 27 <= vo2max < 31:
                    grades.append(3)
                elif 31 <= vo2max < 37:
                    grades.append(4)
                elif vo2max >= 37:
                    grades.append(5)
            return grades

        elif 35 <= row['age'] < 43:
            for vo2max in vo2max_list:
                if vo2max < 17:
                    grades.append(1)
                elif 17 <= vo2max < 25:
                    grades.append(2)
                elif 25 <= vo2max < 31:
                    grades.append(3)
                elif 31 <= vo2max < 36:
                    grades.append(4)
                elif vo2max >= 36:
                    grades.append(5)
            return grades

        elif 43 <= row['age'] < 52:
            for vo2max in vo2max_list:
                if vo2max < 16:
                    grades.append(1)
                elif 16 <= vo2max < 21:
                    grades.append(2)
                elif 21 <= vo2max < 25:
                    grades.append(3)
                elif 25 <= vo2max < 33:
                    grades.append(4)
                elif vo2max >= 33:
                    grades.append(5)
            return grades

        elif 52 <= row['age'] < 60:
            for vo2max in vo2max_list:
                if vo2max < 14:
                    grades.append(1)
                elif 14 <= vo2max < 20:
                    grades.append(2)
                elif 20 <= vo2max < 22:
                    grades.append(3)
                elif 22 <= vo2max < 25:
                    grades.append(4)
                elif vo2max >= 25:
                    grades.append(5)
            return grades

        elif 60 <= row['age'] <= 80:
            for vo2max in vo2max_list:
                if vo2max < 13:
                    grades.append(1)
                elif 13 <= vo2max < 17:
                    grades.append(2)
                elif 17 <= vo2max < 20:
                    grades.append(3)
                elif 20 <= vo2max < 23:
                    grades.append(4)
                elif vo2max >= 23:
                    grades.append(5)
            return grades


def cadrisk_grading(row):
    grades = []
    cad_list = row['Rolling AVG']
    for cad in cad_list:
        if cad > 10:
            grades.append(1)
        elif 7 < cad <= 10:
            grades.append(2)
        elif 4 < cad <= 7:
            grades.append(3)
        elif 1 < cad <= 4:
            grades.append(4)
        elif cad <= 1:
            grades.append(5)
    return grades


def henergy_grade(row):
    grades = []
    val = row['Rolling AVG']
    for x in val:
        if x <= 20:
            grades.append(1)
        elif 20 < x <= 40:
            grades.append(2)
        elif 40 < x <= 60:
            grades.append(3)
        elif 60 < x <= 80:
            grades.append(4)
        elif x > 80:
            grades.append(5)
    return grades


def perceived_stress_grading(row):
    if 11 <= row['pss_value'] <= 16:
        return 1
    elif 6 <= row['pss_value'] <= 10:
        return 2
    elif 0 <= row['pss_value'] <= 5:
        return 3


def rr_grade(df):
    grades = []
    resprate = df['Rolling AVG']
    for val in resprate:
        if 6 < val <= 27:
            grades.append(1)
        elif (6 <= val < 8) or (22 <= val < 27):
            grades.append(2)
        elif 8 <= val < 12:
            grades.append(3)
        elif 18 <= val < 22:
            grades.append(4)
        elif 12 <= val < 18:
            grades.append(5)
    return grades
