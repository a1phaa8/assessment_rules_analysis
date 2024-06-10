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


def extract_ehi_grade(df, patient_data, ehi_val, start_d, end_d):
    df_result = df[['subject', 'ehiGrading', 'effectiveDateTime', 'computedValue']]
    # process the subject column as only the value of reference
    df_result = pd.concat([df_result.drop(['subject'], axis=1), df_result['subject'].apply(pd.Series)], axis=1)
    # inserting column at appropriate place
    ref = df_result['reference']
    df_result.drop(labels=['reference'], axis=1, inplace=True)
    df_result.insert(0, 'subject_reference', ref)
    # df_result['effectiveDateTime'] = df_result['effectiveDateTime'].apply(lambda x: x[0:x.find('.')])
    df_result['effectiveDateTime'] = pd.to_datetime(df_result['effectiveDateTime'], format='mixed')
    df_result['effectiveDateTime'] = df_result['effectiveDateTime'].apply(lambda x: x.replace(microsecond=0, second=0))

    df_result['Date'] = df_result['effectiveDateTime'].dt.date
    df_result['Time'] = df_result['effectiveDateTime'].dt.time
    date_min = min(df_result['Date'])
    date_max = max(df_result['Date'])
    # print("EHI_GRADE")
    # print("Size of df before date filtering:", df_result.shape)
    df_result = filter_df_date_range(df_result, start_d, end_d, date_min, date_max)
    # print("Size of df after date filtering:", df_result.shape)
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
    df_result = pd.merge(patient_data, df_result)
    # print("Size of df after merging:", df_result.shape)
    min_age = min(df_result['age'])
    max_age = max(df_result['age'])
    min_bmi = min(df_result['BMI'])
    max_bmi = max(df_result['BMI'])
    return df_result, date_min, date_max, min_age, max_age, min_bmi, max_bmi


# Define your filter_df_date_range function here

# Call the extract_ehi_grade function with your DataFrame, patient_data, ehi_val, start_d, and end_d

def get_latest_grade(grades_list):
    if not grades_list:
        return None  # Return None if the list is empty
    else:
        return grades_list[-1]


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
        'resprate': rr_grade
        # Add more mappings as needed
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


def affect_grades(df):
    df['cadphr-affect_grade'] = df.apply(lambda row: 1 if row['affect'] > 3 and row['PA'] < row['NA']
    else (2 if row['affect'] <= 3
          else (3 if row['affect'] > 3 and row['PA'] > row['NA']
                else None)), axis=1)
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
    # Grading logic based on age group, gender, and rolling average
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
    # Grading logic based on age group, gender, and rolling average
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

            return grade


def perceived_stress_grading(row):
    if 11 <= row['pss_value'] <= 16:
        return 1
    elif 6 <= row['pss_value'] <= 10:
        return 2
    elif 0 <= row['pss_value'] <= 5:
        return 3


# grading for pulse pressure
def pulsePressure_assign_grade(row):
    gender = row['gender']
    pp = row['Pulse Pressure']

    if gender == 'male':
        if pp >= 56:
            return 1
        elif 45 <= pp < 56:
            return 2
        elif (43 <= pp < 45) or (pp < 37):
            return 3
        elif 42 <= pp < 43:
            return 4
        elif 37 <= pp < 42:
            return 5
    elif gender == 'female':
        if pp >= 50:
            return 1
        elif 37 <= pp < 50:
            return 2
        elif (35 <= pp < 37) or (pp < 26):
            return 3
        elif 33 <= pp < 35:
            return 4
        elif 26 <= pp < 33:
            return 5


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
