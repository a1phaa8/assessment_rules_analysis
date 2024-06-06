import math
import numpy as np


def calculate_average(lst):
    return sum(lst) / len(lst)


def sliding_average(lst, window_size):
    return [sum(lst[i:i + window_size]) / window_size for i in range(len(lst) - window_size + 1)]


def roll_avg_henergy(df, wndw_sz):
    round_lst = lambda lst: [round(x, 2) for x in lst]
    # Check if 'hr' and 'hrv' columns exist
    if 'hr' not in df.columns or 'hrv' not in df.columns:
        raise ValueError("Columns 'hr' and 'hrv' are required in the DataFrame.")
    # df=df[(df['hr'].apply(len) >= 3) & (df['hrv'].apply(len) >= 3)]
    # Calculate rolling average for 'hr' and 'hrv'
    df['hr_rolling_avg'] = df['hr'].apply(lambda x: sliding_average(x, wndw_sz)).reset_index(drop=True).dropna()
    df['hrv_rolling_avg'] = df['hrv'].apply(lambda x: sliding_average(x, wndw_sz)).reset_index(drop=True).dropna()
    df = df[(df['hr_rolling_avg'].apply(len) > 0) & (df['hrv_rolling_avg'].apply(len) > 0)]
    df['hr_rolling_avg'] = df['hr_rolling_avg'].apply(round_lst)
    df['hrv_rolling_avg'] = df['hrv_rolling_avg'].apply(round_lst)
    return df


def processing_instaBeats(df_real):
    hr_min = 0
    hr_max = 130
    hrv_min = 0
    hrv_max = 100

    new_scale_hr_min = -1
    new_scale_hr_max = 1
    new_scale_hrv_min = -1
    new_scale_hrv_max = 1

    beats_scale_min = 0
    beats_scale_max = 100
    ###### Convert the values to a new scale of -1 to +1
    ## hr: y-axis, hrv: x-axis
    df_real['hrv_new_scale'] = df_real['hrv_rolling_avg'].apply(
        lambda x: map_from_one_scale_to_another(hrv_min, hrv_max, new_scale_hrv_min, new_scale_hrv_max, x))
    df_real['hr_new_scale'] = df_real['hr_rolling_avg'].apply(
        lambda x: map_from_one_scale_to_another(hr_min, hr_max, new_scale_hr_min, new_scale_hr_max, x))

    df_real['hrv_opt_new_scale'] = df_real['hrv_opt'].apply(
        lambda x: map_from_one_scale_to_another(hrv_min, hrv_max, new_scale_hrv_min, new_scale_hrv_max, x))
    df_real['hr_opt_new_scale'] = df_real['hr_opt'].apply(
        lambda x: map_from_one_scale_to_another(hr_min, hr_max, new_scale_hr_min, new_scale_hr_max, x))

    ###### Extract the extreme boundaries and  gold standard point
    boundary_point_1, boundary_point_2, boundary_point_3, boundary_point_4 = \
        define_extreme_boundary_points(new_scale_hr_min, new_scale_hr_max, new_scale_hrv_min, new_scale_hrv_max)
    boundary_point_polar_1 = convert_cartesian_to_polar_point(boundary_point_1[0], boundary_point_1[1])
    boundary_point_polar_2 = convert_cartesian_to_polar_point(boundary_point_2[0], boundary_point_2[1])
    boundary_point_polar_3 = convert_cartesian_to_polar_point(boundary_point_3[0], boundary_point_3[1])
    boundary_point_polar_4 = convert_cartesian_to_polar_point(boundary_point_4[0], boundary_point_4[1])

    ###### Convert cartessian to polar co-ordinate
    df_real['polar_new_scale'] = df_real.apply(
        lambda x: convert_cartesian_to_polar_point(x['hrv_new_scale'], x['hr_new_scale']), axis=1)

    df_real['gold_point_polar_new_scale'] = df_real.apply(
        lambda x: convert_cartesian_to_polar_point(x['hrv_opt_new_scale'], x['hr_opt_new_scale']), axis=1)

    ##*********************************************
    ###### Calculate the normalizing factor

    df_real['normalizing_factor'] = df_real.apply(
        lambda x: calculate_normalized_factor(boundary_point_polar_1, boundary_point_polar_2,
                                              boundary_point_polar_3, boundary_point_polar_4,
                                              x['gold_point_polar_new_scale']), axis=1)

    # ###### Calculate the polar distance of each point from Gold-standard
    df_real['polar_dist_from_gold_std'] = df_real.apply(
        lambda x: calculate_polar_distance_list(x.gold_point_polar_new_scale[0], x.gold_point_polar_new_scale[1],
                                                x.polar_new_scale), axis=1)

    ###### calculate the score as normalized relative distance from Gold-standard
    df_real['score'] = df_real.apply(lambda x: calculate_score(x.polar_dist_from_gold_std, x.normalizing_factor),
                                     axis=1)

    ###### Calculate the final InstaBEATS for the users
    df_real['InstaBEATS'] = df_real['score'].apply(
        lambda x: map_score_to_beats_scale(beats_scale_min, beats_scale_max, x))
    return df_real


def truncate(f, n):
    return np.round(f, n)


#     return f
#     return math.floor(f * 10 ** n) / 10 ** n

def get_optimal_HR(age, gender):
    ## optimal heart rate
    if gender == 'male':
        if 18 <= age <= 25:
            hr_opt = 49
        elif 26 <= age <= 35:
            hr_opt = 49
        elif 36 <= age <= 45:
            hr_opt = 50
        elif 46 <= age <= 55:
            hr_opt = 50
        elif 56 <= age <= 65:
            hr_opt = 51
        elif age >= 66:
            hr_opt = 50
    elif gender == 'female':
        if 18 <= age <= 25:
            hr_opt = 54
        elif 26 <= age <= 35:
            hr_opt = 54
        elif 36 <= age <= 45:
            hr_opt = 54
        elif 46 <= age <= 55:
            hr_opt = 54
        elif 56 <= age <= 65:
            hr_opt = 54
        elif age >= 66:
            hr_opt = 54
    return hr_opt


def get_optimal_HRV(age, gender):
    ## optimal heart rate variability
    if gender == 'male':
        if 18 <= age <= 34:
            hrv_opt = 59.6
        elif 35 <= age <= 44:
            hrv_opt = 48.5
        elif 45 <= age <= 54:
            hrv_opt = 33.9
        elif 55 <= age <= 64:
            hrv_opt = 31
        elif age >= 65:
            hrv_opt = 29.8
    elif gender == 'female':
        if 18 <= age <= 34:
            hrv_opt = 65.7
        elif 35 <= age <= 44:
            hrv_opt = 53.9
        elif 45 <= age <= 54:
            hrv_opt = 39.9
        elif 55 <= age <= 64:
            hrv_opt = 33.3
        elif age >= 65:
            hrv_opt = 30.9
    return hrv_opt


def map_from_one_scale_to_another(s1_min, s1_max, s2_min, s2_max, val_s1_list):
    # Ensure val_s1_list is iterable
    if not isinstance(val_s1_list, (list, tuple, np.ndarray)):
        val_s1_list = [val_s1_list]

    val_s2_list = []
    for val_s1 in val_s1_list:
        factor = (s2_max - s2_min) / (s1_max - s1_min)
        val_s2 = ((val_s1 - s1_min) * factor) + s2_min
        val_s2_list.append(val_s2)

    # Return a single value if input was a single value, otherwise return the list
    return val_s2_list[0] if len(val_s2_list) == 1 else val_s2_list


def define_extreme_boundary_points(new_scale_hr_min, new_scale_hr_max, new_scale_hrv_min, new_scale_hrv_max):
    boundary_point_1 = [new_scale_hr_min, new_scale_hrv_min]
    boundary_point_2 = [new_scale_hr_min, new_scale_hrv_max]
    boundary_point_3 = [new_scale_hr_max, new_scale_hrv_min]
    boundary_point_4 = [new_scale_hr_max, new_scale_hrv_max]
    return boundary_point_1, boundary_point_2, boundary_point_3, boundary_point_4


def convert_cartesian_to_polar_point(point_x, point_y):
    # Ensure point_x and point_y are iterable (lists or single values)
    if not isinstance(point_x, (list, tuple, np.ndarray)):
        point_x = [point_x]
    if not isinstance(point_y, (list, tuple, np.ndarray)):
        point_y = [point_y]

    polar_points = []
    for x, y in zip(point_x, point_y):
        val = (x * x) + (y * y)
        r = np.sqrt(val)
        r = truncate(r, 2)
        theta = np.arctan2(y, x)
        polar_points.append((r, theta))

    # Return a single point if input was a single point, otherwise return the list
    return polar_points[0] if len(polar_points) == 1 else polar_points


def calculate_polar_distance_list(r1, theta1, polar_point):
    f_dist = []
    pi = 22 / 7
    if type(polar_point) == list:
        for polar in polar_point:
            theta2 = polar[1]
            radian1 = theta1 * (pi / 180)
            radian2 = theta2 * (pi / 180)
            r2 = polar[0]

            temp_val = (r1 * r1) + (r2 * r2) - (2 * r1 * r2 * math.cos((radian2 - radian1)))
            dist = np.sqrt(temp_val)
            dist = truncate(dist, 2)
            f_dist.append(dist)
    else:
        theta2 = polar_point[1]
        radian1 = theta1 * (pi / 180)
        radian2 = theta2 * (pi / 180)
        r2 = polar_point[0]

        temp_val = (r1 * r1) + (r2 * r2) - (2 * r1 * r2 * math.cos((radian2 - radian1)))
        dist = np.sqrt(temp_val)
        dist = truncate(dist, 2)
        f_dist.append(dist)

    return f_dist


def calculate_polar_distance(r1, theta1, r2, theta2):
    pi = 22 / 7
    radian1 = theta1 * (pi / 180)
    radian2 = theta2 * (pi / 180)

    temp_val = (r1 * r1) + (r2 * r2) - (2 * r1 * r2 * math.cos((radian2 - radian1)))
    dist = np.sqrt(temp_val)
    dist = truncate(dist, 2)

    return dist


def calculate_normalized_factor(boundary_point_polar_1, boundary_point_polar_2,
                                boundary_point_polar_3, boundary_point_polar_4, gold_point_polar):
    dist1 = calculate_polar_distance(gold_point_polar[0], gold_point_polar[1],
                                     boundary_point_polar_1[0], boundary_point_polar_1[1])
    dist2 = calculate_polar_distance(gold_point_polar[0], gold_point_polar[1],
                                     boundary_point_polar_2[0], boundary_point_polar_2[1])
    dist3 = calculate_polar_distance(gold_point_polar[0], gold_point_polar[1],
                                     boundary_point_polar_3[0], boundary_point_polar_3[1])
    dist4 = calculate_polar_distance(gold_point_polar[0], gold_point_polar[1],
                                     boundary_point_polar_4[0], boundary_point_polar_4[1])
    normalized_factor = max(dist1, dist2, dist3, dist4)

    return normalized_factor


def calculate_score(present_distance, normalized_factor):
    score_lst = []
    for val in present_distance:
        normalized_dist = val / normalized_factor
        normalized_dist = truncate(normalized_dist, 2)
        score = 1.0 - normalized_dist
        score_lst.append(score)

    return score_lst


def map_score_to_beats_scale(beats_scale_min, beats_scale_max, score):
    lst_BEATS = []
    for x in score:
        instaBEATS = beats_scale_min + ((beats_scale_max - beats_scale_min) * x)
        lst_BEATS.append(instaBEATS)
    return lst_BEATS


def get_instabeats_grade(val):
    grade_lst = []
    for x in val:
        if x <= 20:
            grade = 1
        elif 20 < x <= 40:
            grade = 2
        elif 40 < x <= 60:
            grade = 3
        elif 60 < x <= 80:
            grade = 4
        elif x > 80:
            grade = 5
        grade_lst.append(grade)
    return grade_lst
