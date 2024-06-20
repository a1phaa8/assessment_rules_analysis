from data_processing.descriptive_stats import *
from data_processing.return_required_dataframes import *
from data_processing.visual_stats import *

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

start_date = data["start_Date"]
end_date = data["end_Date"]
# gender = data["gender"]
age_lower, age_upper = data["start_age"], data["end_age"]
BMI_lower_limit, BMI_upper_limit = data["BMI_start"], data["BMI_end"]
des_opt = data["display_options"]
rules_names = data["rule_name_list"]

# converting to required data types
if start_date:
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
if end_date:
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
if age_lower:
    age_lower = int(age_lower)
if age_upper:
    age_upper = int(age_upper)
if BMI_lower_limit:
    BMI_lower_limit = float(BMI_lower_limit)
if BMI_upper_limit:
    BMI_upper_limit = float(BMI_upper_limit)


def compare_dates(strt_d, end_d, min_d, max_d):
    if strt_d != "" and end_d != "":
        if (strt_d < min_d and end_d > max_d) or (strt_d < min_d and end_d < min_d) or (
                strt_d > max_d and end_d > max_d) or (strt_d > max_d and end_d < min_d):
            strt_d = min_d
            end_d = max_d
        elif end_d > max_d:
            end_d = max_d
        elif strt_d < min_d:
            strt_d = min_d
    elif strt_d != "":
        if strt_d < min_d or strt_d > max_d:
            strt_d = min_d
        end_d = max_d
    elif end_d != "":
        if end_d > max_d or end_d < min_d:
            end_d = max_d
        strt_d = min_d
    return strt_d, end_d


def compare_age(start_age, end_age, min_age, max_age):
    if start_age and end_age:
        if start_age > max_age or start_age < min_age:
            print("WARNING\n")
            start_age = min_age
        if end_age < min_age or end_age > max_age:
            end_age = max_age
    if start_age == "" and end_age == "":
        start_age = min_age
        end_age = max_age
    elif start_age == "":
        if end_age > max_age or end_age < min_age:
            end_age = max_age
        start_age = min_age
    elif end_age == "":
        if start_age < min_age or start_age > max_age:
            start_age = min_age
        end_age = max_age
    else:
        if ((start_age < min_age and end_age > max_age) or (start_age > max_age and end_age > max_age) or (
                start_age < min_age and end_age < min_age) or
                (start_age > max_age and end_age < min_age)):
            start_age = min_age
            end_age = max_age
        elif start_age < min_age:
            start_age = min_age
        elif end_age > max_age:
            end_age = max_age
    return start_age, end_age


def assessment_rules_function(criteria_name, age_lower_bound, age_upper_bound, bmi_lower, bmi_upper, start_dte, end_dte,
                              gender, display_option):
    output_list = []
    required_dataframes, min_date, max_date, min_age, max_age, min_bmi, max_bmi, no_rows = process_data(criteria_name,
                                                                                                        age_lower_bound,
                                                                                                        age_upper_bound,
                                                                                                        start_dte,
                                                                                                        end_dte,
                                                                                                        bmi_lower,
                                                                                                        bmi_upper,
                                                                                                        gender)
    # print(required_dataframes.items())

    age_lower_bound, age_upper_bound = compare_age(age_lower_bound, age_upper_bound, min_age, max_age)
    start_d, end_d = compare_dates(start_dte, end_dte, min_date, max_date)
    if no_rows:
        age_lower_bound = min_age
        age_upper_bound = max_age
        start_d = min_date
        end_d = max_date

    for rule_name, dataframe in required_dataframes.items():
        print(f"Processed DataFrame for EHI {rule_name}:")
        print(dataframe[0])
    if ('descriptive_stats' in display_option) & ('visual_stats' in display_option):
        for rule_name, dataframe in required_dataframes.items():
            KF_bar = ""
            KF_pie = ""
            unique_KF_bar = ""
            unique_bar = assessment_rule_unique_criteria_bar(dataframe[0], rule_name, age_lower_bound, age_upper_bound,
                                                             start_d,
                                                             end_d, gender, dataframe[1])
            unique_pie = assessment_rule_unique_criteria_pie_chart(dataframe[0], rule_name, age_lower_bound,
                                                                   age_upper_bound, start_d,
                                                                   end_d, gender, dataframe[1])
            grouped_bar = assessment_rule_grouped_criteria_bar(dataframe[0], rule_name, age_lower_bound,
                                                               age_upper_bound,
                                                               start_d,
                                                               end_d, gender, dataframe[1])
            grouped_pie = assessment_rule_grouped_criteria_pie_chart(dataframe[0], rule_name, age_lower_bound,
                                                                     age_upper_bound, start_d,
                                                                     end_d,
                                                                     gender, dataframe[1])

            if rule_name in ['ruleset6_V4_combinatorial_rules_PulsePressure_SBP_rule',
                             'ruleset1_V2_1_Fitness_VO2max_Fatness_BMI_rule',
                             'ruleset3_V4_emotional_influx_Stress_Disorder_rule',
                             'ruleset6_V3_1_combinatorial_rules_SBP_DBP_rule',
                             'ruleset6_V3_1_combinatorial_hr_hrv_rule',
                             'ruleset3_V4_emotion_influx_anxiety_disorder_predictor_rule',
                             'ruleset3_V4_emotion_influx_depression_early_symptom_rule',
                             'ruleset6_V4_combinatorial_rules_target_HR_rule',
                             'ruleset6_V4_combinatorial_rules_cardiac_power_index_rule']:
                KF_bar = assessment_rule_KF_criteria_bar(dataframe[0], rule_name, age_lower_bound, age_upper_bound,
                                                         start_d,
                                                         end_d,
                                                         gender, dataframe[1])
                KF_pie = assessment_rule_KF_criteria_pie_chart(dataframe[0], rule_name, age_lower_bound,
                                                               age_upper_bound,
                                                               start_d,
                                                               end_d, gender, dataframe[1])
                unique_KF_bar = assessment_rule_unique_KF_criteria_bar(dataframe[0], rule_name, age_lower_bound,
                                                                       age_upper_bound, start_d,
                                                                       end_d, gender, dataframe[1])
            else:
                print("Key Finding doesn't exist for", rule_name)
            # print(assessment_unique_description(dataframe, rule_name)["unique_rules"])
            descriptive_unique_rule = assessment_unique_description(dataframe[0], rule_name, dataframe[1])
            descriptive_grouped_rule = assessment_grouped_description(dataframe[0], rule_name, dataframe[1])
            combined_list = [[unique_bar, unique_pie], [grouped_bar, grouped_pie], [KF_bar, KF_pie], [unique_KF_bar],
                             [descriptive_unique_rule, descriptive_grouped_rule], rule_name]
            output_list.append(combined_list)
        return output_list, min_date, max_date, min_age, max_age, min_bmi, max_bmi, no_rows
    elif 'descriptive_stats' in display_option:
        for rule_name, dataframe in required_dataframes.items():
            descriptive_unique_rule = assessment_unique_description(dataframe[0], rule_name, dataframe[1])
            descriptive_grouped_rule = assessment_grouped_description(dataframe[0], rule_name, dataframe[1])
            combined_list = [descriptive_unique_rule, descriptive_grouped_rule, rule_name]
            output_list.append(combined_list)
        return output_list, min_date, max_date, min_age, max_age, min_bmi, max_bmi, no_rows

    elif 'visual_stats' in display_option:
        for rule_name, dataframe in required_dataframes.items():
            KF_bar = ""
            KF_pie = ""
            unique_KF_bar = ""
            unique_bar = assessment_rule_unique_criteria_bar(dataframe[0], rule_name, age_lower_bound, age_upper_bound,
                                                             start_d,
                                                             end_d, gender, dataframe[1])
            grouped_bar = assessment_rule_grouped_criteria_bar(dataframe[0], rule_name, age_lower_bound,
                                                               age_upper_bound,
                                                               start_d,
                                                               end_d, gender, dataframe[1])

            unique_pie = assessment_rule_unique_criteria_pie_chart(dataframe[0], rule_name, age_lower_bound,
                                                                   age_upper_bound, start_d,
                                                                   end_d, gender, dataframe[1])
            grouped_pie = assessment_rule_grouped_criteria_pie_chart(dataframe[0], rule_name, age_lower_bound,
                                                                     age_upper_bound, start_d,
                                                                     end_d,
                                                                     gender, dataframe[1])
            if rule_name in ['ruleset6_V4_combinatorial_rules_PulsePressure_SBP_rule',
                             'ruleset1_V2_1_Fitness_VO2max_Fatness_BMI_rule',
                             'ruleset3_V4_emotional_influx_Stress_Disorder_rule',
                             'ruleset6_V3_1_combinatorial_rules_SBP_DBP_rule',
                             'ruleset6_V3_1_combinatorial_hr_hrv_rule',
                             'ruleset3_V4_emotion_influx_anxiety_disorder_predictor_rule',
                             'ruleset3_V4_emotion_influx_depression_early_symptom_rule',
                             'ruleset6_V4_combinatorial_rules_target_HR_rule',
                             'ruleset6_V4_combinatorial_rules_cardiac_power_index_rule']:
                KF_bar = assessment_rule_KF_criteria_bar(dataframe[0], rule_name, age_lower_bound, age_upper_bound,
                                                         start_d,
                                                         end_d,
                                                         gender, dataframe[1])
                KF_pie = assessment_rule_KF_criteria_pie_chart(dataframe[0], rule_name, age_lower_bound,
                                                               age_upper_bound,
                                                               "",
                                                               "", gender, dataframe[1])
                unique_KF_bar = assessment_rule_unique_KF_criteria_bar(dataframe[0], rule_name, age_lower_bound,
                                                                       age_upper_bound, start_d,
                                                                       end_d, gender, dataframe[1])
            else:
                print("Key Finding doesn't exist for", rule_name)
            combined_list = [[unique_bar, unique_pie], [grouped_bar, grouped_pie], [KF_bar, KF_pie], [unique_KF_bar],
                             rule_name]
            output_list.append(combined_list)
        return output_list, min_date, max_date, min_age, max_age, min_bmi, max_bmi, no_rows

# assessment_rules_function(rule_name, age_lower_limit, age_upper_limit, BMI_lower_limit, BMI_upper_limit, start_date,
#                           end_date,
#                           gender, des_opt)
