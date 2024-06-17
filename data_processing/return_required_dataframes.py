from assessment_rules_definition.assessment_rule_functions import *
from .optimized_dataframes_processing import *

data = pd.read_json('configuration.json', typ='series')
sliding_window = data["sliding_window"]
day_condition = data["day_condition"]
min_sufficiency = data["min_sufficiency"]
max_sufficiency = data["max_sufficiency"]
user_data_file = data["User_data"]
trend_window = data["trend_window_size"]


def process_data(rule_names, start_age, end_age, start_date, end_date, BMI_lower_limit, BMI_upper_limit, data_select):
    assessment_rules_dataframes = {}
    common_params = (start_date, end_date, sliding_window, day_condition, min_sufficiency, max_sufficiency)
    df_weight = pd.read_json('input/demographic_data/cadphr-bodyweight.json')
    df_height = pd.read_json('input/demographic_data/cadphr-bodyheight.json')
    # processing Patient dataframe considering all age and gender filters
    df_patient, all_user_data, no_rows_patient = get_demographic_data(user_data_file, df_weight, df_height, data_select,
                                                              start_age, end_age,
                                                              BMI_lower_limit, BMI_upper_limit)

    processed_dataframes, min_date, max_date = process_common_df(rule_names, df_patient, all_user_data, common_params,
                                                                 trend_window, data_select, start_age, end_age)

    # main function to check which rule is called/passed by user
    for rule_name in rule_names:
        if rule_name == 'ruleset1_V2_1_Fitness_VO2max_Fatness_BMI_rule':
            merged_df = processed_dataframes['vo2max_df'][0]
            min_age, max_age = processed_dataframes['vo2max_df'][1], processed_dataframes['vo2max_df'][2]
            min_bmi, max_bmi = processed_dataframes['vo2max_df'][3], processed_dataframes['vo2max_df'][4]
            no_rows = processed_dataframes['vo2max_df'][5]
            merged_df = assign_assessment_rule_KF_flag(merged_df, ruleset1_V2_1_Fitness_VO2max_Fatness_BMI_rule)
            assessment_rules_dataframes[rule_name] = [merged_df, no_rows]

        elif rule_name == 'ruleset1_V2_1_Heart_Rate_Recovery_rule':
            processed_hrrra_dataframe = processed_dataframes['hrr_df'][0]
            min_age, max_age = processed_dataframes['hrr_df'][1], processed_dataframes['hrr_df'][2]
            min_bmi, max_bmi = processed_dataframes['hrr_df'][3], processed_dataframes['hrr_df'][4]
            no_rows = processed_dataframes['hrr_df'][5]
            processed_hrrra_dataframe['assessment'] = processed_hrrra_dataframe.apply(
                ruleset1_V2_1_Heart_Rate_Recovery_rule, axis=1)
            assessment_rules_dataframes[rule_name] = [processed_hrrra_dataframe, no_rows]

        elif rule_name == 'ruleset2_V2_1_diabetes_risk_assessment_rule':
            processed_drs_dataframe = processed_dataframes['drs_df'][0]
            min_age, max_age = processed_dataframes['drs_df'][1], processed_dataframes['drs_df'][2]
            min_bmi, max_bmi = processed_dataframes['drs_df'][3], processed_dataframes['drs_df'][4]
            no_rows = processed_dataframes['drs_df'][5]
            processed_drs_dataframe['assessment'] = processed_drs_dataframe.apply(
                ruleset2_V2_1_diabetes_risk_assessment_rule, axis=1)
            assessment_rules_dataframes[rule_name] = [processed_drs_dataframe, no_rows]

        elif rule_name == 'ruleset2_V2_1_osa_risk_assessment_rule':
            processed_osa_dataframe = processed_dataframes['osa_df'][0]
            min_age, max_age = processed_dataframes['osa_df'][1], processed_dataframes['osa_df'][2]
            min_bmi, max_bmi = processed_dataframes['osa_df'][3], processed_dataframes['osa_df'][4]
            no_rows = processed_dataframes['osa_df'][5]
            processed_osa_dataframe['assessment'] = processed_osa_dataframe.apply(
                ruleset2_V2_1_osa_risk_assessment_rule, axis=1)
            assessment_rules_dataframes[rule_name] = [processed_osa_dataframe, no_rows]

        elif rule_name == 'ruleset2_V2_1_cad_risk_assessment_rule':
            processed_cad_dataframe = processed_dataframes['cad_df'][0]
            min_age, max_age = processed_dataframes['cad_df'][1], processed_dataframes['cad_df'][2]
            min_bmi, max_bmi = processed_dataframes['cad_df'][3], processed_dataframes['cad_df'][4]
            no_rows = processed_dataframes['cad_df'][5]
            processed_cad_dataframe['assessment'] = processed_cad_dataframe.apply(
                ruleset2_V2_1_cad_risk_assessment_rule, axis=1)
            assessment_rules_dataframes[rule_name] = [processed_cad_dataframe, no_rows]

        elif rule_name == 'ruleset4_V2_1_extreme_Heart_Rate_rule':
            processed_hr_dataframe = processed_dataframes['heartrate_df'][0]
            min_age, max_age = processed_dataframes['heartrate_df'][1], processed_dataframes['heartrate_df'][2]
            min_bmi, max_bmi = processed_dataframes['heartrate_df'][3], processed_dataframes['heartrate_df'][4]
            no_rows = processed_dataframes['heartrate_df'][5]
            processed_hr_dataframe['assessment'] = processed_hr_dataframe.apply(ruleset4_V2_1_extreme_Heart_Rate_rule,
                                                                                axis=1)
            assessment_rules_dataframes[rule_name] = [processed_hr_dataframe, no_rows]
        elif rule_name == 'ruleset4_V2_1_extreme_Heart_Rate_Variability_rule':
            processed_hrv_dataframe = processed_dataframes['hrv_df'][0]
            min_age, max_age = processed_dataframes['hrv_df'][1], processed_dataframes['hrv_df'][2]
            min_bmi, max_bmi = processed_dataframes['hrv_df'][3], processed_dataframes['hrv_df'][4]
            no_rows = processed_dataframes['hrv_df'][5]
            processed_hrv_dataframe['assessment'] = processed_hrv_dataframe.apply(
                ruleset4_V2_1_extreme_Heart_Rate_Variability_rule, axis=1)
            assessment_rules_dataframes[rule_name] = [processed_hrv_dataframe, no_rows]
        elif rule_name == 'ruleset4_V2_1_extreme_PRQ_rule':
            processed_prq_dataframe = processed_dataframes['prq_df'][0]
            min_age, max_age = processed_dataframes['prq_df'][1], processed_dataframes['prq_df'][2]
            min_bmi, max_bmi = processed_dataframes['prq_df'][3], processed_dataframes['prq_df'][4]
            no_rows = processed_dataframes['prq_df'][5]
            processed_prq_dataframe['assessment'] = processed_prq_dataframe.apply(ruleset4_V2_1_extreme_PRQ_rule,
                                                                                  axis=1)
            assessment_rules_dataframes[rule_name] = [processed_prq_dataframe, no_rows]
        elif rule_name == 'ruleset1_V4_cardio_respiratory_fitness_rule':
            processed_cv_dataframe = processed_dataframes['cvreactivity'][0]
            min_age, max_age = processed_dataframes['cvreactivity'][1], processed_dataframes['cvreactivity'][2]
            min_bmi, max_bmi = processed_dataframes['cvreactivity'][3], processed_dataframes['cvreactivity'][4]
            no_rows = processed_dataframes['cvreactivity'][5]
            processed_cv_dataframe['assessment'] = processed_cv_dataframe.apply(
                ruleset1_V4_cardio_respiratory_fitness_rule, axis=1)
            assessment_rules_dataframes[rule_name] = [processed_cv_dataframe, no_rows]

        elif rule_name == 'ruleset4_V3_1_extreme_SBP_rule':
            processed_sbp_dataframe = processed_dataframes['sbp_df'][0]
            min_age, max_age = processed_dataframes['sbp_df'][1], processed_dataframes['sbp_df'][2]
            min_bmi, max_bmi = processed_dataframes['sbp_df'][3], processed_dataframes['sbp_df'][4]
            no_rows = processed_dataframes['sbp_df'][5]
            processed_sbp_dataframe['assessment'] = processed_sbp_dataframe.apply(ruleset4_V3_1_extreme_SBP_rule,
                                                                                  axis=1)
            assessment_rules_dataframes[rule_name] = [processed_sbp_dataframe, no_rows]
        elif rule_name == 'ruleset4_V3_1_extreme_DBP_rule':
            processed_dbp_dataframe = processed_dataframes['dbp_df'][0]
            min_age, max_age = processed_dataframes['dbp_df'][1], processed_dataframes['dbp_df'][2]
            min_bmi, max_bmi = processed_dataframes['dbp_df'][3], processed_dataframes['dbp_df'][4]
            no_rows = processed_dataframes['dbp_df'][5]
            processed_dbp_dataframe['assessment'] = processed_dbp_dataframe.apply(ruleset4_V3_1_extreme_DBP_rule,
                                                                                  axis=1)
            assessment_rules_dataframes[rule_name] = [processed_dbp_dataframe, no_rows]
        elif rule_name == 'ruleset6_V3_1_combinatorial_hr_hrv_rule':
            merged_hr_hrv = processed_dataframes['hr_hrv_merged'][0]
            min_age, max_age = processed_dataframes['hr_hrv_merged'][1], processed_dataframes['hr_hrv_merged'][2]
            min_bmi, max_bmi = processed_dataframes['hr_hrv_merged'][3], processed_dataframes['hr_hrv_merged'][4]
            no_rows = processed_dataframes['hr_hrv_merged'][5]
            merged_hr_hrv = assign_assessment_rule_KF_flag(merged_hr_hrv, ruleset6_V3_1_combinatorial_hr_hrv_rule)
            assessment_rules_dataframes[rule_name] = [merged_hr_hrv, no_rows]

        elif rule_name == 'ruleset6_V3_1_combinatorial_rules_SBP_DBP_rule':
            merged_sbp_dbp = processed_dataframes['sbp_dbp_merged'][0]
            min_age, max_age = processed_dataframes['sbp_dbp_merged'][1], processed_dataframes['sbp_dbp_merged'][2]
            min_bmi, max_bmi = processed_dataframes['sbp_dbp_merged'][3], processed_dataframes['sbp_dbp_merged'][4]
            no_rows = processed_dataframes['sbp_dbp_merged'][5]
            # Applying the function to each row
            merged_sbp_dbp = assign_assessment_rule_KF_flag(merged_sbp_dbp,
                                                            ruleset6_V3_1_combinatorial_rules_SBP_DBP_rule)
            assessment_rules_dataframes[rule_name] = [merged_sbp_dbp, no_rows]

        elif rule_name == 'ruleset3_V4_emotional_influx_Stress_Disorder_rule':
            merged_stress = processed_dataframes['stress_disorder_df'][0]
            min_age, max_age = processed_dataframes['stress_disorder_df'][1], \
                processed_dataframes['stress_disorder_df'][2]
            min_bmi, max_bmi = processed_dataframes['stress_disorder_df'][3], \
                processed_dataframes['stress_disorder_df'][4]
            no_rows = processed_dataframes['stress_disorder_df'][5]
            # Applying the function to each row
            merged_stress = assign_assessment_rule_KF_flag(merged_stress,
                                                           ruleset3_V4_emotional_flux_Stress_Disorder_rule)
            assessment_rules_dataframes[rule_name] = [merged_stress, no_rows]

        elif rule_name == 'ruleset6_V4_combinatorial_rules_PulsePressure_SBP_rule':
            processed_pulsepressure_dataframe = processed_dataframes['pp_df'][0]
            min_age, max_age = processed_dataframes['pp_df'][1], processed_dataframes['pp_df'][2]
            min_bmi, max_bmi = processed_dataframes['pp_df'][3], processed_dataframes['pp_df'][4]
            no_rows = processed_dataframes['pp_df'][5]
            # Applying the function to each row
            processed_pulsepressure_dataframe = assign_assessment_rule_KF_flag(processed_pulsepressure_dataframe,
                                                                               ruleset6_V4_combinatorial_rules_PulsePressure_SBP_rule)
            assessment_rules_dataframes[rule_name] = [processed_pulsepressure_dataframe, no_rows]

        elif rule_name == 'ruleset3_V4_emotion_influx_anxiety_disorder_predictor_rule':
            processed_einstability_dataframe = processed_dataframes['emotion_instability_df'][0]
            processed_einstability_dataframe = assign_assessment_rule_KF_flag(processed_einstability_dataframe,
                                                                              ruleset3_V4_emotion_influx_anxiety_disorder_predictor_rule)
            min_age, max_age = processed_dataframes['emotion_instability_df'][1], \
                processed_dataframes['emotion_instability_df'][2]
            min_bmi, max_bmi = processed_dataframes['emotion_instability_df'][3], \
                processed_dataframes['emotion_instability_df'][4]
            no_rows = processed_dataframes['emotion_instability_df'][5]
            assessment_rules_dataframes[rule_name] = [processed_einstability_dataframe, no_rows]

        elif rule_name == 'ruleset3_V4_emotion_influx_depression_early_symptom_rule':
            processed_ei_dataframe = processed_dataframes['emotion_inertia_df'][0]
            processed_ei_dataframe = assign_assessment_rule_KF_flag(processed_ei_dataframe,
                                                                    ruleset3_V4_emotion_influx_depression_early_symptom_rule)
            min_age, max_age = processed_dataframes['emotion_inertia_df'][1], \
                processed_dataframes['emotion_inertia_df'][2]
            min_bmi, max_bmi = processed_dataframes['emotion_inertia_df'][3], \
                processed_dataframes['emotion_inertia_df'][4]
            no_rows = processed_dataframes['emotion_inertia_df'][5]
            assessment_rules_dataframes[rule_name] = [processed_ei_dataframe, no_rows]

        elif rule_name == 'ruleset3_V2_1_emotional_variability_rule':
            processed_ev_dataframe = processed_dataframes['emotion_variability_df'][0]
            processed_ev_dataframe = assign_assessment_rule_KF_flag(processed_ev_dataframe,
                                                                    ruleset3_V2_1_emotional_variability_rule)
            min_age, max_age = processed_dataframes['emotion_variability_df'][1], \
                processed_dataframes['emotion_variability_df'][2]
            min_bmi, max_bmi = processed_dataframes['emotion_variability_df'][3], \
                processed_dataframes['emotion_variability_df'][4]
            no_rows = processed_dataframes['emotion_variability_df'][5]
            assessment_rules_dataframes[rule_name] = [processed_ev_dataframe, no_rows]
        elif rule_name == 'ruleset6_V4_combinatorial_rules_target_HR_rule':
            processed_target_hr_dataframe = processed_dataframes['targetHR_df'][0]
            min_age, max_age = processed_dataframes['targetHR_df'][1], processed_dataframes['targetHR_df'][2]
            min_bmi, max_bmi = processed_dataframes['targetHR_df'][3], processed_dataframes['targetHR_df'][4]
            no_rows = processed_dataframes['targetHR_df'][5]
            processed_target_hr_dataframe = assign_assessment_rule_KF_flag(processed_target_hr_dataframe,
                                                                           ruleset6_V4_combinatorial_rules_target_HR_rule)
            assessment_rules_dataframes[rule_name] = [processed_target_hr_dataframe, no_rows]
        elif rule_name == 'ruleset5_V3_1_progression_hrv_rule':
            processed_hrv_dataframe = processed_dataframes['hrv_progression_df'][0]
            min_age, max_age = processed_dataframes['hrv_progression_df'][1], \
                processed_dataframes['hrv_progression_df'][2]
            min_bmi, max_bmi = processed_dataframes['hrv_progression_df'][3], \
                processed_dataframes['hrv_progression_df'][4]
            no_rows = processed_dataframes['hrv_progression_df'][5]
            processed_hrv_dataframe['assessment'] = processed_hrv_dataframe.apply(ruleset5_V3_1_progression_hrv_rule,
                                                                                  axis=1)
            assessment_rules_dataframes[rule_name] = [processed_hrv_dataframe, no_rows]
        elif rule_name == 'ruleset5_V3_1_progression_rhr_rule':
            processed_hr_dataframe = processed_dataframes['rhr_progression_df'][0]
            min_age, max_age = processed_dataframes['rhr_progression_df'][1], \
                processed_dataframes['rhr_progression_df'][2]
            min_bmi, max_bmi = processed_dataframes['rhr_progression_df'][3], \
                processed_dataframes['rhr_progression_df'][4]
            no_rows = processed_dataframes['rhr_progression_df'][5]
            processed_hr_dataframe['assessment'] = processed_hr_dataframe.apply(ruleset5_V3_1_progression_rhr_rule,
                                                                                axis=1)
            assessment_rules_dataframes[rule_name] = [processed_hr_dataframe, no_rows]
        elif rule_name == 'ruleset5_V3_1_progression_sbp_rule':
            processed_sbp_dataframe = processed_dataframes['sbp_progression_df'][0]
            min_age, max_age = processed_dataframes['rhr_progression_df'][1], \
                processed_dataframes['rhr_progression_df'][2]
            min_bmi, max_bmi = processed_dataframes['rhr_progression_df'][3], \
                processed_dataframes['rhr_progression_df'][4]
            no_rows = processed_dataframes['rhr_progression_df'][5]
            processed_sbp_dataframe['assessment'] = processed_sbp_dataframe.apply(ruleset5_V3_1_progression_sbp_rule,
                                                                                  axis=1)
            assessment_rules_dataframes[rule_name] = [processed_sbp_dataframe, no_rows]
        elif rule_name == 'ruleset5_V3_1_progression_dbp_rule':
            processed_dbp_dataframe = processed_dataframes['dbp_progression_df'][0]
            min_age, max_age = processed_dataframes['dbp_progression_df'][1], \
                processed_dataframes['dbp_progression_df'][2]
            min_bmi, max_bmi = processed_dataframes['dbp_progression_df'][3], \
                processed_dataframes['dbp_progression_df'][4]
            no_rows = processed_dataframes['dbp_progression_df'][5]
            processed_dbp_dataframe['assessment'] = processed_dbp_dataframe.apply(ruleset5_V3_1_progression_dbp_rule,
                                                                                  axis=1)
            assessment_rules_dataframes[rule_name] = [processed_dbp_dataframe, no_rows]
        elif rule_name == 'ruleset5_V4_progression_prq_rule':
            processed_prq_dataframe = processed_dataframes['prq_progression_df'][0]
            min_age, max_age = processed_dataframes['prq_progression_df'][1], \
                processed_dataframes['prq_progression_df'][2]
            min_bmi, max_bmi = processed_dataframes['prq_progression_df'][3], \
                processed_dataframes['prq_progression_df'][4]
            no_rows = processed_dataframes['prq_progression_df'][5]
            processed_prq_dataframe = processed_prq_dataframe.apply(
                lambda x: extract_format_find_trend(x, trend_window), axis=1)
            processed_prq_dataframe['assessment'] = processed_prq_dataframe.apply(ruleset5_V4_progression_prq_rule,
                                                                                  axis=1)
            assessment_rules_dataframes[rule_name] = [processed_prq_dataframe, no_rows]
        elif rule_name == 'ruleset5_V4_progression_resprate_rule':
            processed_rr_dataframe = processed_dataframes['resprate_progression_df'][0]
            min_age, max_age = processed_dataframes['resprate_progression_df'][1], \
                processed_dataframes['resprate_progression_df'][2]
            min_bmi, max_bmi = processed_dataframes['resprate_progression_df'][3], \
                processed_dataframes['resprate_progression_df'][4]
            no_rows = processed_dataframes['resprate_progression_df'][5]
            processed_rr_dataframe['assessment'] = processed_rr_dataframe.apply(
                ruleset5_V4_progression_resprate_rule,
                axis=1)
            assessment_rules_dataframes[rule_name] = [processed_rr_dataframe, no_rows]
        elif rule_name == 'ruleset2_V4_risk_clinical_risk_score_rule':
            merged_df = processed_dataframes['cad_heartage_df'][0]
            min_age, max_age = processed_dataframes['cad_heartage_df'][1], processed_dataframes['cad_heartage_df'][2]
            min_bmi, max_bmi = processed_dataframes['cad_heartage_df'][3], processed_dataframes['cad_heartage_df'][4]
            no_rows = processed_dataframes['cad_heartage_df'][5]
            merged_df['assessment'] = merged_df.apply(ruleset2_V4_risk_clinical_risk_score_rule, axis=1)
            assessment_rules_dataframes[rule_name] = [merged_df, no_rows]
        elif rule_name == 'ruleset6_V4_combinatorial_rules_cardiac_power_index_rule':
            df_cardiac_index = processed_dataframes['cpi_df'][0]
            min_age, max_age = processed_dataframes['cpi_df'][1], processed_dataframes['cpi_df'][2]
            min_bmi, max_bmi = processed_dataframes['cpi_df'][3], processed_dataframes['cpi_df'][4]
            no_rows = processed_dataframes['cpi_df'][5]
            df_cardiac_index = assign_assessment_rule_KF_flag(df_cardiac_index,
                                                              ruleset6_V4_combinatorial_rules_cardiac_power_index_rule)
            assessment_rules_dataframes[rule_name] = [df_cardiac_index, no_rows]
    no_rows_final = no_rows or no_rows_patient
    return assessment_rules_dataframes, min_date, max_date, min_age, max_age, min_bmi, max_bmi, no_rows_final
